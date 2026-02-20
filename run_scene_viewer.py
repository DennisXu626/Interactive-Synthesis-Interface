#!/usr/bin/env python3
"""
run_scene_viewer.py  –  Subtask 2: Interactive room viewer for SceneSmith scenes.

Loads scene_manifest.json produced by parse_scene.py and renders all objects
with full PBR textures in Viser (via GLB). Supports:
  - Full room view with textured objects
  - Click to select/highlight any object
  - Drag to articulate joints (prismatic / revolute) on articulated objects
  - Animate / Reset buttons for articulated joints

Usage:
    python run_scene_viewer.py
    python run_scene_viewer.py --manifest scenesmith_sample/scene_manifest.json --port 8080
"""

import os
import sys
import io
import json
import math
import time
import argparse
import threading
import asyncio
import dataclasses
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tr
import viser
import viser.transforms as vtf
from viser._messages import RunJavascriptMessage

import websockets
import websockets.asyncio.server

# ── SDF Z-up  ↔  Viser Y-up coordinate conversion ───────────────────────────
#
# SDF uses Z-up (X=right, Y=forward, Z=up).
# Viser / GLTF use Y-up (X=right, Y=up, Z=back).
# All GLTF mesh geometry is in Y-up convention.
# We keep FK / raycasting in SDF Z-up and convert only at the Viser interface.
#
#   R_SDF_TO_VISER:  [x,y,z]_sdf  →  [x, z, -y]_viser
#   R_VISER_TO_SDF:  [x,y,z]_viser → [x, -z, y]_sdf   (inverse)

R_SDF_TO_VISER = np.array([[1,  0,  0],
                            [0,  0,  1],
                            [0, -1,  0]], dtype=float)
R_VISER_TO_SDF = R_SDF_TO_VISER.T   # orthogonal, so inverse = transpose


def sdf_pos_to_viser(p):
    return (R_SDF_TO_VISER @ np.array(p, dtype=float)).astype(np.float64)


def sdf_T_to_viser_wxyz_pos(T):
    """Return (wxyz, position) in Viser Y-up from a 4×4 SDF Z-up transform."""
    R_v = R_SDF_TO_VISER @ T[:3, :3] @ R_SDF_TO_VISER.T
    p_v = R_SDF_TO_VISER @ T[:3, 3]
    so3  = vtf.SO3.from_matrix(np.array(R_v, dtype=float))
    return np.array(so3.wxyz, dtype=np.float64), np.array(p_v, dtype=np.float64)


def viser_cam_to_sdf(pos_v, look_at_v):
    """Convert Viser Y-up camera vectors to SDF Z-up."""
    return (R_VISER_TO_SDF @ np.array(pos_v, dtype=float),
            R_VISER_TO_SDF @ np.array(look_at_v, dtype=float))


# ── Generic helpers ───────────────────────────────────────────────────────────

def make_transform_4x4(translation, rotation_matrix):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.array(rotation_matrix, dtype=float)
    T[:3,  3] = np.array(translation, dtype=float)
    return T


def rpy_to_matrix(roll, pitch, yaw):
    """RPY (intrinsic Rz*Ry*Rx, SDF convention) → 3×3 rotation matrix."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [0,  0,  1]])
    Ry = np.array([[cp,  0, sp], [0,   1,  0], [-sp, 0, cp]])
    Rx = np.array([[1,   0,  0], [0,  cr, -sr],[0,  sr,  cr]])
    return Rz @ Ry @ Rx


def visual_offset_to_T(visual_offset):
    """Convert [x,y,z,rx,ry,rz] visual pose offset → 4×4 transform (SDF Z-up)."""
    x, y, z, rx, ry, rz = visual_offset
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rx, ry, rz)
    T[:3,  3] = [x, y, z]
    return T


def angle_axis_to_matrix3(angle_rad, axis):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis) + 1e-12
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def joint_delta_transform(joint, t):
    """
    Compute the 4×4 SDF-frame child-relative-to-parent transform for parameter t ∈ [0,1].

    Convention:
      t = 0  →  joint at rest (angle/displacement = 0, as SDF defines neutral)
      t = 1  →  joint fully open (toward the dominant end of the range)
    """
    lo, hi = joint["limit"]
    axis   = np.array(joint["axis"], dtype=float)
    origin = np.array(joint["origin"], dtype=float)
    jtype  = joint["type"]

    T = np.eye(4)
    if jtype == "prismatic":
        # Drawers always have lo=0 in SceneSmith, so t=0 → disp=0 (closed). ✓
        disp = lo + t * (hi - lo)
        T[:3, 3] = disp * axis / (np.linalg.norm(axis) + 1e-12)
    elif jtype == "revolute":
        # SDF neutral = angle 0.  Map t=0→0, t=1→dominant end.
        if lo <= 0.0 <= hi:
            open_end = lo if abs(lo) > abs(hi) else hi
            angle = t * open_end
        else:
            # 0 outside range — just interpolate through full range
            angle = lo + t * (hi - lo)
        R = angle_axis_to_matrix3(angle, axis)
        T[:3, :3] = R
        T[:3,  3] = origin - R @ origin
    return T


# ── GLTF / GLB loading ───────────────────────────────────────────────────────

def load_glb_bytes(gltf_abs_path):
    """Load a GLTF file and return GLB bytes (preserving PBR textures)."""
    try:
        scene = trimesh.load(gltf_abs_path, process=False)
        buf = io.BytesIO()
        scene.export(buf, file_type="glb")
        return buf.getvalue()
    except Exception as e:
        print(f"  [WARN] Could not load {gltf_abs_path}: {e}")
        return None


def load_trimesh_for_raycast(gltf_abs_path):
    """Load GLTF and return a single concatenated trimesh.Trimesh for raycasting."""
    try:
        scene = trimesh.load(gltf_abs_path, process=False)
        if isinstance(scene, trimesh.Scene):
            meshes = [m for m in scene.geometry.values()
                      if isinstance(m, trimesh.Trimesh) and not m.is_empty]
            if not meshes:
                return None
            return trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            return scene
    except Exception as e:
        print(f"  [WARN] Raycast mesh load failed {gltf_abs_path}: {e}")
    return None


# ── Per-client drag state ────────────────────────────────────────────────────

@dataclasses.dataclass
class _ClientDragState:
    client:           object = None
    drag_obj_name:    str | None = None
    drag_joint_idx:   int | None = None
    dragging:         bool = False
    drag_start_screen: tuple | None = None
    drag_start_param: float = 0.0


# ── WebSocket drag server (bidirectional, same as v3.0) ─────────────────────

class DragWebSocketServer:
    def __init__(self, port, viewer):
        self.port   = port
        self.viewer = viewer
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._loop   = None
        self._ws_by_client = {}

    def start(self):
        self._thread.start()

    def send_to_client(self, client_id, data):
        ws = self._ws_by_client.get(client_id)
        if ws is None or self._loop is None:
            return
        raw = json.dumps(data)
        self._loop.call_soon_threadsafe(asyncio.ensure_future, self._async_send(ws, raw))

    async def _async_send(self, ws, raw):
        try:
            await ws.send(raw)
        except Exception:
            pass

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.asyncio.server.serve(self._handler, "0.0.0.0", self.port):
            print(f"[DragWS] Listening on port {self.port}")
            await asyncio.Future()

    async def _handler(self, websocket):
        client_id = None
        try:
            async for raw in websocket:
                data = json.loads(raw)
                mt = data.get("type")
                if mt == "identify":
                    client_id = data.get("client_id")
                    self._ws_by_client[client_id] = websocket
                elif client_id is None:
                    continue
                elif mt == "hit_test":
                    self.viewer._on_hit_test(client_id, data)
                elif mt == "drag_start":
                    self.viewer._on_drag_start(client_id, data)
                elif mt == "drag_move":
                    self.viewer._on_drag_move(client_id, data)
                elif mt == "drag_end":
                    self.viewer._on_drag_end(client_id, data)
                elif mt == "deselect":
                    self.viewer._on_deselect(client_id, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if client_id is not None:
                self._ws_by_client.pop(client_id, None)


# ── JS injection (same capture-phase pattern as v3.0) ───────────────────────

DRAG_JS_TEMPLATE = r"""
(function() {
    var WS_PORT = __DRAG_WS_PORT__;
    var CLIENT_ID = __CLIENT_ID__;
    var ws = null, _canvas = null;
    var mode = 'idle';
    var savedPointerId = 0;
    var startX = 0, startY = 0;
    var startNorm = {x: 0.5, y: 0.5};
    var lastClientX = 0, lastClientY = 0;

    function connectWS() {
        try {
            ws = new WebSocket("ws://" + window.location.hostname + ":" + WS_PORT);
            ws.onopen = function() {
                ws.send(JSON.stringify({type:"identify", client_id:CLIENT_ID}));
            };
            ws.onmessage = onWsMessage;
            ws.onclose = function() { setTimeout(connectWS, 2000); };
            ws.onerror = function() {};
        } catch(e) { setTimeout(connectWS, 2000); }
    }
    connectWS();

    function sendMsg(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
    }
    function getCanvas() {
        if (_canvas && _canvas.isConnected) return _canvas;
        _canvas = document.querySelector('canvas');
        return _canvas;
    }
    function getNorm(cx, cy) {
        var c = getCanvas(); if (!c) return {x:0.5, y:0.5};
        var r = c.getBoundingClientRect();
        return {x:(cx-r.left)/Math.max(r.width,1), y:(cy-r.top)/Math.max(r.height,1)};
    }
    function isOnCanvas(e) {
        var c = getCanvas(); if (!c) return false;
        var r = c.getBoundingClientRect();
        return e.clientX>=r.left && e.clientX<=r.right && e.clientY>=r.top && e.clientY<=r.bottom;
    }

    document.addEventListener('pointerdown', function(e) {
        if (e.button !== 0 || mode !== 'idle' || !isOnCanvas(e)) return;
        savedPointerId = e.pointerId;
        startX = e.clientX; startY = e.clientY;
        lastClientX = e.clientX; lastClientY = e.clientY;
        startNorm = getNorm(e.clientX, e.clientY);
        mode = 'pending';
        sendMsg({type:"hit_test", screen_x:startNorm.x, screen_y:startNorm.y});
    }, {capture:true});

    document.addEventListener('pointermove', function(e) {
        lastClientX = e.clientX; lastClientY = e.clientY;
        if (mode === 'drag') {
            e.stopPropagation(); e.preventDefault();
            var norm = getNorm(e.clientX, e.clientY);
            sendMsg({type:"drag_move", screen_x:norm.x, screen_y:norm.y,
                     start_x:startNorm.x, start_y:startNorm.y});
        }
    }, {capture:true});

    document.addEventListener('pointerup', function(e) {
        if (mode === 'drag') {
            e.stopPropagation(); e.preventDefault();
            sendMsg({type:"drag_end"});
            document.body.style.cursor = '';
            mode = 'idle'; return;
        }
        if (mode === 'pending') mode = 'idle';
    }, {capture:true});

    function onWsMessage(event) {
        var data; try { data = JSON.parse(event.data); } catch(ex) { return; }
        if (data.type === 'hit_result') {
            if (mode !== 'pending') return;
            if (data.obj_name !== null && data.joint_idx !== null) {
                // Articulated hit — start drag
                var dx = lastClientX - startX, dy = lastClientY - startY;
                if (Math.sqrt(dx*dx+dy*dy) > 30) { mode='idle'; return; }
                var c = getCanvas();
                if (c) {
                    c.dispatchEvent(new PointerEvent('pointerup', {
                        clientX:lastClientX, clientY:lastClientY,
                        button:0, buttons:0, bubbles:true, cancelable:true,
                        pointerId:savedPointerId, pointerType:'mouse',
                        isPrimary:true, view:window
                    }));
                }
                mode = 'drag'; document.body.style.cursor = 'grabbing';
                sendMsg({type:"drag_start", screen_x:startNorm.x, screen_y:startNorm.y,
                         obj_name:data.obj_name, joint_idx:data.joint_idx});
                var norm = getNorm(lastClientX, lastClientY);
                if (Math.abs(norm.x-startNorm.x)>0.001 || Math.abs(norm.y-startNorm.y)>0.001) {
                    sendMsg({type:"drag_move", screen_x:norm.x, screen_y:norm.y,
                             start_x:startNorm.x, start_y:startNorm.y});
                }
            } else if (data.obj_name !== null) {
                // Static object hit — let Viser's native on_click handle selection
                mode = 'idle';
            } else {
                // Empty space — orbit pass-through, no deselect
                mode = 'idle';
            }
        }
    }
})();
"""


# ── Mini panel JS (injected once per client into the MAIN Viser page) ────────
#
# Defines window.__miniPanelAPI = { show(objName), hide() }
# Called from Python via RunJavascriptMessage when an articulated object is selected.

MINI_PANEL_JS_TEMPLATE = r"""
(function() {
    var MINI_PORT = __MINI_PORT__;
    if (window.__miniPanelAPI) return;   // guard: only run once per page

    // ── Move Viser's GUI sidebar to the left ─────────────────────────────────
    var style = document.createElement('style');
    style.textContent = [
        '.viser-panel, [class*="SidebarPanel"] { left:0!important; right:auto!important; }',
        'div[style*="position: fixed"][style*="right: 0px"][style*="z-index:"] {',
        '  left:0!important; right:auto!important; }',
        '#__viser_mini_panel__ iframe { pointer-events:auto; }'
    ].join('\n');
    document.head.appendChild(style);
    // Also try moving any existing fixed panel via DOM scan
    setTimeout(function() {
        document.querySelectorAll('div[style]').forEach(function(el) {
            var s = el.style;
            if (s.position === 'fixed' && s.right === '0px' && s.zIndex &&
                !el.id && el !== panel) {
                s.left = '0px'; s.right = 'auto';
            }
        });
    }, 1500);

    // ── Build the mini-panel (right side, draggable) ─────────────────────────
    var panel = document.createElement('div');
    panel.id = '__viser_mini_panel__';
    panel.style.cssText = [
        'position:fixed', 'right:0', 'top:0',
        'width:360px', 'height:100vh',
        'background:#1a1a1a',
        'border-left:2px solid #555',
        'z-index:9998',
        'display:flex', 'flex-direction:column',
        'box-shadow:-6px 0 20px rgba(0,0,0,0.6)'
    ].join(';');

    var hdr = document.createElement('div');
    hdr.style.cssText = [
        'flex-shrink:0', 'padding:8px 12px',
        'background:#252525', 'border-bottom:1px solid #444',
        'display:flex', 'justify-content:space-between', 'align-items:center',
        'cursor:grab', 'user-select:none'
    ].join(';');
    hdr.innerHTML =
        '<span style="color:#eee;font-family:sans-serif;font-size:13px;font-weight:600;">' +
        '\u2630 Object View</span>' +
        '<span id="__mini_title__" style="color:#999;font-family:sans-serif;font-size:11px;' +
        'max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' +
        'margin-left:8px;"></span>';

    // ── Make the panel draggable by header ────────────────────────────────────
    (function() {
        var dragging = false, offX = 0, offY = 0;
        hdr.addEventListener('pointerdown', function(e) {
            if (e.button !== 0) return;
            dragging = true; hdr.style.cursor = 'grabbing';
            var br = panel.getBoundingClientRect();
            offX = e.clientX - br.left; offY = e.clientY - br.top;
            // Switch from right-anchored to left-anchored so drag works naturally
            panel.style.left = br.left + 'px'; panel.style.right = 'auto';
            hdr.setPointerCapture(e.pointerId);
            e.stopPropagation(); e.preventDefault();
        });
        hdr.addEventListener('pointermove', function(e) {
            if (!dragging) return;
            var nx = Math.max(0, Math.min(e.clientX - offX, window.innerWidth - 60));
            var ny = Math.max(0, Math.min(e.clientY - offY, window.innerHeight - 60));
            panel.style.left = nx + 'px'; panel.style.top = ny + 'px';
            e.stopPropagation(); e.preventDefault();
        });
        hdr.addEventListener('pointerup', function(e) {
            dragging = false; hdr.style.cursor = 'grab';
        });
    })();

    // Drag hint – only shown for articulated objects
    var hint = document.createElement('div');
    hint.id = '__mini_hint__';
    hint.style.cssText = [
        'flex-shrink:0', 'padding:3px 12px',
        'background:#1e1e1e', 'color:#888',
        'font-family:sans-serif', 'font-size:10px',
        'border-bottom:1px solid #333',
        'display:none'    // hidden until Python says object is articulated
    ].join(';');
    hint.textContent =
        'Drag part to articulate  \u00B7  Scroll to zoom  \u00B7  Right-drag to orbit';

    var iframe = document.createElement('iframe');
    iframe.id = '__mini_iframe__';
    iframe.style.cssText = 'flex:1;border:none;background:#111;';
    iframe.src = 'http://' + window.location.hostname + ':' + MINI_PORT;

    panel.appendChild(hdr);
    panel.appendChild(hint);
    panel.appendChild(iframe);
    document.body.appendChild(panel);

    // ── Public API ────────────────────────────────────────────────────────────
    window.__miniPanelAPI = {
        /** Called by Python whenever the selected object changes. */
        updateTitle: function(objName, isArticulated) {
            var t = document.getElementById('__mini_title__');
            if (t) { t.textContent = objName || ''; t.title = objName || ''; }
            var h = document.getElementById('__mini_hint__');
            if (h) h.style.display = (isArticulated ? 'block' : 'none');
        }
    };
})();
"""


# ── Screen-delta → joint parameter (shared by both viewers) ─────────────────

def _compute_joint_param_delta(obj, jnt, start_param, cam_pos, cam_la,
                                start_xy, cur_xy, origin_offset=None):
    """
    Map a 2-D normalised screen drag to a joint parameter change.

    Parameters
    ----------
    obj           : object dict (has world_pose, joints)
    jnt           : joint dict (type, axis, origin, limit)
    start_param   : t at drag start (0..1)
    cam_pos       : camera position in Viser Y-up world space
    cam_la        : camera look-at in Viser Y-up world space
    start_xy      : (x, y) normalised screen at drag start
    cur_xy        : (x, y) normalised screen now
    origin_offset : optional Vec3 subtracted from joint origin
                    (used by the mini-viewer which centres the object)

    Returns
    -------
    float : new joint parameter t in [0, 1]
    """
    vd = np.asarray(cam_la, float) - np.asarray(cam_pos, float)
    vn = np.linalg.norm(vd)
    if vn < 1e-8:
        return start_param
    vd /= vn

    # Camera frame in Viser Y-up world
    right = np.cross(vd, [0, 1, 0])
    rn    = np.linalg.norm(right)
    right = np.array([1, 0, 0]) if rn < 1e-6 else right / rn
    up    = np.cross(right, vd)
    up   /= np.linalg.norm(up)

    dx = cur_xy[0] - start_xy[0]
    dy = -(cur_xy[1] - start_xy[1])
    sd = np.array([dx, dy])

    # Joint axis/origin: SDF Z-up world -> Viser Y-up world
    T_base        = make_transform_4x4(obj["world_pose"]["translation"],
                                       obj["world_pose"]["rotation_matrix"])
    axis_w        = R_SDF_TO_VISER @ (T_base[:3, :3] @ np.array(jnt["axis"], float))
    origin_w_true = R_SDF_TO_VISER @ (T_base @ np.array([*jnt["origin"], 1.0], float))[:3]
    origin_w      = origin_w_true.copy()
    if origin_offset is not None:
        origin_w = origin_w - np.asarray(origin_offset, float)

    cd   = max(np.linalg.norm(np.asarray(cam_pos, float) - origin_w), 1e-6)
    sens = cd * 2.5
    lo, hi = jnt["limit"]

    if jnt["type"] == "prismatic":
        sa = np.array([np.dot(axis_w, right), np.dot(axis_w, up)])
        al = np.linalg.norm(sa)
        if al < 1e-6:
            return start_param
        proj = np.dot(sd, sa / al)
        dr   = abs(hi - lo)
        return float(np.clip(start_param + proj * sens / max(dr, 1e-6), 0, 1))

    elif jnt["type"] == "revolute":
        # tang = cross(axis, vd) always has good screen projection (it is
        # perpendicular to both the axis and the view direction).  Its sign,
        # however, is arbitrary.  We align it with the *physical* opening
        # direction:  cross(axis, free_dir), where free_dir goes from the
        # joint origin toward the body centre, giving the velocity of the
        # door's free end under positive rotation.
        tang = np.cross(axis_w, vd)
        obj_pos_v = R_SDF_TO_VISER @ np.asarray(obj["world_pose"]["translation"], float)
        free_dir  = obj_pos_v - origin_w_true          # joint → body centre
        free_dir -= axis_w * np.dot(free_dir, axis_w)  # project out axis component
        fn = np.linalg.norm(free_dir)
        if fn > 1e-6:
            tang_phys = np.cross(axis_w, free_dir / fn)
            if np.dot(tang, tang_phys) < 0:
                tang = -tang
        tn   = np.linalg.norm(tang)
        if tn < 1e-6:
            tang = np.cross(axis_w, up)
            tn   = np.linalg.norm(tang)
        if tn < 1e-6:
            return start_param
        tang /= tn
        st = np.array([np.dot(tang, right), np.dot(tang, up)])
        sn = np.linalg.norm(st)
        if sn < 1e-6:
            return start_param
        proj = np.dot(sd, st / sn)
        ta   = abs(hi - lo)
        return float(np.clip(start_param + proj * sens / max(ta, 1e-6), 0, 1))

    return start_param


# ── Isolated object mini-viewer (Subtask 3 side panel) ───────────────────────

class ObjectMiniViewer:
    """
    Secondary Viser server rendering one selected object in isolation.
    Shown as an iframe side panel embedded in the main room viewer page.

    Drag interactions articulate joints and sync joint parameters back
    to the main SceneViewer in real time, so the room scene also updates.
    """

    def __init__(self, port: int, scene_viewer):
        self.port  = port
        self.scene = scene_viewer   # reference to parent SceneViewer
        self._lock = threading.Lock()

        self.current_obj_name: str | None = None
        self.glb_handles: dict  = {}      # (obj_name, vname) -> GlbHandle
        self.offset_viser       = np.zeros(3)   # centroid subtracted from positions
        self.client_states: dict = {}
        self._cam_pos  = np.array([0.0, 0.5, 2.0])
        self._cam_look = np.zeros(3)

        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+y")

        # Placeholder shown until an object is focused
        self._placeholder = self.server.gui.add_markdown(
            "*Click an articulated object in the room to inspect it here.*"
        )

        self._setup_client_handler()
        self.drag_ws = DragWebSocketServer(port + 1, self)
        self.drag_ws.start()

    # ── Focus / clear ────────────────────────────────────────────────────────

    def focus_object(self, obj_name: str):
        """Load `obj_name` centred at the origin of the mini viewer."""
        with self._lock:
            if obj_name == self.current_obj_name:
                return
            self._clear_nolock()
            self.current_obj_name = obj_name

        obj = self.scene.objects[obj_name]

        # Compute AABB in Viser Y-up to find centroid
        all_verts = []
        for link in obj["links"]:
            vname = link["name"]
            tm = self.scene.ray_meshes.get((obj_name, vname))
            if tm is None:
                continue
            T_sdf = self.scene._visual_world_T(obj, link)
            R_v = R_SDF_TO_VISER @ T_sdf[:3, :3] @ R_SDF_TO_VISER.T
            p_v = R_SDF_TO_VISER @ T_sdf[:3, 3]
            T_v = np.eye(4); T_v[:3, :3] = R_v; T_v[:3, 3] = p_v
            vh  = np.hstack([tm.vertices, np.ones((len(tm.vertices), 1))])
            all_verts.append((T_v @ vh.T).T[:, :3])

        if not all_verts:
            return

        pts      = np.vstack(all_verts)
        mn, mx   = pts.min(0), pts.max(0)
        centroid = (mn + mx) / 2.0
        size     = mx - mn
        with self._lock:
            self.offset_viser = centroid.copy()

        # Hide placeholder
        try:
            self._placeholder.visible = False
        except Exception:
            pass

        # Load meshes offset so centroid -> origin
        joint_params = self.scene.joint_params.get(obj_name, {})
        for link in obj["links"]:
            vname    = link["name"]
            gltf_rel = link.get("gltf")
            scale    = float(link.get("mesh_scale", 1.0))
            if not gltf_rel:
                continue
            gltf_abs = os.path.join(self.scene.extract_dir, gltf_rel)
            if not os.path.exists(gltf_abs):
                continue
            glb_bytes = load_glb_bytes(gltf_abs)
            if glb_bytes is None:
                continue
            T_sdf = self.scene._visual_world_T(obj, link, joint_params)
            wxyz, pos = sdf_T_to_viser_wxyz_pos(T_sdf)
            pos = pos - centroid
            handle = self.server.scene.add_glb(
                f"/mini/{obj_name}/{vname}",
                glb_data=glb_bytes, wxyz=wxyz, position=pos,
                scale=scale, visible=True,
            )
            with self._lock:
                self.glb_handles[(obj_name, vname)] = handle

        # Position camera to frame the object
        radius   = float(np.max(size)) * 0.65
        cam_pos  = np.array([0.0, float(size[1]) * 0.05, max(radius * 2.2, 0.5)])
        cam_look = np.zeros(3)
        with self._lock:
            self._cam_pos  = cam_pos.copy()
            self._cam_look = cam_look.copy()

        self.server.initial_camera.position = cam_pos
        self.server.initial_camera.look_at  = cam_look
        for state in list(self.client_states.values()):
            try:    state.client.camera.position = cam_pos
            except Exception: pass
            try:    state.client.camera.look_at  = cam_look
            except Exception: pass

    def _clear_nolock(self):
        """Remove all mini scene handles (call with self._lock held)."""
        for handle in self.glb_handles.values():
            try:    handle.remove()
            except Exception: pass
        self.glb_handles.clear()
        self.offset_viser     = np.zeros(3)
        self.current_obj_name = None

    def clear_object(self):
        """Reset mini viewer to placeholder (called on deselect)."""
        with self._lock:
            self._clear_nolock()
        try:
            self._placeholder.visible = True
        except Exception:
            pass

    def _update_transforms(self):
        """Recompute FK transforms for the current object in mini world space."""
        with self._lock:
            obj_name = self.current_obj_name
            offset   = self.offset_viser.copy()
            handles  = dict(self.glb_handles)
        if obj_name is None:
            return
        obj          = self.scene.objects[obj_name]
        joint_params = self.scene.joint_params.get(obj_name, {})
        for link in obj["links"]:
            vname  = link["name"]
            handle = handles.get((obj_name, vname))
            if handle is None:
                continue
            T_sdf = self.scene._visual_world_T(obj, link, joint_params)
            wxyz, pos = sdf_T_to_viser_wxyz_pos(T_sdf)
            pos = pos - offset
            handle.wxyz     = wxyz
            handle.position = pos

    # ── Raycasting ────────────────────────────────────────────────────────────

    def _screen_to_ray(self, sx, sy, client):
        try:
            cp  = np.array(client.camera.position, float)
            cla = np.array(client.camera.look_at,  float)
            fov = float(client.camera.fov)
            asp = float(client.camera.aspect)
        except Exception:
            return None, None
        fwd = cla - cp
        fd  = np.linalg.norm(fwd)
        if fd < 1e-8:
            return None, None
        fwd /= fd
        right = np.cross(fwd, [0, 1, 0])
        rn    = np.linalg.norm(right)
        right = np.array([1, 0, 0]) if rn < 1e-6 else right / rn
        up    = np.cross(right, fwd)
        up   /= np.linalg.norm(up)
        ndc_x = (sx - 0.5) * 2.0
        ndc_y = (0.5 - sy) * 2.0
        thf   = math.tan(fov / 2.0)
        d     = fwd + ndc_x * asp * thf * right + ndc_y * thf * up
        return cp, d / np.linalg.norm(d)

    def _hit_test(self, sx, sy, client):
        """Ray cast in mini-viewer space (meshes are offset by self.offset_viser)."""
        ro, rd = self._screen_to_ray(sx, sy, client)
        if ro is None:
            return None, None
        with self._lock:
            obj_name = self.current_obj_name
            offset   = self.offset_viser.copy()
        if obj_name is None:
            return None, None

        obj          = self.scene.objects[obj_name]
        joint_params = self.scene.joint_params.get(obj_name, {})
        best_dist, best_joint = float("inf"), None
        hit = False

        for link in obj["links"]:
            vname = link["name"]
            tm    = self.scene.ray_meshes.get((obj_name, vname))
            if tm is None:
                continue
            T_sdf = self.scene._visual_world_T(obj, link, joint_params)
            R_v   = R_SDF_TO_VISER @ T_sdf[:3, :3] @ R_SDF_TO_VISER.T
            p_v   = R_SDF_TO_VISER @ T_sdf[:3, 3] - offset
            T     = np.eye(4); T[:3, :3] = R_v; T[:3, 3] = p_v
            try:
                T_inv = np.linalg.inv(T)
            except Exception:
                continue
            ol = (T_inv @ np.array([*ro, 1.0]))[:3]
            dl = T_inv[:3, :3] @ rd
            dn = np.linalg.norm(dl)
            if dn < 1e-8:
                continue
            dl /= dn
            try:
                locs, _, _ = tm.ray.intersects_location(
                    ray_origins=ol.reshape(1, 3),
                    ray_directions=dl.reshape(1, 3))
            except Exception:
                continue
            for loc in locs:
                hw = (T @ np.array([*loc, 1.0]))[:3]
                d  = np.linalg.norm(hw - ro)
                if d < best_dist:
                    best_dist  = d
                    hit        = True
                    fk_link    = link.get("fk_link", link["name"])
                    best_joint = None
                    for i, jnt in enumerate(obj["joints"]):
                        if jnt["child"] == fk_link:
                            best_joint = i
                            break
                    if best_joint is None and obj["is_articulated"]:
                        best_joint = 0

        return (obj_name, best_joint) if hit else (None, None)

    # ── Drag handlers (called by DragWebSocketServer) ─────────────────────────

    def _on_hit_test(self, cid, data):
        state = self.client_states.get(cid)
        if state is None:
            self.drag_ws.send_to_client(cid, {"type": "hit_result", "obj_name": None})
            return
        obj_name, joint_idx = self._hit_test(
            data["screen_x"], data["screen_y"], state.client)
        # Return null for non-articulated objects so orbit is not blocked
        if obj_name and not self.scene.objects.get(obj_name, {}).get("is_articulated", False):
            obj_name, joint_idx = None, None
        self.drag_ws.send_to_client(cid, {
            "type": "hit_result", "obj_name": obj_name, "joint_idx": joint_idx
        })

    def _on_drag_start(self, cid, data):
        state = self.client_states.get(cid)
        if state is None:
            return
        obj_name  = data.get("obj_name")
        joint_idx = data.get("joint_idx")
        if obj_name is None or joint_idx is None:
            return
        state.drag_obj_name     = obj_name
        state.drag_joint_idx    = joint_idx
        state.dragging          = True
        state.drag_start_screen = (data["screen_x"], data["screen_y"])
        state.drag_start_param  = self.scene.joint_params.get(obj_name, {}).get(joint_idx, 0.0)

    def _on_drag_move(self, cid, data):
        state = self.client_states.get(cid)
        if state is None or not state.dragging:
            return
        with self._lock:
            obj_name = self.current_obj_name
            offset   = self.offset_viser.copy()
        if obj_name is None:
            return
        joint_idx = state.drag_joint_idx
        if joint_idx is None:
            return
        obj = self.scene.objects[obj_name]
        jnt = obj["joints"][joint_idx]
        try:
            cp  = np.array(state.client.camera.position, float)
            cla = np.array(state.client.camera.look_at,  float)
        except Exception:
            return
        new_t = _compute_joint_param_delta(
            obj, jnt, state.drag_start_param,
            cp, cla,
            (data["start_x"], data["start_y"]),
            (data["screen_x"], data["screen_y"]),
            origin_offset=offset,
        )
        # Sync to main viewer and update both scenes
        if obj_name in self.scene.joint_params:
            self.scene.joint_params[obj_name][joint_idx] = new_t
        self.scene._update_object_transforms(obj)
        self._update_transforms()
        self.scene._update_info_panel(obj)

    def _on_drag_end(self, cid, data):
        state = self.client_states.get(cid)
        if state is None:
            return
        state.drag_obj_name    = None
        state.drag_joint_idx   = None
        state.dragging          = False

    def _on_deselect(self, cid, data):
        pass  # deselect is handled by the main SceneViewer; no action needed here

    # ── Client management ─────────────────────────────────────────────────────

    def _inject_drag_js(self, client):
        js = (DRAG_JS_TEMPLATE
              .replace("__DRAG_WS_PORT__", str(self.port + 1))
              .replace("__CLIENT_ID__", str(client.client_id)))
        client._websock_connection.queue_message(RunJavascriptMessage(source=js))

    def _setup_client_handler(self):
        @self.server.on_client_connect
        def _(client):
            self.client_states[client.client_id] = _ClientDragState(client=client)
            with self._lock:
                cp  = self._cam_pos.copy()
                cla = self._cam_look.copy()
            try:    client.camera.position = cp
            except Exception: pass
            try:    client.camera.look_at  = cla
            except Exception: pass
            self._inject_drag_js(client)

        @self.server.on_client_disconnect
        def _(client):
            self.client_states.pop(client.client_id, None)


# ── Main scene viewer class ───────────────────────────────────────────────────

class SceneViewer:
    def __init__(self, manifest_path, port=8080, skip_categories=None):
        self.port = port
        self.skip_categories = skip_categories or set()

        print(f"[Viewer] Loading manifest: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.extract_dir = self.manifest["extract_dir"]

        # ── Build data structures ──
        self.objects     = {}  # name -> object dict from manifest
        self.glb_handles = {}  # (obj_name, link_name) -> GlbHandle
        self.ray_meshes  = {}  # (obj_name, link_name) -> trimesh.Trimesh (world-space)
        self.joint_params = {} # obj_name -> {joint_idx -> t ∈ [0,1]}
        self.highlight_boxes = {}  # obj_name -> BoxHandle (wireframe bbox)

        # Selected object (for highlight and GUI info)
        self.selected_obj = None
        self.highlight_on = False  # whether articulated-object highlight is active

        # Per-client drag state
        self.client_states = {}

        self.animating   = False
        self.anim_thread = None
        self.mini_viewer = None   # set after loading (Subtask 3 side panel)
        self.info_md     = None   # set by _setup_gui(); guard early-connect clicks

        # Filter and index objects
        for obj in self.manifest["objects"]:
            if obj["category"] in self.skip_categories:
                continue
            if not obj["links"]:
                continue
            name = obj["name"]
            self.objects[name] = obj
            if obj["is_articulated"]:
                self.joint_params[name] = {i: 0.0 for i in range(len(obj["joints"]))}

        print(f"[Viewer] {len(self.objects)} objects to load "
              f"({sum(1 for o in self.objects.values() if o['is_articulated'])} articulated)")

        # ── Compute room centre from wall visual offsets ──
        # The room_geometry link has visual offsets for each wall; the room
        # model's world_pose gives its world position in SDF Z-up.
        # We infer the room AABB by looking at the perpendicular wall positions.
        self.room_center_viser = self._compute_room_center()

        # ── Viser server ──
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+y")
        cx, cy, cz = self.room_center_viser
        self.server.scene.add_grid(
            "/grid", width=10.0, height=10.0,
            position=(cx, -0.01, cz),     # floor level, room XZ center
            cell_color=(180, 180, 180), plane="xz"
        )

        # ── Load and place all objects ──
        self._load_all_objects()

        # ── GUI ──
        self._setup_gui()

        # ── Client handler ──
        self._setup_client_handler()

        # ── Drag WebSocket ──
        self.drag_ws = DragWebSocketServer(port + 1, self)
        self.drag_ws.start()

        # ── Mini object viewer (Subtask 3 side panel) ──
        # Only created if there are articulated objects to inspect.
        # Runs on port+2 (Viser) and port+3 (drag WS).
        has_art = any(o["is_articulated"] for o in self.objects.values())
        if has_art:
            self.mini_viewer = ObjectMiniViewer(port=port + 2, scene_viewer=self)

    # ── Room centre computation ───────────────────────────────────────────────

    def _compute_room_center(self):
        """
        Infer the room centre in Viser Y-up from the two pairs of perpendicular walls
        stored as visual_offset entries in the room_geometry object.

        Returns [cx, mid_height, cz] in Viser Y-up.
        """
        # Find room_geometry object (may have been skipped)
        room_obj = None
        for obj in self.manifest["objects"]:
            if obj["category"] == "room_geometry":
                room_obj = obj
                break

        if room_obj is None:
            # Fall back to manifest room_frame_translation
            rt = np.array(self.manifest.get("room_frame_translation", [0, 0, 0]))
            return sdf_pos_to_viser(rt).tolist()

        wp = room_obj["world_pose"]
        T_world = make_transform_4x4(wp["translation"], wp["rotation_matrix"])

        xs, zs = [], []
        for link in room_obj["links"]:
            vo = link.get("visual_offset", [0]*6)
            T_off = visual_offset_to_T(vo)
            T = T_world @ T_off
            p_viser = R_SDF_TO_VISER @ T[:3, 3]
            n = link["name"].lower()
            if "north_wall" in n or "south_wall" in n:
                zs.append(float(p_viser[2]))
            elif "east_wall" in n or "west_wall" in n:
                xs.append(float(p_viser[0]))

        # Fall back to room_frame_translation if walls not found
        if not xs or not zs:
            rt = np.array(self.manifest.get("room_frame_translation", [0, 0, 0]))
            return sdf_pos_to_viser(rt).tolist()

        cx  = (min(xs) + max(xs)) / 2.0
        cz  = (min(zs) + max(zs)) / 2.0
        cy  = 1.35  # mid-height of a 2.7m room
        print(f"[Viewer] Room centre (Viser Y-up): X={cx:.2f}  Y={cy:.2f}  Z={cz:.2f}")
        return [cx, cy, cz]

    # ── Loading ──────────────────────────────────────────────────────────────

    def _world_T_for_link(self, obj, link_name, joint_params=None):
        """
        Compute the 4x4 world transform for a given link.
        Uses FK: base_link gets world_pose; child links get parent_T @ joint_delta.
        """
        if joint_params is None:
            joint_params = self.joint_params.get(obj["name"], {})

        world_pose = obj["world_pose"]
        base_T = make_transform_4x4(world_pose["translation"],
                                    world_pose["rotation_matrix"])

        if link_name == obj["base_link"]:
            return base_T

        # FK: find path from base_link to link_name via joints
        # Build parent map: child_link -> (joint_idx, joint)
        parent_of = {}
        for i, jnt in enumerate(obj["joints"]):
            parent_of[jnt["child"]] = (i, jnt, jnt["parent"])

        # Walk from target link up to base
        path = []
        cur = link_name
        visited = set()
        while cur != obj["base_link"] and cur not in visited:
            visited.add(cur)
            if cur not in parent_of:
                break
            idx, jnt, par = parent_of[cur]
            path.append((idx, jnt))
            cur = par
        path.reverse()

        T = base_T
        for idx, jnt in path:
            t = joint_params.get(idx, 0.0)
            # Joint origin/axis are in parent-link frame; delta is in parent frame
            delta_T = joint_delta_transform(jnt, t)
            T = T @ delta_T
        return T

    def _load_all_objects(self):
        n = len(self.objects)
        for i, (name, obj) in enumerate(self.objects.items()):
            sys.stdout.write(f"\r  Loading {i+1}/{n}: {name[:40]:<40}")
            sys.stdout.flush()
            self._load_object(obj)
            if obj["is_articulated"]:
                self._add_highlight_box(obj)
        print()
        print(f"[Viewer] All objects loaded.")

    def _visual_world_T(self, obj, link_dict, joint_params=None):
        """
        Full SDF Z-up world transform for a single visual:
          FK of fk_link  ⊗  visual_offset
        """
        fk_link = link_dict.get("fk_link", link_dict["name"])
        T = self._world_T_for_link(obj, fk_link, joint_params)
        vo = link_dict.get("visual_offset", [0, 0, 0, 0, 0, 0])
        if any(v != 0.0 for v in vo):
            T = T @ visual_offset_to_T(vo)
        return T

    def _load_object(self, obj):
        name = obj["name"]
        for link in obj["links"]:
            vname      = link["name"]          # unique visual ID
            gltf_rel   = link["gltf"]
            mesh_scale = float(link.get("mesh_scale", 1.0))
            if gltf_rel is None:
                continue
            gltf_abs = os.path.join(self.extract_dir, gltf_rel)
            if not os.path.exists(gltf_abs):
                continue

            glb_bytes = load_glb_bytes(gltf_abs)
            if glb_bytes is None:
                continue

            # FK + visual_offset in SDF Z-up → convert to Viser Y-up for add_glb
            T = self._visual_world_T(obj, link)
            wxyz, pos = sdf_T_to_viser_wxyz_pos(T)

            scene_name = f"/scene/{name}/{vname}"
            handle = self.server.scene.add_glb(
                scene_name, glb_data=glb_bytes,
                wxyz=wxyz, position=pos,
                scale=mesh_scale,
                visible=True
            )

            @handle.on_click
            def _on_click(event, _name=name):
                self._select_object(_name)

            self.glb_handles[(name, vname)] = handle

            # Lightweight trimesh for raycasting (GLTF Y-up local coords)
            rm = load_trimesh_for_raycast(gltf_abs)
            if rm is not None:
                if mesh_scale != 1.0:
                    rm = rm.copy()
                    rm.vertices *= mesh_scale
                self.ray_meshes[(name, vname)] = rm

    def _add_highlight_box(self, obj):
        """Add an orange wireframe bounding box around an articulated object (initially hidden)."""
        name = obj["name"]
        # Collect all vertex positions in Viser Y-up world space
        all_verts = []
        for link in obj["links"]:
            vname = link["name"]
            tm = self.ray_meshes.get((name, vname))
            if tm is None:
                continue
            link_dict = next((l for l in obj["links"] if l["name"] == vname), None)
            if link_dict is None:
                continue
            T_sdf = self._visual_world_T(obj, link_dict)
            R_v = R_SDF_TO_VISER @ T_sdf[:3, :3] @ R_SDF_TO_VISER.T
            p_v = R_SDF_TO_VISER @ T_sdf[:3, 3]
            T_vis = np.eye(4); T_vis[:3, :3] = R_v; T_vis[:3, 3] = p_v
            verts_h = np.hstack([tm.vertices, np.ones((len(tm.vertices), 1))])
            all_verts.append((T_vis @ verts_h.T).T[:, :3])

        if not all_verts:
            return
        pts = np.vstack(all_verts)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        dims = (mx - mn).tolist()
        center = ((mn + mx) / 2.0).tolist()

        handle = self.server.scene.add_box(
            f"/highlight/{name}",
            color=(255, 165, 0),           # orange
            dimensions=tuple(max(d, 0.01) for d in dims),
            wireframe=True,
            position=tuple(center),
            visible=False,                  # hidden until button toggled
        )
        self.highlight_boxes[name] = handle

    def _toggle_highlight(self):
        self.highlight_on = not self.highlight_on
        for handle in self.highlight_boxes.values():
            handle.visible = self.highlight_on
        self.highlight_btn.label = "Hide Articulated" if self.highlight_on else "Highlight Articulated"

    # ── Object selection & highlight ─────────────────────────────────────────

    def _select_object(self, obj_name):
        obj = self.objects.get(obj_name)
        if obj is None or obj["category"] == "room_geometry":
            return  # don't select walls / floor
        if obj_name == self.selected_obj:
            return
        self.selected_obj = obj_name
        self._update_info_panel(obj)
        print(f"\n[Select] {obj_name}  articulated={obj['is_articulated']}")

        # Load this object into the mini panel (works for static and articulated)
        if self.mini_viewer is not None:
            self.mini_viewer.focus_object(obj_name)
            is_art = "true" if obj["is_articulated"] else "false"
            js = (f"window.__miniPanelAPI && "
                  f"window.__miniPanelAPI.updateTitle({json.dumps(obj_name)}, {is_art});")
            for state in list(self.client_states.values()):
                try:
                    state.client._websock_connection.queue_message(
                        RunJavascriptMessage(source=js))
                except Exception:
                    pass

    def _on_deselect(self, cid, data):
        """User clicked empty space — clear selection but keep mini panel."""
        self.selected_obj = None
        if self.info_md is not None:
            self.info_md.content = "*Click an object to select it*"
        # Keep mini panel showing the last selected object (don't clear it)

    def _update_info_panel(self, obj):
        name = obj["name"]
        cat  = obj["category"]
        art  = "Yes" if obj["is_articulated"] else "No"
        n_links = len(obj["links"])
        txt = f"**{name}**\n\nCategory: `{cat}`  |  Links: {n_links}  |  Articulated: {art}\n\n"
        if obj["is_articulated"]:
            txt += "**Joints:**\n"
            params = self.joint_params.get(name, {})
            for i, jnt in enumerate(obj["joints"]):
                t = params.get(i, 0.0)
                lo, hi = jnt["limit"]
                val = lo + t * (hi - lo)
                txt += f"- `{jnt['type']}` {jnt['parent']}→{jnt['child']}  "
                txt += f"val={val:.3f}  [{lo:.2f}, {hi:.2f}]\n"
        if self.info_md is not None:
            self.info_md.content = txt

    # ── Hit testing (Python raycasting) ──────────────────────────────────────

    def _screen_to_ray(self, sx, sy, client):
        """Return (ray_origin, ray_dir) in Viser Y-up world space."""
        try:
            cam_pos = np.array(client.camera.position, dtype=float)
            cam_la  = np.array(client.camera.look_at,  dtype=float)
            fov     = float(client.camera.fov)
            aspect  = float(client.camera.aspect)
        except Exception:
            return None, None
        fwd = cam_la - cam_pos
        fd  = np.linalg.norm(fwd)
        if fd < 1e-8:
            return None, None
        fwd /= fd
        right = np.cross(fwd, [0, 1, 0])  # Y-up world
        rn = np.linalg.norm(right)
        right = np.array([1, 0, 0]) if rn < 1e-6 else right / rn
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up)
        ndc_x = (sx - 0.5) * 2.0
        ndc_y = (0.5 - sy) * 2.0
        thf = math.tan(fov / 2.0)
        d = fwd + ndc_x * aspect * thf * right + ndc_y * thf * up
        return cam_pos, d / np.linalg.norm(d)

    def _hit_test(self, sx, sy, client):
        """Return (obj_name, best_joint_idx_or_None) for the frontmost hit.

        Ray and world transforms are all in Viser Y-up space.
        Trimesh local coords are GLTF Y-up = same as Viser Y-up, so consistent.
        """
        ro, rd = self._screen_to_ray(sx, sy, client)
        if ro is None:
            return None, None

        best_obj, best_dist, best_joint = None, float("inf"), None

        for (obj_name, vname), tm in self.ray_meshes.items():
            obj = self.objects[obj_name]
            # Find the link dict for this visual
            link_dict = next((l for l in obj["links"] if l["name"] == vname), None)
            if link_dict is None:
                continue
            fk_link = link_dict.get("fk_link", vname)

            # Build Viser Y-up world transform for this visual (FK + visual_offset)
            T_sdf = self._visual_world_T(obj, link_dict)
            R_v = R_SDF_TO_VISER @ T_sdf[:3, :3] @ R_SDF_TO_VISER.T
            p_v = R_SDF_TO_VISER @ T_sdf[:3, 3]
            T = np.eye(4)
            T[:3, :3] = R_v
            T[:3,  3] = p_v
            try:
                T_inv = np.linalg.inv(T)
            except np.linalg.LinAlgError:
                continue
            ol = (T_inv @ np.array([*ro, 1.0]))[:3]
            dl = T_inv[:3, :3] @ rd
            dn = np.linalg.norm(dl)
            if dn < 1e-8:
                continue
            dl /= dn
            try:
                locs, _, _ = tm.ray.intersects_location(
                    ray_origins=ol.reshape(1, 3), ray_directions=dl.reshape(1, 3))
            except Exception:
                continue
            for loc in locs:
                hit_world = (T @ np.array([*loc, 1.0]))[:3]
                d = np.linalg.norm(hit_world - ro)
                if d < best_dist:
                    best_dist = d
                    best_obj  = obj_name
                    # Find which joint controls this visual's fk_link
                    best_joint = None
                    for i, jnt in enumerate(obj["joints"]):
                        if jnt["child"] == fk_link:
                            best_joint = i
                            break
                    if best_joint is None and obj["is_articulated"]:
                        best_joint = 0

        return best_obj, best_joint

    # ── Drag handlers ─────────────────────────────────────────────────────────

    def _on_hit_test(self, cid, data):
        state = self.client_states.get(cid)
        if state is None or self.animating:
            self.drag_ws.send_to_client(cid, {"type": "hit_result", "obj_name": None})
            return
        obj_name, joint_idx = self._hit_test(data["screen_x"], data["screen_y"], state.client)
        # Room geometry → treat as empty space (null)
        if obj_name and self.objects.get(obj_name, {}).get("category") == "room_geometry":
            obj_name, joint_idx = None, None
        # Non-articulated objects → return obj_name so JS knows it's a real hit
        # (prevents deselect race), but joint_idx stays None so no drag starts
        elif obj_name and not self.objects[obj_name]["is_articulated"]:
            joint_idx = None
        self.drag_ws.send_to_client(cid, {
            "type": "hit_result", "obj_name": obj_name, "joint_idx": joint_idx
        })

    def _on_drag_start(self, cid, data):
        state = self.client_states.get(cid)
        if state is None:
            return
        obj_name  = data.get("obj_name")
        joint_idx = data.get("joint_idx")
        if obj_name is None or joint_idx is None:
            return
        state.drag_obj_name    = obj_name
        state.drag_joint_idx   = joint_idx
        state.dragging         = True
        state.drag_start_screen = (data["screen_x"], data["screen_y"])
        state.drag_start_param  = self.joint_params.get(obj_name, {}).get(joint_idx, 0.0)
        # Select the object
        self._select_object(obj_name)

    def _on_drag_move(self, cid, data):
        state = self.client_states.get(cid)
        if state is None or not state.dragging or self.animating:
            return
        obj_name  = state.drag_obj_name
        joint_idx = state.drag_joint_idx
        if obj_name is None or joint_idx is None:
            return
        obj = self.objects[obj_name]
        jnt = obj["joints"][joint_idx]
        new_t = self._screen_delta_to_joint_param(
            obj, jnt, joint_idx, state,
            (data["start_x"], data["start_y"]),
            (data["screen_x"], data["screen_y"])
        )
        self.joint_params[obj_name][joint_idx] = new_t
        self._update_object_transforms(obj)
        self._update_info_panel(obj)

    def _on_drag_end(self, cid, data):
        state = self.client_states.get(cid)
        if state is None:
            return
        state.drag_obj_name  = None
        state.drag_joint_idx = None
        state.dragging        = False

    def _screen_delta_to_joint_param(self, obj, jnt, joint_idx, state, start_xy, cur_xy):
        """Thin wrapper: reads camera from state.client and calls the module-level helper."""
        client = state.client
        try:
            cp  = np.array(client.camera.position, float)
            cla = np.array(client.camera.look_at,  float)
        except Exception:
            return state.drag_start_param
        return _compute_joint_param_delta(
            obj, jnt, state.drag_start_param, cp, cla, start_xy, cur_xy)

    # ── Transform update (FK) ─────────────────────────────────────────────────

    def _update_object_transforms(self, obj):
        name = obj["name"]
        params = self.joint_params.get(name, {})
        for link in obj["links"]:
            vname = link["name"]
            handle = self.glb_handles.get((name, vname))
            if handle is None:
                continue
            T = self._visual_world_T(obj, link, params)
            wxyz, pos = sdf_T_to_viser_wxyz_pos(T)
            handle.wxyz     = wxyz
            handle.position = pos

    # ── Animation ─────────────────────────────────────────────────────────────

    def _toggle_animation(self):
        if self.animating:
            self.animating = False
            if self.anim_thread:
                self.anim_thread.join(timeout=2.0)
            self.anim_btn.label = "Animate All"
        else:
            self.animating = True
            self.anim_btn.label = "Stop Animation"
            self.anim_thread = threading.Thread(target=self._anim_loop, daemon=True)
            self.anim_thread.start()

    def _anim_loop(self):
        n_frames, fps, frame = 120, 30, 0
        while self.animating:
            t = 0.5 * (1 - math.cos(2 * math.pi * frame / max(n_frames - 1, 1)))
            for name, params in self.joint_params.items():
                for idx in params:
                    params[idx] = t
                self._update_object_transforms(self.objects[name])
            frame = (frame + 1) % n_frames
            time.sleep(1.0 / fps)

    def _reset_all_joints(self):
        for name, params in self.joint_params.items():
            for idx in params:
                params[idx] = 0.0
            self._update_object_transforms(self.objects[name])

    # ── GUI ───────────────────────────────────────────────────────────────────

    def _setup_gui(self):
        with self.server.gui.add_folder("Scene Info"):
            n_art = sum(1 for o in self.objects.values() if o["is_articulated"])
            self.server.gui.add_markdown(
                f"**{len(self.objects)} objects** loaded\n\n"
                f"Articulated: {n_art}  |  Static: {len(self.objects)-n_art}\n\n"
                f"**Drag** a drawer/door to open it.\n"
                f"**Drag** empty space to orbit. **Scroll** to zoom."
            )

        with self.server.gui.add_folder("Selected Object"):
            self.info_md = self.server.gui.add_markdown("*Click an object to select it*")

        with self.server.gui.add_folder("Controls"):
            reset_btn = self.server.gui.add_button("Reset All Joints")
            self.anim_btn = self.server.gui.add_button("Animate All")
            self.highlight_btn = self.server.gui.add_button("Highlight Articulated")

            @reset_btn.on_click
            def _(_): self._reset_all_joints()

            @self.anim_btn.on_click
            def _(_): self._toggle_animation()

            @self.highlight_btn.on_click
            def _(_): self._toggle_highlight()

        with self.server.gui.add_folder("Articulated Objects"):
            txt = ""
            for name, obj in self.objects.items():
                if obj["is_articulated"]:
                    txt += f"- **{name}**\n"
                    for jnt in obj["joints"]:
                        txt += f"  - `{jnt['type']}` {jnt['parent']}→{jnt['child']}\n"
            self.server.gui.add_markdown(txt or "*None*")

    # ── Client handler ────────────────────────────────────────────────────────

    def _setup_client_handler(self):
        # Camera: place viewer outside the room, looking at the computed room centre.
        # Room centre is in Viser Y-up; camera sits ~4.5m back and ~2.5m up.
        cx, cy, cz = self.room_center_viser
        _cam_look = np.array([cx, cy * 0.6, cz])          # look at 60% room height
        _cam_pos  = np.array([cx, cy * 1.8, cz + 4.5])    # ~4.5m in front, elevated

        # Set for "Reset View" button
        self.server.initial_camera.position = _cam_pos
        self.server.initial_camera.look_at  = _cam_look

        @self.server.on_client_connect
        def _(client):
            self.client_states[client.client_id] = _ClientDragState(client=client)
            # camera.position setter reads current position (may not be ready yet);
            # wrap in try/except so look_at always fires even if position setter fails.
            try:
                client.camera.position = _cam_pos
            except Exception:
                pass
            # look_at setter sends SetCameraLookAtMessage(initial=False) which
            # calls cameraControls.setTarget() directly → sets orbit pivot.
            try:
                client.camera.look_at = _cam_look
            except Exception:
                pass
            self._inject_drag_js(client)

        @self.server.on_client_disconnect
        def _(client):
            self.client_states.pop(client.client_id, None)

    def _inject_drag_js(self, client):
        js = (DRAG_JS_TEMPLATE
              .replace("__DRAG_WS_PORT__", str(self.port + 1))
              .replace("__CLIENT_ID__", str(client.client_id)))
        client._websock_connection.queue_message(RunJavascriptMessage(source=js))
        # Inject mini panel JS whenever there are articulated objects.
        # Use self.joint_params (populated before client handler is registered)
        # rather than self.mini_viewer (created afterwards) to avoid the race.
        if self.joint_params:
            mini_js = MINI_PANEL_JS_TEMPLATE.replace(
                "__MINI_PORT__", str(self.port + 2))
            client._websock_connection.queue_message(
                RunJavascriptMessage(source=mini_js))

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        print()
        print("=" * 60)
        print(f"Scene Viewer  →  http://localhost:{self.port}")
        print(f"Drag WebSocket on port {self.port + 1}")
        if self.mini_viewer is not None:
            print(f"Mini Viewer   →  http://localhost:{self.port + 2}  (side panel)")
        print()
        print("  Drag a drawer/door (articulated) to move it")
        print("  Drag empty space to orbit  |  Scroll to zoom")
        print("  Click any object to select and show info")
        print()
        print("Press Ctrl+C to stop.")
        print("=" * 60)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.animating = False
            print("\nShutting down.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive SceneSmith room viewer")
    parser.add_argument("--manifest",
                        default="D:/4YP/singapo/Viser_trial/scenesmith_sample/scene_manifest.json")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--skip", nargs="*", default=["ceiling_mounted"],
                        help="Categories to skip. "
                             "Options: furniture manipuland wall_mounted ceiling_mounted room_geometry. "
                             "Default skips ceiling lights only; room walls are loaded.")
    args = parser.parse_args()

    viewer = SceneViewer(
        manifest_path=args.manifest,
        port=args.port,
        skip_categories=set(args.skip),
    )
    viewer.run()


if __name__ == "__main__":
    main()
