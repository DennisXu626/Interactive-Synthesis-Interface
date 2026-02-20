#!/usr/bin/env python3
"""
run_visualizer_3.0.py - Interactive 3D articulated object viewer using Viser.

v3.0: Direct drag interaction — no selection step. Just grab a movable part and
drag it. Dragging empty space orbits the camera as normal.

Architecture:
    - Viser server (port 8080): 3D rendering, mesh display
    - Side-channel WebSocket (port 8081): Bidirectional — receives pointer events,
      sends hit-test results back to the browser JS
    - Injected JavaScript: Document-level capture-phase listeners with hit-test
      on pointerdown.  Orbit/zoom always work.  On hit, orbit is canceled and
      drag takes over.

Usage:
    conda activate 4yp
    cd D:\\4YP\\singapo\\Viser_trial
    python run_visualizer_3.0.py --object_dir output/0

Then open http://localhost:8080 in your browser.

Controls:
    - Drag a movable part: articulate the joint directly
    - Drag empty space / non-movable part: orbit camera
    - Scroll: zoom
    - GUI panel: Reset, Animate, Export buttons
"""

import os
import sys
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
import viser
import viser.transforms as tf
from viser._messages import RunJavascriptMessage

import websockets
import websockets.asyncio.server


# ────────────────────────────────────────────────────────────────────
# Articulation utilities
# ────────────────────────────────────────────────────────────────────

def load_object(object_dir):
    """Load object.json and part PLY meshes."""
    json_path = os.path.join(object_dir, "object.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["diffuse_tree"]
    id2node = {n["id"]: n for n in nodes}

    for node in nodes:
        ply_paths = node.get("plys", [])
        meshes = []
        for rel in ply_paths:
            mesh_path = os.path.join(object_dir, rel)
            if not os.path.exists(mesh_path):
                print(f"[WARN] Missing mesh: {mesh_path}")
                continue
            tm = trimesh.load(mesh_path, force="mesh")
            if tm.is_empty:
                print(f"[WARN] Empty mesh: {mesh_path}")
                continue
            meshes.append(tm)
        node["_meshes"] = meshes

    return nodes, id2node


def compute_center_and_scale(nodes, target_size=1.0):
    """Compute bounding box center and scale factor for normalization."""
    mins, maxs = [], []
    for node in nodes:
        for m in node["_meshes"]:
            bmin, bmax = m.bounds
            mins.append(bmin)
            maxs.append(bmax)
    if not mins:
        return np.zeros(3), 1.0

    mins = np.vstack(mins)
    maxs = np.vstack(maxs)
    overall_min = mins.min(axis=0)
    overall_max = maxs.max(axis=0)
    center = (overall_min + overall_max) / 2.0
    max_dim = float((overall_max - overall_min).max())
    scale = 1.0 if max_dim == 0 else target_size / max_dim
    return center, scale


def joint_transform_matrix(node, t):
    """Compute 4x4 joint transform for a given parameter t in [0, 1]."""
    T = np.eye(4)
    joint = node.get("joint", None)
    if joint is None:
        return T

    jtype = joint.get("type", "fixed")
    jr = joint.get("range", [0.0, 0.0])
    axis = joint.get("axis", {})
    direction = np.array(axis.get("direction", [0, 0, 0]), dtype=float)
    origin = np.array(axis.get("origin", [0, 0, 0]), dtype=float)

    if jtype == "fixed":
        return T

    if jtype == "prismatic":
        disp_max = jr[1] - jr[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return T
        d_hat = direction / norm
        T[:3, 3] = t * disp_max * d_hat
        return T

    if jtype == "revolute":
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return T
        d_hat = direction / norm
        angle_deg = jr[0] + t * (jr[1] - jr[0])
        theta = math.radians(angle_deg)
        K = np.array([
            [0, -d_hat[2], d_hat[1]],
            [d_hat[2], 0, -d_hat[0]],
            [-d_hat[1], d_hat[0], 0],
        ])
        R3 = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        T[:3, :3] = R3
        T[:3, 3] = origin - R3 @ origin
        return T

    return T


def build_global_transforms(nodes, id2node, center, scale, joint_params):
    """Build global 4x4 transforms for all nodes."""
    base_transform = np.eye(4)
    base_transform[:3, 3] = -center
    scale_mat = np.eye(4)
    scale_mat[0, 0] = scale
    scale_mat[1, 1] = scale
    scale_mat[2, 2] = scale
    base_transform = scale_mat @ base_transform

    children_map = {}
    root_ids = []
    for node in nodes:
        nid = node["id"]
        pid = node["parent"]
        if pid < 0:
            root_ids.append(nid)
        children_map.setdefault(pid, []).append(nid)

    global_T = {}

    def dfs(nid, parent_T):
        node = id2node[nid]
        t = joint_params.get(nid, 0.0)
        J = joint_transform_matrix(node, t)
        Tn = parent_T @ J
        global_T[nid] = Tn
        for cid in children_map.get(nid, []):
            dfs(cid, Tn)

    for rid in root_ids:
        dfs(rid, base_transform)

    return global_T


def matrix_to_wxyz_position(T, uniform_scale=1.0):
    """Extract quaternion (w,x,y,z) and position from a 4x4 transform matrix."""
    R = T[:3, :3]
    pos = T[:3, 3]
    if abs(uniform_scale) > 1e-8 and abs(uniform_scale - 1.0) > 1e-8:
        R = R / uniform_scale
    so3 = tf.SO3.from_matrix(R)
    return so3.wxyz, pos


# ────────────────────────────────────────────────────────────────────
# Color palette
# ────────────────────────────────────────────────────────────────────

PALETTE = [
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (0, 130, 200),    # blue
    (255, 225, 25),   # yellow
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (188, 143, 143),  # rosy brown
    (128, 128, 0),    # olive
]

HIGHLIGHT_COLOR = (255, 255, 255)


# ────────────────────────────────────────────────────────────────────
# Per-client drag state
# ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _ClientDragState:
    """Per-client interaction state."""
    drag_part_id: int | None = None       # part being dragged right now
    dragging: bool = False
    drag_start_screen: tuple | None = None
    drag_start_param: float = 0.0
    client: object = None                 # viser.ClientHandle


# ────────────────────────────────────────────────────────────────────
# Side-channel WebSocket server for drag events (bidirectional)
# ────────────────────────────────────────────────────────────────────

class DragWebSocketServer:
    """Bidirectional WebSocket: receives pointer events, sends hit-test results."""

    def __init__(self, port: int, viewer):
        self.port = port
        self.viewer = viewer
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        # Map client_id -> websocket connection (for sending responses)
        self._ws_by_client: dict[int, object] = {}

    def start(self):
        self._thread.start()

    def send_to_client(self, client_id: int, data: dict):
        """Send a JSON message to a specific browser client (thread-safe)."""
        ws = self._ws_by_client.get(client_id)
        if ws is None or self._loop is None:
            return
        raw = json.dumps(data)
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._async_send(ws, raw),
        )

    async def _async_send(self, ws, raw: str):
        try:
            await ws.send(raw)
        except Exception:
            pass

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"[DragWS] Server error: {e}")

    async def _serve(self):
        async with websockets.asyncio.server.serve(
            self._handler, "0.0.0.0", self.port
        ):
            print(f"[DragWS] Listening on port {self.port}")
            await asyncio.Future()

    async def _handler(self, websocket):
        client_id = None
        try:
            async for raw in websocket:
                data = json.loads(raw)
                msg_type = data.get("type")
                if msg_type == "identify":
                    client_id = data.get("client_id")
                    self._ws_by_client[client_id] = websocket
                    continue
                if client_id is None:
                    continue
                if msg_type == "hit_test":
                    self.viewer._on_hit_test(client_id, data)
                elif msg_type == "drag_start":
                    self.viewer._on_drag_start(client_id, data)
                elif msg_type == "drag_move":
                    self.viewer._on_drag_move(client_id, data)
                elif msg_type == "drag_end":
                    self.viewer._on_drag_end(client_id, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[DragWS] Handler error: {e}")
        finally:
            if client_id is not None:
                self._ws_by_client.pop(client_id, None)


# ────────────────────────────────────────────────────────────────────
# JavaScript template: document-level capture listeners, no overlay
#
# Design: Events always reach the canvas by default (orbit/zoom work
# naturally).  On left-button pointerdown we send a hit_test to Python.
# While waiting, orbit may start — this is intentional and the ~10 ms of
# rotation is imperceptible.  When the hit_result arrives:
#   HIT  → cancel orbit with a synthetic pointerup on the canvas, enter
#          drag mode where document capture listeners call stopPropagation
#          to keep Three.js from seeing subsequent move/up events.
#   MISS → do nothing, orbit continues untouched.
# ────────────────────────────────────────────────────────────────────

DRAG_JS_TEMPLATE = r"""
(function() {
    var WS_PORT = __DRAG_WS_PORT__;
    var CLIENT_ID = __CLIENT_ID__;

    var ws = null;
    var _canvas = null;

    // Pointer / drag state
    var mode = 'idle';           // 'idle' | 'pending' | 'drag'
    var savedPointerId = 0;
    var startX = 0, startY = 0;
    var startNorm = {x: 0.5, y: 0.5};
    var lastClientX = 0, lastClientY = 0;
    var hitPartId = null;

    // ── WebSocket ──

    function connectWS() {
        try {
            ws = new WebSocket("ws://" + window.location.hostname + ":" + WS_PORT);
            ws.onopen = function() {
                ws.send(JSON.stringify({type: "identify", client_id: CLIENT_ID}));
            };
            ws.onmessage = onWsMessage;
            ws.onclose = function() { setTimeout(connectWS, 2000); };
            ws.onerror = function() {};
        } catch(e) {
            setTimeout(connectWS, 2000);
        }
    }
    connectWS();

    function sendMsg(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(obj));
        }
    }

    // ── Canvas helpers ──

    function getCanvas() {
        if (_canvas && _canvas.isConnected) return _canvas;
        _canvas = document.querySelector('canvas');
        return _canvas;
    }

    function getNorm(cx, cy) {
        var c = getCanvas();
        if (!c) return {x: 0.5, y: 0.5};
        var r = c.getBoundingClientRect();
        return {
            x: (cx - r.left) / Math.max(r.width, 1),
            y: (cy - r.top) / Math.max(r.height, 1)
        };
    }

    function isOnCanvas(e) {
        var c = getCanvas();
        if (!c) return false;
        var r = c.getBoundingClientRect();
        return e.clientX >= r.left && e.clientX <= r.right &&
               e.clientY >= r.top  && e.clientY <= r.bottom;
    }

    // ── Document-level capture-phase listeners ──
    // These fire BEFORE any target/bubble listeners (including Three.js).
    // During 'idle'/'pending' we do NOT stop propagation → orbit works.
    // During 'drag' we stop propagation → Three.js sees nothing.

    document.addEventListener('pointerdown', function(e) {
        if (e.button !== 0) return;          // left button only
        if (mode !== 'idle') return;
        if (!isOnCanvas(e)) return;          // ignore GUI panel clicks

        savedPointerId = e.pointerId;
        startX = e.clientX;
        startY = e.clientY;
        lastClientX = e.clientX;
        lastClientY = e.clientY;
        startNorm = getNorm(e.clientX, e.clientY);
        mode = 'pending';
        hitPartId = null;

        // Send hit test — do NOT prevent or stop, so orbit starts naturally
        sendMsg({
            type: "hit_test",
            screen_x: startNorm.x,
            screen_y: startNorm.y
        });
    }, {capture: true});

    document.addEventListener('pointermove', function(e) {
        lastClientX = e.clientX;
        lastClientY = e.clientY;

        if (mode === 'drag') {
            // Block Three.js from seeing this event
            e.stopPropagation();
            e.preventDefault();
            var norm = getNorm(e.clientX, e.clientY);
            sendMsg({
                type: "drag_move",
                screen_x: norm.x,
                screen_y: norm.y,
                start_x: startNorm.x,
                start_y: startNorm.y
            });
        }
        // 'pending' and 'idle': don't touch — orbit/hover works normally
    }, {capture: true});

    document.addEventListener('pointerup', function(e) {
        if (mode === 'drag') {
            e.stopPropagation();
            e.preventDefault();
            var norm = getNorm(e.clientX, e.clientY);
            sendMsg({type: "drag_end", screen_x: norm.x, screen_y: norm.y});
            document.body.style.cursor = '';
            mode = 'idle';
            hitPartId = null;
            return;
        }
        if (mode === 'pending') {
            // User released before hit_test returned — just reset.
            // Don't stop propagation so Three.js can clean up.
            mode = 'idle';
        }
    }, {capture: true});

    // ── Hit-test response from Python ──

    function onWsMessage(event) {
        var data;
        try { data = JSON.parse(event.data); } catch(ex) { return; }

        if (data.type === 'hit_result') {
            if (mode !== 'pending') return;     // too late (user released)

            if (data.part_id !== null) {
                // HIT a movable part

                // If user already moved a lot, they are orbiting — don't hijack
                var dx = lastClientX - startX;
                var dy = lastClientY - startY;
                if (Math.sqrt(dx * dx + dy * dy) > 30) {
                    mode = 'idle';
                    return;
                }

                // Cancel the orbit that Three.js may have started: send a
                // synthetic pointerup on the canvas.  dispatchEvent is
                // synchronous, and our capture listener will see mode='pending'
                // so it won't block the event — it reaches Three.js, which
                // ends orbit and releases pointer capture.
                var c = getCanvas();
                if (c) {
                    var cancelUp = new PointerEvent('pointerup', {
                        clientX: lastClientX, clientY: lastClientY,
                        button: 0, buttons: 0,
                        bubbles: true, cancelable: true,
                        pointerId: savedPointerId,
                        pointerType: 'mouse',
                        isPrimary: true,
                        view: window
                    });
                    c.dispatchEvent(cancelUp);
                }

                // NOW enter drag mode (subsequent real events will be stopped
                // by our capture listeners above)
                mode = 'drag';
                hitPartId = data.part_id;
                document.body.style.cursor = 'grabbing';

                sendMsg({
                    type: "drag_start",
                    screen_x: startNorm.x,
                    screen_y: startNorm.y,
                    part_id: data.part_id
                });

                // Replay movement that occurred during pending as a drag_move
                var norm = getNorm(lastClientX, lastClientY);
                if (Math.abs(norm.x - startNorm.x) > 0.001 ||
                    Math.abs(norm.y - startNorm.y) > 0.001) {
                    sendMsg({
                        type: "drag_move",
                        screen_x: norm.x,
                        screen_y: norm.y,
                        start_x: startNorm.x,
                        start_y: startNorm.y
                    });
                }
            } else {
                // MISS — orbit continues, nothing to do
                mode = 'idle';
            }
        }
    }
})();
"""


# ────────────────────────────────────────────────────────────────────
# Main Visualizer
# ────────────────────────────────────────────────────────────────────

class ArticulatedObjectViewer:
    def __init__(self, object_dir, img_path=None, port=8080):
        self.object_dir = object_dir
        self.img_path = img_path
        self.port = port

        # Load object data
        print(f"Loading object from {object_dir}...")
        self.nodes, self.id2node = load_object(object_dir)
        self.center, self.scale = compute_center_and_scale(self.nodes)
        print(f"  Loaded {len(self.nodes)} parts, center={self.center}, scale={self.scale:.4f}")

        # Joint parameters: node_id -> t (0 to 1)
        self.joint_params = {}
        for node in self.nodes:
            self.joint_params[node["id"]] = 0.0

        # Identify movable joints
        self.movable_joints = set()
        for node in self.nodes:
            joint = node.get("joint", {})
            if joint.get("type", "fixed") not in ("fixed", ""):
                self.movable_joints.add(node["id"])
        print(f"  Movable joints: {sorted(self.movable_joints)}")

        # Build movable-ancestor lookup: for each node, the nearest movable
        # ancestor (including itself). Allows dragging a handle to move its
        # parent drawer.
        self.effective_movable = {}   # node_id -> movable_node_id or None
        self._build_movable_lookup()

        # Viser handles storage
        self.mesh_handles = {}          # node_id -> list of MeshHandle
        self.client_drag_states = {}    # client_id -> _ClientDragState
        self.joint_frames = {}          # node_id -> {origin_world, direction_world, ...}

        # Animation state
        self.animating = False
        self.anim_thread = None

        # Create Viser server
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+y")

        self._compute_joint_frames()
        self._setup_scene()
        self._setup_gui()
        self._setup_client_handler()

        # Start bidirectional drag WebSocket server
        self.drag_ws = DragWebSocketServer(port + 1, self)
        self.drag_ws.start()

    # ── Movable ancestor lookup ─────────────────────────────────────

    def _build_movable_lookup(self):
        """For each node, find the nearest movable ancestor (or itself)."""
        for node in self.nodes:
            nid = node["id"]
            # Walk up the tree
            cur = nid
            found = None
            while cur >= 0:
                if cur in self.movable_joints:
                    found = cur
                    break
                parent = self.id2node[cur].get("parent", -1)
                if parent < 0 or parent not in self.id2node:
                    break
                cur = parent
            self.effective_movable[nid] = found

        print(f"  Effective movable map: {self.effective_movable}")

    # ── Joint frame precomputation ──────────────────────────────────

    def _compute_joint_frames(self):
        """Precompute joint origin and direction in display (normalized) space."""
        base_T = np.eye(4)
        base_T[:3, 3] = -self.center
        scale_mat = np.eye(4) * self.scale
        scale_mat[3, 3] = 1.0
        base_T = scale_mat @ base_T

        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params
        )

        for nid in self.movable_joints:
            node = self.id2node[nid]
            joint = node.get("joint", {})
            jtype = joint.get("type", "fixed")
            jr = joint.get("range", [0.0, 0.0])
            axis_info = joint.get("axis", {})
            direction = np.array(axis_info.get("direction", [0, 0, 0]), dtype=float)
            origin = np.array(axis_info.get("origin", [0, 0, 0]), dtype=float)

            pid = node["parent"]
            parent_T = global_T[pid] if (pid >= 0 and pid in global_T) else base_T

            origin_h = np.array([*origin, 1.0])
            origin_world = (parent_T @ origin_h)[:3]

            R_parent = parent_T[:3, :3]
            dir_world = R_parent @ direction
            dn = np.linalg.norm(dir_world)
            if dn > 1e-6:
                dir_world = dir_world / dn

            self.joint_frames[nid] = {
                "origin_world": origin_world,
                "direction_world": dir_world,
                "joint_type": jtype,
                "joint_range": jr,
            }

    # ── Scene setup ─────────────────────────────────────────────────

    def _setup_scene(self):
        """Create all meshes in the Viser scene."""
        self.server.scene.add_grid(
            "/grid", width=2.0, height=2.0,
            position=(0.0, -0.6, 0.0),
            cell_color=(200, 200, 200), plane="xz",
        )

        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params
        )

        for node in self.nodes:
            nid = node["id"]
            color = PALETTE[nid % len(PALETTE)]
            T = global_T[nid]
            wxyz, position = matrix_to_wxyz_position(T, self.scale)

            handles = []
            for mi, tm in enumerate(node["_meshes"]):
                vertices = np.array(tm.vertices, dtype=np.float32)
                faces = np.array(tm.faces, dtype=np.uint32)

                handle = self.server.scene.add_mesh_simple(
                    name=f"/object/part_{nid}/mesh_{mi}",
                    vertices=vertices, faces=faces,
                    color=color, flat_shading=False, side="double",
                    scale=self.scale, wxyz=wxyz, position=position,
                )
                handles.append(handle)

            self.mesh_handles[nid] = handles

    # ── Hit testing (ray-mesh intersection) ─────────────────────────

    def _screen_to_ray(self, screen_x, screen_y, client):
        """Convert normalized screen coords [0,1] to a 3D ray in display space.

        Returns (ray_origin, ray_direction) or (None, None) if camera not ready.
        """
        try:
            cam_pos = np.array(client.camera.position, dtype=float)
            cam_look_at = np.array(client.camera.look_at, dtype=float)
            fov = float(client.camera.fov)        # vertical FOV in radians
            aspect = float(client.camera.aspect)   # width / height
        except Exception:
            return None, None

        forward = cam_look_at - cam_pos
        fd = np.linalg.norm(forward)
        if fd < 1e-8:
            return None, None
        forward = forward / fd

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / rn
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # NDC: (0,0) = top-left, (1,1) = bottom-right  →  [-1,1] range
        ndc_x = (screen_x - 0.5) * 2.0
        ndc_y = (0.5 - screen_y) * 2.0   # invert Y

        half_fov = fov / 2.0
        tan_hf = math.tan(half_fov)
        ray_dir = forward + ndc_x * aspect * tan_hf * right + ndc_y * tan_hf * up
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        return cam_pos, ray_dir

    def _hit_test(self, screen_x, screen_y, client):
        """Test if a screen position hits a movable mesh.

        Returns the effective movable node_id, or None.
        """
        ray_origin, ray_dir = self._screen_to_ray(screen_x, screen_y, client)
        if ray_origin is None:
            return None

        # Build current global transforms (for current joint state)
        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params
        )

        best_nid = None
        best_dist = float("inf")

        for node in self.nodes:
            nid = node["id"]
            # Skip nodes with no effective movable ancestor
            if self.effective_movable.get(nid) is None:
                continue

            T = global_T[nid]
            try:
                T_inv = np.linalg.inv(T)
            except np.linalg.LinAlgError:
                continue

            # Transform ray to mesh-local space
            origin_local = (T_inv @ np.array([*ray_origin, 1.0]))[:3]
            dir_local = T_inv[:3, :3] @ ray_dir
            dn = np.linalg.norm(dir_local)
            if dn < 1e-8:
                continue
            dir_local = dir_local / dn

            for tm in node["_meshes"]:
                try:
                    locations, index_ray, index_tri = tm.ray.intersects_location(
                        ray_origins=origin_local.reshape(1, 3),
                        ray_directions=dir_local.reshape(1, 3),
                    )
                except Exception:
                    continue
                if len(locations) > 0:
                    # Distance in display space
                    for loc in locations:
                        # Transform hit point back to display space
                        hit_display = (T @ np.array([*loc, 1.0]))[:3]
                        d = np.linalg.norm(hit_display - ray_origin)
                        if d < best_dist:
                            best_dist = d
                            best_nid = self.effective_movable[nid]

        return best_nid

    # ── JavaScript injection ────────────────────────────────────────

    def _inject_drag_js(self, client):
        """Send the drag-handling JavaScript to a specific client."""
        js = DRAG_JS_TEMPLATE.replace(
            "__DRAG_WS_PORT__", str(self.port + 1)
        ).replace(
            "__CLIENT_ID__", str(client.client_id)
        )
        client._websock_connection.queue_message(
            RunJavascriptMessage(source=js)
        )

    # ── Drag event handlers (called from DragWebSocketServer) ──────

    def _on_hit_test(self, client_id, data):
        """Handle hit_test request: raycast and send result back to JS."""
        state = self.client_drag_states.get(client_id)
        if state is None:
            self.drag_ws.send_to_client(client_id, {
                "type": "hit_result", "part_id": None
            })
            return

        if self.animating:
            self.drag_ws.send_to_client(client_id, {
                "type": "hit_result", "part_id": None
            })
            return

        part_id = self._hit_test(data["screen_x"], data["screen_y"], state.client)

        self.drag_ws.send_to_client(client_id, {
            "type": "hit_result",
            "part_id": part_id,
        })

    def _on_drag_start(self, client_id, data):
        """Called when JS confirms drag on a movable part."""
        state = self.client_drag_states.get(client_id)
        if state is None:
            return
        nid = data.get("part_id")
        if nid is None or nid not in self.movable_joints:
            return

        state.drag_part_id = nid
        state.dragging = True
        state.drag_start_screen = (data["screen_x"], data["screen_y"])
        state.drag_start_param = self.joint_params.get(nid, 0.0)

        # Highlight the part being dragged
        if nid in self.mesh_handles:
            for handle in self.mesh_handles[nid]:
                handle.color = HIGHLIGHT_COLOR

        # Show info
        node = self.id2node[nid]
        joint = node.get("joint", {})
        jtype = joint.get("type", "fixed")
        jr = joint.get("range", [0.0, 0.0])
        self.info_text.content = (
            f"**Dragging Part {nid}: {node.get('name', 'unknown')}**\n\n"
            f"- Joint: `{jtype}` [{jr[0]:.1f}, {jr[1]:.1f}]"
        )

    def _on_drag_move(self, client_id, data):
        """Called on each pointer move during drag."""
        state = self.client_drag_states.get(client_id)
        if state is None or not state.dragging or state.drag_part_id is None:
            return
        if self.animating:
            return

        nid = state.drag_part_id
        start_xy = (data["start_x"], data["start_y"])
        current_xy = (data["screen_x"], data["screen_y"])

        new_t = self._screen_delta_to_joint_param(nid, start_xy, current_xy, state)
        self.joint_params[nid] = new_t
        self._update_meshes()

    def _on_drag_end(self, client_id, data):
        """Called when drag finishes."""
        state = self.client_drag_states.get(client_id)
        if state is None:
            return

        # Unhighlight
        nid = state.drag_part_id
        if nid is not None and nid in self.mesh_handles:
            color = PALETTE[nid % len(PALETTE)]
            for handle in self.mesh_handles[nid]:
                handle.color = color

        state.drag_part_id = None
        state.dragging = False
        state.drag_start_screen = None
        self.info_text.content = "*Drag a part to articulate it*"

    # ── Screen-delta to joint parameter mapping ─────────────────────

    def _screen_delta_to_joint_param(self, nid, start_xy, current_xy, state):
        """Map a 2D screen-space drag to a joint parameter value."""
        frame = self.joint_frames.get(nid)
        if frame is None:
            return self.joint_params.get(nid, 0.0)

        jtype = frame["joint_type"]
        jr = frame["joint_range"]
        joint_dir = frame["direction_world"]
        joint_origin = frame["origin_world"]

        client = state.client
        try:
            cam_pos = np.array(client.camera.position)
            cam_look_at = np.array(client.camera.look_at)
        except Exception:
            return self.joint_params.get(nid, 0.0)

        view_dir = cam_look_at - cam_pos
        vd_norm = np.linalg.norm(view_dir)
        if vd_norm < 1e-8:
            return self.joint_params.get(nid, 0.0)
        view_dir = view_dir / vd_norm

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / r_norm
        up = np.cross(right, view_dir)
        up = up / np.linalg.norm(up)

        dx = current_xy[0] - start_xy[0]
        dy = -(current_xy[1] - start_xy[1])
        screen_delta = np.array([dx, dy])

        cam_dist = np.linalg.norm(cam_pos - joint_origin)
        if cam_dist < 1e-6:
            cam_dist = 1.0

        if jtype == "prismatic":
            screen_axis = np.array([
                np.dot(joint_dir, right),
                np.dot(joint_dir, up)
            ])
            axis_len = np.linalg.norm(screen_axis)
            if axis_len < 1e-6:
                return self.joint_params.get(nid, 0.0)
            screen_axis = screen_axis / axis_len

            projection = np.dot(screen_delta, screen_axis)
            disp_range = abs(jr[1] - jr[0]) * self.scale
            if disp_range < 1e-8:
                return 0.0
            sensitivity = cam_dist * 2.5
            dt = projection * sensitivity / max(disp_range, 1e-6)
            return float(np.clip(state.drag_start_param + dt, 0.0, 1.0))

        elif jtype == "revolute":
            tangent = np.cross(joint_dir, view_dir)
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-6:
                tangent = np.cross(joint_dir, up)
                t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-6:
                return self.joint_params.get(nid, 0.0)
            tangent = tangent / t_norm

            screen_tangent = np.array([
                np.dot(tangent, right),
                np.dot(tangent, up)
            ])
            st_norm = np.linalg.norm(screen_tangent)
            if st_norm < 1e-6:
                return self.joint_params.get(nid, 0.0)
            screen_tangent = screen_tangent / st_norm

            projection = np.dot(screen_delta, screen_tangent)
            total_angle = math.radians(abs(jr[1] - jr[0]))
            if total_angle < 1e-6:
                return 0.0
            sensitivity = cam_dist * 2.5
            dt = projection * sensitivity / max(total_angle, 1e-6)
            return float(np.clip(state.drag_start_param + dt, 0.0, 1.0))

        return self.joint_params.get(nid, 0.0)

    # ── Mesh transform updates ──────────────────────────────────────

    def _update_meshes(self):
        """Recompute all global transforms and update mesh positions."""
        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params
        )
        for node in self.nodes:
            nid = node["id"]
            T = global_T[nid]
            wxyz, position = matrix_to_wxyz_position(T, self.scale)
            if nid in self.mesh_handles:
                for handle in self.mesh_handles[nid]:
                    handle.wxyz = wxyz
                    handle.position = position

    # ── Reset / Animation ───────────────────────────────────────────

    def _reset_joints(self):
        """Reset all joints to rest state."""
        for nid in self.joint_params:
            self.joint_params[nid] = 0.0
        self._update_meshes()

    def _toggle_animation(self):
        if self.animating:
            self.animating = False
            if self.anim_thread:
                self.anim_thread.join(timeout=2.0)
                self.anim_thread = None
            self.anim_button.label = "Animate"
        else:
            self.animating = True
            self.anim_button.label = "Stop"
            self.anim_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self.anim_thread.start()

    def _animation_loop(self):
        n_frames = 120
        fps = 30
        frame = 0
        while self.animating:
            phase = frame / max(n_frames - 1, 1)
            t = 0.5 * (1 - math.cos(2 * math.pi * phase))
            for nid in self.movable_joints:
                self.joint_params[nid] = t
            self._update_meshes()
            frame = (frame + 1) % n_frames
            time.sleep(1.0 / fps)

    def _export_state(self):
        export_path = os.path.join(self.object_dir, "joint_state.json")
        state = {
            "joint_params": {str(k): v for k, v in self.joint_params.items()},
            "movable_joints": sorted(self.movable_joints),
        }
        with open(export_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Exported joint state to {export_path}")

    # ── GUI setup ───────────────────────────────────────────────────

    def _setup_gui(self):
        if self.img_path and os.path.exists(self.img_path):
            from PIL import Image as PILImage
            img = PILImage.open(self.img_path)
            img = img.resize((200, 200))
            img_array = np.array(img)
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            with self.server.gui.add_folder("Input Image"):
                self.server.gui.add_image(img_array)

        with self.server.gui.add_folder("Info"):
            self.info_text = self.server.gui.add_markdown(
                "*Drag a part to articulate it*"
            )

        with self.server.gui.add_folder("Controls"):
            self.server.gui.add_markdown(
                "**Drag** a movable part to articulate its joint.\n"
                "**Drag** empty space to orbit the camera.\n"
                "**Scroll** to zoom."
            )
            reset_btn = self.server.gui.add_button("Reset All Joints")
            self.anim_button = self.server.gui.add_button("Animate")
            export_btn = self.server.gui.add_button("Export State")

            @reset_btn.on_click
            def _(_):
                self._reset_joints()

            @self.anim_button.on_click
            def _(_):
                self._toggle_animation()

            @export_btn.on_click
            def _(_):
                self._export_state()

        with self.server.gui.add_folder("Part Hierarchy"):
            hierarchy_text = ""
            for node in self.nodes:
                nid = node["id"]
                name = node.get("name", "?")
                joint = node.get("joint", {})
                jtype = joint.get("type", "fixed")
                parent = node.get("parent", -1)
                indent = "  " if parent >= 0 else ""
                movable = " (movable)" if nid in self.movable_joints else ""
                hierarchy_text += f"{indent}- Part {nid}: **{name}** [{jtype}]{movable}\n"
            self.server.gui.add_markdown(hierarchy_text)

    # ── Client connection handler ───────────────────────────────────

    def _setup_client_handler(self):
        self.server.initial_camera.position = np.array([0.0, 0.3, 1.5])
        self.server.initial_camera.look_at = np.array([0.0, 0.0, 0.0])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            cid = client.client_id
            self.client_drag_states[cid] = _ClientDragState(client=client)
            client.camera.position = np.array([0.0, 0.3, 1.5])
            client.camera.look_at = np.array([0.0, 0.0, 0.0])
            self._inject_drag_js(client)

        @self.server.on_client_disconnect
        def _(client: viser.ClientHandle):
            self.client_drag_states.pop(client.client_id, None)

    # ── Main loop ───────────────────────────────────────────────────

    def run(self):
        print()
        print("=" * 60)
        print(f"Viser viewer running at http://localhost:{self.port}")
        print(f"Drag WebSocket running on port {self.port + 1}")
        print()
        print("Controls:")
        print("  - Drag a movable part: articulate the joint")
        print("  - Drag empty space: orbit camera")
        print("  - Scroll: zoom")
        print("  - GUI panel: Reset, Animate, Export")
        print()
        print("Press Ctrl+C to stop.")
        print("=" * 60)

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.animating = False
            print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description="Viser articulated object viewer v3.0 (direct drag)")
    parser.add_argument(
        "--object_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "output", "0"),
        help="Directory containing object.json and plys/",
    )
    parser.add_argument(
        "--img_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "demo_input.png"),
        help="Path to input image (displayed in GUI)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for the Viser web server (drag WS uses port+1)",
    )
    args = parser.parse_args()

    assert os.path.exists(args.object_dir), f"Object dir not found: {args.object_dir}"
    assert os.path.exists(os.path.join(args.object_dir, "object.json")), \
        f"object.json not found in {args.object_dir}"

    viewer = ArticulatedObjectViewer(
        object_dir=args.object_dir,
        img_path=args.img_path,
        port=args.port,
    )
    viewer.run()


if __name__ == "__main__":
    main()
