#!/usr/bin/env python3
"""
run_visualizer_drag.py - Interactive 3D articulated object viewer using Viser.

Loads object.json + PLY meshes from the inference output, displays them in a
web-based 3D viewer with direct drag interaction on mesh parts.

Architecture:
    - Viser server (port 8080): 3D rendering, mesh display, click-to-select
    - Side-channel WebSocket (port 8081): Real-time drag events from browser JS
    - Injected JavaScript: Transparent overlay for drag capture

Usage:
    conda activate 4yp
    cd D:\\4YP\\singapo\\Viser_trial
    python run_visualizer_drag.py --object_dir output/0

Then open http://localhost:8080 in your browser.

Controls:
    - Left-drag on viewport: orbit camera (when no movable part is selected)
    - Scroll: zoom
    - Click on a part: select it (highlighted, info shown)
    - Drag a selected movable part: articulate the joint
    - Click without dragging / press Escape: deselect, orbit restored
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

# Side-channel WebSocket for drag events
import websockets
import websockets.asyncio.server


# ────────────────────────────────────────────────────────────────────
# Articulation utilities (adapted from render_iso_seq_auto.py)
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
    """
    Build global 4x4 transforms for all nodes.
    joint_params: dict mapping node_id -> t value in [0, 1]
    """
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
    """Per-client interaction state for direct drag."""
    selected_id: int | None = None
    dragging: bool = False
    drag_start_screen: tuple | None = None   # (x, y) normalized [0,1]
    drag_start_param: float = 0.0
    client: object = None  # viser.ClientHandle


# ────────────────────────────────────────────────────────────────────
# Side-channel WebSocket server for drag events
# ────────────────────────────────────────────────────────────────────

class DragWebSocketServer:
    """Receives real-time drag events from injected browser JavaScript."""

    def __init__(self, port: int, viewer):
        self.port = port
        self.viewer = viewer
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

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
            await asyncio.Future()  # run forever

    async def _handler(self, websocket):
        """Handle one WebSocket connection from a browser tab."""
        client_id = None
        try:
            async for raw in websocket:
                data = json.loads(raw)
                msg_type = data.get("type")
                if msg_type == "identify":
                    client_id = data.get("client_id")
                    continue
                if client_id is None:
                    continue
                if msg_type == "drag_start":
                    self.viewer._on_drag_start(client_id, data)
                elif msg_type == "drag_move":
                    self.viewer._on_drag_move(client_id, data)
                elif msg_type == "drag_end":
                    self.viewer._on_drag_end(client_id, data)
                elif msg_type == "deselect":
                    self.viewer._on_drag_deselect(client_id)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[DragWS] Handler error: {e}")


# ────────────────────────────────────────────────────────────────────
# JavaScript template for drag overlay
# ────────────────────────────────────────────────────────────────────

DRAG_JS_TEMPLATE = r"""
(function() {
    var WS_PORT = __DRAG_WS_PORT__;
    var CLIENT_ID = __CLIENT_ID__;
    var DRAG_THRESHOLD = 5;

    var ws = null;
    var overlay = null;
    var pointerDown = false;
    var startX = 0, startY = 0;
    var isDragging = false;
    var startNorm = {x: 0.5, y: 0.5};
    var _canvas = null;

    function connectWS() {
        try {
            ws = new WebSocket("ws://" + window.location.hostname + ":" + WS_PORT);
            ws.onopen = function() {
                ws.send(JSON.stringify({type: "identify", client_id: CLIENT_ID}));
            };
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

    function createOverlay() {
        if (overlay) return;
        overlay = document.createElement('div');
        overlay.id = '__viser_drag_overlay__';
        overlay.style.cssText = [
            'position:fixed', 'top:0', 'left:0', 'width:100vw', 'height:100vh',
            'z-index:99999', 'cursor:grab', 'background:transparent',
            'touch-action:none', 'user-select:none'
        ].join(';');
        document.body.appendChild(overlay);

        overlay.addEventListener('pointerdown', onPointerDown);
        overlay.addEventListener('pointermove', onPointerMove);
        overlay.addEventListener('pointerup', onPointerUp);
        overlay.addEventListener('pointercancel', onPointerUp);
        overlay.addEventListener('contextmenu', function(e) { e.preventDefault(); });
    }

    function removeOverlay() {
        if (!overlay) return;
        overlay.removeEventListener('pointerdown', onPointerDown);
        overlay.removeEventListener('pointermove', onPointerMove);
        overlay.removeEventListener('pointerup', onPointerUp);
        overlay.removeEventListener('pointercancel', onPointerUp);
        overlay.remove();
        overlay = null;
        pointerDown = false;
        isDragging = false;
    }

    function onPointerDown(e) {
        if (e.button !== 0) return;
        e.preventDefault();
        overlay.setPointerCapture(e.pointerId);
        pointerDown = true;
        isDragging = false;
        startX = e.clientX;
        startY = e.clientY;
        startNorm = getNorm(e.clientX, e.clientY);
    }

    function onPointerMove(e) {
        if (!pointerDown) return;
        e.preventDefault();
        var dx = e.clientX - startX;
        var dy = e.clientY - startY;

        if (!isDragging) {
            if (Math.sqrt(dx * dx + dy * dy) < DRAG_THRESHOLD) return;
            isDragging = true;
            overlay.style.cursor = 'grabbing';
            sendMsg({
                type: "drag_start",
                screen_x: startNorm.x,
                screen_y: startNorm.y
            });
        }

        var norm = getNorm(e.clientX, e.clientY);
        sendMsg({
            type: "drag_move",
            screen_x: norm.x,
            screen_y: norm.y,
            start_x: startNorm.x,
            start_y: startNorm.y
        });
    }

    function onPointerUp(e) {
        if (!pointerDown) return;
        e.preventDefault();
        try { overlay.releasePointerCapture(e.pointerId); } catch(ex) {}
        pointerDown = false;

        if (isDragging) {
            var norm = getNorm(e.clientX, e.clientY);
            sendMsg({type: "drag_end", screen_x: norm.x, screen_y: norm.y});
            isDragging = false;
            if (overlay) overlay.style.cursor = 'grab';
        } else {
            sendMsg({type: "deselect"});
        }
    }

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && overlay) {
            sendMsg({type: "deselect"});
        }
    });

    window.__enableDragOverlay = function(partId) {
        createOverlay();
    };

    window.__disableDragOverlay = function() {
        removeOverlay();
    };
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
        self.movable_joints = []
        for node in self.nodes:
            joint = node.get("joint", {})
            if joint.get("type", "fixed") not in ("fixed", ""):
                self.movable_joints.append(node["id"])
        print(f"  Movable joints: {self.movable_joints}")

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

        # Precompute joint frames
        self._compute_joint_frames()

        self._setup_scene()
        self._setup_gui()
        self._setup_client_handler()

        # Start side-channel WebSocket server for drag events
        self.drag_ws = DragWebSocketServer(port + 1, self)
        self.drag_ws.start()

    # ── Joint frame precomputation ──────────────────────────────────

    def _compute_joint_frames(self):
        """Precompute joint origin and direction in display (normalized) space."""
        # Build base transform (center + scale)
        base_T = np.eye(4)
        base_T[:3, 3] = -self.center
        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale
        scale_mat[1, 1] = self.scale
        scale_mat[2, 2] = self.scale
        base_T = scale_mat @ base_T

        # Build global transforms at rest pose
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

            # Get parent's global transform
            pid = node["parent"]
            if pid >= 0 and pid in global_T:
                parent_T = global_T[pid]
            else:
                parent_T = base_T

            # Transform joint origin to display space
            origin_h = np.array([*origin, 1.0])
            origin_world = (parent_T @ origin_h)[:3]

            # Transform direction (rotation part only, then normalize)
            R_parent = parent_T[:3, :3]
            dir_world = R_parent @ direction
            dir_norm = np.linalg.norm(dir_world)
            if dir_norm > 1e-6:
                dir_world = dir_world / dir_norm

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
            "/grid",
            width=2.0,
            height=2.0,
            position=(0.0, -0.6, 0.0),
            cell_color=(200, 200, 200),
            plane="xz",
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
                    vertices=vertices,
                    faces=faces,
                    color=color,
                    flat_shading=False,
                    side="double",
                    scale=self.scale,
                    wxyz=wxyz,
                    position=position,
                )

                self._register_click(handle, nid)
                handles.append(handle)

            self.mesh_handles[nid] = handles

    def _register_click(self, handle, nid):
        """Register a click handler that passes the full event."""
        @handle.on_click
        def _(event, _nid=nid):
            self._on_part_clicked(_nid, event)

    # ── Click / selection handling ──────────────────────────────────

    def _on_part_clicked(self, nid, event):
        """Handle clicking on a mesh part (via Viser's mesh.on_click)."""
        client = event.client
        cid = client.client_id
        state = self.client_drag_states.get(cid)
        if state is None:
            return
        if self.animating:
            return

        old_selected = state.selected_id
        is_movable = nid in self.movable_joints

        # Restore old selection color
        if old_selected is not None and old_selected in self.mesh_handles:
            old_color = PALETTE[old_selected % len(PALETTE)]
            for handle in self.mesh_handles[old_selected]:
                handle.color = old_color

        # Set new selection
        state.selected_id = nid

        # Highlight new selection
        if nid in self.mesh_handles:
            for handle in self.mesh_handles[nid]:
                handle.color = HIGHLIGHT_COLOR

        # Update info text
        node = self.id2node[nid]
        joint = node.get("joint", {})
        jtype = joint.get("type", "fixed")
        jr = joint.get("range", [0.0, 0.0])
        info = (
            f"**Part {nid}: {node.get('name', 'unknown')}**\n\n"
            f"- Joint type: `{jtype}`\n"
            f"- Range: [{jr[0]:.1f}, {jr[1]:.1f}]\n"
            f"- Parent: {node.get('parent', -1)}\n"
            f"- Children: {node.get('children', [])}\n"
        )
        if is_movable:
            info += "\n*Drag to articulate. Click empty space or Escape to deselect.*"
        self.info_text.content = info

        # Enable/disable drag overlay
        if is_movable:
            self._send_enable_overlay(client, nid)
        else:
            self._send_disable_overlay(client)

    def _deselect(self, client_id):
        """Deselect the current part for a given client."""
        state = self.client_drag_states.get(client_id)
        if state is None:
            return
        old_selected = state.selected_id
        state.selected_id = None
        state.dragging = False
        state.drag_start_screen = None

        # Restore mesh color
        if old_selected is not None and old_selected in self.mesh_handles:
            old_color = PALETTE[old_selected % len(PALETTE)]
            for handle in self.mesh_handles[old_selected]:
                handle.color = old_color

        # Remove overlay
        if state.client is not None:
            self._send_disable_overlay(state.client)

        self.info_text.content = "*Click on a part to select it*"

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

    def _send_enable_overlay(self, client, part_id):
        """Tell the client's JS to show the drag overlay."""
        client._websock_connection.queue_message(
            RunJavascriptMessage(source=f"window.__enableDragOverlay({part_id})")
        )

    def _send_disable_overlay(self, client):
        """Tell the client's JS to hide the drag overlay."""
        client._websock_connection.queue_message(
            RunJavascriptMessage(source="window.__disableDragOverlay()")
        )

    # ── Drag event handlers (called from DragWebSocketServer) ──────

    def _on_drag_start(self, client_id, data):
        """Called when the user starts dragging (crossed 5px threshold)."""
        state = self.client_drag_states.get(client_id)
        if state is None or state.selected_id is None:
            return
        nid = state.selected_id
        state.dragging = True
        state.drag_start_screen = (data["screen_x"], data["screen_y"])
        state.drag_start_param = self.joint_params.get(nid, 0.0)

    def _on_drag_move(self, client_id, data):
        """Called on each pointer move during drag."""
        state = self.client_drag_states.get(client_id)
        if state is None or not state.dragging or state.selected_id is None:
            return
        if self.animating:
            return

        nid = state.selected_id
        start_xy = (data["start_x"], data["start_y"])
        current_xy = (data["screen_x"], data["screen_y"])

        new_t = self._screen_delta_to_joint_param(
            nid, start_xy, current_xy, state
        )
        self.joint_params[nid] = new_t
        self._update_meshes()

    def _on_drag_end(self, client_id, data):
        """Called when drag finishes (pointer up after dragging)."""
        state = self.client_drag_states.get(client_id)
        if state is None:
            return
        state.dragging = False
        state.drag_start_screen = None

    def _on_drag_deselect(self, client_id):
        """Called when user clicks overlay without dragging, or presses Escape."""
        self._deselect(client_id)

    # ── Screen-delta to joint parameter mapping ─────────────────────

    def _screen_delta_to_joint_param(self, nid, start_xy, current_xy, state):
        """Map a 2D screen-space drag to a joint parameter value.

        Projects the joint axis (prismatic) or rotation tangent (revolute) into
        screen space, then maps the screen drag component along that direction
        to a parameter delta.
        """
        frame = self.joint_frames.get(nid)
        if frame is None:
            return self.joint_params.get(nid, 0.0)

        jtype = frame["joint_type"]
        jr = frame["joint_range"]
        joint_dir = frame["direction_world"]
        joint_origin = frame["origin_world"]

        # Read camera state
        client = state.client
        try:
            cam_pos = np.array(client.camera.position)
            cam_look_at = np.array(client.camera.look_at)
        except Exception:
            return self.joint_params.get(nid, 0.0)

        # View coordinate frame
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

        # Screen delta (normalized coords, Y inverted so up is positive)
        dx = current_xy[0] - start_xy[0]
        dy = -(current_xy[1] - start_xy[1])  # negate: screen Y goes down
        screen_delta = np.array([dx, dy])

        # Camera distance for sensitivity scaling
        cam_dist = np.linalg.norm(cam_pos - joint_origin)
        if cam_dist < 1e-6:
            cam_dist = 1.0

        if jtype == "prismatic":
            # Project joint axis to screen space
            screen_axis = np.array([
                np.dot(joint_dir, right),
                np.dot(joint_dir, up)
            ])
            axis_len = np.linalg.norm(screen_axis)
            if axis_len < 1e-6:
                return self.joint_params.get(nid, 0.0)
            screen_axis = screen_axis / axis_len

            # Project screen delta onto the axis direction
            projection = np.dot(screen_delta, screen_axis)

            # Map: drag across ~40% of screen = full joint range
            # Sensitivity adjusts with zoom (camera distance)
            disp_range = abs(jr[1] - jr[0]) * self.scale
            if disp_range < 1e-8:
                return 0.0
            sensitivity = cam_dist * 2.5
            dt = projection * sensitivity / max(disp_range, 1e-6)

            return float(np.clip(state.drag_start_param + dt, 0.0, 1.0))

        elif jtype == "revolute":
            # Tangent direction: perpendicular to axis in the rotation plane,
            # as seen from the current view direction
            tangent = np.cross(joint_dir, view_dir)
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-6:
                tangent = np.cross(joint_dir, up)
                t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-6:
                return self.joint_params.get(nid, 0.0)
            tangent = tangent / t_norm

            # Project tangent to screen space
            screen_tangent = np.array([
                np.dot(tangent, right),
                np.dot(tangent, up)
            ])
            st_norm = np.linalg.norm(screen_tangent)
            if st_norm < 1e-6:
                return self.joint_params.get(nid, 0.0)
            screen_tangent = screen_tangent / st_norm

            projection = np.dot(screen_delta, screen_tangent)

            # Map: drag across ~40% of screen = full angular range
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
        """Reset all joints to rest state (t=0)."""
        for nid in self.joint_params:
            self.joint_params[nid] = 0.0
        self._update_meshes()

    def _toggle_animation(self):
        """Toggle cyclic animation of all movable joints."""
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
        """Run cyclic animation in a background thread."""
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
        """Export current joint parameters to a JSON file."""
        export_path = os.path.join(self.object_dir, "joint_state.json")
        state = {
            "joint_params": {str(k): v for k, v in self.joint_params.items()},
            "movable_joints": self.movable_joints,
        }
        with open(export_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Exported joint state to {export_path}")

    # ── GUI setup ───────────────────────────────────────────────────

    def _setup_gui(self):
        """Set up the GUI panel."""
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

        with self.server.gui.add_folder("Selected Part"):
            self.info_text = self.server.gui.add_markdown(
                "*Click on a part to select it*"
            )

        with self.server.gui.add_folder("Controls"):
            self.server.gui.add_markdown(
                "**Click** a part to select it.\n"
                "**Drag** a selected movable part to articulate its joint.\n"
                "**Click empty space** or press **Escape** to deselect."
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
        """Set up handler for new client connections."""
        self.server.initial_camera.position = np.array([0.0, 0.3, 1.5])
        self.server.initial_camera.look_at = np.array([0.0, 0.0, 0.0])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            cid = client.client_id
            self.client_drag_states[cid] = _ClientDragState(client=client)
            client.camera.position = np.array([0.0, 0.3, 1.5])
            client.camera.look_at = np.array([0.0, 0.0, 0.0])
            # Inject drag JavaScript
            self._inject_drag_js(client)

        @self.server.on_client_disconnect
        def _(client: viser.ClientHandle):
            self.client_drag_states.pop(client.client_id, None)

    # ── Main loop ───────────────────────────────────────────────────

    def run(self):
        """Start the Viser server and block."""
        print()
        print("=" * 60)
        print(f"Viser viewer running at http://localhost:{self.port}")
        print(f"Drag WebSocket running on port {self.port + 1}")
        print("Open the URL above in your browser to interact.")
        print()
        print("Controls:")
        print("  - Left-drag on viewport: orbit camera")
        print("  - Scroll: zoom")
        print("  - Click a part: select it")
        print("  - Drag a selected movable part: articulate the joint")
        print("  - Click without dragging / Escape: deselect")
        print("  - GUI panel: Reset, Animate, Export buttons")
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
    parser = argparse.ArgumentParser(description="Viser-based articulated object viewer with direct drag")
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
