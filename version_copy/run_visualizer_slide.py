#!/usr/bin/env python3
"""
run_visualizer.py - Interactive 3D articulated object viewer using Viser.

Loads object.json + PLY meshes from the inference output, displays them in a
web-based 3D viewer with click-to-position interaction.

Usage:
    conda activate 4yp
    cd D:\4YP\singapo\Viser_trial
    python run_visualizer.py --object_dir output/0

Then open http://localhost:8080 in your browser.

Controls:
    - Left-click drag on viewport: orbit camera
    - Scroll: zoom
    - Click on a part: select it (highlighted, info shown)
    - Click a selected movable part again: reposition via ray projection
    - Slider in sidebar: fine-tune the selected joint
    - GUI panel: Reset, Animate, Export buttons
"""

import os
import sys
import json
import math
import time
import argparse
import threading
import dataclasses
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as tf


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
    """Extract quaternion (w,x,y,z) and position from a 4x4 transform matrix.

    If the matrix has a uniform scale baked in (from compute_center_and_scale),
    divide it out before extracting rotation to ensure orthonormality.
    """
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
# Per-client interaction state
# ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _ClientState:
    """Tracks selection and GUI elements for a single browser tab."""
    selected_id: int | None = None
    slider_folder: object = None   # GuiFolderHandle
    slider_handle: object = None   # GuiSliderHandle


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
        self.mesh_handles = {}       # node_id -> list of MeshHandle

        # Per-client interaction state
        self.client_states = {}      # client_id -> _ClientState

        # Animation state
        self.animating = False
        self.anim_thread = None

        # Create server
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+y")

        # Precompute joint geometry in display space
        self.joint_frames = {}       # node_id -> dict with origin/direction in world
        self._compute_joint_frames()

        self._setup_scene()
        self._setup_gui()
        self._setup_client_handler()

    # ── Joint geometry ────────────────────────────────────────────

    def _compute_joint_frames(self):
        """Precompute joint origin and direction in display (normalized) space."""
        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params
        )

        # Build the base transform for fallback
        base_T = np.eye(4)
        base_T[:3, 3] = -self.center
        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale
        scale_mat[1, 1] = self.scale
        scale_mat[2, 2] = self.scale
        base_T = scale_mat @ base_T

        for nid in self.movable_joints:
            node = self.id2node[nid]
            joint = node.get("joint", {})
            jtype = joint.get("type", "fixed")
            jr = joint.get("range", [0.0, 0.0])
            axis_info = joint.get("axis", {})
            direction = np.array(axis_info.get("direction", [0, 0, 0]), dtype=float)
            origin = np.array(axis_info.get("origin", [0, 0, 0]), dtype=float)

            # Transform joint origin to display space via parent's global transform
            origin_h = np.array([*origin, 1.0])
            pid = node["parent"]
            parent_T = global_T[pid] if (pid >= 0 and pid in global_T) else base_T

            origin_world = (parent_T @ origin_h)[:3]

            # Direction in display space
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

    # ── Scene setup ───────────────────────────────────────────────

    def _setup_scene(self):
        """Create all meshes in the Viser scene (no gizmos)."""
        # Add ground grid
        self.server.scene.add_grid(
            "/grid",
            width=2.0,
            height=2.0,
            position=(0.0, -0.6, 0.0),
            cell_color=(200, 200, 200),
            plane="xz",
        )

        # Compute initial global transforms
        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params
        )

        # Add meshes for each part
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

                # Register click handler (receives ray info for repositioning)
                self._register_click(handle, nid)
                handles.append(handle)

            self.mesh_handles[nid] = handles

    def _register_click(self, handle, nid):
        """Register a click handler that passes the full event (with ray info)."""
        @handle.on_click
        def _(event, _nid=nid):
            self._on_part_clicked(_nid, event)

    # ── Click handling ────────────────────────────────────────────

    def _on_part_clicked(self, nid, event):
        """Handle clicking on a part.

        First click on a part: select it (highlight + info + slider).
        Click on the already-selected movable part: reposition via ray projection.
        """
        client = event.client
        cid = client.client_id
        state = self.client_states.get(cid)
        if state is None:
            return

        if self.animating:
            return

        if state.selected_id == nid and nid in self.movable_joints:
            # Already selected + movable → reposition via ray projection
            ray_origin = np.array(event.ray_origin)
            ray_direction = np.array(event.ray_direction)
            t = self._ray_to_joint_param(nid, ray_origin, ray_direction)
            if t is not None:
                self.joint_params[nid] = t
                self._update_meshes()
                # Sync the slider to new value
                if state.slider_handle is not None:
                    state.slider_handle.value = t
        else:
            # Select (or switch selection to) this part
            self._select_part(client, nid)

    def _select_part(self, client, nid):
        """Select a part for a specific client: highlight, info, slider."""
        cid = client.client_id
        state = self.client_states.get(cid)
        if state is None:
            return

        old_selected = state.selected_id
        state.selected_id = nid

        # Restore old selection color
        if old_selected is not None and old_selected in self.mesh_handles:
            old_color = PALETTE[old_selected % len(PALETTE)]
            for handle in self.mesh_handles[old_selected]:
                handle.color = old_color

        # Highlight new selection
        if nid in self.mesh_handles:
            for handle in self.mesh_handles[nid]:
                handle.color = HIGHLIGHT_COLOR

        # Update GUI info
        node = self.id2node[nid]
        joint = node.get("joint", {})
        jtype = joint.get("type", "fixed")
        jr = joint.get("range", [0.0, 0.0])
        movable = nid in self.movable_joints
        info = (
            f"**Part {nid}: {node.get('name', 'unknown')}**\n\n"
            f"- Joint type: `{jtype}`\n"
            f"- Range: [{jr[0]:.1f}, {jr[1]:.1f}]\n"
            f"- Parent: {node.get('parent', -1)}\n"
            f"- Children: {node.get('children', [])}\n"
        )
        if movable:
            info += "\n*Click again to reposition via ray projection.*"
        self.info_text.content = info

        # Create/update contextual slider for movable joints
        self._remove_slider(state)
        if movable:
            self._create_joint_slider(client, state, nid)

    def _create_joint_slider(self, client, state, nid):
        """Create a per-client GUI slider for the selected movable joint."""
        node = self.id2node[nid]
        name = node.get("name", f"part_{nid}")
        joint = node.get("joint", {})
        jtype = joint.get("type", "fixed")

        folder = client.gui.add_folder(f"Joint: {name} ({jtype})")
        slider = client.gui.add_slider(
            label="Position",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=self.joint_params.get(nid, 0.0),
        )

        @slider.on_update
        def _(event, _nid=nid):
            if self.animating:
                return
            self.joint_params[_nid] = slider.value
            self._update_meshes()

        state.slider_folder = folder
        state.slider_handle = slider

    def _remove_slider(self, state):
        """Remove the per-client slider if it exists."""
        if state.slider_folder is not None:
            state.slider_folder.remove()
            state.slider_folder = None
            state.slider_handle = None

    # ── Ray-to-joint projection ───────────────────────────────────

    def _ray_to_joint_param(self, nid, ray_origin, ray_direction):
        """Project a click ray onto a joint's constraint axis and return t in [0,1]."""
        frame = self.joint_frames.get(nid)
        if frame is None:
            return None

        jtype = frame["joint_type"]
        jr = frame["joint_range"]
        O = frame["origin_world"]       # joint origin in display space
        d = frame["direction_world"]    # joint axis direction (unit)
        R = ray_origin
        D = ray_direction / np.linalg.norm(ray_direction)

        if jtype == "prismatic":
            # Closest point on joint axis to the click ray (two-line closest approach).
            # L1(s) = O + s*d,  L2(t) = R + t*D
            w = R - O
            b = np.dot(d, D)
            denom = 1.0 - b * b
            if abs(denom) < 1e-8:
                # Lines are parallel; can't determine position
                return None
            s = (b * np.dot(w, D) - np.dot(w, d)) / denom

            # Map displacement s to joint parameter t ∈ [0, 1].
            disp_range = (jr[1] - jr[0]) * self.scale
            if abs(disp_range) < 1e-8:
                return 0.0
            t = float(np.clip(s / disp_range, 0.0, 1.0))
            return t

        if jtype == "revolute":
            # Intersect ray with the rotation plane (perpendicular to axis, at origin).
            denom = np.dot(D, d)
            if abs(denom) < 1e-8:
                # Ray parallel to rotation plane — can't determine angle
                return None
            t_ray = np.dot(O - R, d) / denom
            P = R + t_ray * D

            # Vector from joint origin to intersection point, projected onto the plane.
            v = P - O
            v_proj = v - np.dot(v, d) * d
            v_norm = np.linalg.norm(v_proj)
            if v_norm < 1e-8:
                return None

            # Need a reference direction in the plane for angle measurement.
            # Use a stable perpendicular to d as the "zero angle" direction.
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(ref, d)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            ref = ref - np.dot(ref, d) * d
            ref = ref / np.linalg.norm(ref)
            perp = np.cross(d, ref)

            angle = math.atan2(np.dot(v_proj, perp), np.dot(v_proj, ref))
            angle_deg = math.degrees(angle)

            # Map angle to t ∈ [0, 1] based on joint range.
            range_span = jr[1] - jr[0]
            if abs(range_span) < 1e-6:
                return 0.0
            t = float(np.clip((angle_deg - jr[0]) / range_span, 0.0, 1.0))
            return t

        return None

    # ── Mesh update ───────────────────────────────────────────────

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

    # ── Reset / animate / export ──────────────────────────────────

    def _reset_joints(self):
        """Reset all joints to rest state (t=0)."""
        for nid in self.joint_params:
            self.joint_params[nid] = 0.0
        self._update_meshes()

        # Reset sliders for all clients
        for state in self.client_states.values():
            if state.slider_handle is not None:
                state.slider_handle.value = 0.0

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

            # Sync sliders for all clients
            for state in self.client_states.values():
                if (
                    state.slider_handle is not None
                    and state.selected_id in self.movable_joints
                ):
                    state.slider_handle.value = t

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

    # ── GUI setup ─────────────────────────────────────────────────

    def _setup_gui(self):
        """Set up the GUI panel."""
        # Input image display
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

        # Part info
        with self.server.gui.add_folder("Selected Part"):
            self.info_text = self.server.gui.add_markdown(
                "*Click on a part to see its info.*\n\n"
                "*Click a movable part twice to reposition it.*"
            )

        # Controls
        with self.server.gui.add_folder("Controls"):
            self.server.gui.add_markdown(
                "**Click** a movable part to select it. "
                "**Click again** to reposition, or use the slider."
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

        # Part hierarchy info
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

    # ── Client handler ────────────────────────────────────────────

    def _setup_client_handler(self):
        """Set up handler for new client connections."""
        self.server.initial_camera.position = np.array([0.0, 0.3, 1.5])
        self.server.initial_camera.look_at = np.array([0.0, 0.0, 0.0])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            # Per-client state
            self.client_states[client.client_id] = _ClientState()

            # Camera
            client.camera.position = np.array([0.0, 0.3, 1.5])
            client.camera.look_at = np.array([0.0, 0.0, 0.0])

        @self.server.on_client_disconnect
        def _(client: viser.ClientHandle):
            self.client_states.pop(client.client_id, None)

    # ── Run ───────────────────────────────────────────────────────

    def run(self):
        """Start the Viser server and block."""
        print()
        print("=" * 60)
        print(f"Viser viewer running at http://localhost:{self.port}")
        print("Open this URL in your browser to interact with the 3D model.")
        print()
        print("Controls:")
        print("  - Left-drag: orbit camera")
        print("  - Scroll: zoom")
        print("  - Click a part: select it")
        print("  - Click a selected movable part: reposition")
        print("  - Slider: fine-tune joint position")
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
    parser = argparse.ArgumentParser(description="Viser-based articulated object viewer")
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
        help="Port for the Viser web server",
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
