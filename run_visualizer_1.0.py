#!/usr/bin/env python3
"""
run_visualizer.py - Interactive 3D articulated object viewer using Viser.

Loads object.json + PLY meshes from the inference output, displays them in a
web-based 3D viewer with direct drag interaction via transform gizmos.

Usage:
    conda activate 4yp
    cd D:\4YP\singapo\Viser_trial
    python run_visualizer.py --object_dir output/0

Then open http://localhost:8080 in your browser.

Controls:
    - Left-click drag on viewport: orbit camera
    - Scroll: zoom
    - Click on a part: select it (highlighted in GUI)
    - Drag a gizmo (colored ring/arrow on movable joints): articulate the joint
    - GUI panel: Reset, Animate, Export buttons
"""

import os
import sys
import json
import math
import time
import argparse
import threading
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


def align_z_to_direction(direction):
    """
    Compute a rotation quaternion (w,x,y,z) that rotates [0,0,1] to align with the given direction.
    Returns the quaternion as a numpy array.
    """
    d = np.array(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = d / norm

    z = np.array([0.0, 0.0, 1.0])
    dot = np.clip(np.dot(z, d), -1.0, 1.0)

    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:
        # 180 degree rotation around x-axis
        return np.array([0.0, 1.0, 0.0, 0.0])

    axis = np.cross(z, d)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    so3 = tf.SO3.exp(axis * angle)
    return so3.wxyz


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
        self.gizmo_handles = {}      # node_id -> TransformControlsHandle
        self.gizmo_frames = {}       # node_id -> (wxyz_alignment, origin_in_normalized_space)
        self.selected_id = None

        # Animation state
        self.animating = False
        self.anim_thread = None

        # Create server
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+y")

        self._setup_scene()
        self._setup_gui()
        self._setup_client_handler()

    def _setup_scene(self):
        """Create all meshes and gizmos in the Viser scene."""
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

                # Register click handler
                self._register_click(handle, nid)
                handles.append(handle)

            self.mesh_handles[nid] = handles

        # Add transform gizmos for movable joints
        for nid in self.movable_joints:
            self._create_gizmo(nid, global_T)

    def _register_click(self, handle, nid):
        """Register a click handler for a mesh."""
        @handle.on_click
        def _(event):
            self._on_part_clicked(nid)

    def _create_gizmo(self, nid, global_T):
        """Create a constrained transform gizmo for a movable joint."""
        node = self.id2node[nid]
        joint = node.get("joint", {})
        jtype = joint.get("type", "fixed")
        jr = joint.get("range", [0.0, 0.0])
        axis_info = joint.get("axis", {})
        direction = np.array(axis_info.get("direction", [0, 0, 0]), dtype=float)
        origin = np.array(axis_info.get("origin", [0, 0, 0]), dtype=float)

        # Transform the joint origin to normalized space
        origin_h = np.array([*origin, 1.0])
        # The base transform moves center to origin and scales
        base_T = np.eye(4)
        base_T[:3, 3] = -self.center
        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale
        scale_mat[1, 1] = self.scale
        scale_mat[2, 2] = self.scale
        base_T = scale_mat @ base_T

        # Get the parent's global transform to position the gizmo correctly
        pid = node["parent"]
        if pid >= 0 and pid in global_T:
            parent_T = global_T[pid]
        else:
            parent_T = base_T

        # Gizmo position: joint origin transformed by parent's global transform
        origin_world = (parent_T @ origin_h)[:3]

        # Direction in world space (only rotation part of parent transform, plus scale)
        R_parent = parent_T[:3, :3]
        dir_world = R_parent @ direction
        dir_norm = np.linalg.norm(dir_world)
        if dir_norm > 1e-6:
            dir_world = dir_world / dir_norm

        # Compute quaternion that aligns Z-axis to joint direction in world space
        align_wxyz = align_z_to_direction(dir_world)

        # Store alignment info for reading gizmo state later
        self.gizmo_frames[nid] = {
            "align_wxyz": align_wxyz,
            "origin_world": origin_world,
            "direction_world": dir_world,
            "joint_type": jtype,
            "joint_range": jr,
        }

        # Configure gizmo based on joint type.
        # Scale relative to object: big enough to be visible and draggable.
        gizmo_scale = 0.4

        if jtype == "revolute":
            # Allow rotation only.  We let the user rotate freely and clamp
            # the Z-axis component (= rotation around the joint axis) in the
            # on_update callback.  Overly tight per-axis limits made gizmos
            # feel unresponsive, so we leave them unconstrained here.
            gizmo = self.server.scene.add_transform_controls(
                name=f"/gizmo/joint_{nid}",
                scale=gizmo_scale,
                line_width=4.0,
                disable_axes=True,       # no translation
                disable_sliders=True,    # no plane sliders
                disable_rotations=False, # rotation enabled
                depth_test=False,        # always visible on top
                opacity=1.0,
                wxyz=align_wxyz,
                position=tuple(origin_world),
            )
        elif jtype == "prismatic":
            # Allow translation only along the joint direction (= gizmo Z).
            disp_range = (jr[1] - jr[0]) * self.scale
            trans_min = min(0, disp_range)
            trans_max = max(0, disp_range)

            gizmo = self.server.scene.add_transform_controls(
                name=f"/gizmo/joint_{nid}",
                scale=gizmo_scale,
                line_width=4.0,
                disable_axes=False,       # translation axes enabled
                disable_sliders=True,     # no plane sliders
                disable_rotations=True,   # no rotation
                active_axes=(False, False, True),  # only Z-axis translation
                translation_limits=(
                    (-0.001, 0.001),       # X locked
                    (-0.001, 0.001),       # Y locked
                    (trans_min, trans_max), # Z = joint range
                ),
                depth_test=False,
                opacity=1.0,
                wxyz=align_wxyz,
                position=tuple(origin_world),
            )
        else:
            return

        # Register gizmo update callback
        @gizmo.on_update
        def _(event, _nid=nid):
            self._on_gizmo_update(_nid, event)

        self.gizmo_handles[nid] = gizmo

    def _on_gizmo_update(self, nid, event):
        """Called when a gizmo is dragged. Compute new joint parameter and update scene."""
        if self.animating:
            return

        frame_info = self.gizmo_frames[nid]
        jtype = frame_info["joint_type"]
        jr = frame_info["joint_range"]
        gizmo = event.target

        if jtype == "revolute":
            # Compute relative rotation from initial orientation to current.
            gizmo_wxyz = np.array(gizmo.wxyz)
            align_wxyz = frame_info["align_wxyz"]

            R_gizmo = tf.SO3(gizmo_wxyz)
            R_align = tf.SO3(align_wxyz)
            R_rel = R_gizmo @ R_align.inverse()

            # Extract the Z-axis component (= rotation around the joint axis).
            log_vec = R_rel.log()  # 3-vector (axis * angle)
            angle_rad = float(log_vec[2])

            # Clamp to the joint's angular range.
            range_min_rad = math.radians(jr[0])
            range_max_rad = math.radians(jr[1])
            lo = min(range_min_rad, range_max_rad)
            hi = max(range_min_rad, range_max_rad)
            angle_rad = float(np.clip(angle_rad, lo, hi))

            # Snap the gizmo to the clamped Z-only rotation so it doesn't
            # drift in X/Y directions during drag.
            R_clamped = tf.SO3.exp(np.array([0.0, 0.0, angle_rad]))
            gizmo.wxyz = (R_clamped @ R_align).wxyz

            # Map angle to t parameter [0, 1].
            total_range = range_max_rad - range_min_rad
            if abs(total_range) > 1e-6:
                t = np.clip((angle_rad - range_min_rad) / total_range, 0.0, 1.0)
            else:
                t = 0.0

            self.joint_params[nid] = float(t)

        elif jtype == "prismatic":
            # Read Z translation from gizmo position relative to initial position
            gizmo_pos = np.array(gizmo.position)
            origin_world = frame_info["origin_world"]
            delta = gizmo_pos - origin_world

            # Project onto direction
            dir_world = frame_info["direction_world"]
            displacement = np.dot(delta, dir_world)

            # Map to t parameter
            disp_range = (jr[1] - jr[0]) * self.scale
            if abs(disp_range) > 1e-6:
                t = np.clip(displacement / disp_range, 0.0, 1.0)
            else:
                t = 0.0

            self.joint_params[nid] = float(t)

        # Update all mesh transforms
        self._update_meshes()

    def _on_part_clicked(self, nid):
        """Handle clicking on a part."""
        # Update selection
        old_selected = self.selected_id
        self.selected_id = nid

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
        info = (
            f"**Part {nid}: {node.get('name', 'unknown')}**\n\n"
            f"- Joint type: `{jtype}`\n"
            f"- Range: [{jr[0]:.1f}, {jr[1]:.1f}]\n"
            f"- Parent: {node.get('parent', -1)}\n"
            f"- Children: {node.get('children', [])}"
        )
        self.info_text.content = info

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

    def _reset_joints(self):
        """Reset all joints to rest state (t=0)."""
        for nid in self.joint_params:
            self.joint_params[nid] = 0.0

        # Reset gizmo positions
        for nid, gizmo in self.gizmo_handles.items():
            frame_info = self.gizmo_frames[nid]
            gizmo.wxyz = frame_info["align_wxyz"]
            gizmo.position = tuple(frame_info["origin_world"])

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

            # Also update gizmos to match
            global_T = build_global_transforms(
                self.nodes, self.id2node, self.center, self.scale, self.joint_params
            )
            for nid in self.movable_joints:
                if nid not in self.gizmo_handles:
                    continue
                frame_info = self.gizmo_frames[nid]
                node = self.id2node[nid]
                joint = node.get("joint", {})
                jtype = joint.get("type", "fixed")
                jr = joint.get("range", [0.0, 0.0])

                gizmo = self.gizmo_handles[nid]

                if jtype == "revolute":
                    angle_deg = jr[0] + t * (jr[1] - jr[0])
                    angle_rad = math.radians(angle_deg)
                    # Rotate the align frame by angle_rad around Z
                    R_align = tf.SO3(frame_info["align_wxyz"])
                    R_z = tf.SO3.exp(np.array([0.0, 0.0, angle_rad]))
                    R_new = R_z @ R_align
                    gizmo.wxyz = R_new.wxyz

                elif jtype == "prismatic":
                    disp = t * (jr[1] - jr[0]) * self.scale
                    dir_world = frame_info["direction_world"]
                    new_pos = frame_info["origin_world"] + disp * dir_world
                    gizmo.position = tuple(new_pos)

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
                "*Click on a part to see its info*"
            )

        # Controls
        with self.server.gui.add_folder("Controls"):
            self.server.gui.add_markdown(
                "**Drag the colored gizmos** (rings/arrows) near "
                "joint hinges to articulate parts."
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

    def _setup_client_handler(self):
        """Set up handler for new client connections."""
        # Set initial camera defaults on the server (applies to all new connections
        # and the "Reset View" button).
        self.server.initial_camera.position = np.array([0.0, 0.3, 1.5])
        self.server.initial_camera.look_at = np.array([0.0, 0.0, 0.0])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            # Set camera to orbit around the object center (0,0,0 after normalization).
            # Important: set position first, then look_at (Viser preserves direction
            # when position changes, so look_at must be set after).
            client.camera.position = np.array([0.0, 0.3, 1.5])
            client.camera.look_at = np.array([0.0, 0.0, 0.0])

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
        print("  - Click on a part: select it")
        print("  - Drag a gizmo: articulate the joint")
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
