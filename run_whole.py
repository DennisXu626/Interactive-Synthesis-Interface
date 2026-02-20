#!/usr/bin/env python3
"""
run_whole.py - Full SINGAPO pipeline: inference + interactive visualizer.

Provides two modes:
    1. Run inference: takes an input image + graph, runs SINGAPO model to
       produce articulated 3D object, then launches the interactive viewer.
    2. Use existing output: skips inference and directly launches the viewer
       on a pre-computed output directory.

Usage:
    conda activate 4yp
    cd D:\\4YP\\singapo\\Viser_trial

    # Mode 1: Full pipeline (inference + visualization)
    python run_whole.py --mode infer --img_path demo_input.png --graph_path example_graph.json

    # Mode 2: Visualize existing output
    python run_whole.py --mode visualize --object_dir output/0

    # Interactive mode (asks which mode to use)
    python run_whole.py
"""

import os
import sys
import json
import math
import time
import argparse
import subprocess
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

# Add the singapo root directory to sys.path so we can import its modules
SINGAPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SINGAPO_ROOT)


# ════════════════════════════════════════════════════════════════════
# PART 1: INFERENCE  (adapted from run_inference_1.0.py)
# ════════════════════════════════════════════════════════════════════

def run_inference(img_path, graph_path, save_dir, ckpt_path, config_path,
                  gt_data_root, n_samples=1, omega=0.5, n_denoise_steps=100):
    """Run the full SINGAPO inference pipeline.

    Returns the path to the first sample's output directory.
    """
    import torch
    from PIL import Image
    import torchvision.transforms as T

    from utils.misc import load_config
    from data.utils import make_white_background, load_input_from, convert_data_range, parse_tree
    from diffusers import DDPMScheduler
    from models.denoiser import Denoiser

    # ── Helper functions ──

    def load_img(path):
        transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        with Image.open(path) as img:
            if img.mode == "RGBA":
                img = make_white_background(img)
            img = transform(img)
        return img.unsqueeze(0).cuda()

    def extract_dino_feature(path):
        print("[Inference] Extracting DINO features...")
        input_img = load_img(path)
        dinov2 = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_reg", pretrained=True
        ).cuda()
        with torch.no_grad():
            feat = dinov2.forward_features(input_img)["x_norm_patchtokens"]
        torch.cuda.empty_cache()
        return feat

    def forward_diffusion(model, scheduler, inputs, omega_val):
        print("[Inference] Running diffusion denoising...")
        noisy_x = inputs["noise"]
        for t in scheduler.timesteps:
            timesteps = torch.tensor([t], device=inputs["noise"].device)
            out_cond = model(
                x=noisy_x, cat=inputs["cat"], timesteps=timesteps,
                feat=inputs["f"], key_pad_mask=inputs["key_pad_mask"],
                graph_mask=inputs["graph_mask"], attr_mask=inputs["attr_mask"],
                label_free=True,
            )
            if omega_val != 0:
                out_free = model(
                    x=noisy_x, cat=inputs["cat"], timesteps=timesteps,
                    feat=inputs["dummy_f"], key_pad_mask=inputs["key_pad_mask"],
                    graph_mask=inputs["graph_mask"], attr_mask=inputs["attr_mask"],
                    label_free=True,
                )
                noise_pred = ((1 + omega_val) * out_cond["noise_pred"]
                              - omega_val * out_free["noise_pred"])
            else:
                noise_pred = out_cond["noise_pred"]
            noisy_x = scheduler.step(noise_pred, t, noisy_x).prev_sample
        return noisy_x

    def convert_json(x, c):
        out = {"meta": {}, "diffuse_tree": []}
        if "obj_cat" in c:
            out["meta"]["obj_cat"] = c["obj_cat"]
        data = convert_data_range(x)
        out["diffuse_tree"] = parse_tree(data, c["n_nodes"], c["parents"].tolist(), c["adj"])
        return out

    # ── Main inference flow ──

    print()
    print("=" * 60)
    print("SINGAPO Inference")
    print("=" * 60)
    print(f"  Image:      {img_path}")
    print(f"  Graph:      {graph_path}")
    print(f"  Save dir:   {save_dir}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  GT data:    {gt_data_root}")
    print(f"  Samples:    {n_samples}")
    print()

    # Step 1: DINO features
    feat = extract_dino_feature(img_path)

    # Step 2: Load graph
    with open(graph_path, "r") as f:
        pred_graph = json.load(f)
    print(f"[Inference] Loaded graph with {len(pred_graph['diffuse_tree'])} nodes")

    # Step 3: Load input
    data, cond = load_input_from(pred_graph, K=32)

    # Step 4: Prepare model input
    n = n_samples
    attr_mask = torch.from_numpy(cond["attr_mask"]).unsqueeze(0).repeat(n, 1, 1)
    key_pad_mask = torch.from_numpy(cond["key_pad_mask"]).unsqueeze(0).repeat(n, 1, 1)
    graph_mask = torch.from_numpy(cond["adj_mask"]).unsqueeze(0).repeat(n, 1, 1)
    f = feat.repeat(n, 1, 1)
    noise = torch.randn(data.shape, dtype=torch.float32).repeat(n, 1, 1)
    dummy_feat = torch.from_numpy(
        np.load(os.path.join(SINGAPO_ROOT, "systems", "dino_dummy.npy")).astype(np.float32)
    ).unsqueeze(0).repeat(n, 1, 1)
    cat = torch.zeros(1, dtype=torch.long).repeat(n)
    inputs = {
        "noise": noise.cuda(), "attr_mask": attr_mask.cuda(),
        "key_pad_mask": key_pad_mask.cuda(), "graph_mask": graph_mask.cuda(),
        "dummy_f": dummy_feat.cuda(), "cat": cat.cuda(), "f": f.cuda(),
    }

    # Step 5: Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )
    scheduler.set_timesteps(n_denoise_steps)

    # Step 6: Load model
    print("[Inference] Loading model...")
    config = load_config(config_path)
    model = Denoiser(config.system.model)
    state_dict = torch.load(ckpt_path, map_location="cuda", weights_only=False)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model = model.cuda()

    # Step 7: Diffusion
    with torch.no_grad():
        output = forward_diffusion(model, scheduler, inputs, omega).cpu().numpy()

    # Step 8: Post-process & mesh retrieval
    print("[Inference] Post-processing and mesh retrieval...")
    os.makedirs(save_dir, exist_ok=True)
    np.fill_diagonal(cond["adj"], 0)

    for i in range(output.shape[0]):
        out_json = convert_json(output, cond)
        sample_dir = os.path.join(save_dir, str(i))
        os.makedirs(sample_dir, exist_ok=True)

        json_path = os.path.join(sample_dir, "object.json")
        with open(json_path, "w") as fout:
            json.dump(out_json, fout, indent=4)
        print(f"  Saved object.json -> {json_path}")

        print(f"  Retrieving part meshes for sample {i}... (this may take a few minutes)")
        retrieve_py = os.path.join(SINGAPO_ROOT, "scripts", "mesh_retrieval", "retrieve.py")
        cmd = [sys.executable, "-u", retrieve_py,
               "--src_dir", sample_dir,
               "--json_name", "object.json",
               "--gt_data_root", gt_data_root]
        proc = subprocess.run(cmd, cwd=SINGAPO_ROOT)
        if proc.returncode != 0:
            print(f"  WARNING: Mesh retrieval exit code {proc.returncode}")
        else:
            print(f"  Mesh retrieval complete for sample {i}.")

    # Free GPU memory before starting viewer
    del model, inputs, output, feat
    torch.cuda.empty_cache()

    first_sample_dir = os.path.join(save_dir, "0")
    print()
    print(f"[Inference] Complete! Output at: {first_sample_dir}")
    return first_sample_dir


# ════════════════════════════════════════════════════════════════════
# PART 2: VISUALIZER  (from run_visualizer_3.0.py — direct drag v3.0)
# ════════════════════════════════════════════════════════════════════

# ── Articulation utilities ──────────────────────────────────────────

def load_object(object_dir):
    json_path = os.path.join(object_dir, "object.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data["diffuse_tree"]
    id2node = {n["id"]: n for n in nodes}
    for node in nodes:
        meshes = []
        for rel in node.get("plys", []):
            mesh_path = os.path.join(object_dir, rel)
            if not os.path.exists(mesh_path):
                print(f"[WARN] Missing mesh: {mesh_path}")
                continue
            tm = trimesh.load(mesh_path, force="mesh")
            if tm.is_empty:
                continue
            meshes.append(tm)
        node["_meshes"] = meshes
    return nodes, id2node


def compute_center_and_scale(nodes, target_size=1.0):
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
        T[:3, 3] = t * disp_max * (direction / norm)
        return T
    if jtype == "revolute":
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return T
        d_hat = direction / norm
        theta = math.radians(jr[0] + t * (jr[1] - jr[0]))
        K = np.array([[0, -d_hat[2], d_hat[1]],
                      [d_hat[2], 0, -d_hat[0]],
                      [-d_hat[1], d_hat[0], 0]])
        R3 = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        T[:3, :3] = R3
        T[:3, 3] = origin - R3 @ origin
        return T
    return T


def build_global_transforms(nodes, id2node, center, scale, joint_params):
    base_transform = np.eye(4)
    base_transform[:3, 3] = -center
    scale_mat = np.eye(4) * scale
    scale_mat[3, 3] = 1.0
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
        Tn = parent_T @ joint_transform_matrix(node, t)
        global_T[nid] = Tn
        for cid in children_map.get(nid, []):
            dfs(cid, Tn)
    for rid in root_ids:
        dfs(rid, base_transform)
    return global_T


def matrix_to_wxyz_position(T, uniform_scale=1.0):
    R = T[:3, :3]
    pos = T[:3, 3]
    if abs(uniform_scale) > 1e-8 and abs(uniform_scale - 1.0) > 1e-8:
        R = R / uniform_scale
    return tf.SO3.from_matrix(R).wxyz, pos


# ── Colors ──

PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (255, 225, 25),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (188, 143, 143), (128, 128, 0),
]
HIGHLIGHT_COLOR = (255, 255, 255)


# ── Per-client state ──

@dataclasses.dataclass
class _ClientDragState:
    drag_part_id: int | None = None
    dragging: bool = False
    drag_start_screen: tuple | None = None
    drag_start_param: float = 0.0
    client: object = None


# ── Side-channel WebSocket ──

class DragWebSocketServer:
    def __init__(self, port, viewer):
        self.port = port
        self.viewer = viewer
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._loop = None
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
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"[DragWS] Server error: {e}")

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
                    continue
                if client_id is None:
                    continue
                if mt == "hit_test":
                    self.viewer._on_hit_test(client_id, data)
                elif mt == "drag_start":
                    self.viewer._on_drag_start(client_id, data)
                elif mt == "drag_move":
                    self.viewer._on_drag_move(client_id, data)
                elif mt == "drag_end":
                    self.viewer._on_drag_end(client_id, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[DragWS] Handler error: {e}")
        finally:
            if client_id is not None:
                self._ws_by_client.pop(client_id, None)


# ── JavaScript template ──

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
    var hitPartId = null;

    function connectWS() {
        try {
            ws = new WebSocket("ws://" + window.location.hostname + ":" + WS_PORT);
            ws.onopen = function() {
                ws.send(JSON.stringify({type: "identify", client_id: CLIENT_ID}));
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
        var c = getCanvas();
        if (!c) return {x:0.5, y:0.5};
        var r = c.getBoundingClientRect();
        return { x: (cx-r.left)/Math.max(r.width,1), y: (cy-r.top)/Math.max(r.height,1) };
    }
    function isOnCanvas(e) {
        var c = getCanvas();
        if (!c) return false;
        var r = c.getBoundingClientRect();
        return e.clientX>=r.left && e.clientX<=r.right && e.clientY>=r.top && e.clientY<=r.bottom;
    }

    document.addEventListener('pointerdown', function(e) {
        if (e.button !== 0) return;
        if (mode !== 'idle') return;
        if (!isOnCanvas(e)) return;
        savedPointerId = e.pointerId;
        startX = e.clientX; startY = e.clientY;
        lastClientX = e.clientX; lastClientY = e.clientY;
        startNorm = getNorm(e.clientX, e.clientY);
        mode = 'pending'; hitPartId = null;
        sendMsg({type:"hit_test", screen_x:startNorm.x, screen_y:startNorm.y});
    }, {capture: true});

    document.addEventListener('pointermove', function(e) {
        lastClientX = e.clientX; lastClientY = e.clientY;
        if (mode === 'drag') {
            e.stopPropagation(); e.preventDefault();
            var norm = getNorm(e.clientX, e.clientY);
            sendMsg({type:"drag_move", screen_x:norm.x, screen_y:norm.y,
                     start_x:startNorm.x, start_y:startNorm.y});
        }
    }, {capture: true});

    document.addEventListener('pointerup', function(e) {
        if (mode === 'drag') {
            e.stopPropagation(); e.preventDefault();
            var norm = getNorm(e.clientX, e.clientY);
            sendMsg({type:"drag_end", screen_x:norm.x, screen_y:norm.y});
            document.body.style.cursor = '';
            mode = 'idle'; hitPartId = null;
            return;
        }
        if (mode === 'pending') mode = 'idle';
    }, {capture: true});

    function onWsMessage(event) {
        var data;
        try { data = JSON.parse(event.data); } catch(ex) { return; }
        if (data.type === 'hit_result') {
            if (mode !== 'pending') return;
            if (data.part_id !== null) {
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
                mode = 'drag'; hitPartId = data.part_id;
                document.body.style.cursor = 'grabbing';
                sendMsg({type:"drag_start", screen_x:startNorm.x, screen_y:startNorm.y,
                         part_id:data.part_id});
                var norm = getNorm(lastClientX, lastClientY);
                if (Math.abs(norm.x-startNorm.x)>0.001 || Math.abs(norm.y-startNorm.y)>0.001) {
                    sendMsg({type:"drag_move", screen_x:norm.x, screen_y:norm.y,
                             start_x:startNorm.x, start_y:startNorm.y});
                }
            } else {
                mode = 'idle';
            }
        }
    }
})();
"""


# ── Main Viewer class ──

class ArticulatedObjectViewer:
    def __init__(self, object_dir, img_path=None, port=8080):
        self.object_dir = object_dir
        self.img_path = img_path
        self.port = port

        print(f"[Viewer] Loading object from {object_dir}...")
        self.nodes, self.id2node = load_object(object_dir)
        self.center, self.scale = compute_center_and_scale(self.nodes)
        print(f"  {len(self.nodes)} parts, scale={self.scale:.4f}")

        self.joint_params = {n["id"]: 0.0 for n in self.nodes}
        self.movable_joints = set()
        for node in self.nodes:
            jt = node.get("joint", {}).get("type", "fixed")
            if jt not in ("fixed", ""):
                self.movable_joints.add(node["id"])
        print(f"  Movable joints: {sorted(self.movable_joints)}")

        self.effective_movable = {}
        self._build_movable_lookup()

        self.mesh_handles = {}
        self.client_drag_states = {}
        self.joint_frames = {}
        self.animating = False
        self.anim_thread = None

        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+y")

        self._compute_joint_frames()
        self._setup_scene()
        self._setup_gui()
        self._setup_client_handler()

        self.drag_ws = DragWebSocketServer(port + 1, self)
        self.drag_ws.start()

    def _build_movable_lookup(self):
        for node in self.nodes:
            nid = node["id"]
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

    def _compute_joint_frames(self):
        base_T = np.eye(4)
        base_T[:3, 3] = -self.center
        s = np.eye(4) * self.scale; s[3, 3] = 1.0
        base_T = s @ base_T
        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params)
        for nid in self.movable_joints:
            node = self.id2node[nid]
            joint = node.get("joint", {})
            axis_info = joint.get("axis", {})
            direction = np.array(axis_info.get("direction", [0, 0, 0]), dtype=float)
            origin = np.array(axis_info.get("origin", [0, 0, 0]), dtype=float)
            pid = node["parent"]
            parent_T = global_T[pid] if (pid >= 0 and pid in global_T) else base_T
            origin_world = (parent_T @ np.array([*origin, 1.0]))[:3]
            dir_world = parent_T[:3, :3] @ direction
            dn = np.linalg.norm(dir_world)
            if dn > 1e-6:
                dir_world = dir_world / dn
            self.joint_frames[nid] = {
                "origin_world": origin_world, "direction_world": dir_world,
                "joint_type": joint.get("type", "fixed"),
                "joint_range": joint.get("range", [0.0, 0.0]),
            }

    def _setup_scene(self):
        self.server.scene.add_grid("/grid", width=2.0, height=2.0,
                                   position=(0.0, -0.6, 0.0),
                                   cell_color=(200, 200, 200), plane="xz")
        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params)
        for node in self.nodes:
            nid = node["id"]
            color = PALETTE[nid % len(PALETTE)]
            wxyz, position = matrix_to_wxyz_position(global_T[nid], self.scale)
            handles = []
            for mi, tm in enumerate(node["_meshes"]):
                h = self.server.scene.add_mesh_simple(
                    name=f"/object/part_{nid}/mesh_{mi}",
                    vertices=np.array(tm.vertices, dtype=np.float32),
                    faces=np.array(tm.faces, dtype=np.uint32),
                    color=color, flat_shading=False, side="double",
                    scale=self.scale, wxyz=wxyz, position=position)
                handles.append(h)
            self.mesh_handles[nid] = handles

    # ── Hit testing ──

    def _screen_to_ray(self, sx, sy, client):
        try:
            cam_pos = np.array(client.camera.position, dtype=float)
            cam_la = np.array(client.camera.look_at, dtype=float)
            fov = float(client.camera.fov)
            aspect = float(client.camera.aspect)
        except Exception:
            return None, None
        fwd = cam_la - cam_pos
        fd = np.linalg.norm(fwd)
        if fd < 1e-8:
            return None, None
        fwd /= fd
        right = np.cross(fwd, [0, 1, 0])
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
        ro, rd = self._screen_to_ray(sx, sy, client)
        if ro is None:
            return None
        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params)
        best_nid, best_dist = None, float("inf")
        for node in self.nodes:
            nid = node["id"]
            if self.effective_movable.get(nid) is None:
                continue
            T = global_T[nid]
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
            for tm in node["_meshes"]:
                try:
                    locs, _, _ = tm.ray.intersects_location(
                        ray_origins=ol.reshape(1, 3), ray_directions=dl.reshape(1, 3))
                except Exception:
                    continue
                for loc in locs:
                    hit_d = np.linalg.norm((T @ np.array([*loc, 1.0]))[:3] - ro)
                    if hit_d < best_dist:
                        best_dist = hit_d
                        best_nid = self.effective_movable[nid]
        return best_nid

    # ── JS injection ──

    def _inject_drag_js(self, client):
        js = DRAG_JS_TEMPLATE.replace("__DRAG_WS_PORT__", str(self.port + 1)).replace(
            "__CLIENT_ID__", str(client.client_id))
        client._websock_connection.queue_message(RunJavascriptMessage(source=js))

    # ── Drag handlers ──

    def _on_hit_test(self, cid, data):
        state = self.client_drag_states.get(cid)
        if state is None or self.animating:
            self.drag_ws.send_to_client(cid, {"type": "hit_result", "part_id": None})
            return
        pid = self._hit_test(data["screen_x"], data["screen_y"], state.client)
        self.drag_ws.send_to_client(cid, {"type": "hit_result", "part_id": pid})

    def _on_drag_start(self, cid, data):
        state = self.client_drag_states.get(cid)
        if state is None:
            return
        nid = data.get("part_id")
        if nid is None or nid not in self.movable_joints:
            return
        state.drag_part_id = nid
        state.dragging = True
        state.drag_start_screen = (data["screen_x"], data["screen_y"])
        state.drag_start_param = self.joint_params.get(nid, 0.0)
        if nid in self.mesh_handles:
            for h in self.mesh_handles[nid]:
                h.color = HIGHLIGHT_COLOR
        node = self.id2node[nid]
        j = node.get("joint", {})
        jr = j.get("range", [0, 0])
        self.info_text.content = (
            f"**Dragging Part {nid}: {node.get('name', '?')}**\n\n"
            f"- Joint: `{j.get('type', 'fixed')}` [{jr[0]:.1f}, {jr[1]:.1f}]")

    def _on_drag_move(self, cid, data):
        state = self.client_drag_states.get(cid)
        if state is None or not state.dragging or state.drag_part_id is None or self.animating:
            return
        nid = state.drag_part_id
        new_t = self._screen_delta_to_joint_param(
            nid, (data["start_x"], data["start_y"]),
            (data["screen_x"], data["screen_y"]), state)
        self.joint_params[nid] = new_t
        self._update_meshes()

    def _on_drag_end(self, cid, data):
        state = self.client_drag_states.get(cid)
        if state is None:
            return
        nid = state.drag_part_id
        if nid is not None and nid in self.mesh_handles:
            for h in self.mesh_handles[nid]:
                h.color = PALETTE[nid % len(PALETTE)]
        state.drag_part_id = None
        state.dragging = False
        self.info_text.content = "*Drag a part to articulate it*"

    # ── Screen delta -> joint param ──

    def _screen_delta_to_joint_param(self, nid, start_xy, cur_xy, state):
        frame = self.joint_frames.get(nid)
        if frame is None:
            return self.joint_params.get(nid, 0.0)
        jtype = frame["joint_type"]
        jr = frame["joint_range"]
        jd = frame["direction_world"]
        jo = frame["origin_world"]
        client = state.client
        try:
            cp = np.array(client.camera.position)
            cl = np.array(client.camera.look_at)
        except Exception:
            return self.joint_params.get(nid, 0.0)
        vd = cl - cp
        vn = np.linalg.norm(vd)
        if vn < 1e-8:
            return self.joint_params.get(nid, 0.0)
        vd /= vn
        wu = np.array([0.0, 1.0, 0.0])
        right = np.cross(vd, wu)
        rn = np.linalg.norm(right)
        right = np.array([1, 0, 0]) if rn < 1e-6 else right / rn
        up = np.cross(right, vd)
        up /= np.linalg.norm(up)
        dx = cur_xy[0] - start_xy[0]
        dy = -(cur_xy[1] - start_xy[1])
        sd = np.array([dx, dy])
        cd = max(np.linalg.norm(cp - jo), 1e-6)
        sens = cd * 2.5

        if jtype == "prismatic":
            sa = np.array([np.dot(jd, right), np.dot(jd, up)])
            al = np.linalg.norm(sa)
            if al < 1e-6:
                return self.joint_params.get(nid, 0.0)
            proj = np.dot(sd, sa / al)
            dr = abs(jr[1] - jr[0]) * self.scale
            if dr < 1e-8:
                return 0.0
            return float(np.clip(state.drag_start_param + proj * sens / max(dr, 1e-6), 0, 1))

        elif jtype == "revolute":
            tang = np.cross(jd, vd)
            tn = np.linalg.norm(tang)
            if tn < 1e-6:
                tang = np.cross(jd, up)
                tn = np.linalg.norm(tang)
            if tn < 1e-6:
                return self.joint_params.get(nid, 0.0)
            tang /= tn
            st = np.array([np.dot(tang, right), np.dot(tang, up)])
            sn = np.linalg.norm(st)
            if sn < 1e-6:
                return self.joint_params.get(nid, 0.0)
            proj = np.dot(sd, st / sn)
            ta = math.radians(abs(jr[1] - jr[0]))
            if ta < 1e-6:
                return 0.0
            return float(np.clip(state.drag_start_param + proj * sens / max(ta, 1e-6), 0, 1))

        return self.joint_params.get(nid, 0.0)

    # ── Mesh update ──

    def _update_meshes(self):
        global_T = build_global_transforms(
            self.nodes, self.id2node, self.center, self.scale, self.joint_params)
        for node in self.nodes:
            nid = node["id"]
            wxyz, pos = matrix_to_wxyz_position(global_T[nid], self.scale)
            if nid in self.mesh_handles:
                for h in self.mesh_handles[nid]:
                    h.wxyz = wxyz
                    h.position = pos

    # ── Reset / Animation ──

    def _reset_joints(self):
        for nid in self.joint_params:
            self.joint_params[nid] = 0.0
        self._update_meshes()

    def _toggle_animation(self):
        if self.animating:
            self.animating = False
            if self.anim_thread:
                self.anim_thread.join(timeout=2.0)
            self.anim_button.label = "Animate"
        else:
            self.animating = True
            self.anim_button.label = "Stop"
            self.anim_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self.anim_thread.start()

    def _animation_loop(self):
        n_frames, fps, frame = 120, 30, 0
        while self.animating:
            t = 0.5 * (1 - math.cos(2 * math.pi * frame / max(n_frames - 1, 1)))
            for nid in self.movable_joints:
                self.joint_params[nid] = t
            self._update_meshes()
            frame = (frame + 1) % n_frames
            time.sleep(1.0 / fps)

    def _export_state(self):
        path = os.path.join(self.object_dir, "joint_state.json")
        with open(path, "w") as f:
            json.dump({"joint_params": {str(k): v for k, v in self.joint_params.items()},
                        "movable_joints": sorted(self.movable_joints)}, f, indent=2)
        print(f"Exported joint state to {path}")

    # ── GUI ──

    def _setup_gui(self):
        if self.img_path and os.path.exists(self.img_path):
            from PIL import Image as PILImage
            img = PILImage.open(self.img_path).resize((200, 200))
            arr = np.array(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            with self.server.gui.add_folder("Input Image"):
                self.server.gui.add_image(arr)
        with self.server.gui.add_folder("Info"):
            self.info_text = self.server.gui.add_markdown("*Drag a part to articulate it*")
        with self.server.gui.add_folder("Controls"):
            self.server.gui.add_markdown(
                "**Drag** a movable part to articulate its joint.\n"
                "**Drag** empty space to orbit the camera.\n**Scroll** to zoom.")
            reset_btn = self.server.gui.add_button("Reset All Joints")
            self.anim_button = self.server.gui.add_button("Animate")
            export_btn = self.server.gui.add_button("Export State")
            @reset_btn.on_click
            def _(_): self._reset_joints()
            @self.anim_button.on_click
            def _(_): self._toggle_animation()
            @export_btn.on_click
            def _(_): self._export_state()
        with self.server.gui.add_folder("Part Hierarchy"):
            txt = ""
            for node in self.nodes:
                nid = node["id"]
                indent = "  " if node.get("parent", -1) >= 0 else ""
                mv = " (movable)" if nid in self.movable_joints else ""
                txt += f"{indent}- Part {nid}: **{node.get('name', '?')}** [{node.get('joint', {}).get('type', 'fixed')}]{mv}\n"
            self.server.gui.add_markdown(txt)

    # ── Client handler ──

    def _setup_client_handler(self):
        self.server.initial_camera.position = np.array([0.0, 0.3, 1.5])
        self.server.initial_camera.look_at = np.array([0.0, 0.0, 0.0])

        @self.server.on_client_connect
        def _(client):
            self.client_drag_states[client.client_id] = _ClientDragState(client=client)
            client.camera.position = np.array([0.0, 0.3, 1.5])
            client.camera.look_at = np.array([0.0, 0.0, 0.0])
            self._inject_drag_js(client)

        @self.server.on_client_disconnect
        def _(client):
            self.client_drag_states.pop(client.client_id, None)

    # ── Run ──

    def run(self):
        print()
        print("=" * 60)
        print(f"Viewer running at http://localhost:{self.port}")
        print(f"Drag WebSocket on port {self.port + 1}")
        print()
        print("  Drag a movable part to articulate it")
        print("  Drag empty space to orbit | Scroll to zoom")
        print()
        print("Press Ctrl+C to stop.")
        print("=" * 60)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.animating = False
            print("\nShutting down...")


# ════════════════════════════════════════════════════════════════════
# PART 3: COMBINED ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SINGAPO full pipeline: inference + interactive visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run full pipeline (inference + visualizer):\n"
            "  python run_whole.py --mode infer\n\n"
            "  # Visualize existing output:\n"
            "  python run_whole.py --mode visualize --object_dir output/0\n\n"
            "  # Interactive mode (asks which mode):\n"
            "  python run_whole.py\n"
        ),
    )

    parser.add_argument(
        "--mode", type=str, choices=["infer", "visualize"],
        default=None,
        help="'infer' = run inference then visualize; 'visualize' = use existing output. "
             "If omitted, prompts interactively.",
    )

    # Inference args
    infer_group = parser.add_argument_group("Inference options (--mode infer)")
    infer_group.add_argument("--img_path", type=str,
                             default=os.path.join(os.path.dirname(__file__), "demo_input.png"))
    infer_group.add_argument("--graph_path", type=str,
                             default=os.path.join(os.path.dirname(__file__), "example_graph.json"))
    infer_group.add_argument("--save_dir", type=str,
                             default=os.path.join(os.path.dirname(__file__), "output"))
    infer_group.add_argument("--ckpt_path", type=str,
                             default=os.path.join(SINGAPO_ROOT, "exps", "singapo", "final", "ckpts", "last.ckpt"))
    infer_group.add_argument("--config_path", type=str,
                             default=os.path.join(SINGAPO_ROOT, "exps", "singapo", "final", "config", "parsed.yaml"))
    infer_group.add_argument("--gt_data_root", type=str,
                             default=os.path.join(SINGAPO_ROOT, "..", "data"))
    infer_group.add_argument("--n_samples", type=int, default=1)
    infer_group.add_argument("--omega", type=float, default=0.5)
    infer_group.add_argument("--n_denoise_steps", type=int, default=100)

    # Visualizer args
    vis_group = parser.add_argument_group("Visualizer options")
    vis_group.add_argument("--object_dir", type=str, default=None,
                           help="Pre-computed output dir (for --mode visualize)")
    vis_group.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    # ── Interactive mode selection ──

    mode = args.mode
    if mode is None:
        print()
        print("=" * 60)
        print("  SINGAPO Pipeline")
        print("=" * 60)
        print()
        print("  [1] Run inference + visualizer  (full pipeline)")
        print("  [2] Visualize existing output    (skip inference)")
        print()
        choice = input("Select mode (1/2): ").strip()
        if choice == "1":
            mode = "infer"
        elif choice == "2":
            mode = "visualize"
            if args.object_dir is None:
                default_dir = os.path.join(os.path.dirname(__file__), "output", "0")
                user_dir = input(f"Object directory [{default_dir}]: ").strip()
                args.object_dir = user_dir if user_dir else default_dir
        else:
            print(f"Invalid choice: '{choice}'. Exiting.")
            sys.exit(1)

    # ── Execute ──

    if mode == "infer":
        # Validate inference paths
        assert os.path.exists(args.img_path), f"Image not found: {args.img_path}"
        assert os.path.exists(args.graph_path), f"Graph not found: {args.graph_path}"
        assert os.path.exists(args.ckpt_path), f"Checkpoint not found: {args.ckpt_path}"
        assert os.path.exists(args.config_path), f"Config not found: {args.config_path}"
        assert os.path.exists(args.gt_data_root), f"GT data not found: {args.gt_data_root}"

        object_dir = run_inference(
            img_path=args.img_path,
            graph_path=args.graph_path,
            save_dir=args.save_dir,
            ckpt_path=args.ckpt_path,
            config_path=args.config_path,
            gt_data_root=args.gt_data_root,
            n_samples=args.n_samples,
            omega=args.omega,
            n_denoise_steps=args.n_denoise_steps,
        )
    else:
        object_dir = args.object_dir
        if object_dir is None:
            object_dir = os.path.join(os.path.dirname(__file__), "output", "0")

    # Validate output exists
    assert os.path.exists(object_dir), f"Object dir not found: {object_dir}"
    assert os.path.exists(os.path.join(object_dir, "object.json")), \
        f"object.json not found in {object_dir}"

    # Launch viewer
    viewer = ArticulatedObjectViewer(
        object_dir=object_dir,
        img_path=args.img_path if mode == "infer" else None,
        port=args.port,
    )
    viewer.run()


if __name__ == "__main__":
    main()
