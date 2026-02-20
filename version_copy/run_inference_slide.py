#!/usr/bin/env python3
"""
run_inference.py - Run SINGAPO inference pipeline for the Viser trial.

Takes demo_input.png + example_graph.json from this folder,
runs the full SINGAPO model (DINO feature extraction -> diffusion -> mesh retrieval),
and saves the output (object.json + PLY meshes) to output/0/.

Usage:
    conda activate 4yp
    cd D:\4YP\singapo\Viser_trial
    python run_inference.py
"""

import os
import sys
import json
import argparse
import subprocess

# Add the singapo root directory to sys.path so we can import its modules
SINGAPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SINGAPO_ROOT)

import torch
import numpy as np
from PIL import Image

from utils.misc import load_config
from utils.refs import joint_ref, sem_ref
from data.utils import make_white_background, load_input_from, convert_data_range, parse_tree
from diffusers import DDPMScheduler
from models.denoiser import Denoiser
import torchvision.transforms as T


# ── Reuse functions from demo/demo.py ──

def load_img(img_path):
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    with Image.open(img_path) as img:
        if img.mode == "RGBA":
            img = make_white_background(img)
        img = transform(img)
    img_batch = img.unsqueeze(0).cuda()
    return img_batch


def extract_dino_feature(img_path):
    print("Extracting DINO feature...")
    input_img = load_img(img_path)
    dinov2_vitb14_reg = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14_reg", pretrained=True
    ).cuda()
    with torch.no_grad():
        feat = dinov2_vitb14_reg.forward_features(input_img)["x_norm_patchtokens"]
    torch.cuda.empty_cache()
    return feat


def set_scheduler(n_steps=100):
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )
    scheduler.set_timesteps(n_steps)
    return scheduler


def prepare_model_input(data, cond, feat, n_samples):
    attr_mask = torch.from_numpy(cond["attr_mask"]).unsqueeze(0).repeat(n_samples, 1, 1)
    key_pad_mask = torch.from_numpy(cond["key_pad_mask"]).unsqueeze(0).repeat(n_samples, 1, 1)
    graph_mask = torch.from_numpy(cond["adj_mask"]).unsqueeze(0).repeat(n_samples, 1, 1)
    f = feat.repeat(n_samples, 1, 1)
    noise = torch.randn(data.shape, dtype=torch.float32).repeat(n_samples, 1, 1)
    dummy_feat = torch.from_numpy(
        np.load(os.path.join(SINGAPO_ROOT, "systems", "dino_dummy.npy")).astype(np.float32)
    )
    dummy_feat = dummy_feat.unsqueeze(0).repeat(n_samples, 1, 1)
    cat = torch.zeros(1, dtype=torch.long).repeat(n_samples)
    return {
        "noise": noise.cuda(),
        "attr_mask": attr_mask.cuda(),
        "key_pad_mask": key_pad_mask.cuda(),
        "graph_mask": graph_mask.cuda(),
        "dummy_f": dummy_feat.cuda(),
        "cat": cat.cuda(),
        "f": f.cuda(),
    }


def forward(model, scheduler, inputs, omega=0.5):
    print("Running diffusion inference...")
    noisy_x = inputs["noise"]
    for t in scheduler.timesteps:
        timesteps = torch.tensor([t], device=inputs["noise"].device)
        outputs_cond = model(
            x=noisy_x,
            cat=inputs["cat"],
            timesteps=timesteps,
            feat=inputs["f"],
            key_pad_mask=inputs["key_pad_mask"],
            graph_mask=inputs["graph_mask"],
            attr_mask=inputs["attr_mask"],
            label_free=True,
        )
        if omega != 0:
            outputs_free = model(
                x=noisy_x,
                cat=inputs["cat"],
                timesteps=timesteps,
                feat=inputs["dummy_f"],
                key_pad_mask=inputs["key_pad_mask"],
                graph_mask=inputs["graph_mask"],
                attr_mask=inputs["attr_mask"],
                label_free=True,
            )
            noise_pred = (1 + omega) * outputs_cond["noise_pred"] - omega * outputs_free["noise_pred"]
        else:
            noise_pred = outputs_cond["noise_pred"]
        noisy_x = scheduler.step(noise_pred, t, noisy_x).prev_sample
    return noisy_x


def _convert_json(x, c):
    out = {"meta": {}, "diffuse_tree": []}
    n_nodes = c["n_nodes"]
    par = c["parents"].tolist()
    adj = c["adj"]
    np.fill_diagonal(adj, 0)
    if "obj_cat" in c:
        out["meta"]["obj_cat"] = c["obj_cat"]
    data = convert_data_range(x)
    out["diffuse_tree"] = parse_tree(data, n_nodes, par, adj)
    return out


def load_model(ckpt_path, config):
    print("Loading model from checkpoint...")
    model = Denoiser(config)
    state_dict = torch.load(
        ckpt_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        weights_only=False,
    )["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.cuda()


def main():
    parser = argparse.ArgumentParser(description="Run SINGAPO inference for Viser trial")
    parser.add_argument(
        "--img_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "demo_input.png"),
        help="Path to input image",
    )
    parser.add_argument(
        "--graph_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "example_graph.json"),
        help="Path to example graph JSON",
    )
    parser.add_argument(
        "--save_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "output"),
        help="Directory to save inference output",
    )
    parser.add_argument(
        "--ckpt_path", type=str,
        default=os.path.join(SINGAPO_ROOT, "exps", "singapo", "final", "ckpts", "last.ckpt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config_path", type=str,
        default=os.path.join(SINGAPO_ROOT, "exps", "singapo", "final", "config", "parsed.yaml"),
        help="Path to model config",
    )
    parser.add_argument(
        "--gt_data_root", type=str,
        default=os.path.join(SINGAPO_ROOT, "..", "data"),
        help="Root directory of ground truth data for mesh retrieval",
    )
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--omega", type=float, default=0.5, help="Classifier-free guidance weight")
    parser.add_argument("--n_denoise_steps", type=int, default=100, help="Denoising steps")
    args = parser.parse_args()

    # Validate paths
    assert os.path.exists(args.img_path), f"Input image not found: {args.img_path}"
    assert os.path.exists(args.graph_path), f"Graph JSON not found: {args.graph_path}"
    assert os.path.exists(args.ckpt_path), f"Checkpoint not found: {args.ckpt_path}"
    assert os.path.exists(args.config_path), f"Config not found: {args.config_path}"
    assert os.path.exists(args.gt_data_root), f"GT data root not found: {args.gt_data_root}"

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("SINGAPO Inference for Viser Trial")
    print("=" * 60)
    print(f"  Image:      {args.img_path}")
    print(f"  Graph:      {args.graph_path}")
    print(f"  Save dir:   {args.save_dir}")
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  GT data:    {args.gt_data_root}")
    print(f"  Samples:    {args.n_samples}")
    print()

    # Step 1: Extract DINO features
    feat = extract_dino_feature(args.img_path)

    # Step 2: Load graph
    with open(args.graph_path, "r") as f:
        pred_graph = json.load(f)
    print(f"Loaded graph with {len(pred_graph['diffuse_tree'])} nodes")

    # Step 3: Load input from graph
    data, cond = load_input_from(pred_graph, K=32)

    # Step 4: Prepare model input
    inputs = prepare_model_input(data, cond, feat, n_samples=args.n_samples)

    # Step 5: Set scheduler
    scheduler = set_scheduler(args.n_denoise_steps)

    # Step 6: Load model
    config = load_config(args.config_path)
    model = load_model(args.ckpt_path, config.system.model)

    # Step 7: Run inference
    with torch.no_grad():
        output = forward(model, scheduler, inputs, omega=args.omega).cpu().numpy()

    # Step 8: Post-process and save
    print("Post-processing...")
    N = output.shape[0]
    for i in range(N):
        out_json = _convert_json(output, cond)
        sample_dir = os.path.join(args.save_dir, str(i))
        os.makedirs(sample_dir, exist_ok=True)

        # Save object.json
        json_path = os.path.join(sample_dir, "object.json")
        with open(json_path, "w") as f:
            json.dump(out_json, f, indent=4)
        print(f"  Saved object.json to {json_path}")

        # Run mesh retrieval
        print(f"  Retrieving part meshes for sample {i}...")

        retrieve_py = os.path.join(SINGAPO_ROOT, "scripts", "mesh_retrieval", "retrieve.py")

        cmd = [
            sys.executable,
            retrieve_py,
            "--src_dir", sample_dir,
            "--json_name", "object.json",
            "--gt_data_root", args.gt_data_root,
        ]

        proc = subprocess.run(cmd, cwd=SINGAPO_ROOT)
        if proc.returncode != 0:
            print(f" !!! === WARNING: Mesh retrieval returned non-zero exit code: {proc.returncode} === !!!")

    print()
    print("=" * 60)
    print("Inference complete!")
    print(f"Output saved to: {args.save_dir}")
    print(f"To visualize, run:")
    print(f"  python run_visualizer.py --object_dir {os.path.join(args.save_dir, '0')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
