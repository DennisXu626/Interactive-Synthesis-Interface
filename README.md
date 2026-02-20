# Viser Trial: Interactive 3D Articulated Object Viewer

End-to-end pipeline that takes a single RGB image, runs SINGAPO inference to produce an articulated 3D object, and displays it in an interactive web-based viewer with direct drag controls.

## Prerequisites

- **Conda environment**: `4yp`
- **GPU**: CUDA-capable GPU (RTX 5080 or similar)
- **Model checkpoint**: `exps/singapo/final/ckpts/last.ckpt`
- **Config**: `exps/singapo/final/config/parsed.yaml`
- **Ground truth data**: `D:\4YP\data` (PartNet-Mobility dataset, ~13GB)
- **Installed packages**: viser, trimesh, torch, torchvision, diffusers, PIL

## Quick Start

### Step 1: Run Inference

```bash
conda activate 4yp
cd D:\4YP\singapo\Viser_trial
python run_inference.py
```

This will:
1. Load `demo_input.png` and `example_graph.json` from this folder
2. Extract DINO features from the input image
3. Run diffusion inference (100 denoising steps)
4. Retrieve part meshes from the PartNet-Mobility dataset
5. Save output to `output/0/` (object.json + plys/part_*.ply)

Takes a few minutes depending on GPU speed.

### Step 2: Run Visualizer

```bash
python run_visualizer.py --object_dir output/0
```

Then open **http://localhost:8080** in your browser.

## Viewer Controls

### Camera
- **Left-click drag** on empty space: Orbit camera around the object
- **Mouse scroll**: Zoom in/out
- **Right-click drag**: Pan camera

### Interacting with Parts
- **Click on a part**: Select it (turns white, shows info in GUI panel)
- **Drag a gizmo**: Directly manipulate joint articulation
  - **Revolute joints** (doors): Drag the rotation ring to open/close
  - **Prismatic joints** (drawers): Drag the arrow to slide open/close
  - Gizmos are the colored rings/arrows visible near movable joints

### GUI Panel (right side)
- **Input Image**: Shows the original input image
- **Selected Part**: Displays info about the clicked part (name, joint type, range)
- **Controls**:
  - **Reset All Joints**: Return all parts to rest position
  - **Animate**: Toggle automatic cyclic animation of all movable joints
  - **Export State**: Save current joint parameters to `joint_state.json`
- **Part Hierarchy**: Overview of all parts and their joint types

## File Structure

```
Viser_trial/
├── demo_input.png          # Input image
├── example_graph.json      # Input graph structure
├── run_inference.py         # Step 1: SINGAPO inference
├── run_visualizer.py        # Step 2: Viser viewer
├── README.md                # This file
└── output/                  # Generated after inference
    └── 0/
        ├── object.json      # Predicted articulation structure
        ├── object.ply       # Merged mesh
        └── plys/
            ├── part_0.ply   # Individual part meshes
            ├── part_1.ply
            └── ...
```

## Command Line Options

### run_inference.py
| Argument | Default | Description |
|----------|---------|-------------|
| `--img_path` | `demo_input.png` | Input image |
| `--graph_path` | `example_graph.json` | Graph structure JSON |
| `--save_dir` | `output` | Output directory |
| `--n_samples` | `1` | Number of output samples |
| `--omega` | `0.5` | Classifier-free guidance weight |
| `--n_denoise_steps` | `100` | Diffusion denoising steps |

### run_visualizer.py
| Argument | Default | Description |
|----------|---------|-------------|
| `--object_dir` | `output/0` | Directory with object.json + plys/ |
| `--img_path` | `demo_input.png` | Input image for GUI display |
| `--port` | `8080` | Viser web server port |

## Troubleshooting

**"Checkpoint not found"**: Make sure `exps/singapo/final/ckpts/last.ckpt` exists relative to the singapo root directory.

**"GT data root not found"**: Ensure the PartNet-Mobility data is at `D:\4YP\data`.

**Port already in use**: Use `--port 8081` or another available port.

**Mesh retrieval returns empty**: This can happen if the predicted structure doesn't match any candidate in the dataset. Try running inference again (different noise seed).

**Browser can't connect**: Make sure no firewall blocks localhost:8080. Try a different browser.

## How It Works

1. **Inference** (`run_inference.py`): Reuses the SINGAPO pipeline - DINOv2 feature extraction, DDPM diffusion model for predicting articulated structure (bounding boxes, joint types, joint axes), and graph-hash-based mesh retrieval from PartNet-Mobility.

2. **Visualization** (`run_visualizer.py`): Uses Viser to create a web-based 3D scene:
   - Each part is loaded as a trimesh and displayed with `add_mesh_simple()`
   - Movable joints get transform gizmos (`add_transform_controls()`) aligned to their axis
   - Gizmo updates trigger recomputation of the kinematic chain
   - The scene graph propagates parent-child transforms automatically
