# Interactive 3D Scene Viewer — Elliott's Catch-Up Guide

> **Note on large files:** The model checkpoint, GT retrieval data, and pre-downloaded SceneSmith scene archives are too large for GitHub. They are shared via Google Drive — **please check the Slack chat for the link**.

---

## What This Is

This repo contains two interactive 3D pipelines built on top of the [SINGAPO](https://github.com/Red-Fairy/SINGAPO) framework, both served through a browser-based viewer ([Viser](https://github.com/nerfstudio-project/viser)):

| Pipeline | Input | What you see |
|---|---|---|
| **SceneSmith Room Viewer** | A pre-generated SceneSmith bedroom scene (`.tar`) | Full textured room with articulated furniture you can drag |
| **SINGAPO Inference** | A single RGB photo + part-graph JSON | Predicted articulated 3D object from the photo, interactively manipulable |

Both open in your browser at `http://localhost:8080`. No GUI installation needed.

---

## Environment Setup

Everything runs in the `4yp` conda environment.

```bash
conda activate 4yp
cd D:\4YP\singapo\Viser_trial
```

If you are setting up from scratch, create the environment and install dependencies as documented separately. The key packages are: `torch`, `trimesh`, `viser` (local copy at `D:\4YP\singapo\viser`), `websockets`, `pyyaml`, `Pillow`.

---

## Pipeline 1 — SceneSmith Interactive Room Viewer

This pipeline takes a pre-generated synthetic bedroom scene from [SceneSmith](https://huggingface.co/datasets/nepfaff/scenesmith-example-scenes) and renders it as a fully interactive room with textured meshes.

### Concepts

- **SceneSmith** generates synthetic indoor scenes consisting of SDF (robot simulation format) + GLTF (textured 3D meshes) + a Drake YAML scene layout.
- **`parse_scene.py`** reads the `.tar` archive and extracts all objects, their world poses, joint definitions, and GLTF paths into a single `scene_manifest.json`.
- **`run_scene_viewer.py`** reads the manifest and:
  - Loads all objects as textured GLB meshes into Viser.
  - Applies forward kinematics (FK) to place every link at its correct world position.
  - Infers the room centre from wall geometry to set the orbit pivot correctly.
  - Launches a drag-to-articulate interaction layer (prismatic drawers, revolute doors).
  - Opens a **side panel mini-viewer** (at port 8082) showing the selected object in isolation — drag parts there too and the main room scene updates in sync.

### Step 1 — Parse the scene archive

```bash
python parse_scene.py --tar scene_001.tar --output_dir scenesmith_sample
```

This produces `scenesmith_sample/scene_manifest.json` and extracts all assets into `scenesmith_sample/scene_extracted/`. You only need to run this once per scene file.

> `scene_001.tar` is the pre-downloaded SceneSmith bedroom scene. Get it from the Google Drive link in Slack.

### Step 2 — Launch the room viewer

```bash
python run_scene_viewer.py
```

Open **`http://localhost:8080`** in your browser.

**Controls:**

| Action | How |
|---|---|
| Orbit the room | Left-drag on empty floor/background |
| Zoom | Scroll wheel |
| Select an object | Left-click any furniture or wall item |
| Articulate a drawer/door | Left-drag directly on a highlighted (orange) part |
| Reset all joints | "Reset All Joints" button in the sidebar |
| Animate all joints | "Animate All" button in the sidebar |
| Highlight articulated objects | "Highlight Articulated" button (shows orange bounding boxes) |
| Inspect an object in isolation | Click it — the right-side mini panel loads it centred |

**Optional flags:**

```bash
# Skip additional categories to speed up loading
python run_scene_viewer.py --skip ceiling_mounted wall_mounted

# Use a different scene manifest
python run_scene_viewer.py --manifest path/to/scene_manifest.json

# Change ports (default: main=8080, drag WS=8081, mini viewer=8082)
python run_scene_viewer.py --port 8090
```

Available category names for `--skip`: `furniture`, `manipuland`, `wall_mounted`, `ceiling_mounted`, `room_geometry`.
Default skips `ceiling_mounted` (light fixtures) only; all walls, floor, and furniture load by default.

---

## Pipeline 2 — SINGAPO Single-Image Inference

This pipeline takes a **single RGB photograph** of an articulated object and predicts its 3D structure using the SINGAPO diffusion model, then lets you interact with the result.

### Concepts

- **DINO features** are extracted from the input image (ViT-B/14 backbone).
- A **diffusion model** denoises a predicted part-graph structure (bounding boxes, joint types, joint axes).
- A **mesh retrieval** step finds the closest matching part meshes from the PartNet-Mobility GT dataset and assembles them according to the predicted structure.
- The result is an articulated 3D object rendered in the same Viser viewer.

### Large files required (get from Google Drive — see Slack)

| File | Default location | Size |
|---|---|---|
| Model checkpoint | `D:\4YP\singapo\exps\singapo\final\ckpts\last.ckpt` | ~500 MB |
| PartNet-Mobility GT data | `D:\4YP\data\` | ~13 GB |

### Mode 1 — Full pipeline (inference + viewer)

```bash
python run_whole.py --mode infer --img_path demo_input.png --graph_path example_graph.json
```

`demo_input.png` and `example_graph.json` are already in the `Viser_trial` folder.
The graph JSON tells the model how many parts to expect and their connectivity — it does **not** need to be exact; it serves as a structural prior.

The pipeline will print progress for each stage (DINO extraction → diffusion → mesh retrieval → viewer launch). Retrieval is the slowest step (up to a few minutes depending on the object category).

### Mode 2 — View a previously computed result

If inference has already run and you just want to re-open the viewer:

```bash
python run_whole.py --mode visualize --object_dir output/0
```

### Interactive mode (asks which mode to use)

```bash
python run_whole.py
```

**Viewer controls** (same Viser interface):

| Action | How |
|---|---|
| Orbit | Left-drag on background |
| Zoom | Scroll wheel |
| Move a part | Left-drag on the part directly |
| Reset pose | "Reset" button in sidebar |

---

## Repository Layout (key files)

```
Viser_trial/
├── parse_scene.py          # Step 1: SceneSmith .tar → scene_manifest.json
├── run_scene_viewer.py     # Step 2: room viewer (main entry point for SceneSmith)
├── run_whole.py            # SINGAPO inference + viewer (single-image pipeline)
├── demo_input.png          # Example input image for SINGAPO
├── example_graph.json      # Example part-graph prior for SINGAPO
├── scenesmith_sample/
│   ├── scene_manifest.json     # Generated by parse_scene.py
│   └── scene_extracted/        # Extracted GLTF assets from the .tar
└── output/
    └── 0/                      # SINGAPO inference output (object.json + PLY meshes)
```

---

## Coordinate System Notes (for reference)

- **SDF / Drake** uses Z-up (X=right, Y=forward, Z=up).
- **GLTF / Viser** uses Y-up (X=right, Y=up, Z=toward viewer).
- All FK and joint calculations are done in SDF Z-up; results are converted to Viser Y-up before rendering. This is handled transparently — no manual conversion needed when using the scripts.
