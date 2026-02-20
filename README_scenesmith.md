# SceneSmith Interactive Room Viewer

A Viser-based interactive 3D room viewer for pre-generated SceneSmith scenes.
Part of the 4YP project: *Controllable Synthetic 3D Motion from a Single RGB Image*.

---

## Overview

This tool loads a SceneSmith scene (SDF + GLTF format) and renders it in a Viser
browser-based 3D viewer. Users can orbit the room, drag to articulate joints on
furniture (drawers, doors), and see real-time updates.

**Subtask 1** (complete) — Parse `.tar` into `scene_manifest.json`
**Subtask 2** (complete) — Render full room with textured meshes, orbit, and drag articulation
**Subtask 3** (complete) — Split-panel interaction: click object → isolated side panel → drag to articulate → update main scene

---

## Files

| File | Description |
|---|---|
| `parse_scene.py` | Parses a SceneSmith `.tar` into `scene_manifest.json` |
| `run_scene_viewer.py` | Main room viewer (Subtask 2) |
| `scenesmith_sample/scene_manifest.json` | Pre-parsed manifest for `scene_001.tar` |
| `scene_001.tar` | Downloaded SceneSmith bedroom scene |

---

## Usage

### Step 1: Parse a scene (only needed once per `.tar`)
```bash
python parse_scene.py --tar scene_001.tar --output_dir scenesmith_sample
```

### Step 2: Run the viewer
```bash
python run_scene_viewer.py
# then open http://localhost:8080
```

Optional flags:
```
--manifest  path to scene_manifest.json  (default: scenesmith_sample/scene_manifest.json)
--port      server port                  (default: 8080)
--skip      space-separated categories to exclude from loading
            choices: furniture manipuland wall_mounted ceiling_mounted room_geometry
            default: ceiling_mounted
```

Example — load only furniture and walls (faster):
```bash
python run_scene_viewer.py --skip ceiling_mounted manipuland wall_mounted
```

---

## Interactions

| Action | Effect |
|---|---|
| Drag on empty space | Orbit camera |
| Scroll | Zoom in/out |
| Drag on a drawer/door | Open/close the joint |
| Click any object | Shows object info in left panel |
| **Highlight Articulated** button | Toggles orange wireframe boxes around all articulated objects |
| **Animate All** button | Cycles all joints open→closed in a loop |
| **Reset All Joints** button | Returns all joints to closed/rest position |
| **Reset View** button (top right) | Returns camera to initial position |

---

## Architecture

### Coordinate Systems

SceneSmith uses two coordinate conventions that must be reconciled:

| System | Convention | Used for |
|---|---|---|
| SDF (Simulation Description Format) | Z-up (X=right, Y=forward, Z=up) | Geometry positions, joint axes, world poses |
| GLTF / Viser | Y-up (X=right, Y=up, Z=back) | Mesh geometry, Three.js renderer |

The conversion matrix applied everywhere:
```
R_SDF_TO_VISER = [[1, 0,  0],
                  [0, 0,  1],
                  [0, -1, 0]]
```
This maps SDF `[x, y, z]` → Viser `[x, z, -y]`.

### Scene Manifest (`scene_manifest.json`)

Produced by `parse_scene.py`. Stores all objects with:
- `world_pose`: translation + rotation matrix in SDF Z-up world
- `links`: list of visual meshes, each with `name`, `gltf` path, `mesh_scale`, and `visual_offset` (pose within the SDF link frame)
- `joints`: prismatic/revolute joint parameters (axis, origin, limit in parent-link frame)

Key design: a single SDF link can have **multiple visuals** (e.g. `room_geometry` has floor + 4 walls + windows as separate visual entries). Each visual becomes its own entry in the `links` list with a `visual_offset`.

### Forward Kinematics

For articulated objects, each link's world transform is:
```
T_world_link = T_world_base  ⊗  Π(joint_delta_i)
```
Where `joint_delta` for prismatic joints is a translation, and for revolute joints is a rotation about the joint origin.

Joint parameter `t ∈ [0, 1]`:
- `t = 0` → joint at SDF rest (neutral) position
- `t = 1` → joint fully open (toward dominant end of limit range)

### Drag Interaction

Since Viser has no native drag events on meshes, drag uses a side-channel WebSocket:
1. `pointerdown` → JS sends `hit_test` message → Python raycasts scene → returns hit object name
2. If hit on articulated object, JS cancels orbit (synthetic `pointerup` to Three.js), enters drag mode
3. `pointermove` → JS sends `drag_move` → Python maps screen delta to joint parameter → updates FK transforms

---

## Known Issues and Bug History

### Bug 1: Wall/ceiling objects not visible (fixed)
**Symptom**: Only furniture was visible; walls, ceiling art, sconces were absent.

**Root cause**: `parse_scene.py` only read the **first** `<visual>` element per SDF link. The `room_geometry` SDF has 11 visuals in a single link (floor, 4 interior walls, 4 exterior walls, 2 windows). All walls were silently dropped.

**Fix**: Updated `parse_scene.py` to iterate all `<visual>` elements per link. Each visual becomes its own entry with a `visual_offset` field (the `<pose>` of that visual within the link frame). Updated `run_scene_viewer.py` to apply `visual_offset` on top of the FK transform.

### Bug 2: Wardrobe doors parallel to ground (fixed)
**Symptom**: Wardrobe doors appeared lying flat instead of standing upright.

**Root cause**: `joint_delta_transform` computed `angle = lo + t*(hi-lo)`. For the left door (limit `[-π/2, 0]`), at `t=0` this gave `angle = -π/2` instead of 0 (SDF rest position).

**Fix**: New formula — at `t=0`, angle=0 (SDF neutral); at `t=1`, angle = dominant end of limit range (`lo if |lo| > |hi| else hi`).

### Bug 3: Room had no walls / wrong coordinate system (fixed)
**Symptom**: The scene appeared with furniture floating in empty space; vertical panels (canvas prints) were lying flat.

**Root cause**: Viewer used `set_up_direction("+z")` in Viser, but GLTF meshes are Y-up. The Three.js world rotation applied to the scene made Y-up panels appear horizontal. All positions/rotations from SDF Z-up were passed directly to Viser without coordinate conversion.

**Fix**: Changed to `set_up_direction("+y")`. Added `R_SDF_TO_VISER` matrix applied at every SDF→Viser boundary (positions, rotations, camera, raycasting, drag delta mapping).

### Bug 4: Wardrobe wrong size (fixed)
**Symptom**: Wardrobe appeared at 1× scale but SDF specified 1.75× scale.

**Root cause**: `parse_scene.py` did not read `<scale>` from `<visual><geometry><mesh>`.

**Fix**: Parser now reads `mesh_scale` per visual. Viewer passes `scale=mesh_scale` to `add_glb` and applies it to the raycasting trimesh.

### Bug 5: Orbit around room corner instead of center (fixed)
**Symptom**: Camera orbited around `[0, 0, 0]` (a room corner in Viser Y-up) rather than the room center.

**Root cause 1**: `client.camera.position` setter calls the `position` getter, which asserts `update_timestamp != 0`. On connect, the camera handle is uninitialized, causing an AssertionError that silently swallowed the `look_at` call too (it was never reached).

**Root cause 2**: Camera was positioned at a fixed hardcoded point `[2.25, 3.5, 3.0]` rather than relative to the computed room center.

**Fix**:
- Added `_compute_room_center()` that finds the room AABB by intersecting wall visual positions from the manifest.
- Wrapped `client.camera.position` in `try/except` so `look_at` is always sent.
- Camera position and look_at now derived from computed room centre.

---

## Subtask 3 — Split-Panel Interaction (implemented)

When an articulated object is clicked or dragged in the main room view, a 360 px side panel slides in from the right edge of the browser window.  The panel embeds a second Viser server (`localhost:{port+2}`) as an `<iframe>` that renders **only the selected object** in isolation, centred at the origin, with its own camera, orbit controls, and drag-to-articulate interaction.

### How it works

| Port | Service |
|---|---|
| `port` (8080) | Main room Viser server |
| `port+1` (8081) | Main room drag WebSocket |
| `port+2` (8082) | Mini object Viser server (side panel) |
| `port+3` (8083) | Mini object drag WebSocket |

1. **Object selection** → `SceneViewer._select_object()` calls `ObjectMiniViewer.focus_object(obj_name)`, which clears old handles, computes the object AABB centroid, loads all its meshes offset so the centroid is at the origin, and positions the mini camera to frame the object.
2. **Panel injection** → Python sends a `RunJavascriptMessage` to all connected main clients calling `window.__miniPanelAPI.show(objName)`, which creates a `<div>` panel with a header, hint bar, and `<iframe src="localhost:8082">`.
3. **Drag in mini panel** → The same `DRAG_JS_TEMPLATE` + `DragWebSocketServer` pattern runs inside the iframe. `ObjectMiniViewer._on_drag_move()` applies the joint delta and syncs `scene.joint_params` back to the main `SceneViewer`, so the room also updates live.
4. **Close panel** → The `✕` button removes the panel div from the DOM.

---

## Dependencies

```
viser==1.0.21   (local at D:/4YP/singapo/viser)
trimesh
numpy
websockets
pyyaml
```

Conda environment: `4yp`
Python: `C:/Users/Dennis/miniconda3/envs/4yp/python.exe`
