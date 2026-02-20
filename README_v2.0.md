# Visualizer v2.0: Click-to-Select + Direct Drag

## What Changed from v1.0

### Problem with v1.0
v1.0 used **TransformControls gizmos** (colored arrows/rings) placed at each joint. Users had to:
1. Find the small gizmo near the joint hinge
2. Click precisely on the correct arrow or ring
3. Drag the gizmo handle to articulate

This was unintuitive — the gizmo arrows were small, hard to click (especially for people with limited dexterity), and cluttered the 3D scene visually.

### v2.0 Improvement
Replaced gizmos with **click-to-select + direct drag on the mesh itself**:
1. Click a part to select it (highlights white, shows joint info)
2. Drag anywhere on the viewport to articulate the selected joint
3. Click empty space or press Escape to deselect and resume orbit

No gizmo arrows visible. Clean 3D view. The drag direction is automatically mapped to the joint's axis of motion.

## Architecture

```
Browser                              Python Server
+----------------------+      +-----------------------------+
|  Viser Frontend      |      |  Viser Server (port 8080)   |
|  + Injected JS       |      |  + Drag WS Server (8081)    |
|                      |      |                              |
|  mesh.on_click ------+------+-> select part (highlight)    |
|                      |      |    -> enable drag overlay    |
|  overlay pointerdown +--ws--+-> drag_start                 |
|  overlay pointermove +--ws--+-> drag_move -> update joint  |
|  overlay pointerup   +--ws--+-> drag_end / deselect        |
|  Escape key          +--ws--+-> deselect -> remove overlay |
|                      |      |                              |
|  <-------------------+------+-- update mesh transforms     |
+----------------------+      +-----------------------------+
```

### Why Two Servers?

Viser v1.0.21 only supports `"click"` events on meshes — no drag, no mousemove, no pointerdown. Enabling `scene.on_pointer_event("click")` disables camera orbit entirely. So we use:

- **Viser** (port 8080): 3D rendering, mesh display, `mesh.on_click` for part selection (does NOT disable orbit)
- **Side-channel WebSocket** (port 8081): Receives real-time drag events from injected JavaScript

### Key Implementation Details

#### 1. JavaScript Injection via `RunJavascriptMessage`

Viser allows injecting arbitrary JS into the browser frontend:
```python
from viser._messages import RunJavascriptMessage
client._websock_connection.queue_message(
    RunJavascriptMessage(source="/* JS code here */")
)
```
The JS runs via `new Function(source)()` in the browser context. We use this to set up the drag overlay and WebSocket connection.

#### 2. Transparent Overlay for Drag Capture

When a movable part is selected, Python tells the JS to create a full-viewport transparent `<div>` (z-index 99999) on top of the 3D canvas. This overlay:
- Captures all pointer events (blocking orbit)
- Sends pointerdown/pointermove/pointerup to Python via the side-channel WebSocket
- Distinguishes click-vs-drag using a 5px movement threshold
- Click without drag = deselect (overlay removed, orbit restored)
- Escape key = deselect

#### 3. Screen-Delta to Joint Parameter Mapping

The core algorithm that maps 2D mouse drag to 1D joint articulation:

**For prismatic joints (drawers):**
1. Read camera position/look_at from `client.camera`
2. Compute view coordinate frame: `right = cross(view_dir, world_up)`, `up = cross(right, view_dir)`
3. Project the joint axis direction to 2D screen space: `screen_axis = (dot(d, right), dot(d, up))`
4. Compute screen delta from drag start to current position
5. Project screen delta onto the screen axis (dot product) -> displacement
6. Scale by camera distance for consistent feel at any zoom
7. Map to parameter delta t in [0, 1]

**For revolute joints (doors):**
1. Compute tangent direction: `cross(joint_axis, view_dir)` — the direction a surface point would move under rotation
2. Project tangent to screen space
3. Same screen delta projection and scaling

#### 4. Per-Client State

Each browser tab gets its own `_ClientDragState` (selection, drag progress), keyed by `client.client_id`. Multiple users can interact independently.

## Usage

```bash
conda activate 4yp
cd D:\4YP\singapo\Viser_trial
python run_visualizer_2.0.py --object_dir output/0
```

Open http://localhost:8080.

| Argument | Default | Description |
|----------|---------|-------------|
| `--object_dir` | `output/0` | Directory with object.json + plys/ |
| `--img_path` | `demo_input.png` | Input image for GUI display |
| `--port` | `8080` | Viser port (drag WS uses port+1) |

## Controls

| Action | Effect |
|--------|--------|
| Left-drag on viewport | Orbit camera |
| Scroll | Zoom |
| Click a part | Select (highlight + info) |
| Drag after selecting movable part | Articulate joint |
| Click without drag / Escape | Deselect, orbit restored |
| GUI: Reset All Joints | Return to rest pose |
| GUI: Animate | Toggle cyclic animation |
| GUI: Export State | Save joint_state.json |

## Limitations

- **Two-step interaction**: Must click to select before dragging. Not ideal for quick manipulation.
- **Orbit blocked during selection**: While a movable part is selected, camera orbit is disabled (overlay captures all events). Must deselect first to orbit.
- **Single joint at a time**: Can only manipulate one joint per interaction cycle.
