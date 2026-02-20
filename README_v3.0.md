# Visualizer v3.0: Direct Drag (No Selection Step)

## What Changed from v2.0

### Problem with v2.0
v2.0 required a **click-to-select** step before dragging:
1. Click a part to select it (overlay appears, orbit disabled)
2. Drag to articulate
3. Click empty space or Escape to deselect (orbit re-enabled)

This three-step cycle (select -> drag -> deselect) interrupted the flow. Users had to constantly switch between "orbit mode" and "manipulation mode".

### v3.0 Improvement
**No selection step.** Just grab a part and drag it. Drag empty space to orbit. The system automatically determines what the user is interacting with on every pointer press.

| v2.0 | v3.0 |
|------|------|
| Click part -> drag -> deselect -> orbit | Just drag part OR drag empty space |
| Two modes (orbit / manipulation) | Single unified mode |
| Overlay appears/disappears | Overlay always present, transparent |
| Must deselect to orbit | Orbit works between any two drags |

## Architecture

```
Browser                              Python Server
+----------------------+      +-----------------------------+
|  Viser Frontend      |      |  Viser Server (port 8080)   |
|  + Injected JS       |      |  + Drag WS Server (8081)    |
|                      |      |                              |
|  pointerdown --------+--ws--+-> hit_test(screen_x, y)     |
|                      |      |    -> ray-mesh intersection  |
|                      |  ws  |                              |
|  hit_result(part) <--+------+-- {part_id: 3} or {null}    |
|                      |      |                              |
|  [if hit]:           |      |                              |
|    drag_start -------+--ws--+-> highlight part, start drag |
|    drag_move  -------+--ws--+-> update joint parameter     |
|    drag_end   -------+--ws--+-> unhighlight, end drag      |
|                      |      |                              |
|  [if miss]:          |      |                              |
|    -> disable overlay|      |                              |
|    -> synthetic pointerdown on canvas                      |
|    -> orbit starts normally                                |
|    -> re-enable overlay on pointerup                       |
+----------------------+      +-----------------------------+
```

### Key Difference: Bidirectional WebSocket

v2.0's WebSocket was **unidirectional** (JS -> Python only). v3.0 is **bidirectional**: Python sends `hit_result` responses back to JS. This enables the hit-test-on-pointerdown pattern.

## Key Implementation Details

### 1. Always-On Overlay with Hit-Test

The transparent overlay is created once when the client connects and never removed. On every `pointerdown`:

1. JS captures the event and sends `hit_test(screen_x, screen_y)` to Python
2. JS enters `'pending'` state, queuing any pointer moves
3. Python computes a 3D ray from the screen coordinates and tests it against all meshes
4. Python responds with `{part_id: X}` (hit) or `{part_id: null}` (miss)
5. JS receives the response (typically within ~5-15ms on localhost):
   - **Hit**: Enter drag mode. Cursor becomes `grabbing`. Replay any queued moves as drag events.
   - **Miss**: Release pointer capture, set `overlay.style.pointerEvents = 'none'`, dispatch synthetic `pointerdown` on the canvas -> orbit starts naturally. Re-enable overlay on `pointerup`.

### 2. Python-Side Raycasting

New `_screen_to_ray()` method converts normalized screen coordinates to a 3D ray:

```python
def _screen_to_ray(self, screen_x, screen_y, client):
    cam_pos = client.camera.position      # 3D camera position
    cam_look_at = client.camera.look_at    # look-at point
    fov = client.camera.fov                # vertical FOV in radians
    aspect = client.camera.aspect           # width / height

    # Build view coordinate frame
    forward = normalize(cam_look_at - cam_pos)
    right = normalize(cross(forward, world_up))
    up = cross(right, forward)

    # NDC -> ray direction (perspective projection)
    ndc_x = (screen_x - 0.5) * 2.0
    ndc_y = (0.5 - screen_y) * 2.0
    ray_dir = normalize(
        forward
        + ndc_x * aspect * tan(fov/2) * right
        + ndc_y * tan(fov/2) * up
    )
    return cam_pos, ray_dir
```

`_hit_test()` then tests this ray against all meshes using trimesh:

```python
def _hit_test(self, screen_x, screen_y, client):
    ray_origin, ray_dir = self._screen_to_ray(screen_x, screen_y, client)

    for each node with a movable ancestor:
        T_inv = inverse(global_transform[node_id])
        # Transform ray to mesh-local coordinates
        origin_local = T_inv @ ray_origin
        dir_local = normalize(T_inv[:3,:3] @ ray_dir)
        # Trimesh ray intersection
        hits = mesh.ray.intersects_location(origin_local, dir_local)
        # Track closest hit
    return closest_movable_ancestor_id  # or None
```

### 3. Movable-Ancestor Lookup (`effective_movable`)

A precomputed map that allows dragging any child part to articulate the nearest movable ancestor's joint:

```
StorageFurniture example:
  Part 0 (base, fixed)        -> effective_movable = None
  Part 1 (drawer_1, prismatic) -> effective_movable = 1
  Part 2 (handle_1, fixed)     -> effective_movable = 1  (parent is drawer_1)
  Part 3 (drawer_2, prismatic) -> effective_movable = 3
  Part 4 (handle_2, fixed)     -> effective_movable = 3  (parent is drawer_2)
  ...
```

This means clicking on a door handle articulates the door's joint, which is the intuitive behavior.

### 4. Orbit Passthrough via Synthetic Events

When the hit test returns "miss", the JS needs to let orbit happen despite the overlay having captured the initial `pointerdown`. The solution:

```javascript
// 1. Release pointer capture from overlay
overlay.releasePointerCapture(savedPointerId);

// 2. Make overlay transparent to events
overlay.style.pointerEvents = 'none';

// 3. Dispatch synthetic pointerdown on the canvas
var canvas = document.querySelector('canvas');
canvas.dispatchEvent(new PointerEvent('pointerdown', {
    clientX: startX, clientY: startY,
    button: 0, buttons: 1,
    bubbles: true, pointerId: savedPointerId,
    pointerType: 'mouse', isPrimary: true
}));

// 4. Subsequent real pointermove events pass through overlay to canvas
//    (pointer-events: none means overlay is invisible to events)

// 5. On pointerup, restore overlay
document.addEventListener('pointerup', function() {
    overlay.style.pointerEvents = 'auto';
}, {once: true, capture: true});
```

### 5. Scroll-Wheel Passthrough

Since the overlay covers the viewport, scroll events would be blocked. The JS clones wheel events and dispatches them on the canvas:

```javascript
overlay.addEventListener('wheel', function(e) {
    var canvas = document.querySelector('canvas');
    canvas.dispatchEvent(new WheelEvent(e.type, e));
}, {passive: false});
```

### 6. Highlight During Drag

No persistent selection state. Instead:
- **Drag starts**: Part turns white, info text shows joint details
- **Drag ends**: Part returns to its original color, info text resets

## Usage

```bash
conda activate 4yp
cd D:\4YP\singapo\Viser_trial
python run_visualizer_3.0.py --object_dir output/0
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
| Drag a movable part | Articulate the joint |
| Drag a handle/child part | Articulate the parent joint |
| Drag empty space | Orbit camera |
| Scroll | Zoom |
| GUI: Reset All Joints | Return to rest pose |
| GUI: Animate | Toggle cyclic animation |
| GUI: Export State | Save joint_state.json |

## Version Comparison

| Feature | v1.0 (Gizmo) | v2.0 (Click+Drag) | v3.0 (Direct Drag) |
|---------|--------------|-------------------|-------------------|
| Selection step | N/A (drag gizmo) | Click to select | None |
| Drag target | Gizmo arrows/rings | Overlay (after select) | Overlay (always) |
| Orbit availability | Always | Only when deselected | Always (between drags) |
| Visual clutter | Gizmo handles visible | Clean | Clean |
| Hit detection | Three.js raycasting | Viser mesh.on_click | Python trimesh raycasting |
| WebSocket | None | Unidirectional (JS->Py) | Bidirectional |
| Handle drag | Must find gizmo | Must select drawer first | Drag handle -> moves drawer |
| Latency | Instant | Instant (after select) | ~5-15ms hit test |

## Technical Notes

- **Hit test latency**: The round-trip (JS -> Python raycasting -> JS) is ~5-15ms on localhost. This is imperceptible — the user's finger hasn't moved 5 pixels yet.
- **Thread safety**: The drag WebSocket handler runs in an asyncio thread. Shared state mutations (`joint_params`, `_update_meshes`) are protected by the GIL and are fast enough to avoid contention.
- **Camera state timing**: `client.camera.fov` and `.aspect` require at least one camera state update from the frontend. This happens automatically on connect; the first drag might fail if attempted within milliseconds of page load (extremely rare).
