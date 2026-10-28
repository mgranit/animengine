# animengine

An OpenGL engine for creating 2D and 3D animations offline, in Python.

Scenes are rendered with a fixed timestep to an offscreen framebuffer and piped to ffmpeg, so a
video looks the same on every machine regardless of how fast it renders. The same scene can be
played in a window in real time for development.

```python
import numpy as np
from animengine import Lines, OrthographicCamera, Scene, preview, render_video

scene = Scene(
    size=(1280, 720),
    camera=OrthographicCamera(width=16 / 9 * 2.0, height=2.0),
)
line = scene.add(Lines(color="#ffcc00"))

def update(t, dt):
    angle = 2.0 * t
    line.set_points([[0, 0, 0], [np.cos(angle), np.sin(angle), 0]])

render_video(scene, update, duration=5, fps=60, out="output/spin.mp4")
# preview(scene, update)
```

An orthographic camera with both `width` and `height` set keeps fixed view bounds. If the output
aspect ratio differs, the view is centered without stretching and the remaining bars use the
scene background color by default; set `Scene(bar_color=...)` to choose another bar color.

## Install

```
pip install -e ".[dev]"
```

ffmpeg is bundled through `imageio-ffmpeg`; nothing else needs to be installed. An OpenGL 3.3
capable GPU and driver is required (offline rendering uses a hidden window).

## Layout

Each layer imports only from the layers above it in this list.

| Package | Responsibility |
|---|---|
| `animengine.linalg` | `Frame` (origin + orthonormal basis), transform, view and projection matrices |
| `animengine.platform` | GLFW window, input events, clocks |
| `animengine.gpu` | Thin OpenGL wrappers: buffers, shaders, framebuffers |
| `animengine.render` | Cameras, meshes, materials, the `Renderer` (GLSL lives in `render/shaders`) |
| `animengine.scene` | `Scene`, `Node` and ready-made objects (`Cube`, `Lines`, `Points`, ...) |
| `animengine.anim` | Easing, video export, `render_video`, real-time `preview` |
| `animengine.physics` | CPU-only 2D particle state, integrators, rectangular walls and fixed stepping |

## Conventions

- The world is right-handed and z-up. Matrices are row-major numpy arrays.
- `update(t, dt)` brings the scene to time `t` after `dt` seconds. Offline, `dt` is always
  `1 / (fps * steps_per_frame)`.
- Output is H.264 (`yuv420p`), so width and height must be even.

## Particle physics

`animengine.physics` is independent of the scene and renderer. A `ParticleState` owns float64
`(N, 2)` arrays for positions, velocities and accelerations in world units. `ParticleSimulation`
advances the state in place; update a `Points` object with `set_points(state.positions)` after
stepping so the renderer sees the changes.

```python
from animengine.physics import (
    ParticleSimulation, ParticleState, RectangularBoundary, forward_euler,
)

state = ParticleState([[0.0, 0.0]], [[0.8, 0.5]])
simulation = ParticleSimulation(
    state,
    integrator=forward_euler,  # or velocity_verlet (the default)
    boundary=RectangularBoundary((-1.0, -1.0), (1.0, 1.0)),
)
simulation.step(1 / 240)
```

By default, acceleration is zero and particles do not interact. To add forces, pass an
`acceleration(positions)` function that returns an `(N, 2)` array in world units per second
squared. The callback must depend only on positions; it can implement force divided by mass.
Acceleration is initialized and refreshed by the simulation. Euler is explicit Euler;
velocity Verlet uses acceleration at the start and end of a step. Both produce the same free
motion, so a nonzero force is needed to compare their integration behavior.

The rectangular boundary reflects particle **centers** elastically. For free motion it preserves
remaining travel, including corners, exact wall hits, and multiple bounces in one step. Point
sizes are in pixels, so their visible edges may extend past the world-space walls. With nonzero
acceleration, wall handling folds the unconstrained endpoint and reflects the final velocity:
this is an approximation, not an impact-time solver. Reduce the timestep near walls; the
acceleration callback must also accept the unconstrained endpoint outside the rectangle. The
simulation recomputes acceleration at the reflected position for the next step.

`FixedStepper(simulation.step, dt)` accumulates elapsed time and advances in fixed steps, keeping
any fractional remainder. This is useful because `preview` supplies variable wall-clock `dt`,
while offline rendering supplies fixed `dt`. The stepper does not discard elapsed time or cap
catch-up steps; a long preview pause can require many steps. With the same seed, integrator and
fixed timestep, both modes follow the same state sequence, although preview can display a
fractional-step-old state. A zero elapsed time does nothing.

[`examples/bouncing_points.py`](examples/bouncing_points.py) demonstrates this setup:

```
python examples/bouncing_points.py --preview --seed 3156 --integrator verlet
python examples/bouncing_points.py --integrator euler --out output/bouncing_points.mp4
```

Its camera and simulation rectangle share fixed world-space bounds; the camera letterboxes when
the window aspect ratio differs. No collisions between particles are calculated.

## Examples

See [examples/](examples). Each script renders a video to `output/`; pass `--preview` to play
it in a window instead.
