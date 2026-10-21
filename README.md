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
| `animengine.gpu` | Thin OpenGL wrappers: buffers, shaders, textures, framebuffers |
| `animengine.render` | Cameras, meshes, materials, the `Renderer` (GLSL lives in `render/shaders`) |
| `animengine.scene` | `Scene`, `Node` and ready-made objects (`Cube`, `Lines`, `Points`, ...) |
| `animengine.anim` | Easing, video export, `render_video`, real-time `preview` |

## Conventions

- The world is right-handed and z-up. Matrices are row-major numpy arrays.
- `update(t, dt)` brings the scene to time `t` after `dt` seconds. Offline, `dt` is always
  `1 / (fps * steps_per_frame)`.
- Output is H.264 (`yuv420p`), so width and height must be even.

## Examples

See [examples/](examples). Each script renders a video to `output/`; pass `--preview` to play
it in a window instead.
