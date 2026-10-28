"""An empty world: a grid on the xy-plane and coordinate axes at the origin.

Fly around with WASD, space/ctrl and the mouse; scroll zooms and ESC quits.
"""

from animengine import Axes, Grid, PerspectiveCamera, Scene, preview


def main() -> None:
    camera = PerspectiveCamera(position=(4.0, -4.0, 3.0), target=(0.0, 0.0, 0.0))
    scene = Scene(size=(1280, 720), background="#202428", camera=camera)
    scene.add(Grid(size=10.0, step=1.0))
    scene.add(Axes(length=1.0, line_width=3.0))
    preview(scene, title="World origin", fly_camera=True)


if __name__ == "__main__":
    main()
