"""A resizable square window showing an XY grid with an orthographic camera."""

from animengine import Grid, OrthographicCamera, Scene, preview


def main() -> None:
    camera = OrthographicCamera(width=12.0, height=12.0)
    scene = Scene(
        size=(800, 800),
        background="#202428",
        camera=camera,
        bar_color="#d05090",
    )
    scene.add(Grid(size=10.0, step=1.0))
    preview(scene, title="Orthographic projection")


if __name__ == "__main__":
    main()
