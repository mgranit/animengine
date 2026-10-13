"""100,000 particles falling towards the origin; fly around with WASD in --preview."""

import argparse

import numpy as np

from animengine import Axes, Grid, PerspectiveCamera, Points, Scene, preview, render_video

N_PARTICLES = 100_000
ORBIT_RADIUS = 12.0
ORBIT_HEIGHT = 6.0
ORBIT_SPEED = 0.3  # rad/s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="play in a window instead")
    parser.add_argument("--out", default="output/particles_3d.mp4")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds of video")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    pos = rng.uniform(-5.0, 5.0, (N_PARTICLES, 3))
    vel = np.zeros_like(pos)

    camera = PerspectiveCamera(position=(ORBIT_RADIUS, 0.0, ORBIT_HEIGHT))
    scene = Scene(size=(1920, 1080), background="#335555", camera=camera)
    scene.add(Grid(size=10.0, step=1.0))
    scene.add(Axes(length=1.0, line_width=2.0))
    particles = scene.add(Points(pos, size=1.5))

    def update(t: float, dt: float) -> None:
        pos[:] += vel * dt
        distance = np.linalg.norm(pos, axis=1, keepdims=True)
        vel[:] += -pos / (distance + 1e-6) * dt
        particles.set_points(pos)

    def update_with_orbit(t: float, dt: float) -> None:
        update(t, dt)
        angle = ORBIT_SPEED * t
        camera.position = np.array([ORBIT_RADIUS * np.cos(angle), ORBIT_RADIUS * np.sin(angle),
                                    ORBIT_HEIGHT])

    if args.preview:
        preview(scene, update, title="Particles 3D", fly_camera=True)
    else:
        render_video(scene, update_with_orbit, duration=args.duration, fps=60, out=args.out)


if __name__ == "__main__":
    main()
