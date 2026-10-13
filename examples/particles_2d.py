"""Softened gravitational attraction between 100 particles, integrated with velocity Verlet."""

import argparse

import numpy as np

from animengine import OrthographicCamera, Points, Scene, preview, render_video

N_PARTICLES = 100
G = 1e-5
MAX_FORCE = 0.1


def calc_force(pos: np.ndarray) -> np.ndarray:
    r_ij = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    distance = np.linalg.norm(r_ij, axis=-1)
    inv_distance = np.divide(1.0, distance, out=np.zeros_like(distance), where=distance != 0.0)
    force = G * (inv_distance**3)[:, :, np.newaxis] * r_ij

    # Cap the magnitude of each pairwise force so close encounters stay finite.
    magnitude = np.linalg.norm(force, axis=-1, keepdims=True)
    scale = np.divide(MAX_FORCE, magnitude, out=np.ones_like(magnitude),
                      where=magnitude >= MAX_FORCE)
    return np.sum(scale * force, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="play in a window instead")
    parser.add_argument("--out", default="output/particles_2d.mp4")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds of video")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    pos = rng.uniform(-0.25, 0.25, (N_PARTICLES, 2))
    vel = np.zeros_like(pos)
    acc = calc_force(pos)

    scene = Scene(size=(1280, 720), camera=OrthographicCamera(height=2.0))
    particles = scene.add(Points(pos, size=6.0))

    def update(t: float, dt: float) -> None:
        nonlocal acc
        pos[:] += vel * dt + 0.5 * acc * dt**2
        new_acc = calc_force(pos)
        vel[:] += 0.5 * (acc + new_acc) * dt
        acc = new_acc
        particles.set_points(pos)

    if args.preview:
        preview(scene, update, title="Particles 2D")
    else:
        render_video(scene, update, duration=args.duration, fps=60, out=args.out)


if __name__ == "__main__":
    main()
