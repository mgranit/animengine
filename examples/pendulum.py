"""Two simple pendulums released from opposite sides, integrated with RK4."""

import argparse

import numpy as np

from animengine import Lines, OrthographicCamera, Scene, preview, render_video

GRAVITY = 1.0
LENGTH = 0.9
INITIAL_ANGLES = (np.pi / 2, -np.pi / 2)


def derivative(state: np.ndarray) -> np.ndarray:
    """state has one [theta, omega] row per pendulum."""
    theta, omega = state[:, 0], state[:, 1]
    return np.stack([omega, -GRAVITY / LENGTH * np.sin(theta)], axis=1)


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rod_segments(state: np.ndarray) -> np.ndarray:
    """Line segments from the pivot to each bob."""
    bobs = LENGTH * np.stack([np.sin(state[:, 0]), -np.cos(state[:, 0])], axis=1)
    return np.stack([np.zeros_like(bobs), bobs], axis=1).reshape(-1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="play in a window instead")
    parser.add_argument("--out", default="output/pendulum.mp4")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds of video")
    args = parser.parse_args()

    scene = Scene(size=(800, 800), camera=OrthographicCamera(height=2.0))
    rods = scene.add(Lines(color="#ffffff"))

    state = np.array([[angle, 0.0] for angle in INITIAL_ANGLES])
    rods.set_points(rod_segments(state))

    def update(t: float, dt: float) -> None:
        nonlocal state
        state = rk4_step(state, dt)
        rods.set_points(rod_segments(state))

    if args.preview:
        preview(scene, update, title="Pendulums")
    else:
        render_video(scene, update, duration=args.duration, fps=60, out=args.out,
                     steps_per_frame=4)


if __name__ == "__main__":
    main()
