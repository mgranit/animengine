"""Random, non-interacting particles bouncing within a 2D orthographic view."""

import argparse

import numpy as np

from animengine import OrthographicCamera, Points, Scene, preview, render_video
from animengine.physics import (
    FixedStepper,
    ParticleSimulation,
    ParticleState,
    RectangularBoundary,
    forward_euler,
    velocity_verlet,
)

SIZE = (1280, 720)
VIEW_HEIGHT = 2.0
VIEW_WIDTH = VIEW_HEIGHT * SIZE[0] / SIZE[1]
FPS = 60
STEPS_PER_FRAME = 4
PHYSICS_DT = 1.0 / (FPS * STEPS_PER_FRAME)
INTEGRATORS = {"euler": forward_euler, "verlet": velocity_verlet}


def build_animation(
    count: int = 250, speed: float = 0.5, seed: int = 3156, integrator: str = "verlet",
) -> tuple[Scene, Points, ParticleSimulation, FixedStepper]:
    if count < 0:
        raise ValueError("particle count must be non-negative")
    if not np.isfinite(speed) or speed < 0:
        raise ValueError("speed must be finite and non-negative")
    rng = np.random.default_rng(seed)
    half_extent = np.array([VIEW_WIDTH, VIEW_HEIGHT]) / 2.0
    positions = rng.uniform(-half_extent, half_extent, size=(count, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
    velocities = speed * np.column_stack((np.cos(angles), np.sin(angles)))

    state = ParticleState(positions, velocities)
    simulation = ParticleSimulation(
        state,
        integrator=INTEGRATORS[integrator],
        boundary=RectangularBoundary(-half_extent, half_extent),
    )
    stepper = FixedStepper(simulation.step, PHYSICS_DT)
    camera = OrthographicCamera(width=VIEW_WIDTH, height=VIEW_HEIGHT)
    scene = Scene(size=SIZE, camera=camera)
    points = Points(state.positions, color="#ffffff", size=5.0)
    scene.add(points)
    return scene, points, simulation, stepper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="play in a window instead")
    parser.add_argument("--out", default="output/bouncing_points.mp4")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds of video")
    parser.add_argument("--seed", type=int, default=3156)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--speed", type=float, default=0.5, help="world units per second")
    parser.add_argument("--integrator", choices=INTEGRATORS, default="verlet")
    args = parser.parse_args()

    scene, points, simulation, stepper = build_animation(
        args.count, args.speed, args.seed, args.integrator
    )

    def update(_: float, dt: float) -> None:
        stepper.advance(dt)
        points.set_points(simulation.state.positions)

    if args.preview:
        preview(scene, update, title="Bouncing points")
    else:
        render_video(
            scene, update, duration=args.duration, fps=FPS, out=args.out,
            steps_per_frame=STEPS_PER_FRAME,
        )


if __name__ == "__main__":
    main()
