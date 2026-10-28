import numpy as np
import pytest

from animengine.physics import (
    FixedStepper,
    ParticleSimulation,
    ParticleState,
    RectangularBoundary,
    forward_euler,
    velocity_verlet,
)

INTEGRATORS = (forward_euler, velocity_verlet)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_force_free_motion_and_zero_dt(integrator):
    state = ParticleState([[0, 1], [1, -1]], [[2, -3], [-1, 0.5]])
    simulation = ParticleSimulation(state, integrator=integrator)
    positions = state.positions
    velocities = state.velocities
    simulation.step(0)
    assert np.array_equal(positions, [[0, 1], [1, -1]])
    simulation.step(0.5)
    assert positions is state.positions
    assert velocities is state.velocities
    assert np.allclose(positions, [[1, -0.5], [0.5, -0.75]])
    assert np.allclose(velocities, [[2, -3], [-1, 0.5]])
    assert np.all(state.accelerations == 0)


@pytest.mark.parametrize(
    ("integrator", "expected_position", "expected_velocity"),
    [(forward_euler, 1.0, -0.1), (velocity_verlet, 0.995, -0.09975)],
)
def test_integrators_with_position_dependent_acceleration(
    integrator, expected_position, expected_velocity
):
    state = ParticleState([[1, 0]], [[0, 0]])
    simulation = ParticleSimulation(state, acceleration=lambda positions: -positions,
                                    integrator=integrator)
    assert np.allclose(state.accelerations, [[-1, 0]])
    simulation.step(0.1)
    assert state.positions[0, 0] == pytest.approx(expected_position)
    assert state.velocities[0, 0] == pytest.approx(expected_velocity)
    assert np.allclose(state.accelerations, -state.positions)


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize(
    ("position", "velocity", "dt", "expected_position", "expected_velocity"),
    [
        ([0, 0], [1, 0], 1, [1, 0], [-1, 0]),
        ([0, 0], [-1, 0], 1, [-1, 0], [1, 0]),
        ([0, 0], [0, 1], 1, [0, 1], [0, -1]),
        ([0, 0], [0, -1], 1, [0, -1], [0, 1]),
        ([0, 0], [1, 1], 1.25, [0.75, 0.75], [-1, -1]),
        ([0, 0], [1, -1], 7.25, [-0.75, 0.75], [1, -1]),
        ([1, 0], [1, 0], 0.25, [0.75, 0], [-1, 0]),
        ([-1, 0], [-1, 0], 0.25, [-0.75, 0], [1, 0]),
        ([1, 0], [-1, 0], 0.25, [0.75, 0], [-1, 0]),
    ],
)
def test_elastic_walls(integrator, position, velocity, dt, expected_position, expected_velocity):
    state = ParticleState([position], [velocity])
    simulation = ParticleSimulation(state, integrator=integrator,
                                    boundary=RectangularBoundary((-1, -1), (1, 1)))
    simulation.step(dt)
    assert np.allclose(state.positions, [expected_position])
    assert np.allclose(state.velocities, [expected_velocity])
    assert np.linalg.norm(state.velocities) == pytest.approx(np.linalg.norm(velocity))


def test_walls_recompute_acceleration_at_reflected_position():
    state = ParticleState([[0.9, 0]], [[1, 0]])
    simulation = ParticleSimulation(
        state, acceleration=lambda positions: -positions, integrator=velocity_verlet,
        boundary=RectangularBoundary((-1, -1), (1, 1)),
    )
    simulation.step(0.3)
    assert state.positions[0, 0] < 1
    assert state.velocities[0, 0] < 0
    assert np.allclose(state.accelerations, -state.positions)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_many_bounces_match_smaller_steps(integrator):
    boundary = RectangularBoundary((-2, 1), (1, 5))
    states = [ParticleState([[0, 2]], [[-7, 11]]) for _ in range(2)]
    simulations = [
        ParticleSimulation(state, integrator=integrator, boundary=boundary) for state in states
    ]
    simulations[0].step(3.0)
    for _ in range(300):
        simulations[1].step(0.01)
    assert np.allclose(states[0].positions, states[1].positions, atol=1e-12)
    assert np.allclose(states[0].velocities, states[1].velocities, atol=1e-12)


def test_particle_state_copies_inputs_and_supports_empty_arrays():
    positions = np.array([[0, 0]], dtype=np.float32)
    state = ParticleState(positions, [[1, 2]])
    positions[0, 0] = 4
    assert state.positions.dtype == np.float64
    assert state.positions[0, 0] == 0
    empty = ParticleState(np.empty((0, 2)), np.empty((0, 2)))
    ParticleSimulation(empty, boundary=RectangularBoundary((-1, -1), (1, 1))).step(1)
    assert empty.positions.shape == (0, 2)


@pytest.mark.parametrize(
    ("positions", "velocities"),
    [([1, 2], [[1, 2]]), ([[1, 2]], [[1, 2, 3]]), ([[float("nan"), 0]], [[0, 0]])],
)
def test_invalid_particle_states(positions, velocities):
    with pytest.raises(ValueError):
        ParticleState(positions, velocities)


@pytest.mark.parametrize("dt", [-1, float("nan"), float("inf")])
def test_invalid_step_does_not_change_state(dt):
    state = ParticleState([[0, 0]], [[1, 0]])
    simulation = ParticleSimulation(state)
    with pytest.raises(ValueError, match="dt"):
        simulation.step(dt)
    assert np.array_equal(state.positions, [[0, 0]])


def test_invalid_boundary_and_outside_initial_position():
    with pytest.raises(ValueError):
        RectangularBoundary((1, 0), (0, 1))
    with pytest.raises(ValueError, match="inside"):
        ParticleSimulation(ParticleState([[2, 0]], [[0, 0]]),
                           boundary=RectangularBoundary((-1, -1), (1, 1)))


def test_invalid_acceleration_is_reported_without_mutating_state():
    state = ParticleState([[0, 0]], [[1, 0]])
    simulation = ParticleSimulation(state)
    simulation.acceleration = lambda positions: np.full_like(positions, np.nan)
    with pytest.raises(ValueError, match="acceleration"):
        simulation.step(1)
    assert np.array_equal(state.positions, [[0, 0]])
    assert np.array_equal(state.velocities, [[1, 0]])


def test_invalid_acceleration_at_end_of_step_does_not_mutate_state():
    state = ParticleState([[0, 0]], [[1, 0]])

    def acceleration(positions):
        if positions[0, 0] > 0:
            return np.full_like(positions, np.nan)
        return np.zeros_like(positions)

    simulation = ParticleSimulation(state, acceleration=acceleration)
    with pytest.raises(ValueError, match="acceleration"):
        simulation.step(1)
    assert np.array_equal(state.positions, [[0, 0]])
    assert np.array_equal(state.velocities, [[1, 0]])
    assert np.array_equal(state.accelerations, [[0, 0]])


def test_fixed_stepper_retains_remainder_and_matches_variable_elapsed_times():
    fixed_calls = []
    variable_calls = []
    fixed = FixedStepper(fixed_calls.append, 0.1)
    variable = FixedStepper(variable_calls.append, 0.1)
    assert fixed.advance(0) == 0
    for _ in range(10):
        fixed.advance(0.1)
    for elapsed in (0.03, 0.22, 0.01, 0.34, 0.4):
        variable.advance(elapsed)
    assert fixed_calls == variable_calls == [0.1] * 10
    assert variable.remainder == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_fixed_stepper_rejects_bad_step_sizes(dt):
    with pytest.raises(ValueError):
        FixedStepper(lambda _: None, dt)


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf")])
def test_fixed_stepper_rejects_bad_elapsed_time(elapsed):
    stepper = FixedStepper(lambda _: None, 0.1)
    with pytest.raises(ValueError):
        stepper.advance(elapsed)
    assert stepper.remainder == 0


def test_fixed_stepper_rejects_unrepresentable_subtraction():
    calls = []
    stepper = FixedStepper(calls.append, 1e-300)
    with pytest.raises(ValueError, match="too small"):
        stepper.advance(1.0)
    assert calls == []
