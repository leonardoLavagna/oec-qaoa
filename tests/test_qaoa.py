"""Regression tests for the ideal statevector QAOA implementation."""

import numpy as np

from oec_qaoa.benchmarks import (
    FROZEN_QAOA_BENCHMARKS,
    build_frozen_benchmark,
    physical_feasibility_mask,
)
from oec_qaoa.qaoa import (
    basis_energies,
    expected_energy,
    optimize_qaoa,
    qaoa_statevector,
)


def test_qaoa_statevector_is_normalized() -> None:
    _, _, model = build_frozen_benchmark(FROZEN_QAOA_BENCHMARKS[0])
    energies = basis_energies(model)

    state = qaoa_statevector(
        energies,
        gammas=np.asarray([0.2]),
        betas=np.asarray([0.4]),
    )

    assert abs(float(np.vdot(state, state).real) - 1.0) < 1e-10


def test_zero_angles_reproduce_uniform_expectation() -> None:
    _, _, model = build_frozen_benchmark(FROZEN_QAOA_BENCHMARKS[0])
    energies = basis_energies(model)

    parameters = np.zeros(2)
    observed = expected_energy(parameters, energies, depth=1)

    assert abs(observed - float(energies.mean())) < 1e-10


def test_physical_feasibility_mask_contains_exact_optimum() -> None:
    instance, retained, model = build_frozen_benchmark(
        FROZEN_QAOA_BENCHMARKS[0]
    )
    energies = basis_energies(model)
    feasible = physical_feasibility_mask(instance, retained, model)

    optimum = float(energies.min())
    optimal_states = np.isclose(energies, optimum, atol=1e-9)

    assert np.any(optimal_states)
    assert np.any(feasible & optimal_states)


def test_small_qaoa_run_returns_valid_probabilities() -> None:
    instance, retained, model = build_frozen_benchmark(
        FROZEN_QAOA_BENCHMARKS[0]
    )
    feasible = physical_feasibility_mask(instance, retained, model)

    result = optimize_qaoa(
        model,
        depth=1,
        seed=11,
        feasible_mask=feasible,
        maxiter=10,
    )

    assert 0.0 <= result.optimal_probability <= 1.0
    assert 0.0 <= result.feasible_probability <= 1.0
    assert result.n_evaluations > 0
