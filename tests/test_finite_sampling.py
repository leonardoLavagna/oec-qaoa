"""Regression tests for finite-shot sampling utilities."""

import numpy as np

from oec_qaoa.finite_sampling import (
    exact_sampling_baseline,
    optimum_detection_probability,
    sample_distribution,
)


def test_sampling_baseline_matches_direct_expectation() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
    energies = np.asarray([0.0, 1.0, 2.0, 3.0])
    optimal = energies == energies.min()
    feasible = np.asarray([True, True, False, True])

    baseline = exact_sampling_baseline(
        probabilities,
        energies,
        optimal,
        feasible,
    )

    assert abs(
        baseline["exact_expected_energy"]
        - float(probabilities @ energies)
    ) < 1e-12
    assert abs(baseline["exact_optimal_probability"] - 0.1) < 1e-12
    assert abs(baseline["exact_feasible_probability"] - 0.7) < 1e-12


def test_detection_probability_formula() -> None:
    observed = optimum_detection_probability(0.25, 2)
    assert abs(observed - 0.4375) < 1e-12


def test_sampling_is_deterministic_for_fixed_seed() -> None:
    probabilities = np.asarray([0.2, 0.8])
    energies = np.asarray([0.0, 1.0])
    optimal = np.asarray([True, False])
    feasible = np.asarray([True, True])

    first = sample_distribution(
        probabilities,
        energies,
        optimal,
        feasible,
        shots=100,
        seed=17,
    )
    second = sample_distribution(
        probabilities,
        energies,
        optimal,
        feasible,
        shots=100,
        seed=17,
    )

    assert first == second
