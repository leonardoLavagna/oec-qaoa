"""Finite-sampling analysis for frozen ideal-QAOA states.

The functions in this module do not re-optimize QAOA parameters from noisy
objective estimates.  They sample a fixed ideal QAOA output distribution, which
isolates measurement uncertainty from ansatz expressivity and classical
optimizer behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingStatistics:
    """Statistics obtained from one finite-shot sample."""

    shots: int
    sampled_expected_energy: float
    sampled_optimal_probability: float
    sampled_feasible_probability: float
    best_sampled_energy: float
    optimum_observed: bool


def sample_distribution(
    probabilities: np.ndarray,
    energies: np.ndarray,
    optimal_mask: np.ndarray,
    feasible_mask: np.ndarray,
    shots: int,
    seed: int,
) -> SamplingStatistics:
    """Sample a computational-basis distribution without hardware noise."""

    if shots < 1:
        raise ValueError("shots must be positive")

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()

    if not (
        len(probabilities)
        == len(energies)
        == len(optimal_mask)
        == len(feasible_mask)
    ):
        raise ValueError("probability, energy and mask arrays must have equal length")

    rng = np.random.default_rng(seed)

    # searchsorted on one cumulative distribution avoids allocating a
    # multinomial count vector with one entry for every basis state.
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    draws = np.searchsorted(
        cumulative,
        rng.random(shots),
        side="right",
    )

    sampled_energies = energies[draws]
    sampled_optimal = optimal_mask[draws]
    sampled_feasible = feasible_mask[draws]

    return SamplingStatistics(
        shots=shots,
        sampled_expected_energy=float(sampled_energies.mean()),
        sampled_optimal_probability=float(sampled_optimal.mean()),
        sampled_feasible_probability=float(sampled_feasible.mean()),
        best_sampled_energy=float(sampled_energies.min()),
        optimum_observed=bool(sampled_optimal.any()),
    )


def exact_sampling_baseline(
    probabilities: np.ndarray,
    energies: np.ndarray,
    optimal_mask: np.ndarray,
    feasible_mask: np.ndarray,
) -> dict[str, float]:
    """Return exact distribution quantities against which sampling is compared."""

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()

    return {
        "exact_expected_energy": float(probabilities @ energies),
        "exact_optimal_probability": float(probabilities[optimal_mask].sum()),
        "exact_feasible_probability": float(probabilities[feasible_mask].sum()),
    }


def optimum_detection_probability(
    optimal_probability: float,
    shots: int,
) -> float:
    """Probability of observing at least one optimal sample."""

    if shots < 1:
        raise ValueError("shots must be positive")
    p = float(optimal_probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("optimal_probability must lie in [0, 1]")
    return float(1.0 - (1.0 - p) ** shots)
