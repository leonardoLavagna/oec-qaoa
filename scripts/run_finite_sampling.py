"""Run finite-shot sampling on frozen ideal-QAOA parameterizations.

For each benchmark and depth, the deterministic restart that produced the lowest
ideal expected energy in the frozen 5-restart protocol is re-optimized once to
recover its parameters.  The resulting ideal state is then sampled repeatedly
at several shot budgets.  This isolates measurement uncertainty from the
variational and classical-optimization errors already characterized in
Notebook 05.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from oec_qaoa.benchmarks import (
    FROZEN_QAOA_BENCHMARKS,
    build_frozen_benchmark,
    physical_feasibility_mask,
)
from oec_qaoa.finite_sampling import (
    exact_sampling_baseline,
    optimum_detection_probability,
    sample_distribution,
)
from oec_qaoa.qaoa import (
    basis_energies,
    optimize_qaoa,
    qaoa_statevector,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "qaoa"
RESULTS.mkdir(parents=True, exist_ok=True)

DEPTHS = [1, 2, 3, 4]
SHOTS = [128, 512, 2048, 8192]
REPETITIONS = 30
MAXITER = 200

# Best-energy restarts from the frozen ideal-QAOA experiment.
BEST_SEEDS = {
    ("B1", 1): 101,
    ("B1", 2): 404,
    ("B1", 3): 303,
    ("B1", 4): 404,
    ("B2", 1): 303,
    ("B2", 2): 303,
    ("B2", 3): 404,
    ("B2", 4): 404,
    ("B3", 1): 202,
    ("B3", 2): 404,
    ("B3", 3): 303,
    ("B3", 4): 404,
}


def _cost_scale(model) -> float:
    coefficients = [
        abs(float(value))
        for value in model.linear
        if abs(float(value)) > 1e-12
    ]
    coefficients.extend(
        abs(float(value))
        for value in model.quadratic.values()
        if abs(float(value)) > 1e-12
    )
    return max(coefficients) if coefficients else 1.0


def main() -> None:
    records = []
    state_rows = []

    for benchmark in FROZEN_QAOA_BENCHMARKS:
        instance, retained, model = build_frozen_benchmark(benchmark)
        feasible_mask = physical_feasibility_mask(
            instance,
            retained,
            model,
        )
        energies = basis_energies(model)
        optimum = float(energies.min())
        optimal_mask = np.isclose(energies, optimum, atol=1e-9)
        scale = _cost_scale(model)
        phase_energies = energies / scale

        for depth in DEPTHS:
            seed = BEST_SEEDS[(benchmark.benchmark_id, depth)]
            ideal = optimize_qaoa(
                model,
                depth=depth,
                seed=seed,
                feasible_mask=feasible_mask,
                optimizer="COBYLA",
                maxiter=MAXITER,
                normalize_cost=True,
            )

            state = qaoa_statevector(
                phase_energies,
                ideal.parameters[:depth],
                ideal.parameters[depth:],
            )
            probabilities = np.abs(state) ** 2
            baseline = exact_sampling_baseline(
                probabilities,
                energies,
                optimal_mask,
                feasible_mask,
            )

            state_rows.append(
                {
                    "benchmark_id": benchmark.benchmark_id,
                    "depth": depth,
                    "seed": seed,
                    "ideal_expected_energy": baseline["exact_expected_energy"],
                    "ideal_optimal_probability": baseline[
                        "exact_optimal_probability"
                    ],
                    "ideal_feasible_probability": baseline[
                        "exact_feasible_probability"
                    ],
                    "optimal_energy": optimum,
                    "n_evaluations": ideal.n_evaluations,
                }
            )

            for shots in SHOTS:
                detection_probability = optimum_detection_probability(
                    baseline["exact_optimal_probability"],
                    shots,
                )

                for repetition in range(REPETITIONS):
                    sample_seed = (
                        1_000_000
                        + 10_000 * int(benchmark.benchmark_id[1:])
                        + 1_000 * depth
                        + 10 * SHOTS.index(shots)
                        + repetition
                    )
                    sampled = sample_distribution(
                        probabilities,
                        energies,
                        optimal_mask,
                        feasible_mask,
                        shots=shots,
                        seed=sample_seed,
                    )

                    records.append(
                        {
                            "benchmark_id": benchmark.benchmark_id,
                            "depth": depth,
                            "shots": shots,
                            "repetition": repetition,
                            "sample_seed": sample_seed,
                            "sampled_expected_energy": (
                                sampled.sampled_expected_energy
                            ),
                            "sampled_optimal_probability": (
                                sampled.sampled_optimal_probability
                            ),
                            "sampled_feasible_probability": (
                                sampled.sampled_feasible_probability
                            ),
                            "best_sampled_energy": sampled.best_sampled_energy,
                            "optimum_observed": sampled.optimum_observed,
                            "ideal_expected_energy": baseline[
                                "exact_expected_energy"
                            ],
                            "ideal_optimal_probability": baseline[
                                "exact_optimal_probability"
                            ],
                            "ideal_feasible_probability": baseline[
                                "exact_feasible_probability"
                            ],
                            "analytic_optimum_detection_probability": (
                                detection_probability
                            ),
                        }
                    )

    runs = pd.DataFrame(records)
    states = pd.DataFrame(state_rows)

    runs.to_csv(
        RESULTS / "finite_sampling_runs.csv",
        index=False,
    )
    states.to_csv(
        RESULTS / "finite_sampling_states.csv",
        index=False,
    )

    summary = (
        runs.groupby(["benchmark_id", "depth", "shots"])
        .agg(
            sampled_expected_energy_mean=(
                "sampled_expected_energy",
                "mean",
            ),
            sampled_expected_energy_std=(
                "sampled_expected_energy",
                "std",
            ),
            sampled_optimal_probability_mean=(
                "sampled_optimal_probability",
                "mean",
            ),
            sampled_feasible_probability_mean=(
                "sampled_feasible_probability",
                "mean",
            ),
            optimum_observation_rate=("optimum_observed", "mean"),
            best_sampled_energy_mean=("best_sampled_energy", "mean"),
            ideal_expected_energy=("ideal_expected_energy", "first"),
            ideal_optimal_probability=(
                "ideal_optimal_probability",
                "first",
            ),
            ideal_feasible_probability=(
                "ideal_feasible_probability",
                "first",
            ),
            analytic_optimum_detection_probability=(
                "analytic_optimum_detection_probability",
                "first",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        RESULTS / "finite_sampling_summary.csv",
        index=False,
    )

    print("=== Frozen ideal states ===")
    print(states.to_string(index=False))
    print("\n=== Finite-sampling summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
