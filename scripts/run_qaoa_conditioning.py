"""Compare raw and normalized ideal-QAOA parameterizations.

This diagnostic isolates optimizer conditioning from the physical benchmark
definition.  The QUBO itself is unchanged; only a positive rescaling of the
cost Hamiltonian is applied during the variational search.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oec_qaoa.benchmarks import (
    FROZEN_QAOA_BENCHMARKS,
    build_frozen_benchmark,
    physical_feasibility_mask,
)
from oec_qaoa.qaoa import basis_energies, optimize_qaoa


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "qaoa"
RESULTS.mkdir(parents=True, exist_ok=True)

DEPTHS = [1, 2, 3]
RESTART_SEEDS = [101, 202, 303]
MAXITER = 150


def main() -> None:
    records = []

    # Calibration uses B1 and B2 first; B3 is retained for the final protocol
    # after the conditioning choice has been fixed.
    for benchmark in FROZEN_QAOA_BENCHMARKS[:2]:
        instance, retained, model = build_frozen_benchmark(benchmark)
        feasible_mask = physical_feasibility_mask(
            instance,
            retained,
            model,
        )
        energies = basis_energies(model)

        uniform_optimal_probability = float(
            (energies == energies.min()).sum() / len(energies)
        )
        uniform_feasible_probability = float(feasible_mask.mean())
        uniform_expected_energy = float(energies.mean())

        for normalize in (False, True):
            for depth in DEPTHS:
                for restart, seed in enumerate(RESTART_SEEDS, start=1):
                    result = optimize_qaoa(
                        model,
                        depth=depth,
                        seed=seed,
                        feasible_mask=feasible_mask,
                        optimizer="COBYLA",
                        maxiter=MAXITER,
                        normalize_cost=normalize,
                    )

                    records.append(
                        {
                            "benchmark_id": benchmark.benchmark_id,
                            "normalize_cost": normalize,
                            "depth": depth,
                            "restart": restart,
                            "seed": seed,
                            "expected_energy": result.expected_energy,
                            "expected_gap": (
                                result.expected_energy
                                - result.optimal_energy
                            ),
                            "optimal_probability": (
                                result.optimal_probability
                            ),
                            "feasible_probability": (
                                result.feasible_probability
                            ),
                            "n_evaluations": result.n_evaluations,
                            "uniform_expected_energy": (
                                uniform_expected_energy
                            ),
                            "uniform_optimal_probability": (
                                uniform_optimal_probability
                            ),
                            "uniform_feasible_probability": (
                                uniform_feasible_probability
                            ),
                        }
                    )

    frame = pd.DataFrame(records)
    frame.to_csv(
        RESULTS / "qaoa_conditioning_runs.csv",
        index=False,
    )

    summary = (
        frame.groupby(
            ["benchmark_id", "normalize_cost", "depth"]
        )
        .agg(
            expected_energy_mean=("expected_energy", "mean"),
            expected_energy_best=("expected_energy", "min"),
            optimal_probability_mean=("optimal_probability", "mean"),
            optimal_probability_best=("optimal_probability", "max"),
            feasible_probability_mean=("feasible_probability", "mean"),
            feasible_probability_best=("feasible_probability", "max"),
            n_evaluations_mean=("n_evaluations", "mean"),
            uniform_expected_energy=("uniform_expected_energy", "first"),
            uniform_optimal_probability=(
                "uniform_optimal_probability",
                "first",
            ),
            uniform_feasible_probability=(
                "uniform_feasible_probability",
                "first",
            ),
        )
        .reset_index()
    )
    summary.to_csv(
        RESULTS / "qaoa_conditioning_summary.csv",
        index=False,
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
