"""Run the frozen ideal-state QAOA benchmark study."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oec_qaoa.benchmarks import (
    FROZEN_QAOA_BENCHMARKS,
    build_frozen_benchmark,
    physical_feasibility_mask,
)
from oec_qaoa.qaoa import optimize_qaoa


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "qaoa"
RESULTS.mkdir(parents=True, exist_ok=True)

DEPTHS = [1, 2, 3, 4]
RESTART_SEEDS = [101, 202, 303]
MAXITER = 100


def main() -> None:
    records = []

    for benchmark in FROZEN_QAOA_BENCHMARKS:
        instance, retained, model = build_frozen_benchmark(benchmark)
        feasible_mask = physical_feasibility_mask(
            instance,
            retained,
            model,
        )

        for depth in DEPTHS:
            for restart, seed in enumerate(RESTART_SEEDS, start=1):
                result = optimize_qaoa(
                    model,
                    depth=depth,
                    seed=seed,
                    feasible_mask=feasible_mask,
                    optimizer="COBYLA",
                    maxiter=MAXITER,
                )

                records.append(
                    {
                        "benchmark_id": benchmark.benchmark_id,
                        "instance": instance.name,
                        "depth": depth,
                        "restart": restart,
                        "seed": seed,
                        "optimizer": result.optimizer,
                        "optimizer_success": result.success,
                        "expected_energy": result.expected_energy,
                        "optimal_energy": result.optimal_energy,
                        "expected_gap": (
                            result.expected_energy - result.optimal_energy
                        ),
                        "optimal_probability": result.optimal_probability,
                        "feasible_probability": result.feasible_probability,
                        "most_likely_energy": result.most_likely_energy,
                        "n_evaluations": result.n_evaluations,
                        "logical_variables": len(model.variable_names),
                    }
                )

                print(
                    benchmark.benchmark_id,
                    f"p={depth}",
                    f"restart={restart}",
                    f"E={result.expected_energy:.6f}",
                    f"Popt={result.optimal_probability:.6f}",
                    f"Pfeas={result.feasible_probability:.6f}",
                    f"evals={result.n_evaluations}",
                )

    frame = pd.DataFrame(records)
    frame.to_csv(
        RESULTS / "ideal_qaoa_runs.csv",
        index=False,
    )

    summary = (
        frame.groupby(
            ["benchmark_id", "depth", "logical_variables"]
        )
        .agg(
            expected_energy_mean=("expected_energy", "mean"),
            expected_energy_best=("expected_energy", "min"),
            expected_gap_mean=("expected_gap", "mean"),
            optimal_probability_mean=("optimal_probability", "mean"),
            optimal_probability_best=("optimal_probability", "max"),
            feasible_probability_mean=("feasible_probability", "mean"),
            feasible_probability_best=("feasible_probability", "max"),
            n_evaluations_mean=("n_evaluations", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        RESULTS / "ideal_qaoa_summary.csv",
        index=False,
    )

    print("\n=== Ideal-QAOA summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
