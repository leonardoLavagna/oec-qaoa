"""Select exact-optimum-known instances for the ideal-QAOA study.

The selection stage is intentionally classical. It searches a controlled pool
of synthetic OEC instances, solves the full constrained reference problem,
solves nested candidate-path reductions, builds the corresponding path QUBOs
and records the resource/faithfulness trade-off. The resulting table is used
to freeze a small benchmark set before any variational optimization is run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oec_qaoa.generators import SyntheticConfig, synthetic_relay_instance
from oec_qaoa.ilp import solve_reference_ilp
from oec_qaoa.metrics import qubo_resource_metrics
from oec_qaoa.path_qubo import build_path_qubo
from oec_qaoa.path_reduction import (
    candidate_sets,
    decomposition_gap,
    solve_path_selection,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "qaoa"
RESULTS.mkdir(parents=True, exist_ok=True)

SATELLITES = [3, 4]
TIME_CYCLES = [3, 4, 5]
SERVICES = [2, 3, 4]
CAPACITY_FACTORS = [0.8, 1.0, 1.2]
SEEDS = [20260929, 20260930, 20261001]
K_VALUES = [1, 2, 3, 4]


def build_pool() -> pd.DataFrame:
    records = []

    for n_satellites in SATELLITES:
        for n_time_cycles in TIME_CYCLES:
            for n_services in SERVICES:
                for capacity_factor in CAPACITY_FACTORS:
                    for seed in SEEDS:
                        config = SyntheticConfig(
                            n_satellites=n_satellites,
                            n_time_cycles=n_time_cycles,
                            n_services=n_services,
                            capacity_factor=capacity_factor,
                            seed=seed,
                        )
                        instance = synthetic_relay_instance(config)
                        reference = solve_reference_ilp(instance)

                        if not reference.success:
                            continue

                        all_paths = candidate_sets(instance)
                        max_available = min(
                            len(paths)
                            for paths in all_paths.values()
                        )

                        for k in K_VALUES:
                            if k > max_available:
                                continue

                            retained = {
                                service: paths[:k]
                                for service, paths in all_paths.items()
                            }
                            reduced = solve_path_selection(
                                instance,
                                retained,
                            )
                            if not reduced.success:
                                records.append(
                                    {
                                        "instance": instance.name,
                                        "n_satellites": n_satellites,
                                        "n_time_cycles": n_time_cycles,
                                        "n_services": n_services,
                                        "capacity_factor": capacity_factor,
                                        "seed": seed,
                                        "K": k,
                                        "reference_objective": reference.objective,
                                        "reduced_feasible": False,
                                        "reduced_objective": None,
                                        "decomposition_gap": None,
                                        "relative_gap": None,
                                        "semantic_variables": None,
                                        "slack_variables": None,
                                        "logical_variables": None,
                                        "quadratic_couplings": None,
                                        "qubo_density": None,
                                        "max_degree": None,
                                        "coefficient_dynamic_range": None,
                                    }
                                )
                                continue

                            model = build_path_qubo(instance, retained)
                            metrics = qubo_resource_metrics(model)
                            gap = decomposition_gap(
                                reference.objective,
                                reduced,
                            )
                            relative_gap = (
                                gap / abs(reference.objective)
                                if abs(reference.objective) > 1e-12
                                else (0.0 if abs(gap) <= 1e-12 else float("inf"))
                            )

                            records.append(
                                {
                                    "instance": instance.name,
                                    "n_satellites": n_satellites,
                                    "n_time_cycles": n_time_cycles,
                                    "n_services": n_services,
                                    "capacity_factor": capacity_factor,
                                    "seed": seed,
                                    "K": k,
                                    "reference_objective": reference.objective,
                                    "reduced_feasible": True,
                                    "reduced_objective": reduced.objective,
                                    "decomposition_gap": gap,
                                    "relative_gap": relative_gap,
                                    "semantic_variables": metrics.semantic_variables,
                                    "slack_variables": metrics.slack_variables,
                                    "logical_variables": metrics.logical_variables,
                                    "quadratic_couplings": metrics.quadratic_couplings,
                                    "qubo_density": metrics.qubo_density,
                                    "max_degree": metrics.max_degree,
                                    "coefficient_dynamic_range": (
                                        metrics.coefficient_dynamic_range
                                    ),
                                }
                            )

    return pd.DataFrame(records)


def choose_benchmarks(pool: pd.DataFrame) -> pd.DataFrame:
    faithful = pool[
        (pool["reduced_feasible"])
        & (pool["decomposition_gap"].abs() <= 1e-9)
        & (pool["logical_variables"].between(6, 20))
        & (pool["reference_objective"] > 1e-9)
    ].copy()

    if faithful.empty:
        return faithful

    faithful["size_band"] = pd.cut(
        faithful["logical_variables"],
        bins=[5, 10, 14, 20],
        labels=["small", "medium", "large"],
        include_lowest=True,
    )

    selected = []
    used_instances = set()
    used_signatures = set()

    # Prefer K=2 when possible: it retains a non-trivial per-service choice
    # while remaining substantially smaller than the direct edge encoding.
    for band in ["small", "medium", "large"]:
        candidates = faithful[faithful["size_band"] == band].copy()
        if candidates.empty:
            continue

        candidates["k_preference"] = (candidates["K"] - 2).abs()
        candidates = candidates.sort_values(
            [
                "k_preference",
                "logical_variables",
                "quadratic_couplings",
                "n_services",
                "seed",
            ]
        )

        for _, row in candidates.iterrows():
            signature = (
                int(row["n_services"]),
                int(row["K"]),
                int(row["logical_variables"]),
                int(row["quadratic_couplings"]),
                round(float(row["reference_objective"]), 9),
                round(float(row["qubo_density"]), 9),
                round(float(row["coefficient_dynamic_range"]), 9),
            )
            if (
                row["instance"] in used_instances
                or signature in used_signatures
            ):
                continue
            selected.append(row)
            used_instances.add(row["instance"])
            used_signatures.add(signature)
            break

    # Add one second medium-scale instance if available so the QAOA conclusions
    # are not tied to a single synthetic realization.
    medium = faithful[faithful["size_band"] == "medium"].copy()
    medium["k_preference"] = (medium["K"] - 2).abs()
    medium = medium.sort_values(
        [
            "k_preference",
            "quadratic_couplings",
            "logical_variables",
            "seed",
        ]
    )
    for _, row in medium.iterrows():
        signature = (
            int(row["n_services"]),
            int(row["K"]),
            int(row["logical_variables"]),
            int(row["quadratic_couplings"]),
            round(float(row["reference_objective"]), 9),
            round(float(row["qubo_density"]), 9),
            round(float(row["coefficient_dynamic_range"]), 9),
        )
        if (
            row["instance"] in used_instances
            or signature in used_signatures
        ):
            continue
        selected.append(row)
        used_instances.add(row["instance"])
        used_signatures.add(signature)
        break

    if not selected:
        return faithful.iloc[0:0]

    result = pd.DataFrame(selected).drop(columns=["k_preference"], errors="ignore")
    result.insert(0, "benchmark_id", [
        f"B{index + 1}" for index in range(len(result))
    ])
    return result


def main() -> None:
    pool = build_pool()
    selected = choose_benchmarks(pool)

    pool.to_csv(
        RESULTS / "benchmark_selection_pool.csv",
        index=False,
    )
    selected.to_csv(
        RESULTS / "qaoa_benchmarks.csv",
        index=False,
    )

    print("=== Candidate pool ===")
    print(f"Rows: {len(pool)}")
    print(
        "Feasible reductions: "
        f"{int(pool['reduced_feasible'].sum())}/{len(pool)}"
    )

    faithful = pool[
        (pool["reduced_feasible"])
        & (pool["decomposition_gap"].abs() <= 1e-9)
        & (pool["reference_objective"] > 1e-9)
    ]
    print(f"Zero-gap reductions: {len(faithful)}")

    if not selected.empty:
        print("\n=== Selected ideal-QAOA benchmarks ===")
        columns = [
            "benchmark_id",
            "instance",
            "n_satellites",
            "n_time_cycles",
            "n_services",
            "capacity_factor",
            "seed",
            "K",
            "logical_variables",
            "quadratic_couplings",
            "qubo_density",
            "reference_objective",
            "decomposition_gap",
        ]
        print(selected[columns].to_string(index=False))
    else:
        print("\nNo benchmark satisfied the current selection criteria.")


if __name__ == "__main__":
    main()
