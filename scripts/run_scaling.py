"""Execute the resource-scaling study used by Notebook 04.

The script mirrors the notebook sweeps in a non-interactive form so that the
numerical outputs can be produced in CI and reviewed before they are transferred
to the manuscript.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from oec_qaoa.edge_qubo import build_edge_qubo
from oec_qaoa.generators import SyntheticConfig, synthetic_relay_instance
from oec_qaoa.metrics import qubo_resource_metrics
from oec_qaoa.path_qubo import build_path_qubo
from oec_qaoa.path_reduction import candidate_sets


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "scaling"
RESULTS.mkdir(parents=True, exist_ok=True)

BASE = {
    "n_satellites": 4,
    "n_time_cycles": 5,
    "n_services": 4,
    "capacity_factor": 1.0,
    "seed": 20260929,
}

SWEEPS = {
    "n_services": [2, 4, 6, 8, 10, 12],
    "n_time_cycles": [3, 4, 5, 6, 8],
    "n_satellites": [3, 4, 5, 6, 8],
}

K_VALUES = [1, 2, 4, 8]
SEEDS = [20260929, 20260930, 20261001]


def run_primary_sweeps() -> pd.DataFrame:
    records = []

    for parameter, values in SWEEPS.items():
        for value in values:
            for seed in SEEDS:
                configuration = dict(BASE)
                configuration[parameter] = value
                configuration["seed"] = seed

                instance = synthetic_relay_instance(
                    SyntheticConfig(**configuration)
                )

                edge_qubo = build_edge_qubo(instance)
                records.append(
                    {
                        "sweep": parameter,
                        "sweep_value": value,
                        "seed": seed,
                        "formulation": "edge",
                        "K": None,
                        **configuration,
                        **qubo_resource_metrics(edge_qubo).to_dict(),
                    }
                )

                all_paths = candidate_sets(instance)
                max_available = min(
                    len(paths)
                    for paths in all_paths.values()
                )

                for k in K_VALUES:
                    if k > max_available:
                        continue

                    paths = {
                        service: candidates[:k]
                        for service, candidates in all_paths.items()
                    }
                    path_qubo = build_path_qubo(instance, paths)

                    records.append(
                        {
                            "sweep": parameter,
                            "sweep_value": value,
                            "seed": seed,
                            "formulation": "path",
                            "K": k,
                            **configuration,
                            **qubo_resource_metrics(path_qubo).to_dict(),
                        }
                    )

    return pd.DataFrame(records)


def reduction_ratios(scaling: pd.DataFrame) -> pd.DataFrame:
    edge_rows = scaling[
        scaling["formulation"] == "edge"
    ][
        [
            "sweep",
            "sweep_value",
            "seed",
            "logical_variables",
            "quadratic_couplings",
        ]
    ].rename(
        columns={
            "logical_variables": "edge_logical_variables",
            "quadratic_couplings": "edge_quadratic_couplings",
        }
    )

    ratios = scaling[
        scaling["formulation"] == "path"
    ].merge(
        edge_rows,
        on=["sweep", "sweep_value", "seed"],
        how="left",
    )

    ratios["logical_variable_ratio"] = (
        ratios["logical_variables"]
        / ratios["edge_logical_variables"]
    )
    ratios["coupling_ratio"] = (
        ratios["quadratic_couplings"]
        / ratios["edge_quadratic_couplings"]
    )
    return ratios


def run_combined_grid() -> pd.DataFrame:
    records = []

    for n_satellites, n_time_cycles, n_services, seed in product(
        [3, 4, 6],
        [4, 6],
        [2, 4, 8],
        SEEDS,
    ):
        config = SyntheticConfig(
            n_satellites=n_satellites,
            n_time_cycles=n_time_cycles,
            n_services=n_services,
            capacity_factor=1.0,
            seed=seed,
        )
        instance = synthetic_relay_instance(config)

        edge = build_edge_qubo(instance)
        records.append(
            {
                "formulation": "edge",
                "K": None,
                "n_satellites": n_satellites,
                "n_time_cycles": n_time_cycles,
                "n_services": n_services,
                "seed": seed,
                **qubo_resource_metrics(edge).to_dict(),
            }
        )

        all_paths = candidate_sets(instance)
        max_available = min(
            len(paths)
            for paths in all_paths.values()
        )

        for k in (2, 4):
            if k > max_available:
                continue

            paths = {
                service: candidates[:k]
                for service, candidates in all_paths.items()
            }
            model = build_path_qubo(instance, paths)

            records.append(
                {
                    "formulation": "path",
                    "K": k,
                    "n_satellites": n_satellites,
                    "n_time_cycles": n_time_cycles,
                    "n_services": n_services,
                    "seed": seed,
                    **qubo_resource_metrics(model).to_dict(),
                }
            )

    return pd.DataFrame(records)


def print_summary(
    scaling: pd.DataFrame,
    ratios: pd.DataFrame,
) -> None:
    print("\n=== Mean logical variables by service count ===")
    table = (
        scaling[scaling["sweep"] == "n_services"]
        .groupby(["sweep_value", "formulation", "K"], dropna=False)
        ["logical_variables"]
        .mean()
        .reset_index()
    )
    print(table.to_string(index=False))

    print("\n=== Mean quadratic couplings by service count ===")
    table = (
        scaling[scaling["sweep"] == "n_services"]
        .groupby(["sweep_value", "formulation", "K"], dropna=False)
        ["quadratic_couplings"]
        .mean()
        .reset_index()
    )
    print(table.to_string(index=False))

    print("\n=== Path/edge reduction ratios by service count ===")
    table = (
        ratios[ratios["sweep"] == "n_services"]
        .groupby(["sweep_value", "K"])[
            ["logical_variable_ratio", "coupling_ratio"]
        ]
        .mean()
        .reset_index()
    )
    print(table.to_string(index=False))

    print("\n=== Time-horizon sensitivity: mean logical variables ===")
    table = (
        scaling[scaling["sweep"] == "n_time_cycles"]
        .groupby(["sweep_value", "formulation", "K"], dropna=False)
        ["logical_variables"]
        .mean()
        .reset_index()
    )
    print(table.to_string(index=False))


def main() -> None:
    scaling = run_primary_sweeps()
    ratios = reduction_ratios(scaling)
    combined = run_combined_grid()

    scaling.to_csv(RESULTS / "resource_scaling.csv", index=False)
    ratios.to_csv(
        RESULTS / "resource_reduction_ratios.csv",
        index=False,
    )
    combined.to_csv(
        RESULTS / "resource_scaling_combined.csv",
        index=False,
    )

    print_summary(scaling, ratios)
    print(f"\nPrimary rows: {len(scaling)}")
    print(f"Ratio rows: {len(ratios)}")
    print(f"Combined rows: {len(combined)}")


if __name__ == "__main__":
    main()
