"""Run noise-free circuit and hardware-connectivity resource analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oec_qaoa.benchmarks import FROZEN_QAOA_BENCHMARKS, build_frozen_benchmark
from oec_qaoa.circuit_resources import connectivity_metrics, cost_layer_metrics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "hardware"
RESULTS.mkdir(parents=True, exist_ok=True)

TOPOLOGIES = ["all_to_all", "grid", "ring", "line"]


def main() -> None:
    logical_rows = []
    connectivity_rows = []

    for benchmark in FROZEN_QAOA_BENCHMARKS:
        instance, _, model = build_frozen_benchmark(benchmark)
        logical = cost_layer_metrics(model)

        logical_rows.append(
            {
                "benchmark_id": benchmark.benchmark_id,
                "instance": instance.name,
                **logical.to_dict(),
            }
        )

        for topology in TOPOLOGIES:
            connectivity = connectivity_metrics(model, topology)
            connectivity_rows.append(
                {
                    "benchmark_id": benchmark.benchmark_id,
                    "instance": instance.name,
                    **connectivity.to_dict(),
                }
            )

    logical_frame = pd.DataFrame(logical_rows)
    connectivity_frame = pd.DataFrame(connectivity_rows)

    logical_frame.to_csv(
        RESULTS / "cost_layer_resources.csv",
        index=False,
    )
    connectivity_frame.to_csv(
        RESULTS / "connectivity_pressure.csv",
        index=False,
    )

    print("=== Logical cost-layer resources ===")
    print(logical_frame.to_string(index=False))
    print("\n=== Representative connectivity pressure ===")
    print(connectivity_frame.to_string(index=False))


if __name__ == "__main__":
    main()