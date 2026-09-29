"""Frozen benchmark definitions for the ideal-QAOA study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generators import SyntheticConfig, synthetic_relay_instance
from .path_qubo import PathQUBOModel, build_path_qubo
from .path_reduction import CandidatePath, candidate_sets


@dataclass(frozen=True)
class QAOABenchmark:
    benchmark_id: str
    config: SyntheticConfig
    k: int


FROZEN_QAOA_BENCHMARKS = (
    QAOABenchmark(
        "B1",
        SyntheticConfig(3, 3, 2, 0.8, 20260929),
        2,
    ),
    QAOABenchmark(
        "B2",
        SyntheticConfig(3, 3, 2, 1.2, 20260929),
        2,
    ),
    QAOABenchmark(
        "B3",
        SyntheticConfig(3, 3, 4, 1.0, 20260929),
        2,
    ),
    QAOABenchmark(
        "B4",
        SyntheticConfig(3, 4, 2, 1.2, 20260929),
        2,
    ),
)


def build_frozen_benchmark(
    benchmark: QAOABenchmark,
) -> tuple[object, dict[str, tuple[CandidatePath, ...]], PathQUBOModel]:
    """Reconstruct one frozen benchmark deterministically."""

    instance = synthetic_relay_instance(benchmark.config)
    all_paths = candidate_sets(instance)
    retained = {
        service: paths[: benchmark.k]
        for service, paths in all_paths.items()
    }
    model = build_path_qubo(instance, retained)
    return instance, retained, model


def physical_feasibility_mask(
    instance,
    retained: dict[str, tuple[CandidatePath, ...]],
    model: PathQUBOModel,
) -> np.ndarray:
    """Return states whose semantic path choices satisfy all physical resources.

    Slack variables are marginalized because they are encoding variables rather
    than physical decisions.  A computational-basis state is counted as
    physically feasible when its semantic prefix selects exactly one retained
    path for every service and the selected paths jointly satisfy every shared
    capacity and energy bound.
    """

    n = len(model.variable_names)
    states = np.arange(1 << n, dtype=np.uint64)
    feasible = np.ones(1 << n, dtype=bool)

    path_lookup = {
        (path.service, path.name): path
        for paths in retained.values()
        for path in paths
    }

    service_indices: dict[str, list[int]] = {}
    for index, (service, _) in enumerate(model.semantic_keys):
        service_indices.setdefault(service, []).append(index)

    for indices in service_indices.values():
        count = np.zeros(1 << n, dtype=np.int16)
        for index in indices:
            count += ((states >> index) & 1).astype(np.int16)
        feasible &= count == 1

    for resource, capacity in instance.resource_capacities_mb.items():
        usage = np.zeros(1 << n, dtype=float)
        for index, key in enumerate(model.semantic_keys):
            path = path_lookup[key]
            coefficient = path.resource_usage.get(resource, 0.0)
            if coefficient:
                usage += coefficient * ((states >> index) & 1)
        feasible &= usage <= float(capacity) + 1e-9

    for resource, budget in instance.energy_budgets_j.items():
        usage = np.zeros(1 << n, dtype=float)
        for index, key in enumerate(model.semantic_keys):
            path = path_lookup[key]
            coefficient = path.energy_usage.get(resource, 0.0)
            if coefficient:
                usage += coefficient * ((states >> index) & 1)
        feasible &= usage <= float(budget) + 1e-9

    return feasible
