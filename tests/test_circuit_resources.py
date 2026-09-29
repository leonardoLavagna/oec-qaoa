"""Regression tests for circuit and connectivity metrics."""

from oec_qaoa.benchmarks import FROZEN_QAOA_BENCHMARKS, build_frozen_benchmark
from oec_qaoa.circuit_resources import (
    connectivity_metrics,
    cost_layer_metrics,
    hardware_graph,
)


def test_all_to_all_supports_every_required_interaction() -> None:
    _, _, model = build_frozen_benchmark(FROZEN_QAOA_BENCHMARKS[0])
    metrics = connectivity_metrics(model, "all_to_all")

    assert metrics.directly_supported_fraction == 1.0
    assert metrics.mean_interaction_distance == 1.0
    assert metrics.max_interaction_distance == 1
    assert metrics.distance_excess_sum == 0


def test_cost_layer_depth_bounds_are_consistent() -> None:
    _, _, model = build_frozen_benchmark(FROZEN_QAOA_BENCHMARKS[0])
    metrics = cost_layer_metrics(model)

    assert metrics.zz_terms == len(model.quadratic)
    assert metrics.greedy_parallel_depth >= metrics.edge_chromatic_lower_bound
    assert metrics.edge_chromatic_lower_bound == metrics.interaction_max_degree


def test_representative_connectivity_graph_sizes() -> None:
    for topology in ("line", "ring", "grid", "all_to_all"):
        graph = hardware_graph(topology, 16)
        assert graph.number_of_nodes() == 16
        assert graph.number_of_edges() > 0


def test_sparse_connectivity_requires_nonlocal_interactions() -> None:
    _, _, model = build_frozen_benchmark(FROZEN_QAOA_BENCHMARKS[0])
    line = connectivity_metrics(model, "line")

    assert line.directly_supported_fraction < 1.0
    assert line.distance_excess_sum > 0