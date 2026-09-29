"""Regression tests for synthetic scaling instances and QUBO metrics."""

from oec_qaoa.edge_qubo import build_edge_qubo
from oec_qaoa.generators import SyntheticConfig, synthetic_relay_instance
from oec_qaoa.metrics import qubo_resource_metrics
from oec_qaoa.path_qubo import build_path_qubo
from oec_qaoa.path_reduction import candidate_sets


def test_synthetic_generator_is_deterministic() -> None:
    config = SyntheticConfig(
        n_satellites=4,
        n_time_cycles=5,
        n_services=4,
        capacity_factor=1.0,
        seed=17,
    )

    first = synthetic_relay_instance(config)
    second = synthetic_relay_instance(config)

    assert first == second
    assert first.name == second.name
    assert len(first.services) == 4


def test_scaling_metrics_are_consistent() -> None:
    instance = synthetic_relay_instance(
        SyntheticConfig(
            n_satellites=4,
            n_time_cycles=5,
            n_services=3,
            capacity_factor=1.0,
            seed=3,
        )
    )

    edge = build_edge_qubo(instance)
    edge_metrics = qubo_resource_metrics(edge)

    paths = candidate_sets(instance, k=2)
    path = build_path_qubo(instance, paths)
    path_metrics = qubo_resource_metrics(path)

    assert edge_metrics.logical_variables == len(edge.variable_names)
    assert edge_metrics.semantic_variables == edge.semantic_count
    assert edge_metrics.slack_variables == (
        len(edge.variable_names) - edge.semantic_count
    )
    assert edge_metrics.quadratic_couplings == len(edge.quadratic)

    assert path_metrics.logical_variables == len(path.variable_names)
    assert path_metrics.semantic_variables == path.semantic_count
    assert path_metrics.slack_variables == (
        len(path.variable_names) - path.semantic_count
    )
    assert path_metrics.quadratic_couplings == len(path.quadratic)

    assert 0.0 <= edge_metrics.qubo_density <= 1.0
    assert 0.0 <= path_metrics.qubo_density <= 1.0
