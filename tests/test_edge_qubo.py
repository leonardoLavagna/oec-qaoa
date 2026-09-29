"""Regression tests for the direct edge-based QUBO."""

from oec_qaoa import canonical_c1
from oec_qaoa.edge_qubo import build_edge_qubo, solve_qubo_exact
from oec_qaoa.ilp import solve_reference_ilp


def test_c1_edge_qubo_matches_reference_optimum() -> None:
    instance = canonical_c1()

    reference = solve_reference_ilp(instance)
    qubo = build_edge_qubo(instance)
    solution = solve_qubo_exact(qubo)

    assert reference.success
    assert solution.success
    assert abs(solution.energy - reference.objective) < 1e-9
    assert solution.selected_edges == reference.selected_edges


def test_c1_edge_qubo_resource_counts() -> None:
    instance = canonical_c1()
    qubo = build_edge_qubo(instance)

    assert qubo.semantic_count == 20
    assert len(qubo.variable_names) == 26
    assert len(qubo.quadratic) == 53
    assert qubo.penalty == 8.0
    assert set(qubo.slack_variables) == {
        "capacity:downlink_S1_G_t2",
        "capacity:proc_S1_t1",
    }


def test_penalty_must_dominate_objective_span() -> None:
    instance = canonical_c1()

    try:
        build_edge_qubo(instance, penalty=7.0)
    except ValueError as error:
        assert "penalty" in str(error)
    else:
        raise AssertionError("an insufficient penalty should be rejected")
