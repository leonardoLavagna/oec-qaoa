"""Regression tests for the candidate-path reduction and path QUBO."""

from oec_qaoa import canonical_c2
from oec_qaoa.ilp import solve_reference_ilp
from oec_qaoa.path_qubo import build_path_qubo, solve_path_qubo_exact
from oec_qaoa.path_reduction import (
    candidate_sets,
    decomposition_gap,
    solve_path_selection,
)


def test_c2_candidate_path_hierarchy() -> None:
    instance = canonical_c2()

    k1 = candidate_sets(instance, k=1)
    k2 = candidate_sets(instance, k=2)
    full = candidate_sets(instance)

    assert len(k1["h0"]) == 1
    assert len(k1["h1"]) == 1
    assert len(k2["h0"]) == 2
    assert len(k2["h1"]) == 2
    assert len(full["h0"]) == 3
    assert len(full["h1"]) == 4

    assert k1["h0"][0].name == "P_h0_0"
    assert k1["h1"][0].name == "P_h1_0"
    assert k2["h1"][1].name == "P_h1_1"


def test_c2_k1_is_jointly_infeasible_but_k2_recovers_optimum() -> None:
    instance = canonical_c2()
    reference = solve_reference_ilp(instance)

    solution_k1 = solve_path_selection(instance, candidate_sets(instance, k=1))
    solution_k2 = solve_path_selection(instance, candidate_sets(instance, k=2))

    assert not solution_k1.success
    assert solution_k2.success
    assert solution_k2.objective == reference.objective == 0.0
    assert decomposition_gap(reference.objective, solution_k2) == 0.0
    assert solution_k2.selected_paths == {
        "h0": "P_h0_0",
        "h1": "P_h1_1",
    }


def test_c2_path_qubo_matches_reduced_exact_model() -> None:
    instance = canonical_c2()
    paths = candidate_sets(instance, k=2)

    reduced = solve_path_selection(instance, paths)
    qubo = build_path_qubo(instance, paths)
    qubo_solution = solve_path_qubo_exact(qubo)

    assert reduced.success
    assert qubo_solution.success
    assert abs(qubo_solution.energy - reduced.objective) < 1e-9
    assert qubo_solution.selected_paths == {
        service: (path,)
        for service, path in reduced.selected_paths.items()
    }
