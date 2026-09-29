"""Regression tests for compact positive-objective QAOA benchmarks."""

from oec_qaoa.generators import compact_constrained_instance
from oec_qaoa.ilp import solve_reference_ilp
from oec_qaoa.path_qubo import build_path_qubo, solve_path_qubo_exact
from oec_qaoa.path_reduction import candidate_sets, decomposition_gap, solve_path_selection


def test_compact_family_has_positive_reference_optimum() -> None:
    instance = compact_constrained_instance(2, processing_capacity_mb=4.0)
    reference = solve_reference_ilp(instance)

    assert reference.success
    assert reference.objective > 0.0


def test_compact_k2_preserves_reference_optimum() -> None:
    instance = compact_constrained_instance(2, processing_capacity_mb=4.0)
    reference = solve_reference_ilp(instance)
    paths = candidate_sets(instance, k=2)
    reduced = solve_path_selection(instance, paths)

    assert reduced.success
    assert decomposition_gap(reference.objective, reduced) == 0.0


def test_compact_path_qubo_matches_reduced_model() -> None:
    instance = compact_constrained_instance(2, processing_capacity_mb=4.0)
    paths = candidate_sets(instance, k=2)
    reduced = solve_path_selection(instance, paths)
    model = build_path_qubo(instance, paths)
    qubo = solve_path_qubo_exact(model)

    assert reduced.success
    assert qubo.success
    assert abs(qubo.energy - reduced.objective) < 1e-9
