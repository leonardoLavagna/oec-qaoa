"""Regression tests for the first canonical OEC instance."""

from oec_qaoa import canonical_c1
from oec_qaoa.ilp import solve_reference_ilp
from oec_qaoa.validation import validate_reference_solution


def test_c1_reference_optimum() -> None:
    instance = canonical_c1()
    solution = solve_reference_ilp(instance)
    report = validate_reference_solution(instance, solution)

    assert solution.success
    assert report.feasible
    assert report.objective == 3.0
    assert report.processing_nodes == {"h0": "S1", "h1": "G"}


def test_c1_downlink_is_saturated_at_optimum() -> None:
    instance = canonical_c1()
    solution = solve_reference_ilp(instance)

    assert solution.uses("h0", "tx_S1_G_t2_p1")
    assert solution.uses("h1", "tx_S1_G_t2_p0")
