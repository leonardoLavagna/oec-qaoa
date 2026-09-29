"""Validation helpers shared by the experimental notebooks."""

from __future__ import annotations

from dataclasses import dataclass

from .ilp import ReferenceSolution, ordered_service_path
from .model import OECInstance


@dataclass(frozen=True)
class ValidationReport:
    feasible: bool
    objective: float
    paths: dict[str, tuple[str, ...]]
    processing_nodes: dict[str, str]


def validate_reference_solution(
    instance: OECInstance,
    solution: ReferenceSolution,
) -> ValidationReport:
    """Decode a reference solution and expose its physical interpretation."""
    if not solution.success:
        raise ValueError(f"reference solver failed: {solution.message}")

    edge_by_name = {edge.name: edge for edge in instance.edges}
    paths: dict[str, tuple[str, ...]] = {}
    processing_nodes: dict[str, str] = {}

    for service in instance.services:
        path = ordered_service_path(instance, solution, service.name)
        paths[service.name] = path

        processing_edges = [
            edge_by_name[name]
            for name in path
            if edge_by_name[name].kind == "processing"
        ]
        if len(processing_edges) != 1:
            raise ValueError(
                f"service {service.name} must contain exactly one processing edge"
            )
        processing_nodes[service.name] = processing_edges[0].source.physical

    return ValidationReport(
        feasible=True,
        objective=solution.objective,
        paths=paths,
        processing_nodes=processing_nodes,
    )
