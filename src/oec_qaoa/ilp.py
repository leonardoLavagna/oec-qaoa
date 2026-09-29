"""Exact classical reference formulation for small OEC instances.

The reference solver is deliberately independent of the later QUBO machinery.
It uses SciPy's mixed-integer interface, backed by HiGHS, and therefore provides
an open-source baseline for the equivalence tests developed in the subsequent
notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import Edge, OECInstance, Service, StateNode


@dataclass(frozen=True)
class ReferenceSolution:
    """Decoded exact solution of the constrained OEC formulation."""

    objective: float
    selected_edges: dict[str, tuple[str, ...]]
    raw_vector: np.ndarray
    variable_order: tuple[tuple[str, str], ...]
    success: bool
    message: str

    def uses(self, service: str, edge: str) -> bool:
        return edge in self.selected_edges.get(service, ())


def _admissible_edge(service: Service, edge: Edge) -> bool:
    """Keep only transitions inside the generation/deadline window."""
    return (
        service.generation_time <= edge.source.time <= service.deadline
        and service.generation_time <= edge.target.time <= service.deadline
    )


def _candidate_variables(
    instance: OECInstance,
) -> tuple[tuple[Service, Edge], ...]:
    return tuple(
        (service, edge)
        for service in instance.services
        for edge in instance.edges
        if _admissible_edge(service, edge)
    )


def _flow_rhs(
    instance: OECInstance,
    service: Service,
    node: StateNode,
) -> float:
    if node == instance.source_node(service):
        return 1.0
    if node == instance.sink_node(service):
        return -1.0
    return 0.0


def solve_reference_ilp(instance: OECInstance) -> ReferenceSolution:
    """Solve the binary edge-flow OEC model exactly for a small instance."""

    variables = _candidate_variables(instance)
    n_variables = len(variables)
    if n_variables == 0:
        raise ValueError("instance has no admissible decision variables")

    objective = np.asarray(
        [
            instance.ground_processing_cost(service, edge)
            for service, edge in variables
        ],
        dtype=float,
    )

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    # Flow conservation is imposed independently for every service.  Source and
    # destination conditions are represented by non-zero right-hand sides.
    for service in instance.services:
        relevant_nodes = {
            node
            for edge in instance.edges
            if _admissible_edge(service, edge)
            for node in (edge.source, edge.target)
        }
        relevant_nodes.add(instance.source_node(service))
        relevant_nodes.add(instance.sink_node(service))

        for node in sorted(relevant_nodes):
            row = np.zeros(n_variables, dtype=float)
            for j, (var_service, edge) in enumerate(variables):
                if var_service.name != service.name:
                    continue
                if edge.source == node:
                    row[j] += 1.0
                if edge.target == node:
                    row[j] -= 1.0

            rhs = _flow_rhs(instance, service, node)
            rows.append(row)
            lower.append(rhs)
            upper.append(rhs)

    # The physical capacity of a transmission or memory resource is shared
    # across processing states whenever the edges carry the same resource key.
    for resource_key, capacity in instance.resource_capacities_mb.items():
        row = np.zeros(n_variables, dtype=float)
        for j, (service, edge) in enumerate(variables):
            if edge.resource_key == resource_key:
                row[j] = instance.edge_data_mb(service, edge)
        if np.any(row):
            rows.append(row)
            lower.append(-np.inf)
            upper.append(float(capacity))

    # Energy is charged to the satellite that transmits or processes the data.
    # Ground energy is excluded here because ground processing defines the
    # objective rather than a bounded on-board resource.
    for node_time, budget in instance.energy_budgets_j.items():
        physical, time = node_time
        row = np.zeros(n_variables, dtype=float)
        for j, (service, edge) in enumerate(variables):
            if edge.source.physical == physical and edge.source.time == time:
                row[j] = instance.onboard_energy(service, edge)
        if np.any(row):
            rows.append(row)
            lower.append(-np.inf)
            upper.append(float(budget))

    A = np.vstack(rows)
    constraints = LinearConstraint(
        A,
        lb=np.asarray(lower, dtype=float),
        ub=np.asarray(upper, dtype=float),
    )

    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=int),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=constraints,
        options={"disp": False},
    )

    if result.x is None:
        return ReferenceSolution(
            objective=float("inf"),
            selected_edges={},
            raw_vector=np.full(n_variables, np.nan),
            variable_order=tuple(
                (service.name, edge.name) for service, edge in variables
            ),
            success=False,
            message=str(result.message),
        )

    selected: dict[str, list[str]] = {
        service.name: [] for service in instance.services
    }
    for value, (service, edge) in zip(result.x, variables):
        if value > 0.5:
            selected[service.name].append(edge.name)

    return ReferenceSolution(
        objective=float(result.fun),
        selected_edges={
            name: tuple(edges) for name, edges in selected.items()
        },
        raw_vector=np.asarray(result.x, dtype=float),
        variable_order=tuple(
            (service.name, edge.name) for service, edge in variables
        ),
        success=bool(result.success),
        message=str(result.message),
    )


def ordered_service_path(
    instance: OECInstance,
    solution: ReferenceSolution,
    service_name: str,
) -> tuple[str, ...]:
    """Return selected edge names in path order.

    This helper intentionally raises when the selected subgraph does not define
    one unambiguous source-to-destination path.  The strict behavior is useful
    for regression tests because it exposes accidental cycles or branching.
    """

    service = next(
        service
        for service in instance.services
        if service.name == service_name
    )
    edge_by_name = {edge.name: edge for edge in instance.edges}
    chosen = [
        edge_by_name[name]
        for name in solution.selected_edges.get(service_name, ())
    ]

    outgoing: dict[StateNode, list[Edge]] = {}
    for edge in chosen:
        outgoing.setdefault(edge.source, []).append(edge)

    node = instance.source_node(service)
    sink = instance.sink_node(service)
    path: list[str] = []
    visited: set[str] = set()

    while node != sink:
        options = outgoing.get(node, [])
        if len(options) != 1:
            raise ValueError(
                f"selected edges for {service_name} do not define a unique path "
                f"at node {node}: found {len(options)} outgoing edges"
            )
        edge = options[0]
        if edge.name in visited:
            raise ValueError(f"cycle detected for service {service_name}")
        visited.add(edge.name)
        path.append(edge.name)
        node = edge.target

    if len(visited) != len(chosen):
        raise ValueError(
            f"selected edges for {service_name} contain disconnected components"
        )
    return tuple(path)
