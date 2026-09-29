"""Exact classical reference formulation for small OEC instances.

The solver uses SciPy's mixed-integer interface backed by HiGHS and is kept
independent of the QUBO implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import Edge, OECInstance, Service, StateNode
from .preprocessing import service_subgraph_edges


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


def solve_reference_ilp(instance: OECInstance) -> ReferenceSolution:
    """Solve the binary edge-flow OEC model exactly for a small instance."""

    service_edges = {
        service.name: service_subgraph_edges(instance, service)
        for service in instance.services
    }
    variables = tuple(
        (service, edge)
        for service in instance.services
        for edge in service_edges[service.name]
    )
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

    # Flow conservation is imposed independently for every service. Source and
    # destination conditions are represented by non-zero right-hand sides.
    for service in instance.services:
        edges = service_edges[service.name]
        nodes = sorted(
            {node for edge in edges for node in (edge.source, edge.target)}
        )

        for node in nodes:
            row = np.zeros(n_variables, dtype=float)
            for j, (var_service, edge) in enumerate(variables):
                if var_service.name != service.name:
                    continue
                if edge.source == node:
                    row[j] += 1.0
                if edge.target == node:
                    row[j] -= 1.0

            if node == instance.source_node(service):
                rhs = 1.0
            elif node == instance.sink_node(service):
                rhs = -1.0
            else:
                rhs = 0.0

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
    # Ground energy is excluded because ground processing defines the objective
    # rather than a bounded on-board resource.
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

    constraints = LinearConstraint(
        np.vstack(rows),
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

    The helper raises when the selected subgraph does not define one
    unambiguous source-to-destination path.  This strict behavior exposes
    accidental cycles, branches or disconnected components during validation.
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
