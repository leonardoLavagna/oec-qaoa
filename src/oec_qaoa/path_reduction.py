"""Candidate-path construction and exact path-selection model.

The reduced formulation moves source, destination, flow-conservation and
processing-state consistency out of the optimization model by enumerating
complete admissible service paths first.  The remaining binary decision selects
one candidate path per service, while resource capacities continue to couple
different services globally.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import Edge, OECInstance, Service, StateNode
from .preprocessing import service_subgraph_edges


@dataclass(frozen=True)
class CandidatePath:
    """One complete admissible route for a service."""

    service: str
    name: str
    edges: tuple[str, ...]
    objective: float
    resource_usage: dict[str, float]
    energy_usage: dict[tuple[str, int], float]


@dataclass(frozen=True)
class PathSelectionSolution:
    """Exact solution of the reduced candidate-path model."""

    objective: float
    selected_paths: dict[str, str]
    raw_vector: np.ndarray
    variable_order: tuple[tuple[str, str], ...]
    success: bool
    message: str


def _enumerate_edge_paths(
    instance: OECInstance,
    service: Service,
) -> list[tuple[Edge, ...]]:
    """Enumerate simple source-to-destination paths in the service subgraph."""

    edges = service_subgraph_edges(instance, service)
    outgoing: dict[StateNode, list[Edge]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source].append(edge)

    for node in outgoing:
        outgoing[node].sort(key=lambda edge: edge.name)

    source = instance.source_node(service)
    sink = instance.sink_node(service)
    paths: list[tuple[Edge, ...]] = []

    def dfs(
        node: StateNode,
        current: list[Edge],
        visited: set[StateNode],
    ) -> None:
        if node == sink:
            paths.append(tuple(current))
            return

        for edge in outgoing.get(node, []):
            if edge.target in visited:
                continue
            current.append(edge)
            visited.add(edge.target)
            dfs(edge.target, current, visited)
            visited.remove(edge.target)
            current.pop()

    dfs(source, [], {source})
    return paths


def _path_statistics(
    instance: OECInstance,
    service: Service,
    edges: tuple[Edge, ...],
) -> tuple[
    float,
    dict[str, float],
    dict[tuple[str, int], float],
]:
    objective = 0.0
    resources: dict[str, float] = defaultdict(float)
    energy: dict[tuple[str, int], float] = defaultdict(float)

    for edge in edges:
        objective += instance.ground_processing_cost(service, edge)
        resources[edge.resource_key] += instance.edge_data_mb(service, edge)

        onboard = instance.onboard_energy(service, edge)
        if onboard:
            energy[(edge.source.physical, edge.source.time)] += onboard

    return objective, dict(resources), dict(energy)


def enumerate_candidate_paths(
    instance: OECInstance,
    service: Service,
) -> tuple[CandidatePath, ...]:
    """Enumerate and rank paths that are individually resource-feasible."""

    records = []

    for edge_path in _enumerate_edge_paths(instance, service):
        objective, resources, energy = _path_statistics(
            instance,
            service,
            edge_path,
        )

        resource_feasible = all(
            value <= instance.resource_capacities_mb[key] + 1e-12
            for key, value in resources.items()
        )
        energy_feasible = all(
            value <= instance.energy_budgets_j.get(key, float("inf")) + 1e-12
            for key, value in energy.items()
        )
        if not (resource_feasible and energy_feasible):
            continue

        edge_names = tuple(edge.name for edge in edge_path)
        records.append(
            (
                objective,
                len(edge_names),
                edge_names,
                resources,
                energy,
            )
        )

    records.sort(key=lambda record: (record[0], record[1], record[2]))

    return tuple(
        CandidatePath(
            service=service.name,
            name=f"P_{service.name}_{rank}",
            edges=edge_names,
            objective=float(objective),
            resource_usage=resources,
            energy_usage=energy,
        )
        for rank, (
            objective,
            _,
            edge_names,
            resources,
            energy,
        ) in enumerate(records)
    )


def candidate_sets(
    instance: OECInstance,
    k: int | None = None,
) -> dict[str, tuple[CandidatePath, ...]]:
    """Return the first k ranked paths for every service.

    When k is None, all individually feasible paths are retained.
    """

    if k is not None and k < 1:
        raise ValueError("k must be positive")

    result = {}
    for service in instance.services:
        paths = enumerate_candidate_paths(instance, service)
        result[service.name] = paths if k is None else paths[:k]
    return result


def solve_path_selection(
    instance: OECInstance,
    paths_by_service: dict[str, tuple[CandidatePath, ...]],
) -> PathSelectionSolution:
    """Solve the reduced one-path-per-service model exactly."""

    for service in instance.services:
        if not paths_by_service.get(service.name):
            return PathSelectionSolution(
                objective=float("inf"),
                selected_paths={},
                raw_vector=np.empty(0),
                variable_order=(),
                success=False,
                message=f"service {service.name} has no retained candidate path",
            )

    variables = tuple(
        (service.name, path)
        for service in instance.services
        for path in paths_by_service[service.name]
    )
    n_variables = len(variables)

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    # Every service selects exactly one complete candidate path.
    for service in instance.services:
        row = np.zeros(n_variables, dtype=float)
        for j, (service_name, _) in enumerate(variables):
            if service_name == service.name:
                row[j] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)

    for resource_key, capacity in instance.resource_capacities_mb.items():
        row = np.asarray(
            [
                path.resource_usage.get(resource_key, 0.0)
                for _, path in variables
            ],
            dtype=float,
        )
        if np.any(row):
            rows.append(row)
            lower.append(-np.inf)
            upper.append(float(capacity))

    for node_time, budget in instance.energy_budgets_j.items():
        row = np.asarray(
            [
                path.energy_usage.get(node_time, 0.0)
                for _, path in variables
            ],
            dtype=float,
        )
        if np.any(row):
            rows.append(row)
            lower.append(-np.inf)
            upper.append(float(budget))

    result = milp(
        c=np.asarray([path.objective for _, path in variables], dtype=float),
        integrality=np.ones(n_variables, dtype=int),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(
            np.vstack(rows),
            lb=np.asarray(lower),
            ub=np.asarray(upper),
        ),
        options={"disp": False},
    )

    if result.x is None:
        return PathSelectionSolution(
            objective=float("inf"),
            selected_paths={},
            raw_vector=np.full(n_variables, np.nan),
            variable_order=tuple(
                (service_name, path.name)
                for service_name, path in variables
            ),
            success=False,
            message=str(result.message),
        )

    selected = {}
    for value, (service_name, path) in zip(result.x, variables):
        if value > 0.5:
            selected[service_name] = path.name

    return PathSelectionSolution(
        objective=float(result.fun),
        selected_paths=selected,
        raw_vector=np.asarray(result.x, dtype=float),
        variable_order=tuple(
            (service_name, path.name)
            for service_name, path in variables
        ),
        success=bool(result.success),
        message=str(result.message),
    )


def decomposition_gap(
    reference_objective: float,
    reduced_solution: PathSelectionSolution,
) -> float:
    """Return the objective loss introduced by candidate-path restriction."""

    if not reduced_solution.success:
        return float("inf")
    return float(reduced_solution.objective - reference_objective)
