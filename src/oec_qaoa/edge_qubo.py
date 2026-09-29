"""Manual edge-based QUBO construction for the OEC model.

Semantic variables, slack variables, penalty coefficients and quadratic
couplings remain explicit so they can be inspected and measured directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import ceil, gcd, log2

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import OECInstance
from .preprocessing import service_subgraph_edges


@dataclass(frozen=True)
class LinearEquality:
    """Integer-valued equality contributing one squared QUBO penalty."""

    name: str
    coefficients: dict[int, int]
    rhs: int


@dataclass(frozen=True)
class QUBOModel:
    """Explicit upper-triangular representation of a binary quadratic model."""

    variable_names: tuple[str, ...]
    semantic_count: int
    constant: float
    linear: np.ndarray
    quadratic: dict[tuple[int, int], float]
    penalty: float
    constraints: tuple[LinearEquality, ...]
    semantic_keys: tuple[tuple[str, str], ...]
    slack_variables: dict[str, tuple[int, ...]]

    def energy(self, bits: np.ndarray) -> float:
        x = np.asarray(bits, dtype=float)
        value = self.constant + float(self.linear @ x)
        for (i, j), coefficient in self.quadratic.items():
            value += coefficient * x[i] * x[j]
        return float(value)


@dataclass(frozen=True)
class QUBOSolution:
    """Exact solution of a QUBO together with decoded semantic variables."""

    energy: float
    bits: np.ndarray
    selected_edges: dict[str, tuple[str, ...]]
    success: bool
    message: str


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def _integerize(
    coefficients: list[float],
    rhs: float,
) -> tuple[list[int], int]:
    """Scale one linear inequality to integer coefficients."""

    values = [
        Fraction(str(value)).limit_denominator(10_000)
        for value in [*coefficients, rhs]
    ]
    scale = reduce(_lcm, (value.denominator for value in values), 1)
    integers = [int(value * scale) for value in values]

    common = reduce(
        gcd,
        (abs(value) for value in integers if value != 0),
        0,
    ) or 1
    integers = [value // common for value in integers]
    return integers[:-1], integers[-1]


def _slack_weights(capacity: int) -> tuple[int, ...]:
    """Use a unique powers-of-two encoding for all slacks up to capacity."""

    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if capacity == 0:
        return ()
    n_bits = ceil(log2(capacity + 1))
    return tuple(1 << bit for bit in range(n_bits))


def _add_squared_penalty(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    coefficients: dict[int, int],
    rhs: int,
    weight: float,
) -> float:
    """Expand weight * (a^T x - rhs)^2 using x_i^2 = x_i."""

    constant = weight * rhs * rhs
    items = sorted(coefficients.items())

    for i, coefficient in items:
        linear[i] += weight * (
            coefficient * coefficient - 2 * rhs * coefficient
        )

    for position, (i, coefficient_i) in enumerate(items):
        for j, coefficient_j in items[position + 1 :]:
            key = (i, j)
            quadratic[key] = quadratic.get(key, 0.0) + (
                2 * weight * coefficient_i * coefficient_j
            )

    return constant


def build_edge_qubo(
    instance: OECInstance,
    penalty: float | None = None,
) -> QUBOModel:
    """Build the direct edge-based QUBO of an OEC instance.

    Non-binding resource inequalities are removed before slack variables are
    introduced. Algebraically identical inequalities are represented once; in
    C1 this prevents the deliberately coincident processing-capacity and energy
    bounds from inflating the encoding without adding a new feasible-set
    restriction.
    """

    semantic_variables = []
    service_edges = {}
    for service in instance.services:
        edges = service_subgraph_edges(instance, service)
        service_edges[service.name] = edges
        semantic_variables.extend((service, edge) for edge in edges)

    semantic_keys = tuple(
        (service.name, edge.name)
        for service, edge in semantic_variables
    )
    variable_names = [
        f"y[{service_name},{edge_name}]"
        for service_name, edge_name in semantic_keys
    ]
    index = {
        key: position
        for position, key in enumerate(semantic_keys)
    }

    objective = np.asarray(
        [
            instance.ground_processing_cost(service, edge)
            for service, edge in semantic_variables
        ],
        dtype=float,
    )
    objective_span = float(
        np.maximum(objective, 0).sum()
        - np.minimum(objective, 0).sum()
    )
    penalty_value = float(
        penalty if penalty is not None else objective_span + 1.0
    )
    if penalty_value <= objective_span:
        raise ValueError(
            "penalty must exceed the range of the unconstrained linear objective"
        )

    constraints: list[LinearEquality] = []

    # Source, destination and intermediate flow equations.
    for service in instance.services:
        edges = service_edges[service.name]
        nodes = sorted(
            {node for edge in edges for node in (edge.source, edge.target)}
        )

        for node in nodes:
            coefficients: dict[int, int] = {}
            for edge in edges:
                position = index[(service.name, edge.name)]
                coefficient = (
                    (1 if edge.source == node else 0)
                    - (1 if edge.target == node else 0)
                )
                if coefficient:
                    coefficients[position] = coefficient

            if node == instance.source_node(service):
                rhs = 1
            elif node == instance.sink_node(service):
                rhs = -1
            else:
                rhs = 0

            constraints.append(
                LinearEquality(
                    name=f"flow:{service.name}:{node}",
                    coefficients=coefficients,
                    rhs=rhs,
                )
            )

    inequalities = []

    # Shared transmission, storage and processing capacities.
    for resource_key, capacity in instance.resource_capacities_mb.items():
        entries = []
        for service, edge in semantic_variables:
            if edge.resource_key == resource_key:
                entries.append(
                    (
                        index[(service.name, edge.name)],
                        instance.edge_data_mb(service, edge),
                    )
                )

        if entries and sum(value for _, value in entries) > capacity + 1e-12:
            inequalities.append(
                (f"capacity:{resource_key}", entries, float(capacity))
            )

    # Per-node, per-cycle on-board energy budgets.
    for (physical, time), budget in instance.energy_budgets_j.items():
        entries = []
        for service, edge in semantic_variables:
            if (
                edge.source.physical == physical
                and edge.source.time == time
            ):
                value = instance.onboard_energy(service, edge)
                if abs(value) > 1e-15:
                    entries.append(
                        (index[(service.name, edge.name)], value)
                    )

        if entries and sum(value for _, value in entries) > budget + 1e-12:
            inequalities.append(
                (f"energy:{physical}:t{time}", entries, float(budget))
            )

    # Different physical constraints can occasionally become the same algebraic
    # inequality on a synthetic validation instance. Keep one copy in that case.
    seen_inequalities = set()
    active_inequalities = []

    for label, entries, rhs in inequalities:
        coefficients, integer_rhs = _integerize(
            [value for _, value in entries],
            rhs,
        )
        pairs = tuple(
            sorted(
                (position, coefficient)
                for (position, _), coefficient
                in zip(entries, coefficients)
                if coefficient
            )
        )
        signature = (pairs, integer_rhs)
        if signature in seen_inequalities:
            continue
        seen_inequalities.add(signature)
        active_inequalities.append((label, pairs, integer_rhs))

    slack_variables: dict[str, tuple[int, ...]] = {}

    for label, pairs, rhs in active_inequalities:
        coefficients = dict(pairs)
        slack_indices = []

        for bit, weight in enumerate(_slack_weights(rhs)):
            position = len(variable_names)
            variable_names.append(f"slack[{label},{bit}]")
            coefficients[position] = weight
            slack_indices.append(position)

        slack_variables[label] = tuple(slack_indices)
        constraints.append(
            LinearEquality(
                name=label,
                coefficients=coefficients,
                rhs=rhs,
            )
        )

    linear = np.zeros(len(variable_names), dtype=float)
    linear[: len(objective)] = objective
    quadratic: dict[tuple[int, int], float] = {}
    constant = 0.0

    for constraint in constraints:
        constant += _add_squared_penalty(
            linear,
            quadratic,
            constraint.coefficients,
            constraint.rhs,
            penalty_value,
        )

    quadratic = {
        key: value
        for key, value in quadratic.items()
        if abs(value) > 1e-12
    }

    return QUBOModel(
        variable_names=tuple(variable_names),
        semantic_count=len(semantic_keys),
        constant=constant,
        linear=linear,
        quadratic=quadratic,
        penalty=penalty_value,
        constraints=tuple(constraints),
        semantic_keys=semantic_keys,
        slack_variables=slack_variables,
    )


def solve_qubo_exact(model: QUBOModel) -> QUBOSolution:
    """Minimize the QUBO exactly through a linearized binary program.

    Each quadratic product receives one auxiliary continuous variable with the
    standard McCormick envelope. Because the original factors are binary, the
    linearization is exact and avoids state-vector enumeration when the direct
    edge encoding already contains several tens of logical variables.
    """

    n_variables = len(model.variable_names)
    quadratic_pairs = tuple(model.quadratic)
    n_products = len(quadratic_pairs)

    objective = np.concatenate(
        [
            model.linear,
            np.asarray(
                [model.quadratic[pair] for pair in quadratic_pairs],
                dtype=float,
            ),
        ]
    )

    rows = []
    lower = []
    upper = []

    for product_index, (i, j) in enumerate(quadratic_pairs):
        z = n_variables + product_index

        row = np.zeros(n_variables + n_products)
        row[z] = 1.0
        row[i] = -1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(n_variables + n_products)
        row[z] = 1.0
        row[j] = -1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(n_variables + n_products)
        row[z] = -1.0
        row[i] = 1.0
        row[j] = 1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1.0)

    constraints = (
        LinearConstraint(
            np.vstack(rows),
            lb=np.asarray(lower),
            ub=np.asarray(upper),
        )
        if rows
        else None
    )

    result = milp(
        c=objective,
        integrality=np.concatenate(
            [
                np.ones(n_variables, dtype=int),
                np.zeros(n_products, dtype=int),
            ]
        ),
        bounds=Bounds(
            np.zeros(n_variables + n_products),
            np.ones(n_variables + n_products),
        ),
        constraints=constraints,
        options={"disp": False},
    )

    if result.x is None:
        return QUBOSolution(
            energy=float("inf"),
            bits=np.full(n_variables, np.nan),
            selected_edges={},
            success=False,
            message=str(result.message),
        )

    bits = np.rint(result.x[:n_variables]).astype(int)
    selected: dict[str, list[str]] = {
        service_name: []
        for service_name, _ in model.semantic_keys
    }

    for bit, (service_name, edge_name) in zip(
        bits[: model.semantic_count],
        model.semantic_keys,
    ):
        if bit:
            selected[service_name].append(edge_name)

    return QUBOSolution(
        energy=model.energy(bits),
        bits=bits,
        selected_edges={
            service: tuple(edges)
            for service, edges in selected.items()
        },
        success=bool(result.success),
        message=str(result.message),
    )
