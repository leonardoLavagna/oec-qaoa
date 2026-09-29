"""QUBO formulation of the reduced candidate-path selection problem."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import ceil, gcd, log2

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import OECInstance
from .path_reduction import CandidatePath


@dataclass(frozen=True)
class PathQUBOModel:
    """Explicit QUBO for one-path-per-service selection."""

    variable_names: tuple[str, ...]
    semantic_count: int
    constant: float
    linear: np.ndarray
    quadratic: dict[tuple[int, int], float]
    penalty: float
    semantic_keys: tuple[tuple[str, str], ...]
    slack_variables: dict[str, tuple[int, ...]]

    def energy(self, bits: np.ndarray) -> float:
        x = np.asarray(bits, dtype=float)
        value = self.constant + float(self.linear @ x)
        for (i, j), coefficient in self.quadratic.items():
            value += coefficient * x[i] * x[j]
        return float(value)


@dataclass(frozen=True)
class PathQUBOSolution:
    """Exact QUBO solution decoded to candidate-path choices."""

    energy: float
    bits: np.ndarray
    selected_paths: dict[str, tuple[str, ...]]
    success: bool
    message: str


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def _integerize(
    coefficients: list[float],
    rhs: float,
) -> tuple[list[int], int]:
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
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if capacity == 0:
        return ()
    return tuple(1 << bit for bit in range(ceil(log2(capacity + 1))))


def _add_square(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    coefficients: dict[int, int],
    rhs: int,
    penalty: float,
) -> float:
    constant = penalty * rhs * rhs
    items = sorted(coefficients.items())

    for i, coefficient in items:
        linear[i] += penalty * (
            coefficient * coefficient - 2 * rhs * coefficient
        )

    for position, (i, coefficient_i) in enumerate(items):
        for j, coefficient_j in items[position + 1 :]:
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + (
                2 * penalty * coefficient_i * coefficient_j
            )

    return constant


def build_path_qubo(
    instance: OECInstance,
    paths_by_service: dict[str, tuple[CandidatePath, ...]],
    penalty: float | None = None,
) -> PathQUBOModel:
    """Build the reduced path-selection QUBO."""

    for service in instance.services:
        if not paths_by_service.get(service.name):
            raise ValueError(
                f"service {service.name} has no retained candidate path"
            )

    variables = tuple(
        (service.name, path)
        for service in instance.services
        for path in paths_by_service[service.name]
    )
    semantic_keys = tuple(
        (service_name, path.name)
        for service_name, path in variables
    )
    variable_names = [
        f"x[{service_name},{path_name}]"
        for service_name, path_name in semantic_keys
    ]
    index = {
        key: position
        for position, key in enumerate(semantic_keys)
    }

    objective = np.asarray(
        [path.objective for _, path in variables],
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

    equalities: list[tuple[str, dict[int, int], int]] = []

    for service in instance.services:
        coefficients = {
            index[(service.name, path.name)]: 1
            for path in paths_by_service[service.name]
        }
        equalities.append(
            (f"choice:{service.name}", coefficients, 1)
        )

    inequalities = []

    for resource_key, capacity in instance.resource_capacities_mb.items():
        entries = [
            (
                index[(service_name, path.name)],
                path.resource_usage.get(resource_key, 0.0),
            )
            for service_name, path in variables
            if path.resource_usage.get(resource_key, 0.0) > 0
        ]
        if entries and sum(value for _, value in entries) > capacity + 1e-12:
            inequalities.append(
                (f"capacity:{resource_key}", entries, float(capacity))
            )

    for node_time, budget in instance.energy_budgets_j.items():
        entries = [
            (
                index[(service_name, path.name)],
                path.energy_usage.get(node_time, 0.0),
            )
            for service_name, path in variables
            if path.energy_usage.get(node_time, 0.0) > 0
        ]
        if entries and sum(value for _, value in entries) > budget + 1e-12:
            physical, time = node_time
            inequalities.append(
                (f"energy:{physical}:t{time}", entries, float(budget))
            )

    seen = set()
    slack_variables: dict[str, tuple[int, ...]] = {}

    for label, entries, rhs in inequalities:
        integer_coefficients, integer_rhs = _integerize(
            [value for _, value in entries],
            rhs,
        )
        pairs = tuple(
            sorted(
                (position, coefficient)
                for (position, _), coefficient
                in zip(entries, integer_coefficients)
                if coefficient
            )
        )
        signature = (pairs, integer_rhs)
        if signature in seen:
            continue
        seen.add(signature)

        coefficients = dict(pairs)
        slack_indices = []
        for bit, weight in enumerate(_slack_weights(integer_rhs)):
            position = len(variable_names)
            variable_names.append(f"slack[{label},{bit}]")
            coefficients[position] = weight
            slack_indices.append(position)

        slack_variables[label] = tuple(slack_indices)
        equalities.append((label, coefficients, integer_rhs))

    linear = np.zeros(len(variable_names), dtype=float)
    linear[: len(objective)] = objective
    quadratic: dict[tuple[int, int], float] = {}
    constant = 0.0

    for _, coefficients, rhs in equalities:
        constant += _add_square(
            linear,
            quadratic,
            coefficients,
            rhs,
            penalty_value,
        )

    quadratic = {
        pair: value
        for pair, value in quadratic.items()
        if abs(value) > 1e-12
    }

    return PathQUBOModel(
        variable_names=tuple(variable_names),
        semantic_count=len(semantic_keys),
        constant=constant,
        linear=linear,
        quadratic=quadratic,
        penalty=penalty_value,
        semantic_keys=semantic_keys,
        slack_variables=slack_variables,
    )


def solve_path_qubo_exact(model: PathQUBOModel) -> PathQUBOSolution:
    """Minimize the path QUBO exactly by linearizing quadratic products."""

    n_variables = len(model.variable_names)
    pairs = tuple(model.quadratic)
    n_products = len(pairs)

    objective = np.concatenate(
        [
            model.linear,
            np.asarray(
                [model.quadratic[pair] for pair in pairs],
                dtype=float,
            ),
        ]
    )

    rows = []
    lower = []
    upper = []

    for product_index, (i, j) in enumerate(pairs):
        z = n_variables + product_index

        row = np.zeros(n_variables + n_products)
        row[z], row[i] = 1.0, -1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(n_variables + n_products)
        row[z], row[j] = 1.0, -1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(n_variables + n_products)
        row[z], row[i], row[j] = -1.0, 1.0, 1.0
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
        return PathQUBOSolution(
            energy=float("inf"),
            bits=np.full(n_variables, np.nan),
            selected_paths={},
            success=False,
            message=str(result.message),
        )

    bits = np.rint(result.x[:n_variables]).astype(int)
    selected: dict[str, list[str]] = {}
    for bit, (service_name, path_name) in zip(
        bits[: model.semantic_count],
        model.semantic_keys,
    ):
        if bit:
            selected.setdefault(service_name, []).append(path_name)

    return PathQUBOSolution(
        energy=model.energy(bits),
        bits=bits,
        selected_paths={
            service: tuple(paths)
            for service, paths in selected.items()
        },
        success=bool(result.success),
        message=str(result.message),
    )
