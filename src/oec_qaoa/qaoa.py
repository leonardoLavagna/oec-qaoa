"""Ideal statevector QAOA for explicit QUBO models.

The implementation is intentionally small and formulation-transparent.  It
works directly with the explicit QUBO coefficients used throughout the study,
so the ideal variational results do not depend on a second optimization-model
conversion layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .path_qubo import PathQUBOModel


@dataclass(frozen=True)
class QAOAResult:
    """Result of one ideal-state QAOA optimization."""

    depth: int
    seed: int
    success: bool
    message: str
    parameters: np.ndarray
    expected_energy: float
    optimal_energy: float
    optimal_probability: float
    feasible_probability: float
    most_likely_energy: float
    best_supported_energy: float
    n_evaluations: int
    optimizer: str


def basis_energies(model: PathQUBOModel) -> np.ndarray:
    """Return QUBO energies for all computational-basis states."""

    n = len(model.variable_names)
    states = np.arange(1 << n, dtype=np.uint64)
    energy = np.full(1 << n, model.constant, dtype=float)

    for index, coefficient in enumerate(model.linear):
        if abs(float(coefficient)) <= 1e-15:
            continue
        bits = ((states >> index) & 1).astype(float)
        energy += float(coefficient) * bits

    for (i, j), coefficient in model.quadratic.items():
        bi = ((states >> i) & 1).astype(float)
        bj = ((states >> j) & 1).astype(float)
        energy += float(coefficient) * bi * bj

    return energy


def _apply_rx_all(
    state: np.ndarray,
    beta: float,
    n_qubits: int,
) -> None:
    """Apply exp(-i beta X) independently to every qubit in place."""

    c = np.cos(beta)
    s = -1j * np.sin(beta)

    for qubit in range(n_qubits):
        step = 1 << qubit
        block = step << 1
        view = state.reshape(-1, block)
        left = view[:, :step].copy()
        right = view[:, step:].copy()
        view[:, :step] = c * left + s * right
        view[:, step:] = s * left + c * right


def qaoa_statevector(
    energies: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Return the ideal QAOA state for a diagonal QUBO cost Hamiltonian."""

    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")

    dimension = len(energies)
    n_qubits = int(np.log2(dimension))
    if 1 << n_qubits != dimension:
        raise ValueError("energy array length must be a power of two")

    state = np.full(
        dimension,
        1.0 / np.sqrt(dimension),
        dtype=np.complex128,
    )

    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * energies)
        _apply_rx_all(state, float(beta), n_qubits)

    return state


def expected_energy(
    parameters: np.ndarray,
    energies: np.ndarray,
    depth: int,
) -> float:
    """Evaluate the exact statevector expectation of the QUBO Hamiltonian."""

    gammas = parameters[:depth]
    betas = parameters[depth:]
    state = qaoa_statevector(energies, gammas, betas)
    probabilities = np.abs(state) ** 2
    return float(probabilities @ energies)


def semantic_feasibility_mask(
    model: PathQUBOModel,
) -> np.ndarray:
    """Return basis-state mask satisfying one-path-per-service semantics.

    The mask evaluates the semantic path-choice equalities only and marginalizes
    over slack bits.  Shared-resource feasibility is encoded through the slack
    penalties and is captured separately by exact QUBO feasibility below.
    """

    n = len(model.variable_names)
    states = np.arange(1 << n, dtype=np.uint64)
    feasible = np.ones(1 << n, dtype=bool)

    service_indices: dict[str, list[int]] = {}
    for index, (service, _) in enumerate(model.semantic_keys):
        service_indices.setdefault(service, []).append(index)

    for indices in service_indices.values():
        count = np.zeros(1 << n, dtype=np.int16)
        for index in indices:
            count += ((states >> index) & 1).astype(np.int16)
        feasible &= count == 1

    return feasible


def qubo_feasibility_mask(
    model: PathQUBOModel,
    energies: np.ndarray | None = None,
) -> np.ndarray:
    """Return states with zero total penalty.

    For this model class the unpenalized objective is the linear contribution
    on semantic path variables.  A state is QUBO-feasible when its full energy
    equals this objective contribution, within numerical tolerance.
    """

    if energies is None:
        energies = basis_energies(model)

    n = len(model.variable_names)
    states = np.arange(1 << n, dtype=np.uint64)
    objective = np.zeros(1 << n, dtype=float)

    for index in range(model.semantic_count):
        coefficient = float(model.linear[index])
        # model.linear already contains penalty contributions, so reconstruct
        # the physical objective from selected path variables is not possible
        # from the expanded coefficients alone.
        del coefficient

    # Feasibility can instead be identified from the penalty lattice: all
    # valid states have energies below the first penalty-separated violation
    # only when the objective span is smaller than model.penalty. We therefore
    # infer validity by checking whether energy modulo the penalty can be
    # represented by the known path objective range. This helper is superseded
    # in experiments by a model-specific callback when physical feasibility is
    # required.
    raise NotImplementedError(
        "Use a model-specific feasibility callback for path-QUBO experiments."
    )


def optimize_qaoa(
    model: PathQUBOModel,
    depth: int,
    seed: int,
    feasible_mask: np.ndarray,
    optimizer: str = "COBYLA",
    maxiter: int = 150,
) -> QAOAResult:
    """Optimize one ideal QAOA run from a deterministic random initialization."""

    if depth < 1:
        raise ValueError("depth must be positive")

    energies = basis_energies(model)
    optimum = float(energies.min())
    optimal_mask = np.isclose(energies, optimum, atol=1e-9)

    rng = np.random.default_rng(seed)
    initial = np.concatenate(
        [
            rng.uniform(0.0, 2.0 * pi, size=depth),
            rng.uniform(0.0, pi, size=depth),
        ]
    )

    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return expected_energy(parameters, energies, depth)

    result = minimize(
        objective,
        initial,
        method=optimizer,
        options={"maxiter": maxiter},
    )

    parameters = np.asarray(result.x, dtype=float)
    state = qaoa_statevector(
        energies,
        parameters[:depth],
        parameters[depth:],
    )
    probabilities = np.abs(state) ** 2

    supported = probabilities > 1e-12
    most_likely = int(np.argmax(probabilities))

    return QAOAResult(
        depth=depth,
        seed=seed,
        success=bool(result.success),
        message=str(result.message),
        parameters=parameters,
        expected_energy=float(probabilities @ energies),
        optimal_energy=optimum,
        optimal_probability=float(probabilities[optimal_mask].sum()),
        feasible_probability=float(probabilities[feasible_mask].sum()),
        most_likely_energy=float(energies[most_likely]),
        best_supported_energy=float(energies[supported].min()),
        n_evaluations=evaluations,
        optimizer=optimizer,
    )
