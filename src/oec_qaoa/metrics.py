"""Resource metrics for edge- and path-based QUBO encodings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

from .edge_qubo import QUBOModel
from .path_qubo import PathQUBOModel


@dataclass(frozen=True)
class QUBOResourceMetrics:
    """Structural quantities associated with one QUBO model."""

    semantic_variables: int
    slack_variables: int
    logical_variables: int
    quadratic_couplings: int
    qubo_density: float
    max_degree: int
    mean_degree: float
    connected_components: int
    coefficient_max_abs: float
    coefficient_min_abs_nonzero: float
    coefficient_dynamic_range: float
    estimated_cost_two_qubit_terms: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def qubo_resource_metrics(
    model: QUBOModel | PathQUBOModel,
) -> QUBOResourceMetrics:
    """Measure logical size and interaction structure of a QUBO."""

    n = len(model.variable_names)
    m = len(model.quadratic)
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(model.quadratic)

    degrees = np.asarray(
        [degree for _, degree in graph.degree()],
        dtype=float,
    )
    max_degree = int(degrees.max()) if len(degrees) else 0
    mean_degree = float(degrees.mean()) if len(degrees) else 0.0
    components = nx.number_connected_components(graph) if n else 0

    coefficients = [
        abs(float(value))
        for value in model.linear
        if abs(float(value)) > 1e-12
    ]
    coefficients.extend(
        abs(float(value))
        for value in model.quadratic.values()
        if abs(float(value)) > 1e-12
    )

    if coefficients:
        max_abs = max(coefficients)
        min_abs = min(coefficients)
        dynamic_range = max_abs / min_abs
    else:
        max_abs = 0.0
        min_abs = 0.0
        dynamic_range = 0.0

    return QUBOResourceMetrics(
        semantic_variables=model.semantic_count,
        slack_variables=n - model.semantic_count,
        logical_variables=n,
        quadratic_couplings=m,
        qubo_density=density,
        max_degree=max_degree,
        mean_degree=mean_degree,
        connected_components=components,
        coefficient_max_abs=max_abs,
        coefficient_min_abs_nonzero=min_abs,
        coefficient_dynamic_range=dynamic_range,
        estimated_cost_two_qubit_terms=m,
    )
