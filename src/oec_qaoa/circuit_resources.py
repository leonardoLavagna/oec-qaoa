"""Circuit- and connectivity-level resource analysis for QUBO cost layers.

The analysis remains noise-free and compilation-agnostic. It translates the
QUBO interaction graph into logical two-qubit requirements and evaluates how
those requirements align with simple representative hardware-connectivity
graphs. Routing quantities are reported as graph-distance proxies rather than
as device-specific SWAP counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, sqrt

import networkx as nx
import numpy as np

from .edge_qubo import QUBOModel
from .path_qubo import PathQUBOModel


@dataclass(frozen=True)
class CostLayerMetrics:
    """Logical structure of one QAOA cost layer."""

    logical_qubits: int
    zz_terms: int
    interaction_max_degree: int
    edge_chromatic_lower_bound: int
    greedy_parallel_depth: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ConnectivityMetrics:
    """Alignment between a logical interaction graph and hardware connectivity."""

    topology: str
    physical_qubits: int
    hardware_edges: int
    directly_supported_fraction: float
    mean_interaction_distance: float
    max_interaction_distance: int
    distance_excess_sum: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def interaction_graph(model: QUBOModel | PathQUBOModel) -> nx.Graph:
    """Return the graph induced by non-zero quadratic QUBO couplings."""

    graph = nx.Graph()
    graph.add_nodes_from(range(len(model.variable_names)))
    graph.add_edges_from(model.quadratic)
    return graph


def cost_layer_metrics(model: QUBOModel | PathQUBOModel) -> CostLayerMetrics:
    """Estimate ideal parallel depth of one commuting ZZ cost layer.

    The maximum interaction degree is a lower bound on the number of two-qubit
    time slots required if each logical qubit participates in at most one
    two-qubit operation per slot. A deterministic greedy coloring of the line
    graph supplies a constructive schedule and therefore an upper estimate for
    the same logical layer under all-to-all connectivity.
    """

    graph = interaction_graph(model)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    max_degree = max((degree for _, degree in graph.degree()), default=0)

    if m == 0:
        greedy_depth = 0
    else:
        line_graph = nx.line_graph(graph)
        colors = nx.coloring.greedy_color(line_graph, strategy="largest_first")
        greedy_depth = 1 + max(colors.values(), default=-1)

    return CostLayerMetrics(
        logical_qubits=n,
        zz_terms=m,
        interaction_max_degree=max_degree,
        edge_chromatic_lower_bound=max_degree,
        greedy_parallel_depth=greedy_depth,
    )


def hardware_graph(topology: str, n_qubits: int) -> nx.Graph:
    """Construct a simple representative hardware-connectivity graph."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if topology == "line":
        return nx.path_graph(n_qubits)
    if topology == "ring":
        if n_qubits == 1:
            return nx.empty_graph(1)
        if n_qubits == 2:
            return nx.path_graph(2)
        return nx.cycle_graph(n_qubits)
    if topology == "grid":
        rows = max(1, int(sqrt(n_qubits)))
        cols = ceil(n_qubits / rows)
        base = nx.grid_2d_graph(rows, cols)
        ordered = sorted(base.nodes())[:n_qubits]
        graph = base.subgraph(ordered).copy()
        return nx.convert_node_labels_to_integers(graph, ordering="sorted")
    if topology == "all_to_all":
        return nx.complete_graph(n_qubits)
    raise ValueError("topology must be one of: line, ring, grid, all_to_all")


def _degree_based_mapping(logical: nx.Graph, hardware: nx.Graph) -> dict[int, int]:
    """Map high-degree logical vertices to central hardware vertices."""

    logical_order = sorted(
        logical.nodes(),
        key=lambda node: (-logical.degree(node), node),
    )
    hardware_distances = dict(nx.all_pairs_shortest_path_length(hardware))
    hardware_order = sorted(
        hardware.nodes(),
        key=lambda node: (
            sum(hardware_distances[node].values()),
            -hardware.degree(node),
            node,
        ),
    )
    if len(hardware_order) < len(logical_order):
        raise ValueError("hardware graph has fewer qubits than logical graph")
    return dict(zip(logical_order, hardware_order))


def connectivity_metrics(
    model: QUBOModel | PathQUBOModel,
    topology: str,
) -> ConnectivityMetrics:
    """Measure graph-distance pressure under one deterministic placement.

    The distance-excess sum is the sum of d-1 over logical ZZ interactions,
    where d is the shortest-path distance between the assigned physical qubits.
    It is a routing-pressure proxy, not a compiled SWAP count.
    """

    logical = interaction_graph(model)
    hardware = hardware_graph(topology, logical.number_of_nodes())
    mapping = _degree_based_mapping(logical, hardware)

    if logical.number_of_edges() == 0:
        return ConnectivityMetrics(
            topology=topology,
            physical_qubits=hardware.number_of_nodes(),
            hardware_edges=hardware.number_of_edges(),
            directly_supported_fraction=1.0,
            mean_interaction_distance=0.0,
            max_interaction_distance=0,
            distance_excess_sum=0,
        )

    distances = dict(nx.all_pairs_shortest_path_length(hardware))
    interaction_distances = np.asarray(
        [distances[mapping[u]][mapping[v]] for u, v in logical.edges()],
        dtype=int,
    )

    return ConnectivityMetrics(
        topology=topology,
        physical_qubits=hardware.number_of_nodes(),
        hardware_edges=hardware.number_of_edges(),
        directly_supported_fraction=float(np.mean(interaction_distances == 1)),
        mean_interaction_distance=float(interaction_distances.mean()),
        max_interaction_distance=int(interaction_distances.max()),
        distance_excess_sum=int(np.maximum(interaction_distances - 1, 0).sum()),
    )