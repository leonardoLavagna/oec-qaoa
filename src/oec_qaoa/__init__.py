"""Tools for the OEC-QAOA numerical study."""

from .edge_qubo import QUBOModel, QUBOSolution, build_edge_qubo, solve_qubo_exact
from .generators import SyntheticConfig, synthetic_relay_instance
from .instances import canonical_c1, canonical_c2
from .metrics import QUBOResourceMetrics, qubo_resource_metrics
from .model import Edge, OECInstance, Service, StateNode
from .path_qubo import (
    PathQUBOModel,
    PathQUBOSolution,
    build_path_qubo,
    solve_path_qubo_exact,
)
from .path_reduction import (
    CandidatePath,
    PathSelectionSolution,
    candidate_sets,
    decomposition_gap,
    enumerate_candidate_paths,
    solve_path_selection,
)

__all__ = [
    "CandidatePath",
    "Edge",
    "OECInstance",
    "PathQUBOModel",
    "PathQUBOSolution",
    "PathSelectionSolution",
    "QUBOModel",
    "QUBOSolution",
    "QUBOResourceMetrics",
    "Service",
    "SyntheticConfig",
    "StateNode",
    "build_edge_qubo",
    "build_path_qubo",
    "candidate_sets",
    "canonical_c1",
    "canonical_c2",
    "decomposition_gap",
    "enumerate_candidate_paths",
    "solve_path_qubo_exact",
    "solve_path_selection",
    "qubo_resource_metrics",
    "solve_qubo_exact",
    "synthetic_relay_instance",
]
