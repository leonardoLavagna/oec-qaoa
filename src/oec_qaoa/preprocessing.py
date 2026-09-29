"""Classical preprocessing shared by the exact and QUBO formulations.

A service does not need binary variables for every edge of the complete
time-expanded graph.  Edges that cannot be reached from its acquisition state
or cannot lead to the processed ground-state destination can be removed before
the optimization model is assembled.  This pruning preserves the set of
source-to-destination paths and avoids interpreting physically irrelevant graph
regions as part of the quantum encoding.
"""

from __future__ import annotations

from collections import defaultdict, deque

from .model import Edge, OECInstance, Service


def service_subgraph_edges(
    instance: OECInstance,
    service: Service,
) -> tuple[Edge, ...]:
    """Return the edges lying on at least one admissible service path."""

    window = [
        edge
        for edge in instance.edges
        if service.generation_time <= edge.source.time <= service.deadline
        and service.generation_time <= edge.target.time <= service.deadline
    ]

    outgoing: dict[object, list[Edge]] = defaultdict(list)
    incoming: dict[object, list[Edge]] = defaultdict(list)
    for edge in window:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    source = instance.source_node(service)
    sink = instance.sink_node(service)

    forward = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for edge in outgoing[node]:
            if edge.target not in forward:
                forward.add(edge.target)
                queue.append(edge.target)

    backward = {sink}
    queue = deque([sink])
    while queue:
        node = queue.popleft()
        for edge in incoming[node]:
            if edge.source not in backward:
                backward.add(edge.source)
                queue.append(edge.source)

    return tuple(
        edge
        for edge in window
        if edge.source in forward and edge.target in backward
    )
