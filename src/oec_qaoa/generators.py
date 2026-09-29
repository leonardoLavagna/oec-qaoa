"""Controlled synthetic OEC families for scaling experiments.

The generated instances are not intended to reproduce a particular orbital
constellation. They preserve the time-expanded routing, storage, processing
and ground-delivery semantics of the reference model while exposing a small
set of parameters that can be varied independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .model import Edge, OECInstance, Service, StateNode


def _n(physical: str, time: int, processed: int) -> StateNode:
    return StateNode(physical, time, processed)


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters defining one controlled scaling instance."""

    n_satellites: int
    n_time_cycles: int
    n_services: int
    capacity_factor: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_satellites < 3:
            raise ValueError("n_satellites must be at least 3")
        if self.n_time_cycles < 3:
            raise ValueError("n_time_cycles must be at least 3")
        if self.n_services < 1:
            raise ValueError("n_services must be positive")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")


def synthetic_relay_instance(config: SyntheticConfig) -> OECInstance:
    """Generate a deterministic source-relay OEC instance.

    One acquisition satellite feeds n_satellites - 1 relay satellites at time
    zero. Every relay can process data near the end of the horizon and downlink
    either processed or unprocessed data at the deadline. Services may
    consequently choose both a relay and a processing location, while the time
    horizon controls the number of storage decisions in the direct encoding.

    The capacity_factor scales balanced per-relay processing and downlink
    capacities. Values below one increase contention and values above one
    relax the shared constraints.
    """

    rng = np.random.default_rng(config.seed)

    source = "S0"
    relays = tuple(f"R{index}" for index in range(1, config.n_satellites))
    ground = "G"
    deadline = config.n_time_cycles - 1
    processing_time = deadline - 1

    unprocessed_sizes = rng.integers(
        low=3, high=7, size=config.n_services
    ).astype(float)
    processed_sizes = np.maximum(
        1.0, np.floor(unprocessed_sizes / 3.0)
    )

    services = tuple(
        Service(
            name=f"h{index}",
            source=source,
            generation_time=0,
            deadline=deadline,
            unprocessed_mb=float(unprocessed_sizes[index]),
            processed_mb=float(processed_sizes[index]),
        )
        for index in range(config.n_services)
    )

    edges: list[Edge] = []

    for relay in relays:
        edges.append(
            Edge(
                f"tx_{source}_{relay}_t0_p0",
                _n(source, 0, 0),
                _n(relay, 0, 0),
                "transmission",
                f"tx_{source}_{relay}_t0",
                0.05,
            )
        )

    physical_nodes = (source, *relays, ground)
    for time in range(deadline):
        for physical in physical_nodes:
            for processed in (0, 1):
                edges.append(
                    Edge(
                        f"st_{physical}_t{time}_p{processed}",
                        _n(physical, time, processed),
                        _n(physical, time + 1, processed),
                        "storage",
                        f"mem_{physical}_t{time}",
                        0.0,
                    )
                )

    for relay in relays:
        edges.append(
            Edge(
                f"proc_{relay}_t{processing_time}",
                _n(relay, processing_time, 0),
                _n(relay, processing_time, 1),
                "processing",
                f"proc_{relay}_t{processing_time}",
                1.0,
            )
        )
        for processed in (0, 1):
            edges.append(
                Edge(
                    f"tx_{relay}_{ground}_t{deadline}_p{processed}",
                    _n(relay, deadline, processed),
                    _n(ground, deadline, processed),
                    "transmission",
                    f"downlink_{relay}_{ground}_t{deadline}",
                    0.05,
                )
            )

    edges.append(
        Edge(
            f"proc_{ground}_t{deadline}",
            _n(ground, deadline, 0),
            _n(ground, deadline, 1),
            "processing",
            f"proc_{ground}_t{deadline}",
            1.0,
        )
    )

    total_unprocessed = float(unprocessed_sizes.sum())
    max_unprocessed = float(unprocessed_sizes.max())
    balanced = total_unprocessed / len(relays)
    shared_capacity = max(
        max_unprocessed,
        config.capacity_factor * balanced,
    )

    capacities: dict[str, float] = {
        f"proc_{ground}_t{deadline}": total_unprocessed,
    }
    for relay in relays:
        capacities[f"tx_{source}_{relay}_t0"] = total_unprocessed
        capacities[f"proc_{relay}_t{processing_time}"] = shared_capacity
        capacities[f"downlink_{relay}_{ground}_t{deadline}"] = shared_capacity

    for time in range(deadline):
        for physical in physical_nodes:
            capacities[f"mem_{physical}_t{time}"] = total_unprocessed

    return OECInstance(
        name=(
            f"SYN_S{config.n_satellites}_T{config.n_time_cycles}_"
            f"H{config.n_services}_F{config.capacity_factor:g}_"
            f"seed{config.seed}"
        ),
        ground_node=ground,
        services=services,
        edges=tuple(edges),
        resource_capacities_mb=capacities,
        energy_budgets_j={},
    )
