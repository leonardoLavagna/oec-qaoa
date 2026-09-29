"""Deterministic instances used to validate the OEC formulations."""

from __future__ import annotations

from .model import Edge, OECInstance, Service, StateNode


def _n(physical: str, time: int, processed: int) -> StateNode:
    return StateNode(physical, time, processed)


def canonical_c1() -> OECInstance:
    """Return the first hand-constructed validation instance.

    C1 contains two EO services and one shared on-board processing opportunity.
    The capacities are chosen so that exactly one structural solution is
    feasible: service h0 is processed on S1, whereas h1 reaches the virtual
    ground station unprocessed and is processed there.  The instance therefore
    tests transmission, storage, processing placement and shared-resource
    coupling without introducing unnecessary combinatorial size.
    """

    services = (
        Service(
            name="h0",
            source="S0",
            generation_time=0,
            deadline=3,
            unprocessed_mb=4.0,
            processed_mb=1.0,
        ),
        Service(
            name="h1",
            source="S2",
            generation_time=0,
            deadline=3,
            unprocessed_mb=3.0,
            processed_mb=1.0,
        ),
    )

    edges: list[Edge] = []

    # Acquisition-time inter-satellite transfers.  Only unprocessed data are
    # present at this point in C1.
    edges.extend(
        [
            Edge(
                "tx_S0_S1_t0_p0",
                _n("S0", 0, 0),
                _n("S1", 0, 0),
                "transmission",
                "tx_S0_S1_t0",
                0.05,
            ),
            Edge(
                "tx_S2_S1_t0_p0",
                _n("S2", 0, 0),
                _n("S1", 0, 0),
                "transmission",
                "tx_S2_S1_t0",
                0.05,
            ),
        ]
    )

    # Storage edges advance the service by one time cycle.  The resource key is
    # shared across the two processing layers, matching one physical memory.
    for t in range(3):
        for physical in ("S0", "S1", "S2", "G"):
            for p in (0, 1):
                edges.append(
                    Edge(
                        f"st_{physical}_t{t}_p{p}",
                        _n(physical, t, p),
                        _n(physical, t + 1, p),
                        "storage",
                        f"mem_{physical}_t{t}",
                        0.0,
                    )
                )

    # C1 deliberately exposes one satellite-processing opportunity.  Processing
    # on S0 and S2 is disabled in this synthetic instance so that the optimum is
    # structurally identifiable and can be used as a regression target.
    edges.append(
        Edge(
            "proc_S1_t1",
            _n("S1", 1, 0),
            _n("S1", 1, 1),
            "processing",
            "proc_S1_t1",
            1.0,
        )
    )

    # Both processing states share the same physical downlink capacity.
    for p in (0, 1):
        edges.append(
            Edge(
                f"tx_S1_G_t2_p{p}",
                _n("S1", 2, p),
                _n("G", 2, p),
                "transmission",
                "downlink_S1_G_t2",
                0.05,
            )
        )

    # Ground processing is available at the delivery cycle.  Its unit energy is
    # also the objective coefficient per unprocessed Mb.
    edges.append(
        Edge(
            "proc_G_t3",
            _n("G", 3, 0),
            _n("G", 3, 1),
            "processing",
            "proc_G_t3",
            1.0,
        )
    )

    capacities = {
        "tx_S0_S1_t0": 10.0,
        "tx_S2_S1_t0": 10.0,
        "downlink_S1_G_t2": 4.0,
        "proc_S1_t1": 4.0,
        "proc_G_t3": 100.0,
    }
    for t in range(3):
        capacities.update(
            {
                f"mem_S0_t{t}": 100.0,
                f"mem_S1_t{t}": 100.0,
                f"mem_S2_t{t}": 100.0,
                f"mem_G_t{t}": 100.0,
            }
        )

    energy_budgets = {
        ("S0", 0): 10.0,
        ("S2", 0): 10.0,
        ("S1", 1): 4.0,
        ("S1", 2): 10.0,
    }

    return OECInstance(
        name="C1",
        ground_node="G",
        services=services,
        edges=tuple(edges),
        resource_capacities_mb=capacities,
        energy_budgets_j=energy_budgets,
    )
