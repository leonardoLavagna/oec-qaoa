# OEC-QAOA

Hybrid quantum-classical optimization of Orbital Edge Computing in LEO satellite networks using QUBO formulations and QAOA.

This repository accompanies the journal study on the reformulation of the time-expanded Orbital Edge Computing resource-allocation problem as a family of QUBO models. The numerical work is organized as a sequence of reproducible Jupyter notebooks supported by a small Python package, so that the physical model, the QUBO encodings, the classical reductions and the quantum experiments can be validated independently.

## Development logic

The study follows the hierarchy

```text
OEC reference model
    -> direct edge-based QUBO
    -> path-based reduction
    -> resource-scaling analysis
    -> ideal QAOA
    -> finite-sampling
    -> circuit and hardware-resource analysis
```

The first implementation milestone is intentionally classical. It establishes deterministic reference OEC instances and exact solutions against which the later QUBO formulations will be checked.

## Repository structure

- `notebooks/`: ordered experimental notebooks.
- `src/oec_qaoa/`: reusable model, solver, validation and metric code.
- `tests/`: regression tests for the mathematical implementation.
- `data/`: canonical and generated instances.
- `results/`: serialized experiment outputs and paper figures.

## Current status

- [x] Repository and reproducibility baseline
- [x] Canonical OEC reference model
- [x] Direct edge-QUBO
- [x] Path-based reduction
- [x] Resource scaling
- [x] QAOA benchmarks
- [x] Finite-sampling analysis
- [ ] Circuit/hardware-resource analysis

The manuscript remains frozen while the simulations are developed. Numerical statements will be transferred to the paper only after the corresponding implementation and regression checks have been reviewed.
