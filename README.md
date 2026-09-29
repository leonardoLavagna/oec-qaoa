# OEC-QAOA

Hybrid quantum-classical optimization of Orbital Edge Computing in LEO satellite networks using QUBO formulations and QAOA.

The repository contains the reference OEC model, direct and path-based QUBO encodings, exact validation routines, resource-scaling experiments, ideal-state QAOA simulations, finite-sampling analysis, and circuit/connectivity metrics.

## Structure

```text
.
├── notebooks/              # ordered numerical notebooks
├── scripts/                # non-interactive experiment and analysis entry points
├── src/oec_qaoa/           # models, encodings, solvers, metrics, and plotting
├── tests/                  # regression and reproducibility tests
├── pyproject.toml
└── requirements.txt
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pytest -q
```

Optional Qiskit dependencies can be installed with:

```bash
pip install -e ".[quantum]"
```

## Notebooks

The notebooks follow the numerical workflow:

1. `00_environment_and_reproducibility.ipynb` — environment and seeds.
2. `01_oec_reference_model.ipynb` — classical time-expanded OEC model.
3. `02_edge_qubo_validation.ipynb` — direct edge-based QUBO.
4. `03_path_based_reduction.ipynb` — candidate-path reduction and path-QUBO.
5. `04_resource_scaling.ipynb` — logical-variable and coupling scaling.
6. `05_qaoa_ideal_simulation.ipynb` — ideal-state QAOA.
7. `06_qaoa_finite_sampling.ipynb` — finite-shot sampling.
8. `07_circuit_and_hardware_resources.ipynb` — cost-layer and connectivity resources.
9. `08_figures_and_tables.ipynb` — consolidated figures and summary tables.

## Reproduce the numerical analyses

The main non-interactive entry points are:

```bash
python scripts/run_scaling.py
python scripts/select_qaoa_benchmarks.py
python scripts/run_ideal_qaoa.py
python scripts/run_finite_sampling.py
python scripts/run_circuit_resources.py
python scripts/generate_outputs.py
```

The scripts write machine-readable CSV outputs under `results/`. GitHub Actions run the same validation and analysis steps on the repository.

## Tests

The test suite covers the reference OEC formulation, direct QUBO equivalence, path reduction, compact QAOA benchmarks, statevector evolution, finite sampling, and circuit-resource metrics.

```bash
pytest -q
```
