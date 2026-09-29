"""Figures and summary tables from validated numerical outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .benchmarks import FROZEN_QAOA_BENCHMARKS, build_frozen_benchmark
from .circuit_resources import cost_layer_metrics
from .metrics import qubo_resource_metrics


def _prepare_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def benchmark_table() -> pd.DataFrame:
    """Return the frozen benchmark metadata used in the QAOA study."""

    rows = []
    for benchmark in FROZEN_QAOA_BENCHMARKS:
        instance, _, model = build_frozen_benchmark(benchmark)
        resources = qubo_resource_metrics(model)
        cost = cost_layer_metrics(model)
        rows.append(
            {
                "benchmark": benchmark.benchmark_id,
                "instance": instance.name,
                "services": benchmark.n_services,
                "K": benchmark.k,
                "logical_qubits": resources.logical_variables,
                "semantic_variables": resources.semantic_variables,
                "slack_variables": resources.slack_variables,
                "quadratic_couplings": resources.quadratic_couplings,
                "qubo_density": resources.qubo_density,
                "cost_layer_depth_lower": cost.edge_chromatic_lower_bound,
                "cost_layer_depth_greedy": cost.greedy_parallel_depth,
            }
        )
    return pd.DataFrame(rows)


def plot_resource_scaling(
    scaling: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create logical-variable and coupling scaling figures."""

    output_dir = _prepare_output(output_dir)
    subset = scaling[scaling["sweep"] == "n_services"].copy()

    grouped = (
        subset.groupby(
            ["sweep_value", "formulation", "K"],
            dropna=False,
        )
        .agg(
            logical_variables=("logical_variables", "mean"),
            quadratic_couplings=("quadratic_couplings", "mean"),
        )
        .reset_index()
    )

    for metric, filename, ylabel in [
        ("logical_variables", "fig_resource_logical_variables.png", "Logical variables"),
        ("quadratic_couplings", "fig_resource_quadratic_couplings.png", "Quadratic couplings"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))

        edge = grouped[grouped["formulation"] == "edge"]
        ax.plot(
            edge["sweep_value"],
            edge[metric],
            marker="o",
            linewidth=1.8,
            label="Edge QUBO",
        )

        markers = {1: "s", 2: "^", 4: "D", 8: "v"}
        linestyles = {1: "--", 2: "-.", 4: ":", 8: (0, (3, 1, 1, 1))}
        for k in sorted(grouped["K"].dropna().unique()):
            k_int = int(k)
            path = grouped[
                (grouped["formulation"] == "path")
                & (grouped["K"] == k)
            ]
            ax.plot(
                path["sweep_value"],
                path[metric],
                marker=markers.get(k_int, "o"),
                linestyle=linestyles.get(k_int, "--"),
                linewidth=1.5,
                label=f"Path QUBO, K={k_int}",
            )

        ax.set_xlabel("Number of services")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, output_dir / filename)


def plot_reduction_ratios(
    ratios: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot path-to-edge logical-variable and coupling ratios separately."""

    output_dir = _prepare_output(output_dir)
    subset = ratios[ratios["sweep"] == "n_services"].copy()
    grouped = (
        subset.groupby(["sweep_value", "K"])[
            ["logical_variable_ratio", "coupling_ratio"]
        ]
        .mean()
        .reset_index()
    )

    for metric, filename, ylabel in [
        (
            "logical_variable_ratio",
            "fig_path_logical_variable_ratio.png",
            "Path / edge logical-variable ratio",
        ),
        (
            "coupling_ratio",
            "fig_path_coupling_ratio.png",
            "Path / edge coupling ratio",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for k in sorted(grouped["K"].dropna().unique()):
            frame = grouped[grouped["K"] == k]
            ax.plot(
                frame["sweep_value"],
                frame[metric],
                marker="o",
                linewidth=1.6,
                label=f"K={int(k)}",
            )

        ax.axhline(1.0, linewidth=1.0, linestyle=":")
        ax.set_xlabel("Number of services")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, output_dir / filename)


def plot_ideal_qaoa(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot ideal-QAOA optimal and feasible probabilities versus depth."""

    output_dir = _prepare_output(output_dir)

    for metric, filename, ylabel in [
        ("optimal_probability_mean", "fig_qaoa_optimal_probability.png", "Mean optimal-state probability"),
        ("feasible_probability_mean", "fig_qaoa_feasible_probability.png", "Mean feasible-state probability"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for benchmark in sorted(summary["benchmark_id"].unique()):
            frame = summary[summary["benchmark_id"] == benchmark]
            ax.plot(
                frame["depth"],
                frame[metric],
                marker="o",
                linewidth=1.6,
                label=benchmark,
            )
        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xticks(sorted(summary["depth"].unique()))
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(frameon=False)
        _save(fig, output_dir / filename)


def plot_finite_sampling(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create finite-shot convergence and optimum-detection figures."""

    output_dir = _prepare_output(output_dir)

    # Use the best-energy depth selected for each benchmark in the frozen ideal study.
    selected_depth = {"B1": 3, "B2": 3, "B3": 3}

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for benchmark, depth in selected_depth.items():
        frame = summary[
            (summary["benchmark_id"] == benchmark)
            & (summary["depth"] == depth)
        ]
        ax.plot(
            frame["shots"],
            frame["sampled_expected_energy_std"],
            marker="o",
            linewidth=1.6,
            label=f"{benchmark}, p={depth}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Shots")
    ax.set_ylabel("Std. dev. of sampled energy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, output_dir / "fig_sampling_energy_uncertainty.png")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for benchmark, depth in selected_depth.items():
        frame = summary[
            (summary["benchmark_id"] == benchmark)
            & (summary["depth"] == depth)
        ]
        ax.plot(
            frame["shots"],
            frame["optimum_observation_rate"],
            marker="o",
            linewidth=1.6,
            label=f"{benchmark} empirical",
        )
        ax.plot(
            frame["shots"],
            frame["analytic_optimum_detection_probability"],
            linestyle="--",
            linewidth=1.3,
            label=f"{benchmark} analytic",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Shots")
    ax.set_ylabel("Probability of observing an optimum")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output_dir / "fig_sampling_optimum_detection.png")


def plot_connectivity(
    connectivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot mean interaction distance for representative topologies."""

    output_dir = _prepare_output(output_dir)
    pivot = connectivity.pivot(
        index="benchmark_id",
        columns="topology",
        values="mean_interaction_distance",
    )
    order = [
        topology
        for topology in ["all_to_all", "grid", "ring", "line"]
        if topology in pivot.columns
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = range(len(pivot.index))
    width = 0.18
    offsets = [
        (index - (len(order) - 1) / 2) * width
        for index in range(len(order))
    ]

    for offset, topology in zip(offsets, order):
        ax.bar(
            [value + offset for value in x],
            pivot[topology],
            width=width,
            label=topology.replace("_", " "),
        )

    ax.set_xticks(list(x), list(pivot.index))
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Mean logical-interaction distance")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, output_dir / "fig_connectivity_distance.png")

    support = connectivity.pivot(
        index="benchmark_id",
        columns="topology",
        values="directly_supported_fraction",
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for offset, topology in zip(offsets, order):
        ax.bar(
            [value + offset for value in x],
            support[topology],
            width=width,
            label=topology.replace("_", " "),
        )

    ax.set_xticks(list(x), list(support.index))
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Directly supported interaction fraction")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, output_dir / "fig_connectivity_direct_support.png")


def create_summary_tables(
    ideal_summary: pd.DataFrame,
    finite_summary: pd.DataFrame,
    connectivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write compact CSV summary tables."""

    output_dir = _prepare_output(output_dir)
    benchmark_table().to_csv(
        output_dir / "table_benchmarks.csv",
        index=False,
    )

    ideal_summary.to_csv(
        output_dir / "table_ideal_qaoa.csv",
        index=False,
    )

    selected = finite_summary[
        finite_summary["shots"].isin([512, 2048, 8192])
    ].copy()
    selected.to_csv(
        output_dir / "table_finite_sampling.csv",
        index=False,
    )

    connectivity.to_csv(
        output_dir / "table_connectivity.csv",
        index=False,
    )


def build_all_outputs(
    results_root: Path,
    output_dir: Path,
) -> None:
    """Create all figures and review tables from validated CSV results."""

    scaling = pd.read_csv(results_root / "scaling" / "resource_scaling.csv")
    ratios = pd.read_csv(results_root / "scaling" / "resource_reduction_ratios.csv")
    ideal = pd.read_csv(results_root / "qaoa" / "ideal_qaoa_summary.csv")
    finite = pd.read_csv(results_root / "qaoa" / "finite_sampling_summary.csv")
    connectivity = pd.read_csv(results_root / "hardware" / "connectivity_pressure.csv")

    plot_resource_scaling(scaling, output_dir)
    plot_reduction_ratios(ratios, output_dir)
    plot_ideal_qaoa(ideal, output_dir)
    plot_finite_sampling(finite, output_dir)
    plot_connectivity(connectivity, output_dir)
    create_summary_tables(ideal, finite, connectivity, output_dir)