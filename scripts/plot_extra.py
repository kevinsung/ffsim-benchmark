import glob
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SLATER_BENCHMARK_NAMES = {
    "dppy": "slater.SampleSlaterBenchmarkReal.time_sample_slater_real_gs_dppy",
    "ffsim": "slater.SampleSlaterBenchmarkReal.time_sample_slater_real_ffsim",
}
NORMAL_ORDER_BENCHMARK_NAMES = {
    "openfermion": "fermion_operator.FermionOperatorBenchmark.time_normal_order_openfermion",
    "ffsim": "fermion_operator.FermionOperatorBenchmark.time_normal_order_ffsim",
}
DESIRED_BENCHMARKS = set(SLATER_BENCHMARK_NAMES.values()) | set(
    NORMAL_ORDER_BENCHMARK_NAMES.values()
)


def find_result_data(
    results_dir: str, num_threads: int, required_benchmarks: set[str] | None = None
) -> dict:
    candidates = []
    for path in glob.glob(f"{results_dir}/*.json"):
        if path.endswith("machine.json"):
            continue
        with open(path) as f:
            data = json.load(f)
        if int(data.get("env_vars", {}).get("OMP_NUM_THREADS", -1)) == num_threads:
            if required_benchmarks is None or required_benchmarks.issubset(
                data.get("results", {}).keys()
            ):
                candidates.append((os.path.getmtime(path), data))
    if not candidates:
        raise FileNotFoundError(
            f"No result file found with OMP_NUM_THREADS={num_threads} in {results_dir}"
        )
    return max(candidates, key=lambda x: x[0])[1]


(machine,) = (
    d["machine"]
    for d in [json.load(open(p)) for p in glob.glob(".asv/results/*/machine.json")]
)
RESULTS_DIR = f".asv/results/{machine}"

DATA = find_result_data(RESULTS_DIR, 1, DESIRED_BENCHMARKS)
print(f"Commit: {DATA['commit_hash'][:8]}")
print(f"Date: {datetime.fromtimestamp(DATA['date'] / 1000)}")

assert (
    int(DATA["env_vars"]["OMP_NUM_THREADS"])
    == int(DATA["env_vars"]["RAYON_NUM_THREADS"])
    == 1
)

print("Benchmarks:")
for k in DATA["results"]:
    print(f"\t{k}")


colors = {
    "ffsim": "#0f62fe",
    "dppy": "#42be65",
    "openfermion": "#3ddbd9",
}
markers = {
    "ffsim": "o",
    "dppy": "s",
    "openfermion": "v",
}

capsize = 4
legend_fontsize = 12
tick_label_fontsize = 13
axis_label_fontsize = 14
title_fontsize = 15


def _load_benchmark_results(benchmark_names: dict[str, str]) -> dict[str, dict]:
    results = {}
    for label, benchmark_name in benchmark_names.items():
        these_results = dict(
            zip(
                DATA["result_columns"],
                DATA["results"][benchmark_name],
            )
        )
        results[label] = dict(
            zip(
                itertools.product(*these_results["params"]),
                zip(
                    [np.nan if x is None else x for x in these_results["result"]],
                    [np.nan if x is None else x for x in these_results["stats_q_25"]],
                    [np.nan if x is None else x for x in these_results["stats_q_75"]],
                ),
            )
        )
    return results


def plot_slater(
    ax,
    norb_range: list[int],
    shots: int,
    filling_fraction: float,
    ylim: tuple[float, float] | None = None,
) -> None:
    benchmark_results = _load_benchmark_results(SLATER_BENCHMARK_NAMES)
    for label, times_by_key in benchmark_results.items():
        data = [
            times_by_key[(str(norb), str(filling_fraction), str(shots))]
            for norb in norb_range
        ]
        times, stats_q_25, stats_q_75 = zip(*data)
        yerr_a = [t - x for t, x in zip(times, stats_q_25)]
        yerr_b = [x - t for t, x in zip(times, stats_q_75)]
        ax.errorbar(
            range(len(norb_range)),
            times,
            yerr=(yerr_a, yerr_b),
            marker=markers[label],
            linestyle="--",
            color=colors[label],
            label=label,
            capsize=capsize,
        )

    ax.set_yscale("log")
    ax.set_xticks(range(len(norb_range)), labels=norb_range)
    if ylim:
        ax.set_ylim(*ylim)


def plot_normal_order(
    ax,
    n_terms_range: list[int],
    ylim: tuple[float, float] | None = None,
) -> None:
    benchmark_results = _load_benchmark_results(NORMAL_ORDER_BENCHMARK_NAMES)

    for label, times_by_key in benchmark_results.items():
        data = [times_by_key[(str(n_terms),)] for n_terms in n_terms_range]
        times, stats_q_25, stats_q_75 = zip(*data)
        yerr_a = [t - x for t, x in zip(times, stats_q_25)]
        yerr_b = [x - t for t, x in zip(times, stats_q_75)]
        ax.errorbar(
            range(len(n_terms_range)),
            times,
            yerr=(yerr_a, yerr_b),
            marker=markers[label],
            linestyle="--",
            color=colors[label],
            label=label,
            capsize=capsize,
        )

    ax.set_yscale("log")
    ax.set_xticks(range(len(n_terms_range)), labels=n_terms_range)
    if ylim:
        ax.set_ylim(*ylim)


norb_range = [100, 200, 400, 800]
n_terms_range = [100, 1_000, 10_000, 100_000]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax_slater, ax_normal = axes
for ax in axes:
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)

plot_slater(ax_slater, norb_range=norb_range, shots=1_000, filling_fraction=0.25)
plot_normal_order(ax_normal, n_terms_range=n_terms_range)

ax_slater.set_title("Sample Slater", fontsize=title_fontsize)
ax_slater.set_xlabel("# orbitals", fontsize=axis_label_fontsize)
ax_slater.set_ylabel("Time (s)", fontsize=axis_label_fontsize)
ax_slater.legend(fontsize=legend_fontsize)

ax_normal.set_title("Normal order", fontsize=title_fontsize)
ax_normal.set_xlabel("# terms", fontsize=axis_label_fontsize)
ax_normal.set_ylabel("Time (s)", fontsize=axis_label_fontsize)
ax_normal.legend(fontsize=legend_fontsize)

filepath = Path("plots/extra.pdf")
os.makedirs(filepath.parent, exist_ok=True)
plt.savefig(filepath, bbox_inches="tight")
print(f"Saved figure to {filepath}.")
