# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import argparse
import glob
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_NAMES_TROTTER = {
    "FQE": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
    "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim",
}
BENCHMARK_NAMES_QUAD_HAM = {
    "FQE": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_fqe",
    "ffsim": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim",
}
BENCHMARK_NAMES_DIAG_COULOMB = {
    "FQE": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_fqe",
    "ffsim": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_ffsim",
}
DESIRED_BENCHMARKS = (
    set(BENCHMARK_NAMES_TROTTER.values())
    | set(BENCHMARK_NAMES_QUAD_HAM.values())
    | set(BENCHMARK_NAMES_DIAG_COULOMB.values())
)
DESIRED_BENCHMARKS_MULTI_THREADED = (
    {BENCHMARK_NAMES_TROTTER["ffsim"]}
    | {BENCHMARK_NAMES_QUAD_HAM["ffsim"]}
    | {BENCHMARK_NAMES_DIAG_COULOMB["ffsim"]}
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


parser = argparse.ArgumentParser()
parser.add_argument("machine", help="Machine name (subdirectory of .asv/results/)")
args = parser.parse_args()
RESULTS_DIR = f".asv/results/{args.machine}"

DATA_SINGLE_THREADED = find_result_data(RESULTS_DIR, 1, DESIRED_BENCHMARKS)
DATA_MULTI_THREADED = find_result_data(
    RESULTS_DIR, 6, DESIRED_BENCHMARKS_MULTI_THREADED
)
print(f"Single threaded commit: {DATA_SINGLE_THREADED['commit_hash'][:8]}")
print(
    f"Single threaded date: {datetime.fromtimestamp(DATA_SINGLE_THREADED['date'] / 1000)}"
)
print(f"Multi threaded commit: {DATA_MULTI_THREADED['commit_hash'][:8]}")
print(
    f"Multi threaded date: {datetime.fromtimestamp(DATA_MULTI_THREADED['date'] / 1000)}"
)

assert (
    int(DATA_SINGLE_THREADED["env_vars"]["OMP_NUM_THREADS"])
    == int(DATA_SINGLE_THREADED["env_vars"]["RAYON_NUM_THREADS"])
    == 1
)
assert (
    int(DATA_MULTI_THREADED["env_vars"]["OMP_NUM_THREADS"])
    == int(DATA_MULTI_THREADED["env_vars"]["RAYON_NUM_THREADS"])
    > 1
)
MULTI_THREADED_NUM_THREADS = int(DATA_MULTI_THREADED["env_vars"]["OMP_NUM_THREADS"])

print("Single-threaded benchmarks:")
for k in DATA_SINGLE_THREADED["results"]:
    print(f"\t{k}")
print("Multi-threaded benchmarks:")
for k in DATA_MULTI_THREADED["results"]:
    print(f"\t{k}")


colors = {
    "FQE": "#be95ff",
    "ffsim": "#0f62fe",
}
fmts = {
    "FQE": "s:",
    "ffsim": "o--",
}

capsize = 4
legend_fontsize = 12
tick_label_fontsize = 13
axis_label_fontsize = 14
title_fontsize = 15


def plot_results(
    axes,
    benchmark_names: dict[str, str],
    norb_range: list[int],
    ylim: tuple[float, float] | None = None,
) -> None:
    benchmark_results_single_threaded = {}
    benchmark_results_multi_threaded = {}
    for label, benchmark_name in benchmark_names.items():
        these_results = dict(
            zip(
                DATA_SINGLE_THREADED["result_columns"],
                DATA_SINGLE_THREADED["results"][benchmark_name],
            )
        )
        benchmark_results_single_threaded[label] = dict(
            zip(
                itertools.product(*these_results["params"]),
                zip(
                    [np.nan if x is None else x for x in these_results["result"]],
                    [np.nan if x is None else x for x in these_results["stats_q_25"]],
                    [np.nan if x is None else x for x in these_results["stats_q_75"]],
                ),
            )
        )
        if label == "ffsim":
            these_results = dict(
                zip(
                    DATA_MULTI_THREADED["result_columns"],
                    DATA_MULTI_THREADED["results"][benchmark_name],
                )
            )
            benchmark_results_multi_threaded[label] = dict(
                zip(
                    itertools.product(*these_results["params"]),
                    zip(
                        [np.nan if x is None else x for x in these_results["result"]],
                        [
                            np.nan if x is None else x
                            for x in these_results["stats_q_25"]
                        ],
                        [
                            np.nan if x is None else x
                            for x in these_results["stats_q_75"]
                        ],
                    ),
                )
            )

    for filling_fraction, ax in zip([0.5, 0.25], axes):
        times_single_threaded = {
            label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
            for label, times in benchmark_results_single_threaded.items()
        }
        times_multi_threaded = {
            label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
            for label, times in benchmark_results_multi_threaded.items()
        }

        for label_single, data_single in times_single_threaded.items():
            times, stats_q_25, stats_q_75 = zip(*data_single)
            yerr_a = [max(0, t - x) for t, x in zip(times, stats_q_25)]
            yerr_b = [max(0, x - t) for t, x in zip(times, stats_q_75)]
            ax.errorbar(
                norb_range,
                times,
                yerr=(yerr_a, yerr_b),
                fmt=fmts[label_single],
                color=colors[label_single],
                label=label_single,
                capsize=capsize,
            )

        for label_multi, data_multi in times_multi_threaded.items():
            times, stats_q_25, stats_q_75 = zip(*data_multi)
            yerr_a = [max(0, t - x) for t, x in zip(times, stats_q_25)]
            yerr_b = [max(0, x - t) for t, x in zip(times, stats_q_75)]
            if any(err < 0 for err in yerr_a):
                pass
            if any(err < 0 for err in yerr_b):
                pass
            ax.errorbar(
                norb_range,
                times,
                yerr=(yerr_a, yerr_b),
                fmt=fmts[label_multi],
                color=colors[label_multi],
                alpha=0.5,
                label=f"{label_multi}, {MULTI_THREADED_NUM_THREADS} threads",
                capsize=capsize,
            )

        ax.set_yscale("log")
        ax.set_xticks(norb_range)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("# orbitals", fontsize=axis_label_fontsize)

    axes[0].set_ylabel("Time (s)", fontsize=axis_label_fontsize)


fig = plt.figure(figsize=(14, 4))
outer_gs = fig.add_gridspec(1, 3, wspace=0.32)

group_positions = [(0, 0), (0, 1), (0, 2)]
axes_groups = []
for row, col in group_positions:
    inner_gs = outer_gs[row, col].subgridspec(1, 2, wspace=0.05)
    ax_left = fig.add_subplot(inner_gs[0, 0])
    ax_right = fig.add_subplot(inner_gs[0, 1], sharey=ax_left)
    ax_right.tick_params(axis="y", labelleft=False)
    axes_groups.append([ax_left, ax_right])

for axes in axes_groups:
    for ax in axes:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

norb_range = [4, 8, 12, 16]

plot_results(
    axes_groups[0],
    benchmark_names=BENCHMARK_NAMES_QUAD_HAM,
    norb_range=norb_range,
    # ylim=(1e-4, 2e2),
)
plot_results(
    axes_groups[1],
    benchmark_names=BENCHMARK_NAMES_DIAG_COULOMB,
    norb_range=norb_range,
)
plot_results(
    axes_groups[2],
    benchmark_names=BENCHMARK_NAMES_TROTTER,
    norb_range=norb_range,
    # ylim=(1e-3, 1e3),
)
group_titles = [
    "Quadratic Hamiltonian evolution",
    "Diagonal Coulomb evolution",
    "Double factorized Trotter",
]

for (row, col), title, axes in zip(group_positions, group_titles, axes_groups):
    axes[0].set_title("1/2 filling", fontsize=title_fontsize)
    axes[1].set_title("1/4 filling", fontsize=title_fontsize)
    ax_span = fig.add_subplot(outer_gs[row, col])
    ax_span.set_axis_off()
    ax_span.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=30)

handles, labels = axes_groups[0][0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    fontsize=legend_fontsize,
)
fig.subplots_adjust(bottom=0.26)

filepath = Path("plots/gates.pdf")
os.makedirs(filepath.parent, exist_ok=True)
plt.savefig(filepath, bbox_inches="tight")
print(f"Saved figure to {filepath}.")
