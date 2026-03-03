import glob
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_result_data(results_dir: str, num_threads: int) -> dict:
    candidates = []
    for path in glob.glob(f"{results_dir}/*.json"):
        if path.endswith("machine.json"):
            continue
        with open(path) as f:
            data = json.load(f)
        if int(data.get("env_vars", {}).get("OMP_NUM_THREADS", -1)) == num_threads:
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

DATA_SINGLE_THREADED = find_result_data(RESULTS_DIR, 1)
DATA_MULTI_THREADED = find_result_data(RESULTS_DIR, 6)
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
    "Aer": "#ff7eb6",
    "FQE": "#be95ff",
    "ffsim": "#0f62fe",
}
fmts = {
    "Aer": "v-.",
    "FQE": "s:",
    "ffsim": "o--",
}


def plot_results(
    axes,
    benchmark_names: dict[str, str],
    norb_range: list[int],
    title: str,
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
                    [np.nan if x is None else x for x in these_results["stats_q_25"]],
                    [np.nan if x is None else x for x in these_results["stats_q_75"]],
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

        for (label_single, data_single), (label_multi, data_multi) in zip(
            times_single_threaded.items(), times_multi_threaded.items()
        ):
            times, stats_q_25, stats_q_75 = zip(*data_single)
            yerr_a = [t - x for t, x in zip(times, stats_q_25)]
            yerr_b = [x - t for t, x in zip(times, stats_q_75)]
            ax.errorbar(
                norb_range,
                times,
                yerr=(yerr_a, yerr_b),
                fmt=fmts[label_single],
                color=colors[label_single],
                label=label_single,
            )
            times, stats_q_25, stats_q_75 = zip(*data_multi)
            yerr_a = [t - x for t, x in zip(times, stats_q_25)]
            yerr_b = [x - t for t, x in zip(times, stats_q_75)]
            ax.errorbar(
                norb_range,
                times,
                yerr=(yerr_a, yerr_b),
                fmt=fmts[label_multi],
                color=colors[label_multi],
                alpha=0.5,
                label=f"{label_multi}, {MULTI_THREADED_NUM_THREADS} threads",
            )

        ax.set_yscale("log")
        ax.set_xticks(norb_range)
        if ylim:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel("Time (s)")
    axes[-1].yaxis.set_label_position("right")
    axes[-1].set_ylabel(
        title,
        rotation=270,
        # labelpad=15,
        va="bottom",
    )


fig, axes = plt.subplots(
    3,
    2,
    # figsize=(8, 7),
)
fig.subplots_adjust(wspace=0.25)
norb_range = [4, 8, 12, 16]

benchmark_names = {
    "Aer": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_qiskit",
    "FQE": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
    "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim",
}
title = "DF Trotter"
plot_results(
    axes[0],
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    ylim=(1e-3, 1e3),
)

benchmark_names = {
    "Aer": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_qiskit",
    "FQE": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_fqe",
    "ffsim": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim",
}
title = "Quad Ham"
plot_results(
    axes[1],
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    ylim=(1e-4, 2e2),
)

benchmark_names = {
    "FQE": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_fqe_restricted",
    "ffsim": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_ffsim",
}
title = "Op action"
plot_results(
    axes[2],
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    ylim=(5e-5, 2e3),
)

for ax_list in axes[:2]:
    for ax in ax_list:
        ax.tick_params(axis="x", labelbottom=False)
for ax in axes[2]:
    ax.set_xlabel("# orbitals")

axes[0, 0].set_title("filling 1/2")
axes[0, 1].set_title("filling 1/4")

handles, labels = axes[0, 0].get_legend_handles_labels()
# Place legend centered below the entire figure
fig.legend(
    handles,
    labels,
    loc="lower center",
    # bbox_to_anchor=(0.5, -0.02),  # slight negative y to sit just below
    ncol=3,  # tweak based on how many entries you have
    # frameon=False,
)

# Reserve extra bottom margin for the legend
fig.subplots_adjust(bottom=0.21)

filepath = Path("plots/sim.pdf")
os.makedirs(filepath.parent, exist_ok=True)
plt.savefig(filepath, bbox_inches="tight")
print(f"Saved figure to {filepath}.")
