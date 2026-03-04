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
    "ffsim (real)": "#0f62fe",
    "dppy (GS)": "#ff7eb6",
    "dppy (GS_bis)": "#be95ff",
}
fmts = {
    "ffsim (real)": "o--",
    "dppy (GS)": "v-.",
    "dppy (GS_bis)": "s:",
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
                range(len(norb_range)),
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
                range(len(norb_range)),
                times,
                yerr=(yerr_a, yerr_b),
                fmt=fmts[label_multi],
                color=colors[label_multi],
                alpha=0.5,
                label=f"{label_multi}, {MULTI_THREADED_NUM_THREADS} threads",
            )

        ax.set_yscale("log")
        ax.set_xticks(range(len(norb_range)), labels=norb_range)
        if ylim:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel("Time (s)")
    axes[-1].yaxis.set_label_position("right")
    axes[-1].set_ylabel(
        title,
        rotation=270,
        va="bottom",
    )


fig, axes = plt.subplots(
    1,
    2,
)
# fig.subplots_adjust(wspace=0.25)
norb_range = [50, 100, 200, 400]

benchmark_names = {
    "ffsim (real)": "slater.SampleSlaterBenchmark.time_sample_slater_real_ffsim",
    "dppy (GS)": "slater.SampleSlaterBenchmark.time_sample_slater_real_gs_dppy",
    "dppy (GS_bis)": "slater.SampleSlaterBenchmark.time_sample_slater_real_gs_bis_dppy",
}
title = "Real"
plot_results(
    axes,
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
)

for ax in axes:
    ax.set_xlabel("# orbitals")

axes[0].set_title("filling 1/2")
axes[1].set_title("filling 1/4")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
)

fig.subplots_adjust(bottom=0.21)

filepath = Path("plots/slater.pdf")
os.makedirs(filepath.parent, exist_ok=True)
plt.savefig(filepath, bbox_inches="tight")
print(f"Saved figure to {filepath}.")
