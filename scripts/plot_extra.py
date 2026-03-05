import glob
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_NAMES = {
    "ffsim": "slater.SampleSlaterBenchmarkReal.time_sample_slater_real_ffsim",
    "dppy": "slater.SampleSlaterBenchmarkReal.time_sample_slater_real_gs_dppy",
}
DESIRED_BENCHMARKS = set(BENCHMARK_NAMES.values())


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
    "dppy": "#be95ff",
    "ffsim": "#0f62fe",
}
fmts = {
    "dppy": "s:",
    "ffsim": "o--",
}


def plot_results(
    axes,
    benchmark_names: dict[str, str],
    norb_range: list[int],
    shots: int,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    benchmark_results = {}
    for label, benchmark_name in benchmark_names.items():
        these_results = dict(
            zip(
                DATA["result_columns"],
                DATA["results"][benchmark_name],
            )
        )
        benchmark_results[label] = dict(
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
        times_dict = {
            label: [
                times[(str(norb), str(filling_fraction), str(shots))]
                for norb in norb_range
            ]
            for label, times in benchmark_results.items()
        }

        for label, data in times_dict.items():
            times, stats_q_25, stats_q_75 = zip(*data)
            yerr_a = [t - x for t, x in zip(times, stats_q_25)]
            yerr_b = [x - t for t, x in zip(times, stats_q_75)]
            ax.errorbar(
                range(len(norb_range)),
                times,
                yerr=(yerr_a, yerr_b),
                fmt=fmts[label],
                color=colors[label],
                label=label,
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


fig, axes = plt.subplots(1, 2)
# fig.subplots_adjust(wspace=0.25)
norb_range = [50, 100, 200, 400]
shots = 1_000

title = "Sample Slater"
plot_results(
    axes,
    benchmark_names=BENCHMARK_NAMES,
    norb_range=norb_range,
    shots=shots,
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
    ncol=2,
)

fig.subplots_adjust(bottom=0.21)

filepath = Path("plots/extra.pdf")
os.makedirs(filepath.parent, exist_ok=True)
plt.savefig(filepath, bbox_inches="tight")
print(f"Saved figure to {filepath}.")
