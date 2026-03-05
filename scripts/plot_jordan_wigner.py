# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import glob
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_NAMES = {
    "ffsim": "jordan_wigner.JordanWignerBenchmark.time_jordan_wigner_ffsim",
    "openfermion": "jordan_wigner.JordanWignerBenchmark.time_jordan_wigner_openfermion",
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

DATA_SINGLE_THREADED = find_result_data(RESULTS_DIR, 1, DESIRED_BENCHMARKS)
DATA_MULTI_THREADED = find_result_data(RESULTS_DIR, 6, DESIRED_BENCHMARKS)
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


def load_results(data: dict, benchmark_names: dict[str, str]) -> dict[str, dict]:
    results = {}
    for label, benchmark_name in benchmark_names.items():
        these_results = dict(
            zip(
                data["result_columns"],
                data["results"][benchmark_name],
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


colors = {
    "ffsim": "#0f62fe",
    "openfermion": "#3ddbd9",
}
markers = {
    "ffsim": "o",
    "openfermion": "v",
}

capsize = 4
legend_fontsize = 12
tick_label_fontsize = 13
axis_label_fontsize = 14
title_fontsize = 15

n_terms_range = [100, 1_000, 10_000]

results_single_threaded = load_results(DATA_SINGLE_THREADED, BENCHMARK_NAMES)
results_multi_threaded = load_results(DATA_MULTI_THREADED, BENCHMARK_NAMES)

fig, ax = plt.subplots(figsize=(6, 4))

for label in BENCHMARK_NAMES:
    for times_by_key, alpha, suffix in [
        (results_single_threaded[label], 1.0, ""),
        (results_multi_threaded[label], 0.5, f", {MULTI_THREADED_NUM_THREADS} threads"),
    ]:
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
            alpha=alpha,
            label=f"{label}{suffix}",
            capsize=capsize,
        )

ax.set_yscale("log")
ax.set_xticks(range(len(n_terms_range)), labels=n_terms_range)
ax.tick_params(axis="both", labelsize=tick_label_fontsize)
ax.set_xlabel("# terms", fontsize=axis_label_fontsize)
ax.set_ylabel("Time (s)", fontsize=axis_label_fontsize)
ax.set_title("Jordan-Wigner transform", fontsize=title_fontsize)
ax.legend(fontsize=legend_fontsize)

filepath = Path("plots/jordan_wigner.pdf")
os.makedirs(filepath.parent, exist_ok=True)
plt.savefig(filepath, bbox_inches="tight")
print(f"Saved figure to {filepath}.")
