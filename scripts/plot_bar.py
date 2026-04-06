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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_PANELS = {
    "Quadratic Hamiltonian evolution": {
        "ffsim": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim",
        "FQE": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_fqe",
    },
    "Diagonal Coulomb evolution": {
        "ffsim": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_ffsim",
        "FQE": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_fqe",
    },
    "Double factorized Trotter": {
        "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim",
        "FQE": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
    },
    "Molecular Hamiltonian operator action": {
        "ffsim": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_ffsim",
        "FQE": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_fqe_restricted",
    },
}

# Mol Ham action has no Mac multi-threaded benchmarks
PANELS_WITH_MAC_MULTI = [True, True, True, False]

NORB = 16
FILLING_FRACTION = 0.5

LINUX_RESULTS_DIR = ".asv/results/kjs-thinkpad-x13"
MAC_RESULTS_DIR = ".asv/results/MacBookPro"


def find_result_data(results_dir: str, num_threads: int) -> dict | None:
    candidates = []
    for path in glob.glob(f"{results_dir}/*.json"):
        if path.endswith("machine.json"):
            continue
        with open(path) as f:
            data = json.load(f)
        if int(data.get("env_vars", {}).get("OMP_NUM_THREADS", -1)) == num_threads:
            candidates.append((os.path.getmtime(path), data))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def get_time(
    data: dict | None, benchmark_name: str
) -> tuple[float, float, float] | None:
    if data is None or benchmark_name not in data.get("results", {}):
        return None
    res = dict(zip(data["result_columns"], data["results"][benchmark_name]))
    key = (str(NORB), str(FILLING_FRACTION))
    result_dict = dict(
        zip(
            itertools.product(*res["params"]),
            zip(
                [np.nan if x is None else x for x in res["result"]],
                [np.nan if x is None else x for x in res["stats_q_25"]],
                [np.nan if x is None else x for x in res["stats_q_75"]],
            ),
        )
    )
    entry = result_dict.get(key)
    if entry is None or np.isnan(entry[0]):
        return None
    return entry


# Load data for both machines, both thread counts
linux_1t = find_result_data(LINUX_RESULTS_DIR, 1)
linux_6t = find_result_data(LINUX_RESULTS_DIR, 6)
mac_1t = find_result_data(MAC_RESULTS_DIR, 1)
mac_6t = find_result_data(MAC_RESULTS_DIR, 6)

for name, data in [
    ("Linux 1T", linux_1t),
    ("Linux 6T", linux_6t),
    ("Mac 1T", mac_1t),
    ("Mac 6T", mac_6t),
]:
    if data:
        print(
            f"{name}: commit {data['commit_hash'][:8]},"
            f" date {datetime.fromtimestamp(data['date'] / 1000)}"
        )

# Bar layout constants
BAR_WIDTH = 0.35
GROUP_GAP = 0.3

g_linux_1t = 0.0
g_linux_6t = g_linux_1t + 2 * BAR_WIDTH + GROUP_GAP
g_mac_1t = g_linux_6t + 2 * BAR_WIDTH + GROUP_GAP
g_mac_6t = g_mac_1t + 2 * BAR_WIDTH + GROUP_GAP

FFSIM_COLOR = "#0f62fe"
FQE_COLOR = "#8a3ffc"
ALPHA_6T = 0.5
EDGECOLOR = "black"

capsize = 4
legend_fontsize = 11
tick_label_fontsize = 11
axis_label_fontsize = 12
title_fontsize = 13

LINUX_HATCH = None
MAC_HATCH = "//"


def draw_bar(ax, x_pos, times, color, alpha=1.0, hatch=None):
    if times is None:
        return
    t, q25, q75 = times
    yerr_low = max(0.0, t - q25)
    yerr_high = max(0.0, q75 - t)
    ax.bar(
        x_pos,
        t,
        BAR_WIDTH,
        color=color,
        edgecolor=EDGECOLOR,
        linewidth=0,
        alpha=alpha,
        hatch=hatch,
    )
    ax.errorbar(
        x_pos,
        t,
        yerr=[[yerr_low], [yerr_high]],
        fmt="none",
        color="black",
        capsize=capsize,
        linewidth=1.5,
    )


def plot_bar_panel(ax, benchmark_names: dict[str, str], has_mac_multi: bool) -> None:
    ffsim_bname = benchmark_names["ffsim"]
    fqe_bname = benchmark_names["FQE"]

    # Linux 1T
    draw_bar(ax, g_linux_1t, get_time(linux_1t, fqe_bname), FQE_COLOR)
    draw_bar(ax, g_linux_1t + BAR_WIDTH, get_time(linux_1t, ffsim_bname), FFSIM_COLOR)

    # Linux 6T
    draw_bar(ax, g_linux_6t, get_time(linux_6t, fqe_bname), FQE_COLOR, alpha=ALPHA_6T)
    draw_bar(
        ax,
        g_linux_6t + BAR_WIDTH,
        get_time(linux_6t, ffsim_bname),
        FFSIM_COLOR,
        alpha=ALPHA_6T,
    )

    # Mac 1T
    draw_bar(ax, g_mac_1t, get_time(mac_1t, fqe_bname), FQE_COLOR, hatch=MAC_HATCH)
    draw_bar(
        ax,
        g_mac_1t + BAR_WIDTH,
        get_time(mac_1t, ffsim_bname),
        FFSIM_COLOR,
        hatch=MAC_HATCH,
    )

    # Mac 6T (ffsim only, and only for panels where it exists)
    if has_mac_multi:
        draw_bar(
            ax,
            g_mac_6t,
            get_time(mac_6t, ffsim_bname),
            FFSIM_COLOR,
            alpha=ALPHA_6T,
            hatch=MAC_HATCH,
        )

    # ax.set_yscale("log")
    ax.set_ylabel("Time (s)", fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)

    group_centers = [
        g_linux_1t + BAR_WIDTH / 2,
        g_linux_6t + BAR_WIDTH / 2,
        g_mac_1t + BAR_WIDTH / 2,
    ]
    group_labels = ["Linux\n1 CPU", "Linux\n6 CPUs", "Mac\n1 CPU"]

    if has_mac_multi:
        group_centers.append(g_mac_6t)
        group_labels.append("Mac\n6 CPUs")
        x_right = g_mac_6t + BAR_WIDTH
    else:
        x_right = g_mac_1t + 2 * BAR_WIDTH

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=tick_label_fontsize)
    ax.set_xlim(-BAR_WIDTH, x_right + BAR_WIDTH * 0.5)


# Build legend handles
legend_handles = [
    mpatches.Patch(
        facecolor=FQE_COLOR, edgecolor=EDGECOLOR, linewidth=0, label="FQE Linux 1 CPU"
    ),
    mpatches.Patch(
        facecolor=FFSIM_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        label="ffsim Linux 1 CPU",
    ),
    mpatches.Patch(
        facecolor=FQE_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        alpha=ALPHA_6T,
        label="FQE Linux 6 CPUs",
    ),
    mpatches.Patch(
        facecolor=FFSIM_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        alpha=ALPHA_6T,
        label="ffsim Linux 6 CPUs",
    ),
    mpatches.Patch(
        facecolor=FQE_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        hatch=MAC_HATCH,
        label="FQE Mac 1 CPU",
    ),
    mpatches.Patch(
        facecolor=FFSIM_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        hatch=MAC_HATCH,
        label="ffsim Mac 1 CPU",
    ),
    mpatches.Patch(fill=False, edgecolor="none", linewidth=0, label=""),
    mpatches.Patch(
        facecolor=FFSIM_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        alpha=ALPHA_6T,
        hatch=MAC_HATCH,
        label="ffsim Mac 6 CPUs",
    ),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (title, bnames), has_mac_multi in zip(
    axes.flat, BENCHMARK_PANELS.items(), PANELS_WITH_MAC_MULTI
):
    plot_bar_panel(ax, bnames, has_mac_multi)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=legend_fontsize,
)
fig.subplots_adjust(bottom=0.15, hspace=0.42, wspace=0.25)

filepath = Path("plots/bar.pdf")
os.makedirs(filepath.parent, exist_ok=True)
plt.savefig(filepath, bbox_inches="tight")
print(f"Saved figure to {filepath}.")
