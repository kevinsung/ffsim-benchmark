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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ─── Sim benchmark names ────────────────────────────────────────────────────────
BENCHMARK_NAMES_QUAD_HAM = {
    "Aer": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_qiskit",
    "FQE": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_fqe",
    "ffsim": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim",
}
BENCHMARK_NAMES_DIAG_COULOMB = {
    "Aer": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_qiskit",
    "FQE": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_fqe",
    "ffsim": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_ffsim",
}
BENCHMARK_NAMES_TROTTER = {
    "Aer": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_qiskit",
    "FQE": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
    "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim",
}
BENCHMARK_NAMES_OP_ACTION = {
    "FQE": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_fqe_restricted",
    "ffsim": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_ffsim",
}
DESIRED_BENCHMARKS = (
    set(BENCHMARK_NAMES_QUAD_HAM.values())
    | set(BENCHMARK_NAMES_DIAG_COULOMB.values())
    | set(BENCHMARK_NAMES_TROTTER.values())
    | set(BENCHMARK_NAMES_OP_ACTION.values())
)

# ─── Bar benchmark names ────────────────────────────────────────────────────────
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
PANELS_WITH_MAC_MULTI = [True, True, True, False]

NORB = 16
FILLING_FRACTION = 0.5
parser = argparse.ArgumentParser()
parser.add_argument(
    "linux_machine",
    help="Linux machine name (subdirectory of .asv/results/) for bar plots",
)
parser.add_argument(
    "mac_machine", help="Mac machine name (subdirectory of .asv/results/) for bar plots"
)
args = parser.parse_args()
LINUX_RESULTS_DIR = f".asv/results/{args.linux_machine}"
MAC_RESULTS_DIR = f".asv/results/{args.mac_machine}"


def find_result_data_sim(
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


def find_result_data_bar(results_dir: str, num_threads: int) -> dict | None:
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


RESULTS_DIR = f".asv/results/{args.linux_machine}"

data_sim = find_result_data_sim(RESULTS_DIR, 1, DESIRED_BENCHMARKS)
print(f"Sim data commit: {data_sim['commit_hash'][:8]}")
print(f"Sim data date: {datetime.fromtimestamp(data_sim['date'] / 1000)}")

assert (
    int(data_sim["env_vars"]["OMP_NUM_THREADS"])
    == int(data_sim["env_vars"]["RAYON_NUM_THREADS"])
    == 1
)

linux_1t = find_result_data_bar(LINUX_RESULTS_DIR, 1)
linux_6t = find_result_data_bar(LINUX_RESULTS_DIR, 6)
mac_1t = find_result_data_bar(MAC_RESULTS_DIR, 1)
mac_6t = find_result_data_bar(MAC_RESULTS_DIR, 6)

for name, d in [
    ("Linux 1T", linux_1t),
    ("Linux 6T", linux_6t),
    ("Mac 1T", mac_1t),
    ("Mac 6T", mac_6t),
]:
    if d:
        print(
            f"{name}: commit {d['commit_hash'][:8]}, date {datetime.fromtimestamp(d['date'] / 1000)}"
        )

# ─── Style constants ────────────────────────────────────────────────────────────
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

BAR_WIDTH = 0.35
GROUP_GAP = 0.3
g_linux_1t = 0.0
g_linux_6t = g_linux_1t + 2 * BAR_WIDTH + GROUP_GAP
g_mac_1t = g_linux_6t + 2 * BAR_WIDTH + GROUP_GAP
g_mac_6t = g_mac_1t + 2 * BAR_WIDTH + GROUP_GAP

FFSIM_COLOR = "#0f62fe"
FQE_COLOR = "#be95ff"
ALPHA_6T = 0.5
EDGECOLOR = "black"
FFSIM_HATCH = "//"
FQE_HATCH = None

capsize = 4
legend_fontsize = 14
tick_label_fontsize = 14
axis_label_fontsize = 16
title_fontsize = 18


def plot_lines(
    axes,
    data: dict,
    benchmark_names: dict[str, str],
    norb_range: list[int],
    ylim: tuple[float, float] | None = None,
) -> None:
    benchmark_results = {}
    for label, benchmark_name in benchmark_names.items():
        these_results = dict(
            zip(
                data["result_columns"],
                data["results"][benchmark_name],
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
        times = {
            label: [results[(str(norb), str(filling_fraction))] for norb in norb_range]
            for label, results in benchmark_results.items()
        }
        for label, data_pts in times.items():
            t, stats_q_25, stats_q_75 = zip(*data_pts)
            yerr_a = [max(0, ti - x) for ti, x in zip(t, stats_q_25)]
            yerr_b = [max(0, x - ti) for ti, x in zip(t, stats_q_75)]
            ax.errorbar(
                norb_range,
                t,
                yerr=(yerr_a, yerr_b),
                fmt=fmts[label],
                color=colors[label],
                label=label,
                capsize=capsize,
            )
        ax.set_yscale("log")
        ax.set_xticks(norb_range)
        if ylim:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel("Time (s)", fontsize=axis_label_fontsize)


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

    draw_bar(ax, g_linux_1t, get_time(linux_1t, fqe_bname), FQE_COLOR, hatch=FQE_HATCH)
    draw_bar(
        ax,
        g_linux_1t + BAR_WIDTH,
        get_time(linux_1t, ffsim_bname),
        FFSIM_COLOR,
        hatch=FFSIM_HATCH,
    )
    draw_bar(
        ax,
        g_linux_6t,
        get_time(linux_6t, fqe_bname),
        FQE_COLOR,
        alpha=ALPHA_6T,
        hatch=FQE_HATCH,
    )
    draw_bar(
        ax,
        g_linux_6t + BAR_WIDTH,
        get_time(linux_6t, ffsim_bname),
        FFSIM_COLOR,
        alpha=ALPHA_6T,
        hatch=FFSIM_HATCH,
    )
    draw_bar(ax, g_mac_1t, get_time(mac_1t, fqe_bname), FQE_COLOR, hatch=FQE_HATCH)
    draw_bar(
        ax,
        g_mac_1t + BAR_WIDTH,
        get_time(mac_1t, ffsim_bname),
        FFSIM_COLOR,
        hatch=FFSIM_HATCH,
    )
    if has_mac_multi:
        draw_bar(
            ax,
            g_mac_6t,
            get_time(mac_6t, ffsim_bname),
            FFSIM_COLOR,
            alpha=ALPHA_6T,
            hatch=FFSIM_HATCH,
        )

    ax.set_ylabel("Time (s)", fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)

    group_centers = [
        g_linux_1t + BAR_WIDTH / 2,
        g_linux_6t + BAR_WIDTH / 2,
        g_mac_1t + BAR_WIDTH / 2,
        g_mac_6t,
    ]
    group_labels = ["Linux\n1 CPU", "Linux\n6 CPUs", "Mac\n1 CPU", "Mac\n6 CPUs"]
    x_right = g_mac_6t + BAR_WIDTH
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=tick_label_fontsize)
    ax.set_xlim(-BAR_WIDTH, x_right + BAR_WIDTH * 0.5)
    if not has_mac_multi:
        ax.text(
            g_mac_6t,
            0.25,
            "N/A",
            ha="center",
            va="center",
            fontsize=tick_label_fontsize,
            color="gray",
            transform=ax.get_xaxis_transform(),
        )


# ─── Build figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
outer_gs = fig.add_gridspec(4, 3, hspace=0.1, wspace=0.35, width_ratios=[0.2, 1, 1])

# Middle column: sim panels (each split into two sub-axes for filling fractions)
sim_axes_groups = []
for row in range(4):
    inner_gs = outer_gs[row, 1].subgridspec(1, 2, wspace=0.07)
    if row == 0:
        ax_left = fig.add_subplot(inner_gs[0, 0])
        ax_right = fig.add_subplot(inner_gs[0, 1], sharey=ax_left)
    else:
        ax_left = fig.add_subplot(inner_gs[0, 0], sharex=sim_axes_groups[0][0])
        ax_right = fig.add_subplot(
            inner_gs[0, 1], sharey=ax_left, sharex=sim_axes_groups[0][1]
        )
    ax_right.tick_params(axis="y", labelleft=False)
    sim_axes_groups.append([ax_left, ax_right])

for axes in sim_axes_groups:
    for ax in axes:
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

# Right column: bar panels
bar_axes = [fig.add_subplot(outer_gs[row, 2]) for row in range(4)]

# Populate sim panels
norb_range = [4, 8, 12, 16]
sim_configs = [
    (BENCHMARK_NAMES_QUAD_HAM, norb_range, None),
    (BENCHMARK_NAMES_DIAG_COULOMB, norb_range, None),
    (BENCHMARK_NAMES_TROTTER, norb_range, None),
    (BENCHMARK_NAMES_OP_ACTION, norb_range, None),
]
for axes, (bnames, norb_r, ylim) in zip(sim_axes_groups, sim_configs):
    plot_lines(axes, data_sim, bnames, norb_r, ylim)

# Populate bar panels
for ax, (title, bnames), has_mac_multi in zip(
    bar_axes, BENCHMARK_PANELS.items(), PANELS_WITH_MAC_MULTI
):
    plot_bar_panel(ax, bnames, has_mac_multi)

# Shared x axes: hide labels on non-bottom rows, label only the bottom
for sim_axes in sim_axes_groups[:-1]:
    for ax in sim_axes:
        ax.tick_params(axis="x", labelbottom=False)
for ax in sim_axes_groups[-1]:
    ax.set_xlabel("# orbitals", fontsize=axis_label_fontsize)

for ax in bar_axes[:-1]:
    ax.tick_params(axis="x", labelbottom=False)

# Titles
group_titles = [
    "Quadratic\nHamiltonian\nevolution",
    "Diagonal\nCoulomb\nevolution",
    "Double\nfactorized\nTrotter",
    "Molecular\nHamiltonian\noperator\naction",
]
for row, (title, sim_axes, bar_ax) in enumerate(
    zip(group_titles, sim_axes_groups, bar_axes)
):
    if row == 0:
        sim_axes[0].set_title("1/2 filling", fontsize=title_fontsize)
        sim_axes[1].set_title("1/4 filling", fontsize=title_fontsize)
    ax_label = fig.add_subplot(outer_gs[row, 0])
    ax_label.set_axis_off()
    ax_label.text(
        0.0,
        0.5,
        title,
        fontsize=title_fontsize,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax_label.transAxes,
        wrap=True,
    )

# Legends
handles_sim, labels_sim = sim_axes_groups[0][0].get_legend_handles_labels()
bar_legend_handles = [
    mpatches.Patch(
        facecolor=FQE_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        hatch=FQE_HATCH,
        label="FQE Linux 1 CPU",
    ),
    mpatches.Patch(
        facecolor=FQE_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        alpha=ALPHA_6T,
        hatch=FQE_HATCH,
        label="FQE Linux 6 CPUs",
    ),
    mpatches.Patch(
        facecolor=FQE_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        hatch=FQE_HATCH,
        label="FQE Mac 1 CPU",
    ),
    mpatches.Patch(fill=False, edgecolor="none", linewidth=0, label=""),
    mpatches.Patch(
        facecolor=FFSIM_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        hatch=FFSIM_HATCH,
        label="ffsim Linux 1 CPU",
    ),
    mpatches.Patch(
        facecolor=FFSIM_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        alpha=ALPHA_6T,
        hatch=FFSIM_HATCH,
        label="ffsim Linux 6 CPUs",
    ),
    mpatches.Patch(
        facecolor=FFSIM_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        hatch=FFSIM_HATCH,
        label="ffsim Mac 1 CPU",
    ),
    mpatches.Patch(
        facecolor=FFSIM_COLOR,
        edgecolor=EDGECOLOR,
        linewidth=0,
        alpha=ALPHA_6T,
        hatch=FFSIM_HATCH,
        label="ffsim Mac 6 CPUs",
    ),
]

fig.canvas.draw()

left_pos = sim_axes_groups[-1][0].get_position()
right_pos = sim_axes_groups[-1][1].get_position()
sim_cx = (left_pos.x0 + right_pos.x1) / 2
sim_bot = min(left_pos.y0, right_pos.y0)

bar_pos = bar_axes[-1].get_position()
bar_cx = (bar_pos.x0 + bar_pos.x1) / 2
bar_bot = bar_pos.y0

fig.legend(
    handles_sim,
    labels_sim,
    loc="upper center",
    bbox_to_anchor=(sim_cx, sim_bot - 0.06),
    bbox_transform=fig.transFigure,
    ncol=3,
    fontsize=legend_fontsize,
)
fig.legend(
    handles=bar_legend_handles,
    loc="upper center",
    bbox_to_anchor=(bar_cx, bar_bot - 0.06),
    bbox_transform=fig.transFigure,
    ncol=2,
    fontsize=legend_fontsize,
)

filepath = Path("plots/sim.pdf")
os.makedirs(filepath.parent, exist_ok=True)
plt.savefig(filepath, bbox_inches="tight")
print(f"Saved figure to {filepath}.")
