import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ENV_SINGLE_THREADED = "724a6e0f-env-63d0c0e50daf914074f8c79983f4fd29"
ENV_MULTI_THREADED = "724a6e0f-env-17fe83cba2670b64c2ccd91d3215b5b7"

with open(f".asv/results/drybiscuit/{ENV_SINGLE_THREADED}.json") as f:
    DATA_SINGLE_THREADED = json.load(f)
with open(f".asv/results/drybiscuit/{ENV_MULTI_THREADED}.json") as f:
    DATA_MULTI_THREADED = json.load(f)

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
    "Aer": "v:",
    "FQE": "s-.",
    "ffsim": "o--",
}


def plot_results(
    benchmark_names: dict[str, str],
    norb_range: list[int],
    title: str,
    filename: str,
    plots_dir: str = "plots",
) -> None:
    fig, axes = plt.subplots(1, 2, layout="constrained")
    # fig.subplots_adjust(wspace=0.25)

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
        filling_denominator = int(1 / filling_fraction)

        times_single_threaded = {
            label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
            for label, times in benchmark_results_single_threaded.items()
        }
        times_multi_threaded = {
            label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
            for label, times in benchmark_results_multi_threaded.items()
        }

        for ((label_single, data_single), (label_multi, data_multi)), fmt in zip(
            zip(times_single_threaded.items(), times_multi_threaded.items()), fmts
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

        ax.set_xticks(norb_range)
        ax.set_yscale("log")
        ax.set_xlabel("Number of orbitals")
        ax.set_title(f"filling 1/{filling_denominator}")
    axes[0].set_ylabel("Time (s)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"{title}", size="x-large")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/{filename}.pdf")


benchmark_names = {
    "Aer": "orbital_rotation.OrbitalRotationBenchmark.time_apply_orbital_rotation_qiskit",
    "FQE": "orbital_rotation.OrbitalRotationBenchmark.time_apply_orbital_rotation_fqe",
    "ffsim": "orbital_rotation.OrbitalRotationBenchmark.time_apply_orbital_rotation_ffsim",
}
norb_range = [4, 8, 12, 16]
title = "Orbital rotation"
filename = "orbital_rotation_givens"
plot_results(
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    filename=filename,
)


benchmark_names = {
    "Aer": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_qiskit",
    "FQE": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
    "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim",
}
norb_range = [4, 8, 12, 16]
title = "Double factorized Trotter"
filename = "df_trotter"
plot_results(
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    filename=filename,
)

benchmark_names = {
    "Aer": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_qiskit",
    "FQE": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_fqe",
    "ffsim": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim",
}
norb_range = [4, 8, 12, 16]
title = "Quadratic Hamiltonian evolution"
filename = "quad_ham_evo"
plot_results(
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    filename=filename,
)

benchmark_names = {
    "FQE": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_fqe_restricted",
    "ffsim": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_ffsim",
}
norb_range = [4, 8, 12, 16]
title = "Molecular Hamiltonian operator action"
filename = "mol_ham_action"
plot_results(
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    filename=filename,
)
