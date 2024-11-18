import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ENV_SINGLE_THREADED = "e98c72a0-virtualenv-py3.12-fqe-openfermion-qiskit-aer-MKL_NUM_THREADS1-OMP_NUM_THREADS1-OPENBLAS_NUM_THREADS1-RAYON_NUM_THREADS1"
ENV_MULTI_THREADED = "e98c72a0-virtualenv-py3.12-fqe-openfermion-qiskit-aer-MKL_NUM_THREADS6-OMP_NUM_THREADS6-OPENBLAS_NUM_THREADS6-RAYON_NUM_THREADS6"

with open(
    f".asv/results/li-7d1035cc-1f16-11b2-a85c-8530f9357908.ibm.com/{ENV_SINGLE_THREADED}.json"
) as f:
    DATA_SINGLE_THREADED = json.load(f)
with open(
    f".asv/results/li-7d1035cc-1f16-11b2-a85c-8530f9357908.ibm.com/{ENV_MULTI_THREADED}.json"
) as f:
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


def plot_results_single(
    benchmark_names: dict[str, str],
    norb_range: list[int],
    title: str,
    filename: str,
    plots_dir: str = "plots",
) -> None:
    fig, axes = plt.subplots(1, 2, layout="constrained")
    # fig.subplots_adjust(wspace=0.25)

    benchmark_results_single_threaded = {}
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

    for filling_fraction, ax in zip([0.5, 0.25], axes):
        filling_denominator = int(1 / filling_fraction)

        times_single_threaded = {
            label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
            for label, times in benchmark_results_single_threaded.items()
        }

        fmts = ["o--", "s--", "v--"]

        for (label, data), fmt in zip(times_single_threaded.items(), fmts):
            times, stats_q_25, stats_q_75 = zip(*data)
            yerr_a = [t - x for t, x in zip(times, stats_q_25)]
            yerr_b = [x - t for t, x in zip(times, stats_q_75)]
            ax.errorbar(norb_range, times, yerr=(yerr_a, yerr_b), fmt=fmt, label=label)
        ax.set_xticks(norb_range)
        ax.set_yscale("log")
        ax.set_xlabel("Number of orbitals")
        ax.set_title(f"filling 1/{filling_denominator}")
    axes[0].set_ylabel("Time (s)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"{title}", size="x-large")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/{filename}_single.pdf")


def plot_results_multi(
    benchmark_names: dict[str, str],
    norb_range: list[int],
    filling_fraction: float,
    title: str,
    filename: str,
    plots_dir: str = "plots",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")
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
                    these_results["result"],
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
            zip(itertools.product(*these_results["params"]), these_results["result"])
        )

    filling_denominator = int(1 / filling_fraction)

    times_single_threaded = {
        label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
        for label, times in benchmark_results_single_threaded.items()
    }
    times_multi_threaded = {
        label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
        for label, times in benchmark_results_multi_threaded.items()
    }

    fmts = ["o--", "s--", "v--"]

    for (label, data), fmt in zip(times_single_threaded.items(), fmts):
        times, stats_q_25, stats_q_75 = zip(*data)
        yerr_a = [t - x for t, x in zip(times, stats_q_25)]
        yerr_b = [x - t for t, x in zip(times, stats_q_75)]
        ax1.errorbar(norb_range, times, yerr=(yerr_a, yerr_b), fmt=fmt, label=label)
    ax1.set_xticks(norb_range)
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of orbitals")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("single-threaded")
    ax1.legend(loc="upper left")
    for (label, times), fmt in zip(times_multi_threaded.items(), fmts):
        ax2.plot(norb_range, times, fmt, label=label)
    ax2.set_xticks(norb_range)
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of orbitals")
    ax2.set_title(f"{MULTI_THREADED_NUM_THREADS} threads")
    fig.suptitle(f"{title}, filling 1/{filling_denominator}", size="x-large")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/{filename}_multi.pdf")


benchmark_names = {
    "Qiskit Aer": "orbital_rotation.OrbitalRotationBenchmark.time_apply_orbital_rotation_qiskit",
    "FQE": "orbital_rotation.OrbitalRotationBenchmark.time_apply_orbital_rotation_fqe",
    "ffsim": "orbital_rotation.OrbitalRotationBenchmark.time_apply_orbital_rotation_ffsim",
}
norb_range = [4, 8, 12, 16]
title = "Orbital rotation"
filename = "orbital_rotation_givens"
plot_results_single(
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    filename=filename,
)

benchmark_names = {
    "Qiskit Aer": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_qiskit",
    "FQE": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
    "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim",
}
norb_range = [4, 8, 12, 16]
title = "Double-factorized Trotter"
filename = "df_trotter"
plot_results_single(
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    filename=filename,
)

benchmark_names = {
    "Qiskit Aer": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_qiskit",
    "FQE": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_fqe",
    "ffsim": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim",
}
norb_range = [4, 8, 12, 16]
title = "Quadratic Hamiltonian evolution"
filename = "quad_ham_evo"
plot_results_single(
    benchmark_names=benchmark_names,
    norb_range=norb_range,
    title=title,
    filename=filename,
)

plots_dir = "plots"

# operator action
filename = "operator_action"
benchmark_names = {
    "FQE": "operator_action.OperatorActionBenchmark.time_operator_action_fqe",
    "ffsim": "operator_action.OperatorActionBenchmark.time_operator_action_ffsim",
}
norb_range = [4, 8, 12, 16]

fig, ax = plt.subplots(1, 1, layout="constrained")

benchmark_results_single_threaded = {}
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

filling_fraction = 0.25
filling_denominator = int(1 / filling_fraction)

times_single_threaded = {
    label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
    for label, times in benchmark_results_single_threaded.items()
}

fmts = ["o--", "s--", "v--"]

for (label, data), fmt in zip(times_single_threaded.items(), fmts):
    times, stats_q_25, stats_q_75 = zip(*data)
    yerr_a = [t - x for t, x in zip(times, stats_q_25)]
    yerr_b = [x - t for t, x in zip(times, stats_q_75)]
    ax.errorbar(norb_range, times, yerr=(yerr_a, yerr_b), fmt=fmt, label=label)
ax.set_xticks(norb_range)
ax.set_yscale("log")
ax.set_xlabel("Number of orbitals")
ax.set_ylabel("Time (s)")
ax.set_title("Operator action")
ax.legend(loc="upper left")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(f"{plots_dir}/{filename}.pdf")


# normal ordering
filename = "normal_ordering"
benchmark_names = {
    "OpenFermion": "fermion_operator.FermionOperatorBenchmark.time_normal_order_openfermion",
    "ffsim": "fermion_operator.FermionOperatorBenchmark.time_normal_order_ffsim",
}
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
                these_results["result"],
                [np.nan if x is None else x for x in these_results["stats_q_25"]],
                [np.nan if x is None else x for x in these_results["stats_q_75"]],
            ),
        )
    )

n_terms_range = [100, 1_000, 10_000, 100_000]

times_single_threaded = {
    label: [times[(str(n_terms),)] for n_terms in n_terms_range]
    for label, times in benchmark_results_single_threaded.items()
}

fmts = ["o--", "s--", "v--"]
fig, ax = plt.subplots(1, 1, layout="constrained")
# fig.subplots_adjust(wspace=0.25)
for (label, data), fmt in zip(times_single_threaded.items(), fmts):
    times, stats_q_25, stats_q_75 = zip(*data)
    yerr_a = [t - x for t, x in zip(times, stats_q_25)]
    yerr_b = [x - t for t, x in zip(times, stats_q_75)]
    ax.errorbar(n_terms_range, times, yerr=(yerr_a, yerr_b), fmt=fmt, label=label)
ax.set_xticks(n_terms_range)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of terms")
ax.set_ylabel("Time (s)")
ax.set_title("Normal ordering, 100 orbitals")
ax.legend(loc="upper left")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(f"{plots_dir}/{filename}.pdf")
