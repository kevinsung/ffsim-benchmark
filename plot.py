import itertools
import json
import os

import matplotlib.pyplot as plt

ENV_SINGLE_THREADED = "2926a93da429bc9414eb12395eaa6358"
ENV_MULTI_THREADED = "4d56a017f9b481e7854105edb0db7ada"

with open(
    f".asv/results/li-7d1035cc-1f16-11b2-a85c-8530f9357908.ibm.com/7de6a34f-env-{ENV_SINGLE_THREADED}.json"
) as f:
    DATA_SINGLE_THREADED = json.load(f)
with open(
    f".asv/results/li-7d1035cc-1f16-11b2-a85c-8530f9357908.ibm.com/7de6a34f-env-{ENV_MULTI_THREADED}.json"
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


def plot_results(
    benchmark_names: dict[str, str],
    norb_range: list[int],
    filling_fraction: float,
    title: str,
    filename: str,
    plots_dir: str = "plots",
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
            zip(itertools.product(*these_results["params"]), these_results["result"])
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
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.25)
    for (label, times), fmt in zip(times_single_threaded.items(), fmts):
        ax1.plot(norb_range, times, fmt, label=label)
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
    plt.savefig(f"{plots_dir}/{filename}_filling-{1 / filling_denominator}.pdf")


# Orbital rotation, Givens
benchmark_names = {
    "Qiskit Aer": "gates.GatesBenchmark.time_apply_orbital_rotation_givens_qiskit",
    "FQE": "gates.GatesBenchmark.time_apply_orbital_rotation_givens_fqe",
    "ffsim": "gates.GatesBenchmark.time_apply_orbital_rotation_givens",
}
norb_range = [4, 8, 12, 16]
title = "Orbital rotation (Givens)"
filename = "orbital_rotation_givens"
for filling_fraction in [0.25, 0.5]:
    plot_results(
        benchmark_names=benchmark_names,
        norb_range=norb_range,
        filling_fraction=filling_fraction,
        title=title,
        filename=filename,
    )


# Quadratic Hamiltonian evolution
benchmark_names = {
    "Qiskit Aer": "gates.GatesBenchmark.time_quad_ham_evolution_qiskit",
    "FQE": "gates.GatesBenchmark.time_quad_ham_evolution_fqe",
    "ffsim": "gates.GatesBenchmark.time_quad_ham_evolution",
}
norb_range = [4, 8, 12, 16]
title = "Quadratic Hamiltonian"
filename = "quadratic_hamiltonian"
for filling_fraction in [0.25, 0.5]:
    plot_results(
        benchmark_names=benchmark_names,
        norb_range=norb_range,
        filling_fraction=filling_fraction,
        title=title,
        filename=filename,
    )

# Double-factorized Trotter simulation
benchmark_names = {
    "Qiskit Aer": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_qiskit",
    "FQE": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
    "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized",
}
norb_range = [4, 8, 12, 16]
title = "Double-factorized Trotter"
filename = "double_factorized_trotter"
for filling_fraction in [0.25, 0.5]:
    plot_results(
        benchmark_names=benchmark_names,
        norb_range=norb_range,
        filling_fraction=filling_fraction,
        title=title,
        filename=filename,
    )

# Givens vs LU orbital rotation
benchmark_names = {
    "Givens": "gates.GatesBenchmark.time_apply_orbital_rotation_givens",
    "LU": "gates.GatesBenchmark.time_apply_orbital_rotation_lu",
}
norb_range = [4, 8, 12, 16]
title = "Orbital rotation, Givens vs LU"
filename = "orbital_rotation_givens_vs_lu"
for filling_fraction in [0.25, 0.5]:
    plot_results(
        benchmark_names=benchmark_names,
        norb_range=norb_range,
        filling_fraction=filling_fraction,
        title=title,
        filename=filename,
    )


def plot_results_ratio(
    benchmark_names: dict[str, str],
    norb_range: list[int],
    filling_fraction: float,
    title: str,
    filename: str,
    plots_dir: str = "plots",
) -> None:
    benchmark_results_single_threaded = {}
    for label, benchmark_name in benchmark_names.items():
        these_results = dict(
            zip(
                DATA_SINGLE_THREADED["result_columns"],
                DATA_SINGLE_THREADED["results"][benchmark_name],
            )
        )
        benchmark_results_single_threaded[label] = dict(
            zip(itertools.product(*these_results["params"]), these_results["result"])
        )

    filling_denominator = int(1 / filling_fraction)

    times_single_threaded = {
        label: [times[(str(norb), str(filling_fraction))] for norb in norb_range]
        for label, times in benchmark_results_single_threaded.items()
    }

    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(wspace=0.25)
    labels = list(times_single_threaded.keys())
    label0 = labels[0]
    label1 = labels[1]
    times0 = times_single_threaded[label0]
    times1 = times_single_threaded[label1]
    ratios = [t0 / t1 for t0, t1 in zip(times0, times1)]
    ax.plot(norb_range, ratios, "o--")
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xticks(norb_range)
    ax.set_xlabel("Number of orbitals")
    ax.set_ylabel(f"Time {label0} / Time {label1}")
    fig.suptitle(f"{title}, filling 1/{filling_denominator}", size="x-large")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/{filename}_filling-{1 / filling_denominator}.pdf")


# Givens vs LU orbital rotation ratio
benchmark_names = {
    "Givens": "gates.GatesBenchmark.time_apply_orbital_rotation_givens",
    "LU": "gates.GatesBenchmark.time_apply_orbital_rotation_lu",
}
norb_range = [4, 8, 12, 16]
title = "Orbital rotation, Givens vs LU"
filename = "orbital_rotation_givens_vs_lu_ratio"
for filling_fraction in [0.25, 0.5]:
    plot_results_ratio(
        benchmark_names=benchmark_names,
        norb_range=norb_range,
        filling_fraction=filling_fraction,
        title=title,
        filename=filename,
    )
