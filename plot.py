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

    labels = ["Qiskit Aer", "FQE", "ffsim"]
    fmts = ["o--", "s--", "v--"]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.25)
    for label, fmt in zip(labels, fmts):
        times = times_single_threaded[label]
        ax1.plot(norb_range[: len(times)], times, fmt, label=label)
    ax1.set_xticks(norb_range)
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of orbitals")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("single-threaded")
    ax1.legend()
    for label, fmt in zip(labels, fmts):
        times = times_multi_threaded[label]
        ax2.plot(norb_range[: len(times)], times, fmt, label=label)
    ax2.set_xticks(norb_range)
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of orbitals")
    ax2.set_title(f"{MULTI_THREADED_NUM_THREADS} threads")
    fig.suptitle(f"{title}, filling 1/{filling_denominator}", size="x-large")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/{filename}_filling-{1 / filling_denominator}.pdf")


# Orbital rotation, Givens
benchmark_names = {
    "ffsim": "gates.GatesBenchmark.time_apply_orbital_rotation_givens",
    "FQE": "gates.GatesBenchmark.time_apply_orbital_rotation_givens_fqe",
    "Qiskit Aer": "gates.GatesBenchmark.time_apply_orbital_rotation_givens_qiskit",
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
    "ffsim": "gates.GatesBenchmark.time_quad_ham_evolution",
    "FQE": "gates.GatesBenchmark.time_quad_ham_evolution_fqe",
    "Qiskit Aer": "gates.GatesBenchmark.time_quad_ham_evolution_qiskit",
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
    "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized",
    "FQE": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
    "Qiskit Aer": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_qiskit",
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
