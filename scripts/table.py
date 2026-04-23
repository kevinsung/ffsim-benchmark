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

import numpy as np

BENCHMARKS = {
    "Quadratic Ham. evo.": {
        "ffsim": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim",
        "other": "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_fqe",
        "other_label": "FQE",
        "key": ("16", "0.5"),
        "has_mac_multi": True,
    },
    "Diagonal Coulomb evo.": {
        "ffsim": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_ffsim",
        "other": "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_fqe",
        "other_label": "FQE",
        "key": ("16", "0.5"),
        "has_mac_multi": True,
    },
    "Double factorized Trotter": {
        "ffsim": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim",
        "other": "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe",
        "other_label": "FQE",
        "key": ("16", "0.5"),
        "has_mac_multi": True,
    },
    "Mol. Ham. operator action": {
        "ffsim": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_ffsim",
        "other": "mol_ham_action.MolecularHamiltonianActionRealBenchmark.time_mol_ham_action_real_fqe_restricted",
        "other_label": "FQE",
        "key": ("16", "0.5"),
        "has_mac_multi": False,
    },
    "Sample Slater det.": {
        "ffsim": "slater.SampleSlaterBenchmarkReal.time_sample_slater_real_ffsim",
        "other": "slater.SampleSlaterBenchmarkReal.time_sample_slater_real_gs_dppy",
        "other_label": "DPPy",
        "key": ("1600", "0.25", "1000"),
        "has_mac_multi": False,
    },
    "Normal order": {
        "ffsim": "fermion_operator.FermionOperatorBenchmark.time_normal_order_ffsim",
        "other": "fermion_operator.FermionOperatorBenchmark.time_normal_order_openfermion",
        "other_label": "OpenFermion",
        "key": ("100000",),
        "has_mac_multi": False,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "linux_machine", help="Linux machine name (subdirectory of .asv/results/)"
)
parser.add_argument(
    "mac_machine", help="Mac machine name (subdirectory of .asv/results/)"
)
args = parser.parse_args()
LINUX_RESULTS_DIR = f".asv/results/{args.linux_machine}"
MAC_RESULTS_DIR = f".asv/results/{args.mac_machine}"


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


def get_time(data: dict | None, benchmark_name: str, key: tuple) -> float | None:
    if data is None or benchmark_name not in data.get("results", {}):
        return None
    res = dict(zip(data["result_columns"], data["results"][benchmark_name]))
    result_dict = dict(
        zip(
            itertools.product(*res["params"]),
            [np.nan if x is None else x for x in res["result"]],
        )
    )
    val = result_dict.get(key)
    if val is None or np.isnan(val):
        return None
    return float(val)


linux_1t = find_result_data(LINUX_RESULTS_DIR, 1)
linux_6t = find_result_data(LINUX_RESULTS_DIR, 6)
mac_1t = find_result_data(MAC_RESULTS_DIR, 1)
mac_6t = find_result_data(MAC_RESULTS_DIR, 6)


def fmt_time(t: float | None) -> str:
    if t is None:
        return "N/A"
    if t >= 1:
        return f"{t:.3f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.3f} ms"
    return f"{t * 1e6:.3f} µs"


def fmt_ratio(fqe: float | None, ffsim: float | None) -> str:
    if fqe is None or ffsim is None:
        return "N/A"
    return f"{fqe / ffsim:.2f}x"


benchmark_col_width = max(
    len(f"[Linux] {name} ({bench['other_label']})") for name, bench in BENCHMARKS.items()
)
col_width = 12
sep = "  "


def make_header_lines():
    group_names = ["1 CPU", "6 CPUs"]
    sub = ["ffsim", "other", "other/ffsim"]

    h1 = f"{'Benchmark':<{benchmark_col_width}}"
    h2 = f"{'':>{benchmark_col_width}}"
    for gname in group_names:
        group_str = f"{gname:^{3 * col_width + 2 * len(sep)}}"
        h1 += sep + group_str
        h2 += sep + sep.join(f"{s:>{col_width}}" for s in sub)

    return h1, h2


def make_row(label: str, cells: list[str]) -> str:
    row = f"{label:<{benchmark_col_width}}"
    for cell in cells:
        row += sep + f"{cell:>{col_width}}"
    return row


h1, h2 = make_header_lines()
divider = "-" * len(h2)

print(divider)
print(h1)
print(h2)
print(divider)

for i, (bench_name, bench) in enumerate(BENCHMARKS.items()):
    ffsim_bname = bench["ffsim"]
    other_bname = bench["other"]
    other_label = bench["other_label"]
    key = bench["key"]
    has_mac_multi = bench["has_mac_multi"]

    linux1_ffsim = get_time(linux_1t, ffsim_bname, key)
    linux1_other = get_time(linux_1t, other_bname, key)

    linux6_ffsim = get_time(linux_6t, ffsim_bname, key)
    linux6_other = get_time(linux_6t, other_bname, key)

    mac1_ffsim = get_time(mac_1t, ffsim_bname, key)
    mac1_other = get_time(mac_1t, other_bname, key)

    # Mac 6 CPU: ffsim from 6T if available; other always from 1T
    mac6_ffsim = get_time(mac_6t, ffsim_bname, key) if has_mac_multi else None
    mac6_other = get_time(mac_1t, other_bname, key)

    if i > 0:
        print()
    print(make_row(f"[Linux] {bench_name} ({other_label})", [
        fmt_time(linux1_ffsim), fmt_time(linux1_other), fmt_ratio(linux1_other, linux1_ffsim),
        fmt_time(linux6_ffsim), fmt_time(linux6_other), fmt_ratio(linux6_other, linux6_ffsim),
    ]))
    print(make_row(f"[Mac]   {bench_name} ({other_label})", [
        fmt_time(mac1_ffsim), fmt_time(mac1_other), fmt_ratio(mac1_other, mac1_ffsim),
        fmt_time(mac6_ffsim), fmt_time(mac6_other), fmt_ratio(mac6_other, mac6_ffsim),
    ]))

print()
print(divider)
print("† Mac 6 CPUs 'other' column uses 1-CPU data (no multi-threaded competitor available on Mac).")
