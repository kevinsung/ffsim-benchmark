# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os

import fqe
import numpy as np
from asv_runner.benchmarks.mark import skip_for_params
from fqe.algorithm.low_rank import double_factor_trotter_evolution
from fqe.algorithm.low_rank_api import LowRankTrotter
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

import ffsim

OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))


def simulate_trotter_double_factorized_fqe(
    vec_fqe,
    time: float,
    n_steps: int,
    basis_change_unitaries: np.ndarray,
    diag_coulomb_mats: np.ndarray,
):
    step_time = time / n_steps
    for _ in range(n_steps):
        vec_fqe = double_factor_trotter_evolution(
            vec_fqe,
            basis_change_unitaries,
            diag_coulomb_mats,
            step_time,
        )
    return vec_fqe


class TrotterBenchmark:
    """Benchmark Trotter."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12, 16),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        # set benchmark parameters
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)
        self.time = 1.0
        self.n_steps = 3
        rank = 3

        # initialize test objects
        rng = np.random.default_rng()
        self.vec = ffsim.hartree_fock_state(self.norb, self.nelec)
        one_body_tensor = ffsim.random.random_hermitian(self.norb, seed=rng)
        two_body_tensor = ffsim.random.random_two_body_tensor(
            self.norb, rank=rank, seed=rng, dtype=float
        )
        mol_hamiltonian = ffsim.MolecularHamiltonian(
            one_body_tensor=one_body_tensor, two_body_tensor=two_body_tensor
        )
        self.df_hamiltonian = (
            ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
                mol_hamiltonian, max_vecs=rank
            )
        )

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

        # prepare FQE
        n_alpha, n_beta = self.nelec
        self.vec_fqe = fqe.Wavefunction([[n_alpha + n_beta, n_alpha - n_beta, norb]])
        self.vec_fqe.set_wfn(strategy="hartree-fock")
        low_rank_trotter = LowRankTrotter(
            oei=one_body_tensor,
            tei=np.asarray(two_body_tensor.transpose(0, 2, 3, 1), order="C"),
        )
        (
            self.basis_change_unitaries,
            self.diag_coulomb_mats,
        ) = low_rank_trotter.prepare_trotter_sequence(self.time)
        fqe.settings.use_accelerated_code = True

        # prepare Qiskit
        if norb <= 12:
            self.aer_sim = AerSimulator(max_parallel_threads=OMP_NUM_THREADS)
            initial_state = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
                ffsim.hartree_fock_state(self.norb, self.nelec),
                norb=self.norb,
                nelec=self.nelec,
            )
            qubits = QuantumRegister(2 * norb)
            circuit = QuantumCircuit(qubits)
            circuit.set_statevector(initial_state)
            gate = ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(
                self.df_hamiltonian, time=self.time, n_steps=self.n_steps, order=0
            )
            circuit.append(gate, qubits)
            circuit.save_state()
            self.circuit = transpile(circuit, self.aer_sim)

    def time_simulate_trotter_double_factorized_ffsim(self, *_):
        ffsim.simulate_trotter_double_factorized(
            self.vec,
            self.df_hamiltonian,
            self.time,
            norb=self.norb,
            nelec=self.nelec,
            n_steps=self.n_steps,
            order=0,
            copy=False,
        )

    @skip_for_params([(16, 0.5)])
    def time_simulate_trotter_double_factorized_fqe(self, *_):
        simulate_trotter_double_factorized_fqe(
            self.vec_fqe,
            time=self.time,
            n_steps=self.n_steps,
            basis_change_unitaries=self.basis_change_unitaries,
            diag_coulomb_mats=self.diag_coulomb_mats,
        )

    @skip_for_params([(16, 0.25), (16, 0.5)])
    def time_simulate_trotter_double_factorized_qiskit(self, *_):
        self.aer_sim.run(self.circuit).result()
