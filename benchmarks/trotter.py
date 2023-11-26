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
from qiskit_sim.trotter import AsymmetricLowRankTrotterStepJW, simulate_trotter

import ffsim

OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))


class TrotterBenchmark:
    """Benchmark Trotter."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12, 16),
        (0.25,),
    ]

    def setup(self, norb: int, filling_fraction: float):
        # benchmark parameters
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)
        self.time = 1.0
        self.n_steps = 3
        rank = 3

        # initialize random objects
        rng = np.random.default_rng()
        self.vec = ffsim.hartree_fock_state(self.norb, self.nelec)
        one_body_tensor = ffsim.random.random_hermitian(self.norb, seed=rng)
        two_body_tensor = ffsim.random.random_two_body_tensor_real(
            self.norb, rank=rank, seed=rng
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
        initial_state = ffsim.random.random_statevector(2 ** (2 * norb), dtype=complex)
        qubits = QuantumRegister(2 * norb)
        trotter_step = AsymmetricLowRankTrotterStepJW(qubits, self.df_hamiltonian)
        circuit = QuantumCircuit(qubits)
        circuit.set_statevector(initial_state)
        for instruction in simulate_trotter(
            trotter_step,
            self.time,
            n_steps=self.n_steps,
        ):
            circuit.append(instruction)
        circuit.save_state()
        self.aer_sim = AerSimulator(max_parallel_threads=OMP_NUM_THREADS)
        self.circuit = transpile(circuit, self.aer_sim)

    def time_simulate_trotter_double_factorized(self, *_):
        ffsim.simulate_trotter_double_factorized(
            self.vec,
            self.df_hamiltonian,
            self.time,
            norb=self.norb,
            nelec=self.nelec,
            n_steps=self.n_steps,
            order=0,
        )

    def time_simulate_trotter_double_factorized_fqe(self, *_):
        step_time = self.time / self.n_steps
        for _ in range(self.n_steps):
            self.vec_fqe = double_factor_trotter_evolution(
                self.vec_fqe,
                self.basis_change_unitaries,
                self.diag_coulomb_mats,
                step_time,
            )

    @skip_for_params([(16, 0.25)])
    def time_simulate_trotter_double_factorized_qiskit(self, *_):
        self.aer_sim.run(self.circuit).result()


TrotterBenchmark().setup(16, 0.25)
