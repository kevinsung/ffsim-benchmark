# (C) Copyright IBM 2024.
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
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

import ffsim

OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))


def num_op_sum_evo_circuit(
    coeffs: np.ndarray,
    initial_state: np.ndarray,
    time: float,
    norb: int,
    orbital_rotation: np.ndarray | None = None,
):
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.set_statevector(initial_state)
    if orbital_rotation is not None:
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation.T.conj()), qubits
        )
    for q, energy in zip(qubits, coeffs):
        circuit.rz(-energy * time, q)
    if orbital_rotation is not None:
        circuit.append(ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation), qubits)
    circuit.save_state()
    return circuit


class QuadHamEvoBenchmark:
    """Benchmark quadratic Hamiltonian time evolution."""

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

        # initialize test objects
        rng = np.random.default_rng()
        self.vec = ffsim.random.random_statevector(
            ffsim.dim(self.norb, self.nelec), seed=rng
        )
        self.one_body_tensor = ffsim.random.random_hermitian(self.norb, seed=rng)
        self.orbital_energies, self.orbital_rotation = np.linalg.eigh(
            self.one_body_tensor
        )

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

        # prepare FQE
        n_alpha, n_beta = self.nelec
        n_particles = n_alpha + n_beta
        spin = n_alpha - n_beta
        self.vec_fqe = fqe.Wavefunction([[n_particles, spin, norb]])
        dim_a, dim_b = ffsim.dims(self.norb, self.nelec)
        self.vec_fqe.set_wfn(
            strategy="from_data",
            raw_data={(n_particles, spin): self.vec.reshape(dim_a, dim_b).copy()},
        )
        self.fqe_ham = RestrictedHamiltonian((self.one_body_tensor,))
        fqe.settings.use_accelerated_code = True

        # prepare Qiskit
        if norb <= 12:
            self.aer_sim = AerSimulator(max_parallel_threads=OMP_NUM_THREADS)
            initial_state = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
                self.vec, norb=norb, nelec=self.nelec
            )
            circuit = num_op_sum_evo_circuit(
                self.orbital_energies,
                initial_state,
                1.0,
                norb=norb,
                orbital_rotation=self.orbital_rotation,
            )
            self.quad_ham_evo_circuit = transpile(circuit, self.aer_sim)

    def time_quad_ham_evolution_ffsim(self, *_):
        ffsim.apply_num_op_sum_evolution(
            self.vec,
            self.orbital_energies,
            time=1.0,
            norb=self.norb,
            nelec=self.nelec,
            orbital_rotation=self.orbital_rotation,
            copy=False,
        )

    def time_quad_ham_evolution_fqe(self, *_):
        self.vec_fqe.time_evolve(1.0, self.fqe_ham)

    @skip_for_params([(16, 0.25), (16, 0.5)])
    def time_quad_ham_evolution_qiskit(self, *_):
        self.aer_sim.run(self.quad_ham_evo_circuit).result()
