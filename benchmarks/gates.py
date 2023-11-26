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
from fqe.algorithm.low_rank import evolve_fqe_givens
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_nature.circuit.library import BogoliubovTransform

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
        circuit.append(BogoliubovTransform(orbital_rotation.conj()), qubits[:norb])
        circuit.append(BogoliubovTransform(orbital_rotation.conj()), qubits[norb:])
    for q, energy in zip(qubits, coeffs):
        circuit.rz(-energy * time, q)
    if orbital_rotation is not None:
        circuit.append(BogoliubovTransform(orbital_rotation.T), qubits[:norb])
        circuit.append(BogoliubovTransform(orbital_rotation.T), qubits[norb:])
    circuit.save_state()
    return circuit


class GatesBenchmark:
    """Benchmark gates."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12, 16),
        (0.25,),
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
        self.diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
            self.norb, seed=rng
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
            initial_state = ffsim.random.random_statevector(2 ** (2 * norb), seed=rng)
            # orbital rotation circuit
            register = QuantumRegister(2 * norb)
            circuit = QuantumCircuit(register)
            circuit.set_statevector(initial_state)
            circuit.append(
                BogoliubovTransform(self.orbital_rotation.T), register[:norb]
            )
            circuit.append(
                BogoliubovTransform(self.orbital_rotation.T), register[norb:]
            )
            circuit.save_state()
            self.orbital_rotation_circuit = transpile(circuit, self.aer_sim)
            # quad ham evolution circuit
            circuit = num_op_sum_evo_circuit(
                self.orbital_energies,
                initial_state,
                1.0,
                norb=norb,
                orbital_rotation=self.orbital_rotation,
            )
            self.quad_ham_evo_circuit = transpile(circuit, self.aer_sim)

    def time_apply_orbital_rotation_givens(self, *_):
        ffsim.apply_orbital_rotation(
            self.vec,
            self.orbital_rotation,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    @skip_for_params([(16, 0.5)])
    def time_apply_orbital_rotation_givens_fqe(self, *_):
        evolve_fqe_givens(self.vec_fqe, self.orbital_rotation)

    @skip_for_params([(16, 0.25), (16, 0.5)])
    def time_apply_orbital_rotation_givens_qiskit(self, *_):
        self.aer_sim.run(self.orbital_rotation_circuit).result()

    def time_apply_orbital_rotation_lu(self, *_):
        ffsim.apply_orbital_rotation(
            self.vec,
            self.orbital_rotation,
            norb=self.norb,
            nelec=self.nelec,
            allow_col_permutation=True,
            copy=False,
        )

    def time_apply_num_op_sum_evolution(self, *_):
        ffsim.apply_num_op_sum_evolution(
            self.vec,
            self.orbital_energies,
            time=1.0,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_diag_coulomb_evolution(self, *_):
        ffsim.apply_diag_coulomb_evolution(
            self.vec,
            self.diag_coulomb_mat,
            time=1.0,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_givens_rotation(self, *_):
        ffsim.apply_givens_rotation(
            self.vec,
            theta=1.0,
            target_orbs=(0, 2),
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_num_interaction(self, *_):
        ffsim.apply_num_interaction(
            self.vec,
            theta=1.0,
            target_orb=1,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_num_num_interaction(self, *_):
        ffsim.apply_num_num_interaction(
            self.vec,
            theta=1.0,
            target_orbs=(0, 1),
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_num_op_prod_interaction(self, *_):
        ffsim.apply_num_op_prod_interaction(
            self.vec,
            theta=1.0,
            target_orbs=([1], [0]),
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_tunneling_interaction(self, *_):
        ffsim.apply_tunneling_interaction(
            self.vec,
            theta=1.0,
            target_orbs=(0, 2),
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_quad_ham_evolution(self, *_):
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
