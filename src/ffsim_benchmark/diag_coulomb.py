# (C) Copyright IBM 2026.
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
from fqe.algorithm.low_rank import evolve_fqe_diagonal_coulomb
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

import ffsim

OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))


class DiagCoulombEvoBenchmark:
    """Benchmark diagonal Coulomb time evolution."""

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
        rng = np.random.default_rng(52854555203949936467026317651539355992)
        self.vec = ffsim.random.random_state_vector(
            ffsim.dim(self.norb, self.nelec), seed=rng
        )
        self.diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
            self.norb, seed=rng
        )
        self.time = 1.0

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
        fqe.settings.use_accelerated_code = True

        # prepare Qiskit
        if norb <= 12:
            self.aer_sim = AerSimulator(max_parallel_threads=OMP_NUM_THREADS)
            initial_state = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
                self.vec, norb=norb, nelec=self.nelec
            )
            register = QuantumRegister(2 * norb)
            circuit = QuantumCircuit(register)
            circuit.set_statevector(initial_state)
            circuit.append(
                ffsim.qiskit.DiagCoulombEvolutionJW(
                    norb, self.diag_coulomb_mat, self.time
                ),
                register,
            )
            circuit.save_state()
            self.diag_coulomb_circuit = transpile(circuit, self.aer_sim)

    def time_diag_coulomb_evolution_ffsim(self, *_):
        _ = ffsim.apply_diag_coulomb_evolution(
            self.vec,
            self.diag_coulomb_mat,
            time=self.time,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_diag_coulomb_evolution_fqe(self, *_):
        _ = evolve_fqe_diagonal_coulomb(self.vec_fqe, self.diag_coulomb_mat, self.time)

    @skip_for_params([(16, 0.25), (16, 0.5)])
    def time_diag_coulomb_evolution_qiskit(self, *_):
        _ = self.aer_sim.run(self.diag_coulomb_circuit).result()
