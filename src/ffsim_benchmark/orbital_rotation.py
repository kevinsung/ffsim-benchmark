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
from fqe.algorithm.low_rank import evolve_fqe_givens
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

import ffsim

OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))


class OrbitalRotationBenchmark:
    """Benchmark orbital rotation."""

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
        self.orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

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
                ffsim.qiskit.OrbitalRotationJW(norb, self.orbital_rotation), register
            )
            circuit.save_state()
            self.orbital_rotation_circuit = transpile(circuit, self.aer_sim)

    def time_apply_orbital_rotation_ffsim(self, *_):
        ffsim.apply_orbital_rotation(
            self.vec,
            self.orbital_rotation,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    @skip_for_params([(16, 0.5)])
    def time_apply_orbital_rotation_fqe(self, *_):
        evolve_fqe_givens(self.vec_fqe, self.orbital_rotation)

    @skip_for_params([(16, 0.25), (16, 0.5)])
    def time_apply_orbital_rotation_qiskit(self, *_):
        self.aer_sim.run(self.orbital_rotation_circuit).result()
