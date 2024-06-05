# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools
import os

import numpy as np

import ffsim

OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))


class UCJBenchmark:
    """Benchmark UCJ."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        # set benchmark parameters
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)
        n_reps = 3

        # initialize test objects
        rng = np.random.default_rng()
        self.vec = ffsim.hartree_fock_state(self.norb, self.nelec)
        self.ucj_op_spin_balanced = ffsim.random.random_ucj_op_spin_balanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        self.ucj_op_spin_unbalanced = ffsim.random.random_ucj_op_spin_unbalanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        self.ucj_op_spinless = ffsim.random.random_ucj_op_spinless(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

    def time_ucj_spin_balanced_ffsim(self, *_):
        ffsim.apply_unitary(
            self.vec,
            self.ucj_op_spin_balanced,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_ucj_spin_unbalanced_ffsim(self, *_):
        ffsim.apply_unitary(
            self.vec,
            self.ucj_op_spin_unbalanced,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_ucj_spinless_ffsim(self, *_):
        ffsim.apply_unitary(
            self.vec,
            self.ucj_op_spinless,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )


class HopGateAnsatzBenchmark:
    """Benchmark hop gate ansatz."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        # set benchmark parameters
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)

        # initialize test objects
        rng = np.random.default_rng()
        self.vec = ffsim.hartree_fock_state(norb, self.nelec)
        interaction_pairs = list(itertools.combinations(range(norb), 2))
        thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
        self.operator = ffsim.HopGateAnsatzOperator(
            norb, interaction_pairs, thetas=thetas
        )

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

    def time_hop_gate_ansatz_ffsim(self, *_):
        ffsim.apply_unitary(
            self.vec, self.operator, norb=self.norb, nelec=self.nelec, copy=False
        )
