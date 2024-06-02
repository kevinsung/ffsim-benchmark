# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Benchmarks for linear operators."""

import numpy as np

import ffsim


class LinearOperatorBenchmark:
    """Benchmark linear operators."""

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
        self.vec = ffsim.hartree_fock_state(self.norb, self.nelec)
        one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
        two_body_tensor = ffsim.random.random_two_body_tensor(
            norb, seed=rng, dtype=float
        )
        constant = rng.standard_normal()
        mol_hamiltonian = ffsim.MolecularHamiltonian(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            constant=constant,
        )
        self.mol_hamiltonian_linop = ffsim.linear_operator(
            mol_hamiltonian, norb=norb, nelec=self.nelec
        )

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

    def time_molecular_hamiltonian(self, *_):
        _ = self.mol_hamiltonian_linop @ self.vec
