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

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

        # initialize test objects
        rng = np.random.default_rng()
        self.vec = ffsim.random.random_state_vector(
            ffsim.dim(self.norb, self.nelec), seed=rng
        )

        # MolecularHamiltonian
        one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
        two_body_tensor = ffsim.random.random_two_body_tensor(
            norb, seed=rng, dtype=float
        )
        constant = rng.standard_normal()
        # Complex
        mol_hamiltonian = ffsim.MolecularHamiltonian(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            constant=constant,
        )
        self.mol_hamiltonian_linop = ffsim.linear_operator(
            mol_hamiltonian, norb=norb, nelec=self.nelec
        )
        # Real
        mol_hamiltonian_real = ffsim.MolecularHamiltonian(
            one_body_tensor=one_body_tensor.real,
            two_body_tensor=two_body_tensor,
            constant=constant,
        )
        self.mol_hamiltonian_real_linop = ffsim.linear_operator(
            mol_hamiltonian_real, norb=norb, nelec=self.nelec
        )

        # DoubleFactorizedHamiltonian
        n_reps = 3
        one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
        diag_coulomb_mats = np.stack(
            [
                ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
                for _ in range(n_reps)
            ]
        )
        orbital_rotations = np.stack(
            [ffsim.random.random_unitary(norb, seed=rng) for _ in range(n_reps)]
        )
        constant = rng.standard_normal()
        df_hamiltonian = ffsim.DoubleFactorizedHamiltonian(
            one_body_tensor=one_body_tensor,
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            constant=constant,
        )
        self.df_hamiltonian_linop = ffsim.linear_operator(
            df_hamiltonian, norb=norb, nelec=self.nelec
        )

        # DiagonalCoulombHamiltonian
        one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
        diag_coulomb_mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        diag_coulomb_mat_ab = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        diag_coulomb_mats = np.stack([diag_coulomb_mat_aa, diag_coulomb_mat_ab])
        constant = rng.standard_normal()
        dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
            one_body_tensor=one_body_tensor,
            diag_coulomb_mats=diag_coulomb_mats,
            constant=constant,
        )
        self.dc_hamiltonian_linop = ffsim.linear_operator(
            dc_hamiltonian, norb=norb, nelec=self.nelec
        )

    def time_molecular_hamiltonian_complex_ffsim(self, *_):
        _ = self.mol_hamiltonian_linop @ self.vec

    def time_molecular_hamiltonian_real_ffsim(self, *_):
        _ = self.mol_hamiltonian_real_linop @ self.vec

    def time_double_factorized_hamiltonian_ffsim(self, *_):
        _ = self.df_hamiltonian_linop @ self.vec

    def time_diagonal_coulomb_hamiltonian_ffsim(self, *_):
        _ = self.dc_hamiltonian_linop @ self.vec
