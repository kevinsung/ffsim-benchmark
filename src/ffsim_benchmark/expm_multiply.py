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
import scipy.sparse.linalg

import ffsim


class ExpmMultiplyBenchmark:
    """Benchmark expm_multiply."""

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

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

        # initialize test objects
        rng = np.random.default_rng()
        self.vec = ffsim.random.random_state_vector(
            ffsim.dim(self.norb, self.nelec), seed=rng
        )
        self.time = 1.0

        # MolecularHamiltonian
        mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(
            norb, seed=rng, dtype=float
        )
        self.mol_hamiltonian_linop = ffsim.linear_operator(
            mol_hamiltonian, norb=norb, nelec=self.nelec
        )
        self.mol_hamiltonian_trace = ffsim.trace(
            mol_hamiltonian, norb=norb, nelec=self.nelec
        )

        # DoubleFactorizedHamiltonian
        df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
            mol_hamiltonian, max_vecs=3
        )
        self.df_hamiltonian_linop = ffsim.linear_operator(
            df_hamiltonian, norb=norb, nelec=self.nelec
        )
        self.df_hamiltonian_trace = ffsim.trace(
            df_hamiltonian, norb=norb, nelec=self.nelec
        )

        # DiagonalCoulombHamiltonian
        one_body_tensor = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
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
        self.dc_hamiltonian_trace = ffsim.trace(
            dc_hamiltonian, norb=norb, nelec=self.nelec
        )

    def time_molecular_hamiltonian_with_trace_ffsim(self, *_):
        _ = scipy.sparse.linalg.expm_multiply(
            -1j * self.time * self.mol_hamiltonian_linop,
            self.vec,
            traceA=-1j * self.time * self.mol_hamiltonian_trace,
        )

    def time_molecular_hamiltonian_without_trace_ffsim(self, *_):
        _ = scipy.sparse.linalg.expm_multiply(
            -1j * self.time * self.mol_hamiltonian_linop,
            self.vec,
        )

    def time_double_factorized_hamiltonian_with_trace_ffsim(self, *_):
        _ = scipy.sparse.linalg.expm_multiply(
            -1j * self.time * self.df_hamiltonian_linop,
            self.vec,
            traceA=-1j * self.time * self.df_hamiltonian_trace,
        )

    def time_double_factorized_hamiltonian_without_trace_ffsim(self, *_):
        _ = scipy.sparse.linalg.expm_multiply(
            -1j * self.time * self.df_hamiltonian_linop,
            self.vec,
        )

    def time_diagonal_coulomb_hamiltonian_with_trace_ffsim(self, *_):
        _ = scipy.sparse.linalg.expm_multiply(
            -1j * self.time * self.dc_hamiltonian_linop,
            self.vec,
            traceA=-1j * self.time * self.dc_hamiltonian_trace,
        )

    def time_diagonal_coulomb_hamiltonian_without_trace_ffsim(self, *_):
        _ = scipy.sparse.linalg.expm_multiply(
            -1j * self.time * self.dc_hamiltonian_linop,
            self.vec,
        )
