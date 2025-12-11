# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import fqe
import numpy as np

import ffsim
from ffsim_benchmark.util.convert import (
    ffsim_op_to_openfermion_op,
    ffsim_vec_to_fqe_wfn,
)


class MolecularHamiltonianActionComplexBenchmark:
    """Benchmark molecular Hamiltonian operator action."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12, 16),
        (0.25,),
    ]

    def setup(self, norb: int, filling_fraction: float):
        rng = np.random.default_rng(215196083997839770748582168260368828406)

        # set benchmark parameters
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)

        # initialize test objects
        self.vec_ffsim = ffsim.random.random_state_vector(
            ffsim.dim(self.norb, self.nelec)
        )
        self.wfn_fqe = ffsim_vec_to_fqe_wfn(
            self.vec_ffsim, norb=self.norb, nelec=self.nelec
        )
        mol_ham = ffsim.random.random_molecular_hamiltonian(self.norb, seed=rng)
        self.linop_ffsim = ffsim.linear_operator(
            mol_ham, norb=self.norb, nelec=self.nelec
        )

        self.op_fqe = ffsim_op_to_openfermion_op(ffsim.fermion_operator(mol_ham))
        self.op_fqe_sparse = fqe.get_sparse_hamiltonian(self.op_fqe)

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

    def time_mol_ham_action_complex_ffsim(self, *_):
        _ = self.linop_ffsim @ self.vec_ffsim

    def time_mol_ham_action_complex_fqe(self, *_):
        _ = self.wfn_fqe.apply(self.op_fqe)

    def time_mol_ham_action_complex_fqe_sparse(self, *_):
        _ = self.wfn_fqe.apply(self.op_fqe_sparse)


class MolecularHamiltonianActionRealBenchmark:
    """Benchmark molecular Hamiltonian operator action."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12, 16),
        (0.25,),
    ]

    def setup(self, norb: int, filling_fraction: float):
        rng = np.random.default_rng(215196083997839770748582168260368828406)

        # set benchmark parameters
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)

        # initialize test objects
        self.vec_ffsim = ffsim.random.random_state_vector(
            ffsim.dim(self.norb, self.nelec)
        )
        self.wfn_fqe = ffsim_vec_to_fqe_wfn(
            self.vec_ffsim, norb=self.norb, nelec=self.nelec
        )
        mol_ham = ffsim.random.random_molecular_hamiltonian(
            self.norb, seed=rng, dtype=float
        )
        self.linop_ffsim = ffsim.linear_operator(
            mol_ham, norb=self.norb, nelec=self.nelec
        )

        self.op_fqe = ffsim_op_to_openfermion_op(ffsim.fermion_operator(mol_ham))
        self.op_fqe_sparse = fqe.get_sparse_hamiltonian(self.op_fqe)

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

    def time_mol_ham_action_real_ffsim(self, *_):
        _ = self.linop_ffsim @ self.vec_ffsim

    def time_mol_ham_action_real_fqe(self, *_):
        _ = self.wfn_fqe.apply(self.op_fqe)

    def time_mol_ham_action_real_fqe_sparse(self, *_):
        _ = self.wfn_fqe.apply(self.op_fqe_sparse)
