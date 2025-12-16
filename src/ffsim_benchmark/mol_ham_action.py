# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from importlib.resources import as_file, files

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
            ffsim.dim(self.norb, self.nelec), seed=rng
        )
        self.wfn_fqe = ffsim_vec_to_fqe_wfn(
            self.vec_ffsim, norb=self.norb, nelec=self.nelec
        )
        mol_ham = ffsim.random.random_molecular_hamiltonian(self.norb, seed=rng)
        self.linop_ffsim = ffsim.linear_operator(
            mol_ham, norb=self.norb, nelec=self.nelec
        )

        self.op_fqe = fqe.build_hamiltonian(
            ffsim_op_to_openfermion_op(ffsim.fermion_operator(mol_ham))
        )

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

    def time_mol_ham_action_complex_ffsim(self, *_):
        _ = self.linop_ffsim @ self.vec_ffsim

    def time_mol_ham_action_complex_fqe(self, *_):
        _ = self.wfn_fqe.apply(self.op_fqe)


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
            ffsim.dim(self.norb, self.nelec), seed=rng
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

        self.op_fqe = fqe.build_hamiltonian(
            ffsim_op_to_openfermion_op(ffsim.fermion_operator(mol_ham))
        )

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

    def time_mol_ham_action_real_ffsim(self, *_):
        _ = self.linop_ffsim @ self.vec_ffsim

    def time_mol_ham_action_real_fqe(self, *_):
        _ = self.wfn_fqe.apply(self.op_fqe)


class MolecularHamiltonianActionN2Benchmark:
    """Benchmark molecular Hamiltonian operator action."""

    def setup(self):
        rng = np.random.default_rng(215196083997839770748582168260368828406)

        data_file = files("ffsim_benchmark.data").joinpath(
            "n2_6-31g_10e16o_d-1.20000.json.xz"
        )
        with as_file(data_file) as path:
            mol_data = ffsim.MolecularData.from_json(path, compression="lzma")

        norb = mol_data.norb
        nelec = mol_data.nelec

        # initialize test objects
        self.vec_ffsim = ffsim.random.random_state_vector(
            ffsim.dim(norb, nelec), seed=rng
        )
        self.wfn_fqe = ffsim_vec_to_fqe_wfn(self.vec_ffsim, norb=norb, nelec=nelec)
        mol_ham = mol_data.hamiltonian
        self.linop_ffsim = ffsim.linear_operator(mol_ham, norb=norb, nelec=nelec)

        self.op_fqe = fqe.build_hamiltonian(
            ffsim_op_to_openfermion_op(ffsim.fermion_operator(mol_ham))
        )

        # initialize ffsim cache
        ffsim.init_cache(norb, nelec)

    def time_mol_ham_action_n2_ffsim(self, *_):
        _ = self.linop_ffsim @ self.vec_ffsim

    def time_mol_ham_action_n2_fqe(self, *_):
        _ = self.wfn_fqe.apply(self.op_fqe)
