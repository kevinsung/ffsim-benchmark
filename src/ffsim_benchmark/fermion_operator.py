# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import openfermion as of

import ffsim
from ffsim_benchmark.util.convert import ffsim_op_to_openfermion_op
from ffsim_benchmark.util.random import random_fermion_operator


class FermionOperatorBenchmark:
    """Benchmark FermionOperator."""

    param_names = [
        "n_terms",
    ]
    params = [
        (100, 1_000, 10_000, 100_000),
    ]

    def setup(self, n_terms: int):
        self.op_ffsim = random_fermion_operator(norb=100, n_terms=n_terms, seed=4142)
        self.op_openfermion = ffsim_op_to_openfermion_op(self.op_ffsim)

    def time_normal_order_ffsim(self, *_):
        self.op_ffsim.normal_ordered()

    def time_normal_order_openfermion(self, *_):
        of.normal_ordered(self.op_openfermion)


class MolHamToFermionOperatorBenchmark:
    """Benchmark converting a molecular Hamiltonian to a FermionOperator."""

    param_names = ["norb"]
    params = [(20, 40, 60, 80)]

    def setup(self, norb: int):
        rng = np.random.default_rng(106662067345669348763419345753470946398)
        self.mol_ham = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)

    def time_mol_ham_to_fermion_operator_ffsim(self, *_):
        _ = ffsim.fermion_operator(self.mol_ham)
