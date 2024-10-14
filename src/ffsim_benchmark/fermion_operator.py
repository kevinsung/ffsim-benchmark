# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import openfermion as of

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
        self.op_ffsim = random_fermion_operator(norb=50, n_terms=n_terms, seed=4142)
        self.op_openfermion = ffsim_op_to_openfermion_op(self.op_ffsim)

    def time_normal_order_ffsim(self, *_):
        self.op_ffsim.normal_ordered()

    def time_normal_order_openfermion(self, *_):
        of.normal_ordered(self.op_openfermion)
