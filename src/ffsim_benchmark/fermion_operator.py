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

from ffsim_benchmark.random import random_fermion_operators


class FermionOperatorBenchmark:
    """Benchmark FermionOperator."""

    def setup(self):
        self.op_ffsim, self.op_openfermion = random_fermion_operators(
            norb=50, n_terms=100, seed=4142
        )

    def time_normal_order_ffsim(self):
        self.op_ffsim.normal_ordered()

    def time_normal_order_openfermion(self):
        of.normal_ordered(self.op_openfermion)
