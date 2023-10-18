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

# from ffsim.slow.fermion_operator import FermionOperator
from ffsim import FermionOperator


class FermionOperatorBenchmark:
    """Benchmark FermionOperator."""

    def setup(self):
        norb = 50
        n_terms = 100
        rng = np.random.default_rng()

        self.op_openfermion = of.FermionOperator()
        coeffs_ffsim = {}

        for _ in range(n_terms):
            term_length = int(rng.integers(1, norb + 1))
            actions = [bool(i) for i in rng.integers(2, size=term_length)]
            indices = [int(i) for i in rng.integers(norb, size=term_length)]
            coeff = rng.standard_normal() + 1j * rng.standard_normal()
            self.op_openfermion += of.FermionOperator(
                tuple(zip(indices, actions)), coeff
            )
            ffsim_tuple = tuple(zip(actions, indices))
            if ffsim_tuple in coeffs_ffsim:
                coeffs_ffsim[ffsim_tuple] += coeff
            else:
                coeffs_ffsim[ffsim_tuple] = coeff

        self.op_ffsim = FermionOperator(coeffs_ffsim)

    def time_normal_order_openfermion(self):
        of.normal_ordered(self.op_openfermion)

    def time_normal_order_ffsim(self):
        self.op_ffsim.normal_ordered()
