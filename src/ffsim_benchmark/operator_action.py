# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Benchmarks for operator action."""

import numpy as np

import ffsim
from ffsim_benchmark.util.convert import (
    ffsim_op_to_openfermion_op,
    ffsim_vec_to_fqe_wfn,
)
from ffsim_benchmark.util.random import random_fermion_hamiltonian


class OperatorActionBenchmark:
    """Benchmark operator action."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12, 16),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        rng = np.random.default_rng(5853)

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
        self.op_ffsim = random_fermion_hamiltonian(norb=norb, n_terms=50, seed=rng)
        self.op_openfermion = ffsim_op_to_openfermion_op(self.op_ffsim)
        self.linop_ffsim = ffsim.linear_operator(
            self.op_ffsim, norb=self.norb, nelec=self.nelec
        )

        # initialize ffsim cache
        ffsim.init_cache(self.norb, self.nelec)

    def time_operator_action_ffsim(self, *_):
        _ = self.linop_ffsim @ self.vec_ffsim

    def time_operator_action_fqe(self, *_):
        _ = self.wfn_fqe.apply(self.op_openfermion)
