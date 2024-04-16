# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for operator action."""

import numpy as np
from ffsim_benchmark.convert import (
    ffsim_op_to_openfermion_op,
    ffsim_vec_to_fqe_wfn,
    fqe_wfn_to_ffsim_vec,
)
from ffsim_benchmark.random import random_fermion_hamiltonian

import ffsim


def test_consistent_results():
    """Test ffsim and OpenFermion give consistent results."""
    rng = np.random.default_rng(9938)
    norb = 5
    nelec = (3, 2)
    n_terms = 10
    op_ffsim = random_fermion_hamiltonian(norb=norb, n_terms=n_terms, seed=rng)
    op_openfermion = ffsim_op_to_openfermion_op(op_ffsim)
    vec_ffsim = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)
    wfn = ffsim_vec_to_fqe_wfn(vec_ffsim, norb, nelec)
    result_ffsim = ffsim.linear_operator(op_ffsim, norb=norb, nelec=nelec) @ vec_ffsim
    result_openfermion = fqe_wfn_to_ffsim_vec(wfn.apply(op_openfermion), nelec)
    np.testing.assert_allclose(result_ffsim, result_openfermion)
