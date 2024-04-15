# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import ffsim
from ffsim_benchmark.random import random_fermion_hamiltonian
from ffsim_benchmark.fqe_convert import pyscf_to_fqe_wf, fqe_to_pyscf
import numpy as np


def test_random_fermion_operators():
    rng = np.random.default_rng(9938)
    norb = 5
    nelec = (3, 2)
    n_terms = 10
    op_ffsim, op_openfermion = random_fermion_hamiltonian(
        norb=norb, n_terms=n_terms, seed=rng
    )
    vec_ffsim = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)
    wfn = pyscf_to_fqe_wf(
        vec_ffsim.reshape(*ffsim.dims(norb, nelec)), norbs=norb, nelec=nelec
    )
    result_ffsim = ffsim.linear_operator(op_ffsim, norb=norb, nelec=nelec) @ vec_ffsim
    result_openfermion = fqe_to_pyscf(wfn.apply(op_openfermion), nelec)
    np.testing.assert_allclose(result_ffsim, result_openfermion.reshape(-1))
