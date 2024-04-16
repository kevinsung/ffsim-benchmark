# (C) Copyright IBM 2024.
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
import openfermion as of

import ffsim
from ffsim_benchmark.convert_fqe import fqe_to_pyscf, pyscf_to_fqe_wf


def ffsim_op_to_openfermion_op(operator: ffsim.FermionOperator) -> of.FermionOperator:
    op_openfermion = of.FermionOperator()
    for term, coeff in operator.items():
        openfermion_term = tuple((2 * orb + spin, action) for action, spin, orb in term)
        op_openfermion += of.FermionOperator(openfermion_term, coeff)
    return op_openfermion


def ffsim_vec_to_fqe_wfn(
    vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> fqe.wavefunction.Wavefunction:
    return pyscf_to_fqe_wf(
        vec.reshape(*ffsim.dims(norb, nelec)), norbs=norb, nelec=nelec
    )


def fqe_wfn_to_ffsim_vec(
    wfn: fqe.wavefunction.Wavefunction, nelec: tuple[int, int]
) -> fqe.wavefunction.Wavefunction:
    return fqe_to_pyscf(wfn, nelec).reshape(-1)
