# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import openfermion as of

import ffsim


def ffsim_op_to_openfermion_op(operator: ffsim.FermionOperator) -> of.FermionOperator:
    op_openfermion = of.FermionOperator()
    for term, coeff in operator.items():
        openfermion_term = tuple((2 * orb + spin, action) for action, spin, orb in term)
        op_openfermion += of.FermionOperator(openfermion_term, coeff)
    return op_openfermion
