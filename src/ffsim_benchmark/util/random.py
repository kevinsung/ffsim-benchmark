# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections import defaultdict

import numpy as np

import ffsim


def random_fermion_operator(
    norb: int, n_terms: int, max_term_length: int | None = None, seed=None
) -> ffsim.FermionOperator:
    """Sample a random fermion operator."""
    rng = np.random.default_rng(seed)

    if max_term_length is None:
        max_term_length = norb

    coeffs = defaultdict(complex)
    for _ in range(n_terms):
        term_length = int(rng.integers(1, max_term_length + 1))
        actions = [bool(i) for i in rng.integers(2, size=term_length)]
        spins = [bool(i) for i in rng.integers(2, size=term_length)]
        indices = [int(i) for i in rng.integers(norb, size=term_length)]
        coeff = rng.standard_normal() + 1j * rng.standard_normal()
        term = tuple(zip(actions, spins, indices))
        coeffs[term] += coeff

    op_ffsim = ffsim.FermionOperator(coeffs)

    return op_ffsim


def random_fermion_hamiltonian(
    norb: int, n_terms: int, seed=None
) -> ffsim.FermionOperator:
    """Sample a random fermion Hamiltonian.

    A fermion Hamiltonian is hermitian and conserves particle number and spin Z.
    """
    rng = np.random.default_rng(seed)
    coeffs = defaultdict(complex)

    for _ in range(n_terms):
        n_excitations = int(rng.integers(1, norb // 2 + 1))
        ffsim_term = _random_num_and_spin_z_conserving_term(
            n_excitations, norb=norb, seed=rng
        )
        ffsim_term_adjoint = _adjoint_term(ffsim_term)
        coeff = rng.standard_normal() + 1j * rng.standard_normal()
        coeffs[ffsim_term] += coeff
        coeffs[ffsim_term_adjoint] += coeff.conjugate()

    op_ffsim = ffsim.FermionOperator(coeffs)

    return op_ffsim


def _random_num_and_spin_z_conserving_term(
    n_excitations: int, norb: int, seed=None
) -> tuple[tuple[bool, bool, int], ...]:
    rng = np.random.default_rng(seed)
    term = []
    for _ in range(n_excitations):
        spin = bool(rng.integers(2))
        orb_1, orb_2 = [int(x) for x in rng.integers(norb, size=2)]
        action_1, action_2 = [
            bool(x) for x in rng.choice([True, False], size=2, replace=False)
        ]
        term.append(ffsim.FermionAction(action_1, spin, orb_1))
        term.append(ffsim.FermionAction(action_2, spin, orb_2))
    return tuple(term)


def _adjoint_term(
    term: tuple[tuple[bool, bool, int], ...],
) -> tuple[tuple[bool, bool, int], ...]:
    return tuple(
        ffsim.FermionAction(bool(1 - action), spin, orb)
        for action, spin, orb in reversed(term)
    )
