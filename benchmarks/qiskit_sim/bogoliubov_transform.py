# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Bogoliubov transform."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Optional

import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import PhaseGate, XXPlusYYGate
from qiskit.quantum_info import Operator
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.utils.linalg import fermionic_gaussian_decomposition_jw

from .linalg import givens_decomposition_square


def _rows_are_orthonormal(
    mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    m, _ = mat.shape
    return np.allclose(mat @ mat.T.conj(), np.eye(m), rtol=rtol, atol=atol)


def _validate_transformation_matrix(
    mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> None:
    if not len(mat.shape) == 2:
        raise ValueError(
            "transformation_matrix must be a 2-dimensional array. "
            f"Instead, got shape {mat.shape}."
        )
    n, p = mat.shape  # pylint: disable=invalid-name
    if p == n:
        if not _rows_are_orthonormal(mat, rtol=rtol, atol=atol):
            raise ValueError("transformation_matrix must have orthonormal rows.")
    elif p == n * 2:
        left = mat[:, :n]
        right = mat[:, n:]
        comm1 = left @ left.T.conj() + right @ right.T.conj()
        comm2 = left @ right.T + right @ left.T
        if not np.allclose(comm1, np.eye(n), rtol=rtol, atol=atol) or not np.allclose(
            comm2, 0.0, atol=atol
        ):
            raise ValueError(
                "transformation_matrix does not describe a valid transformation "
                "of fermionic ladder operators. A valid matrix should have the block form "
                "[W1 W2] where W1 @ W1.T.conj() + W2 @ W2.T.conj() = I and "
                "W1 @ W2.T + W2 @ W1.T = 0."
            )
    else:
        raise ValueError(
            f"transformation_matrix must be N x N or N x 2N. Instead, got shape {mat.shape}."
        )


def _bogoliubov_transform_jw(
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterator[tuple[Gate, tuple[Qubit, ...]]]:
    n, p = transformation_matrix.shape  # pylint: disable=invalid-name
    if p == n:
        yield from _bogoliubov_transform_num_conserving_jw(
            register, transformation_matrix
        )
    else:
        yield from _bogoliubov_transform_general_jw(register, transformation_matrix)


def _bogoliubov_transform_num_conserving_jw(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterator[tuple[Gate, tuple[Qubit, ...]]]:
    givens_rotations, phase_shifts = givens_decomposition_square(transformation_matrix)
    for i, phase_shift in enumerate(phase_shifts):
        yield PhaseGate(np.angle(phase_shift)), (register[i],)
    for mat, i, j in givens_rotations:
        angle = np.arccos(np.real(mat[0, 0]))
        phase_angle = np.angle(mat[0, 1])
        yield XXPlusYYGate(2 * angle, phase_angle - np.pi / 2), (
            register[i],
            register[j],
        )


def _bogoliubov_transform_general_jw(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterator[tuple[Gate, tuple[Qubit, ...]]]:
    # TODO correct global phase
    decomposition, left_unitary = fermionic_gaussian_decomposition_jw(
        register, transformation_matrix
    )
    yield from _bogoliubov_transform_num_conserving_jw(register, left_unitary.T)
    yield from reversed(decomposition)


class BogoliubovTransformJW(Gate):
    """Bogoliubov transform under the Jordan-Wigner transformation."""

    def __init__(
        self,
        transformation_matrix: np.ndarray,
        label: Optional[str] = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        """Create new Bogoliubov transform gate.

        Args:
            transformation_matrix: The matrix :math:`W` that specifies the coefficients of the
                new creation operators in terms of the original creation operators.
                Should be either :math:`N \times N` or :math:`N \times 2N`.
            label: The label of the gate.
        """
        if validate:
            _validate_transformation_matrix(transformation_matrix, rtol=rtol, atol=atol)
        self.transformation_matrix = transformation_matrix
        n, _ = transformation_matrix.shape
        super().__init__("bog_jw", n, [], label=label)

    def _define(self):
        """Gate decomposition."""
        n, _ = self.transformation_matrix.shape
        register = QuantumRegister(n)
        circuit = QuantumCircuit(register, name=self.name)
        for gate, qubits in _bogoliubov_transform_jw(
            register, self.transformation_matrix
        ):
            circuit.append(gate, qubits)
        self.definition = circuit

    def inverse(self):
        """Inverse gate."""
        n, p = self.transformation_matrix.shape  # pylint: disable=invalid-name
        if p == n:
            return BogoliubovTransformJW(self.transformation_matrix.T.conj())
        left = self.transformation_matrix[:, :n]
        right = self.transformation_matrix[:, n:]
        return BogoliubovTransformJW(np.concatenate([left.T.conj(), right.T], axis=1))

    def __array__(self, dtype=None):
        """Gate matrix."""
        n, p = self.transformation_matrix.shape  # pylint: disable=invalid-name
        if p == n:
            return _bog_unitary_num_conserving(self.transformation_matrix)
        return np.array(Operator(self.definition), dtype=dtype)


def _bog_unitary_num_conserving(transformation_matrix: np.ndarray) -> np.ndarray:
    """Compute the unitary of a single-particle basis change using the Thouless Theorem.

    Args:
        transformation_matrix: The matrix that specifies the coefficients of the
            new creation operators in terms of the original creation operators.

    Returns:
        The unitary matrix of the single-particle basis change.
    """
    mat_log = scipy.linalg.logm(transformation_matrix.T)
    n_modes, _ = transformation_matrix.shape
    data = {}
    for p in range(n_modes):
        data[f"+_{p} -_{p}"] = mat_log[p, p]
    for p, q in itertools.combinations(range(n_modes), 2):
        data[f"+_{p} -_{q}"] = mat_log[p, q]
        data[f"+_{q} -_{p}"] = -mat_log[p, q].conj()
    op = FermionicOp(data, num_spin_orbitals=n_modes)
    op_jw = JordanWignerMapper().map(op).primitive.to_matrix()
    return scipy.linalg.expm(op_jw)
