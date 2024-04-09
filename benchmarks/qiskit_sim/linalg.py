# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear algebra utilities."""

from __future__ import annotations

import itertools
from collections import namedtuple
from typing import TYPE_CHECKING, List, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import XGate, XXPlusYYGate

# HACK: Sphinx fails to handle "ellipsis"
# See https://github.com/python/typing/issues/684
if TYPE_CHECKING:
    _SliceAtom = Union[int, slice, "ellipsis"]
else:
    _SliceAtom = Union[int, slice, type(Ellipsis)]

_Slice = Union[_SliceAtom, Tuple[_SliceAtom, ...]]

GivensRotation = namedtuple("GivensRotation", ["matrix", "row", "col"])
PhaseShift = namedtuple("PhaseShift", ["angle", "index"])


def apply_matrix_to_slices(
    target: np.ndarray, matrix: np.ndarray, slices: Sequence[_Slice]
) -> np.ndarray:
    """Apply a matrix to slices of a target tensor.

    Args:
        target: The tensor containing the slices on which to apply the matrix.
        matrix: The matrix to apply to slices of the target tensor.
        slices: The slices of the target tensor on which to apply the matrix.

    Returns:
        The tensor resulting from applying the matrix to the slices of the target tensor.
    """
    result = target.copy()
    for i, slice_i in enumerate(slices):
        result[slice_i] *= matrix[i, i]
        for j, slice_j in enumerate(slices):
            if j != i:
                result[slice_i] += target[slice_j] * matrix[i, j]
    return result


def givens_matrix(a: complex, b: complex) -> np.ndarray:
    r"""Compute the Givens rotation to zero out a row entry.

    Returns a :math:`2 \times 2` unitary matrix G that satisfies

    .. math::

        G
        \begin{pmatrix}
            a \\
            b
        \end{pmatrix}
        =
        \begin{pmatrix}
            r \\
            0
        \end{pmatrix}

    where :math:`r` is a complex number.

    References:
        - `<https://en.wikipedia.org/wiki/Givens_rotation#Stable_calculation>`_
        - `<https://www.netlib.org/lapack/lawnspdf/lawn148.pdf>`_

    Args:
        a: A complex number representing the first row entry
        b: A complex number representing the second row entry

    Returns:
        The Givens rotation matrix.
    """
    # Handle case that a is zero
    if np.isclose(a, 0.0):
        cosine = 0.0
        sine = 1.0
    # Handle case that b is zero and a is nonzero
    elif np.isclose(b, 0.0):
        cosine = 1.0
        sine = 0.0
    # Handle case that a and b are both nonzero
    else:
        hypotenuse = np.hypot(abs(a), abs(b))
        cosine = abs(a) / hypotenuse
        sign_a = a / abs(a)
        sine = sign_a * b.conjugate() / hypotenuse

    return np.array([[cosine, sine], [-sine.conjugate(), cosine]])


def givens_decomposition_square(  # pylint: disable=invalid-name
    mat: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    n, _ = mat.shape
    current_matrix = mat
    left_rotations = []
    right_rotations = []

    # compute left and right Givens rotations
    for i in range(n - 1):
        if i % 2 == 0:
            # rotate columns by right multiplication
            for j in range(i + 1):
                target_index = i - j
                row = n - j - 1
                if not np.isclose(current_matrix[row, target_index], 0.0):
                    # zero out element at target index in given row
                    givens_mat = givens_matrix(
                        current_matrix[row, target_index + 1],
                        current_matrix[row, target_index],
                    )
                    right_rotations.append(
                        (givens_mat, (target_index + 1, target_index))
                    )
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat,
                        [(Ellipsis, target_index + 1), (Ellipsis, target_index)],
                    )
        else:
            # rotate rows by left multiplication
            for j in range(i + 1):
                target_index = n - i + j - 1
                col = j
                if not np.isclose(current_matrix[target_index, col], 0.0):
                    # zero out element at target index in given column
                    givens_mat = givens_matrix(
                        current_matrix[target_index - 1, col],
                        current_matrix[target_index, col],
                    )
                    left_rotations.append(
                        (givens_mat, (target_index - 1, target_index))
                    )
                    current_matrix = apply_matrix_to_slices(
                        current_matrix, givens_mat, [target_index - 1, target_index]
                    )

    # convert left rotations to right rotations
    for givens_mat, (i, j) in reversed(left_rotations):
        givens_mat = givens_mat.T.conj().astype(mat.dtype, copy=False)
        givens_mat[:, 0] *= current_matrix[i, i]
        givens_mat[:, 1] *= current_matrix[j, j]
        new_givens_mat = givens_matrix(givens_mat[1, 1], givens_mat[1, 0])
        right_rotations.append((new_givens_mat.T, (i, j)))
        phase_matrix = givens_mat @ new_givens_mat
        current_matrix[i, i] = phase_matrix[0, 0]
        current_matrix[j, j] = phase_matrix[1, 1]

    # return decomposition
    givens_rotations = [
        GivensRotation(givens_mat, j, i) for givens_mat, (i, j) in right_rotations[::-1]
    ]
    phase_shifts = np.diagonal(current_matrix)
    return givens_rotations, phase_shifts


def fermionic_gaussian_decomposition_jw(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> tuple[list[tuple[Gate, tuple[Qubit, ...]]], np.ndarray]:
    """Matrix decomposition to prepare a fermionic Gaussian state under the Jordan-Wigner Transform.

    Reference: `arXiv:1711.05395`_

    .. _arXiv:1711.05395: https://arxiv.org/abs/1711.05395

    Args:
        register: The register containing the qubits to use
        transformation_matrix: The transformation matrix describing
            the fermionic Gaussian state

    Returns:
        - givens rotation decomposition
        - left unitary left over from the decomposition
    """
    n, _ = transformation_matrix.shape

    # transform matrix to align with exposition in arXiv:1711.05395
    left = transformation_matrix[:, :n]
    right = transformation_matrix[:, n:]
    current_matrix = np.block([right.conj(), left.conj()]).astype(complex, copy=False)

    # compute left_unitary
    left_unitary = np.eye(n, dtype=complex)
    for j in range(n - 1):
        # Zero out entries in column j
        for i in range(n - 1 - j):
            # Zero out entry in row i if needed
            if not np.isclose(current_matrix[i, j], 0.0):
                givens_mat = givens_matrix(
                    current_matrix[i + 1, j], current_matrix[i, j]
                )
                current_matrix = apply_matrix_to_slices(
                    current_matrix, givens_mat, [i + 1, i]
                )
                left_unitary = apply_matrix_to_slices(
                    left_unitary, givens_mat, [i + 1, i]
                )

    # decompose matrix into Givens rotations and particle-hole transformations
    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []
    for i in range(n):
        # zero out the columns in row i
        for j in range(n - 1 - i, n):
            if not np.isclose(current_matrix[i, j], 0.0):
                if j == n - 1:
                    # particle-hole transformation
                    decomposition.append((XGate(), (register[-1],)))
                    _swap_columns(current_matrix, n - 1, 2 * n - 1)
                else:
                    # compute Givens rotation
                    givens_mat = givens_matrix(
                        current_matrix[i, j + 1], current_matrix[i, j]
                    )
                    theta = np.arccos(np.real(givens_mat[0, 0]))
                    phi = np.angle(givens_mat[0, 1])
                    # add operations
                    decomposition.append(
                        (
                            XXPlusYYGate(2 * theta, phi - np.pi / 2),
                            (register[j], register[j + 1]),
                        )
                    )
                    # update matrix
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat,
                        [(Ellipsis, j + 1), (Ellipsis, j)],
                    )
                    current_matrix = apply_matrix_to_slices(
                        current_matrix,
                        givens_mat.conj(),
                        [(Ellipsis, n + j + 1), (Ellipsis, n + j)],
                    )

    for i in range(n):
        left_unitary[i] *= current_matrix[i, n + i].conj()

    return decomposition, left_unitary


def _swap_columns(matrix: np.ndarray, i: int, j: int) -> None:
    """Swap columns of a matrix, mutating it."""
    column_i = matrix[:, i].copy()
    column_j = matrix[:, j].copy()
    matrix[:, i], matrix[:, j] = column_j, column_i


def modified_cholesky(
    mat: np.ndarray, *, error_threshold: float = 1e-8, max_vecs: int | None = None
) -> np.ndarray:
    r"""Modified Cholesky decomposition.

    The modified Cholesky decomposition of a square matrix :math:`M` has the form

    .. math::

        M = \sum_{i=1}^N v_i v_i^\dagger

    where each :math:`v_i` is a vector. `M` must be positive definite.
    No checking is performed to verify whether `M` is positive definite.
    The number of terms :math:`N` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the `max_vecs` parameter specifies an optional upper bound
    on :math:`N`. The `max_vecs` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    References:
        - `arXiv:1711.02242`_

    Args:
        mat: The matrix to decompose.
        error_threshold: Threshold for allowed error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: The maximum number of vectors to include in the decomposition.

    Returns:
        The Cholesky vectors v_i assembled into a 2-dimensional Numpy array
        whose columns are the vectors.

    .. _arXiv:1711.02242: https://arxiv.org/abs/1711.02242
    """
    dim, _ = mat.shape

    if max_vecs is None:
        max_vecs = dim

    cholesky_vecs = np.zeros((dim, max_vecs + 1), dtype=mat.dtype)
    errors = np.real(np.diagonal(mat).copy())
    for index in range(max_vecs + 1):
        max_error_index = np.argmax(errors)
        max_error = errors[max_error_index]
        if max_error < error_threshold:
            break
        cholesky_vecs[:, index] = mat[:, max_error_index]
        if index:
            cholesky_vecs[:, index] -= (
                cholesky_vecs[:, 0:index]
                @ cholesky_vecs[max_error_index, 0:index].conj()
            )
        cholesky_vecs[:, index] /= np.sqrt(max_error)
        errors -= np.abs(cholesky_vecs[:, index]) ** 2

    return cholesky_vecs[:, :index]


def double_factorized(
    two_body_tensor: np.ndarray,
    *,
    error_threshold: float = 1e-8,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Double-factorized decomposition of a two-body tensor.

    The double-factorized decomposition is a representation of a two-body tensor
    :math:`h_{pqrs}` as

    .. math::

        h_{pqrs} = \sum_{t=1}^N \sum_{k\ell} U^{t}_{pk} U^{t}_{qk}
            Z^{t}_{k\ell} U^{t}_{r\ell} U^{t}_{s\ell}

    Here each :math:`U^{t}` is a unitary matrix, referred to as a "leaf tensor,"
    and each :math:`Z^{(t)}` is a Hermitian matrix, referred to as a "core tensor."

    The number of terms :math:`N` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the `max_vecs` parameter specifies an optional upper bound
    on :math:`N`. The `max_vecs` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    Args:
        two_body_tensor: The two-body tensor to decompose.
        error_threshold: Threshold for allowed error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: An optional limit on the number of terms to keep in the decomposition
            of the two-body tensor.

    Returns:
        The core tensors and the leaf tensors. Each list of tensors is collected into
        a numpy array, so this method returns a tuple of two numpy arrays,
        the first containing the core tensors and the second containing the leaf tensors.
        Each numpy array will have shape (t, n, n) where t is the rank of the
        decomposition and n is the number of orbitals.

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    """
    n_modes, _, _, _ = two_body_tensor.shape

    if max_vecs is None:
        max_vecs = n_modes * (n_modes + 1) // 2

    reshaped_tensor = np.reshape(two_body_tensor, (n_modes**2, n_modes**2))
    cholesky_vecs = modified_cholesky(
        reshaped_tensor, error_threshold=error_threshold, max_vecs=max_vecs
    )

    _, rank = cholesky_vecs.shape
    diag_coulomb_mats = np.zeros((rank, n_modes, n_modes))
    orbital_rotations = np.zeros((rank, n_modes, n_modes))
    for i in range(rank):
        mat = np.reshape(cholesky_vecs[:, i], (n_modes, n_modes))
        eigs, vecs = np.linalg.eigh(mat)
        diag_coulomb_mats[i] = np.outer(eigs, eigs.conj())
        orbital_rotations[i] = vecs

    return diag_coulomb_mats, orbital_rotations


def is_hermitian(mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determine if a matrix is approximately Hermitian.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is Hermitian within the given tolerance.
    """
    m, n = mat.shape
    return m == n and np.allclose(mat, mat.T.conj(), rtol=rtol, atol=atol)


def is_real_symmetric(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Determine if a matrix is real and approximately symmetric.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is real and symmetric within the given tolerance.
    """
    m, n = mat.shape
    return (
        m == n
        and np.all(np.isreal(mat))
        and np.allclose(mat, mat.T, rtol=rtol, atol=atol)
    )


def is_unitary(mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determine if a matrix is approximately unitary.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is unitary within the given tolerance.
    """
    m, n = mat.shape
    return m == n and np.allclose(mat @ mat.T.conj(), np.eye(m), rtol=rtol, atol=atol)


def is_orthogonal(mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determine if a matrix is approximately orthogonal.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is orthogonal within the given tolerance.
    """
    m, n = mat.shape
    return (
        m == n
        and np.all(np.isreal(mat))
        and np.allclose(mat @ mat.T, np.eye(m), rtol=rtol, atol=atol)
    )


def is_special_orthogonal(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Determine if a matrix is approximately special orthogonal.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is special orthogonal within the given tolerance.
    """
    m, n = mat.shape
    return (
        m == n
        and np.all(np.isreal(mat))
        and np.allclose(mat @ mat.T, np.eye(m), rtol=rtol, atol=atol)
        and (m == 0 or np.allclose(np.linalg.det(mat), 1, rtol=rtol, atol=atol))
    )


def jw_number_indices(n_modes: int, n_particles: int) -> list[int]:
    """Return the indices of states with a fixed number of particles.

    Args:
        n_modes: The number of modes.
        n_particles: The number of particles.

    Returns:
        The indices of the computational basis vectors.
    """
    return [
        sum(1 << i for i in occupied_orbitals)
        for occupied_orbitals in itertools.combinations(range(n_modes), n_particles)
    ]


def jw_number_restrict_matrix(
    mat: scipy.sparse.csr_array, n_particles: int
) -> scipy.sparse.csr_array:
    """Restrict a matrix to the subspace of a fixed particle number.

    Args:
        mat: The matrix.
        n_particles: The number of particles.

    Returns:
        The restricted matrix.
    """
    dim, _ = mat.shape
    n_modes = dim.bit_length() - 1
    indices = jw_number_indices(n_modes=n_modes, n_particles=n_particles)
    return mat[np.ix_(indices, indices)]
