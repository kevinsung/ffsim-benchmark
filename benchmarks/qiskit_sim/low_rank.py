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

"""Low rank decomposition utilities."""

from __future__ import annotations

import dataclasses
import itertools
from collections import defaultdict
from typing import Any, Optional, cast

import numpy as np
import scipy.linalg
import scipy.optimize
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators.tensor_ordering import to_chemist_ordering

from qiskit_sim.linalg import double_factorized


def one_body_tensor_to_fermionic_op(
    one_body_tensor: np.ndarray, expand_spin: bool = False
) -> FermionicOp:
    """Convert a one-body tensor to a FermionicOp.

    Args:
        one_body_tensor: The one-body tensor.

    Returns:
        The FermionicOp corresponding to the one-body tensor.
    """
    n_modes, _ = one_body_tensor.shape
    data = {}
    for p, q in itertools.product(range(n_modes), repeat=2):
        coeff = one_body_tensor[p, q]
        for sigma in range(1 + expand_spin):
            data[f"+_{p + sigma * n_modes} -_{q + sigma * n_modes}"] = coeff
    return FermionicOp(data)


def two_body_tensor_to_fermionic_op(
    two_body_tensor: np.ndarray, expand_spin: bool = False
) -> FermionicOp:
    """Convert a two-body tensor to a FermionicOp.

    Args:
        two_body_tensor: The two-body tensor.

    Returns:
        The FermionicOp corresponding to the two-body tensor.
    """
    n_modes, _, _, _ = two_body_tensor.shape
    data = defaultdict(float)
    for p, q, r, s in itertools.product(range(n_modes), repeat=4):
        coeff = two_body_tensor[p, q, r, s]
        for sigma, tau in itertools.product(range(1 + expand_spin), repeat=2):
            data[
                f"+_{p + sigma * n_modes} -_{q + sigma * n_modes} +_{r + tau * n_modes} -_{s + tau * n_modes}"
            ] += (0.5 * coeff)
    return FermionicOp(data)


def two_body_term_to_fermionic_op(
    diag_coulomb_mat: np.ndarray,
    orbital_rotation: np.ndarray,
    z_representation: bool = False,
) -> FermionicOp:
    """Convert a two-body term to a FermionicOp.

    Args:
        orbital_rotation: The leaf tensor.
        diag_coulomb_mat: The core tensor.
        z_representation: Whether to use the "Z" representation
            (see :func:`~.low_rank_decomposition` for details).

    Returns:
        The FermionicOp corresponding to the two-body term.
    """
    op = FermionicOp.zero()
    n_modes, _ = orbital_rotation.shape
    num_ops = []
    for sigma, i in itertools.product(range(2), range(n_modes)):
        data = {}
        for p, q in itertools.product(range(n_modes), repeat=2):
            data[f"+_{p + sigma * n_modes} -_{q + sigma * n_modes}"] = (
                orbital_rotation[p, i] * orbital_rotation[q, i].conj()
            )
        num_ops.append(FermionicOp(data))
    if z_representation:
        z_ops = [FermionicOp.one() - 2 * num_op for num_op in num_ops]
        for a, b in itertools.combinations(range(2 * n_modes), 2):
            sigma, i = divmod(a, n_modes)
            tau, j = divmod(b, n_modes)
            # TODO: this cast should be unnecessary
            op += cast(
                FermionicOp,
                # TODO remove cast to float once
                # https://github.com/Qiskit/qiskit-nature/issues/953
                # is fixed
                0.25
                * float(diag_coulomb_mat[i, j])
                * z_ops[i + sigma * n_modes]
                @ z_ops[j + tau * n_modes],
            )
    else:
        for i, j in itertools.product(range(n_modes), repeat=2):
            for sigma, tau in itertools.product(range(2), repeat=2):
                op += (
                    0.5
                    # TODO remove cast to float once
                    # https://github.com/Qiskit/qiskit-nature/issues/953
                    # is fixed
                    * float(diag_coulomb_mat[i, j])
                    * num_ops[i + sigma * n_modes]
                    @ num_ops[j + tau * n_modes]
                )
    return op


def one_body_square_decomposition(
    diag_coulomb_mat: np.ndarray,
    orbital_rotation: np.ndarray | None = None,
    truncation_threshold: float = 1e-12,
) -> np.ndarray:
    """Decompose a two-body term as a sum of squared one-body operators.

    Args:
        diag_coulomb_mat: The core tensor.
        orbital_rotation: The leaf tensor.
        truncation_threshold: Eigenvalues of the core tensor whose absolute value
            is less than this value are truncated.
    """
    if orbital_rotation is None:
        orbital_rotation = np.eye(diag_coulomb_mat.shape[0])
    eigs, vecs = np.linalg.eigh(diag_coulomb_mat)
    index = np.abs(eigs) >= truncation_threshold
    eigs = eigs[index]
    vecs = vecs[:, index]
    return np.einsum(
        "t,it,ji,ki->tjk",
        np.emath.sqrt(0.5 * eigs),
        vecs,
        orbital_rotation,
        orbital_rotation.conj(),
    )


@dataclasses.dataclass
class DoubleFactorizedHamiltonian:
    """A Hamiltonian in the double-factorized form of the low rank decomposition.
    See :func:`~.low_rank_decomposition` for a description of the data
    stored in this class.

    Attributes:
        one_body_tensor: The one-body tensor.
        orbital_rotations: The leaf tensors.
        diag_coulomb_mats: The core tensors.
        constant: The constant.
        z_representation: Whether the Hamiltonian is in the "Z" representation.
    """

    one_body_tensor: np.ndarray
    diag_coulomb_mats: np.ndarray
    orbital_rotations: np.ndarray
    constant: float = 0.0
    z_representation: bool = False

    @property
    def n_orbitals(self):
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    @property
    def two_body_tensor(self):
        """The two-body tensor."""
        return np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            self.orbital_rotations,
            self.orbital_rotations,
            self.diag_coulomb_mats,
            self.orbital_rotations,
            self.orbital_rotations,
        )

    def to_fermionic_op(self) -> FermionicOp:
        """Return a FermionicOp representing the Hamiltonian."""
        one_body_tensor = self.one_body_tensor
        two_body_tensor = self.two_body_tensor
        constant = self.constant

        if self.z_representation:
            one_body_tensor = one_body_tensor - 0.5 * (
                np.einsum(
                    "tij,tpi,tqi->pq",
                    self.diag_coulomb_mats,
                    self.orbital_rotations,
                    self.orbital_rotations.conj(),
                )
                + np.einsum(
                    "tij,tpj,tqj->pq",
                    self.diag_coulomb_mats,
                    self.orbital_rotations,
                    self.orbital_rotations.conj(),
                )
            )
            constant -= 0.25 * np.einsum(
                "ijj->", self.diag_coulomb_mats
            ) - 0.5 * np.sum(self.diag_coulomb_mats)

        op = one_body_tensor_to_fermionic_op(one_body_tensor, expand_spin=True)
        op += two_body_tensor_to_fermionic_op(two_body_tensor, expand_spin=True)

        return op

    def to_z_representation(self) -> DoubleFactorizedHamiltonian:
        """Return the Hamiltonian in the "Z" representation."""
        if self.z_representation:
            return self

        one_body_correction, constant_correction = _low_rank_z_representation(
            self.diag_coulomb_mats, self.orbital_rotations
        )
        return DoubleFactorizedHamiltonian(
            one_body_tensor=self.one_body_tensor + one_body_correction,
            diag_coulomb_mats=self.diag_coulomb_mats,
            orbital_rotations=self.orbital_rotations,
            constant=self.constant + constant_correction,
            z_representation=True,
        )

    def to_number_representation(self) -> DoubleFactorizedHamiltonian:
        """Return the Hamiltonian in the "number" representation."""
        if not self.z_representation:
            return self

        one_body_correction, constant_correction = _low_rank_z_representation(
            self.diag_coulomb_mats, self.orbital_rotations
        )
        return DoubleFactorizedHamiltonian(
            one_body_tensor=self.one_body_tensor - one_body_correction,
            diag_coulomb_mats=self.diag_coulomb_mats,
            orbital_rotations=self.orbital_rotations,
            constant=self.constant - constant_correction,
            z_representation=False,
        )


def low_rank_decomposition(
    hamiltonian: ElectronicEnergy,
    *,
    error_threshold: float = 1e-8,
    max_vecs: Optional[int] = None,
    z_representation: bool = False,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    options: Optional[dict] = None,
    diag_coulomb_mat_mask: Optional[np.ndarray] = None,
    seed: Any = None,
) -> DoubleFactorizedHamiltonian:
    r"""Low rank decomposition of a molecular Hamiltonian.

    The low rank decomposition acts on a Hamiltonian of the form

    .. math::

        H = \sum_{pq, \sigma} h_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
            + \frac12 \sum_{pqrs, \sigma} h_{pqrs, \sigma\tau}
            a^\dagger_{p, \sigma} a^\dagger_{r, \tau} a_{s, \tau} a_{q, \sigma}.

    The Hamiltonian is decomposed into the double-factorized form

    .. math::

        H = \sum_{pq, \sigma} \kappa_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \frac12 \sum_t \sum_{ij, \sigma\tau} Z^{(t)}_{ij} n^{(t)}_{i, \sigma} n^{t}_{j, \tau}
        + \text{constant}.

    where

    .. math::

        n^{(t)}_{i, \sigma} = \sum_{pq} U^{(t)}_{pi}
        a^\dagger_{p, \sigma} a^\dagger_{q, \sigma} U^{(t)}_{qi}.

    Here :math:`U^{(t)}_{ij}` and :math:`Z^{(t)}_{ij}` are tensors that are output by the decomposition,
    and :math:`\kappa_{pq}` is an updated one-body tensor.
    Each matrix :math:`U^{(t)}` is guaranteed to be unitary so that the :math:`n^{(t)}_{i, \sigma}` are
    number operators in a rotated basis.
    The number of terms :math:`t` in the decomposition depends on the allowed
    error threshold. A larger error threshold leads to a smaller number of terms.
    Furthermore, the `max_vecs` parameter specifies an optional upper bound
    on :math:`t`.

    The default behavior of this routine is to perform a straightforward
    "exact" factorization of the two-body tensor based on a nested
    eigenvalue decomposition. Additionally, one can choose to optimize the
    coefficients stored in the tensor to achieve a "compressed" factorization.
    This option is enabled by setting the `optimize` parameter to `True`.
    The optimization attempts to minimize a least-squares objective function
    quantifying the error in the low rank decomposition.
    It uses `scipy.optimize.minimize`, passing both the objective function
    and its gradient. The core tensors returned by the optimization can be optionally constrained
    to have only certain elements allowed to be nonzero. This is achieved by passing the
    `diag_coulomb_mat_mask` parameter, which is an :math:`N \times N` matrix of boolean values
    where :math:`N` is the number of orbitals. The nonzero elements of this matrix indicate
    where the core tensors are allowed to be nonzero. Only the upper triangular part of the
    matrix is used because the core tensors are symmetric.

    **"Z" representation**

    The "Z" representation of the low rank decomposition is an alternative
    decomposition that sometimes yields simpler quantum circuits.

    Under the Jordan-Wigner transformation, the number operators take the form

    .. math::

        n^{(t)}_{i, \sigma} = \frac{(1 - z^{(t)}_{i, \sigma})}{2}

    where :math:`z^{(t)}_{i, \sigma}` is the Pauli Z operator in the rotated basis.
    The "Z" representation is obtained by rewriting the two-body part in terms
    of these Pauli Z operators:

    .. math::

        H = \sum_{pq, \sigma} \kappa_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \sum_{pq, \sigma} \tilde{\kappa}_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \frac18 \sum_t \sum_{ij, \sigma\tau}^* Z^{(t)}_{ij} z^{(t)}_{i, \sigma} z^{t}_{j, \tau}
        + \text{constant}

    where the asterisk denotes summation over indices $ij, \sigma\tau$
    where $i \neq j$ or $\sigma \neq \tau$.
    Here :math:`\tilde{\kappa}_{pq}` is a correction to the one-body term.

    Note: Currently, only real-valued two-body tensors are supported.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    Args:
        hamiltonian: The Hamiltonian to decompose.
        error_threshold: Threshold for allowed error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: An optional limit on the number of terms to keep in the decomposition
            of the two-body tensor.
        z_representation: Whether to use the "Z" representation of the
            low rank decomposition.
        optimize: Whether to optimize the tensors returned by the decomposition.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        diag_coulomb_mat_mask: Core tensor mask to use in the optimization. This is a matrix of
            boolean values where the nonzero elements indicate where the core tensors returned by
            optimization are allowed to be nonzero. This parameter is only used if `optimize` is
            set to `True`, and only the upper triangular part of the matrix is used.
        seed: The pseudorandom number generator or seed. Randomness is used to generate
            an initial guess for the optimization.
            Should be an instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        An instance of DoubleFactorizedHamiltonian which stores the decomposition in
        the attributes `one_body_tensor`, `orbital_rotations`, `diag_coulomb_mats`,
        and `constant.

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    .. _scipy.optimize.minimize:
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    one_body_tensor = hamiltonian.electronic_integrals.alpha["+-"].copy()
    # TODO is this copy necessary?
    two_body_tensor = to_chemist_ordering(
        hamiltonian.electronic_integrals.alpha["++--"].copy()
    )

    one_body_tensor -= 0.5 * np.einsum("prqr", two_body_tensor)
    # TODO get constant from ElectronicEnergy
    constant = 0.0

    if optimize:
        (
            diag_coulomb_mats,
            orbital_rotations,
        ) = _low_rank_compressed_two_body_decomposition(
            two_body_tensor,
            max_vecs=max_vecs,
            error_threshold=error_threshold,
            method=method,
            options=options,
            diag_coulomb_mat_mask=diag_coulomb_mat_mask,
            seed=seed,
        )
    else:
        diag_coulomb_mats, orbital_rotations = double_factorized(
            two_body_tensor, max_vecs=max_vecs, error_threshold=error_threshold
        )

    df_hamiltonian = DoubleFactorizedHamiltonian(
        one_body_tensor=one_body_tensor,
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        constant=constant,
    )

    if z_representation:
        df_hamiltonian = df_hamiltonian.to_z_representation()

    return df_hamiltonian


def _low_rank_z_representation(
    diag_coulomb_mats: np.ndarray, orbital_rotations: np.ndarray
) -> tuple[np.ndarray, float]:
    one_body_correction = 0.5 * (
        np.einsum(
            "tij,tpi,tqi->pq",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
        )
        + np.einsum(
            "tij,tpj,tqj->pq",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
        )
    )
    constant_correction = 0.25 * np.einsum("ijj->", diag_coulomb_mats) - 0.5 * np.sum(
        diag_coulomb_mats
    )
    return one_body_correction, constant_correction


def _low_rank_optimal_diag_coulomb_mats(
    two_body_tensor: np.ndarray,
    orbital_rotations: np.ndarray,
    cutoff_threshold: float = 1e-8,
) -> np.ndarray:
    """Compute optimal low rank core tensors given fixed leaf tensors."""
    n_modes, _, _, _ = two_body_tensor.shape
    n_tensors, _, _ = orbital_rotations.shape

    dim = n_tensors * n_modes**2
    target = np.einsum(
        "pqrs,tpk,tqk,trl,tsl->tkl",
        two_body_tensor,
        orbital_rotations,
        orbital_rotations,
        orbital_rotations,
        orbital_rotations,
    )
    target = np.reshape(target, (dim,))
    coeffs = np.zeros((n_tensors, n_modes, n_modes, n_tensors, n_modes, n_modes))
    for i in range(n_tensors):
        for j in range(i, n_tensors):
            metric = (orbital_rotations[i].T @ orbital_rotations[j]) ** 2
            coeffs[i, :, :, j, :, :] = np.einsum("kl,mn->kmln", metric, metric)
            coeffs[j, :, :, i, :, :] = np.einsum("kl,mn->kmln", metric.T, metric.T)
    coeffs = np.reshape(coeffs, (dim, dim))

    eigs, vecs = np.linalg.eigh(coeffs)
    pseudoinverse = np.zeros_like(eigs)
    pseudoinverse[eigs > cutoff_threshold] = eigs[eigs > cutoff_threshold] ** -1
    solution = vecs @ (vecs.T @ target * pseudoinverse)

    return np.reshape(solution, (n_tensors, n_modes, n_modes))


def _low_rank_compressed_two_body_decomposition(  # pylint: disable=invalid-name
    two_body_tensor: np.ndarray,
    *,
    error_threshold: float = 1e-8,
    max_vecs: Optional[int] = None,
    method="L-BFGS-B",
    options: Optional[dict] = None,
    diag_coulomb_mat_mask: Optional[np.ndarray] = None,
    seed: Any = None,
):
    rng = np.random.default_rng(seed)
    _, orbital_rotations = double_factorized(
        two_body_tensor, error_threshold=error_threshold, max_vecs=max_vecs
    )
    n_tensors, n_modes, _ = orbital_rotations.shape
    if diag_coulomb_mat_mask is None:
        diag_coulomb_mat_mask = np.ones((n_modes, n_modes), dtype=bool)
    diag_coulomb_mat_mask = np.triu(diag_coulomb_mat_mask)

    def fun(x):
        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            x, n_tensors, n_modes, diag_coulomb_mat_mask
        )
        diff = two_body_tensor - np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            orbital_rotations,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
        )
        return 0.5 * np.sum(diff**2)

    def jac(x):
        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            x, n_tensors, n_modes, diag_coulomb_mat_mask
        )
        diff = two_body_tensor - np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            orbital_rotations,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
        )
        grad_leaf = -4 * np.einsum(
            "pqrs,tqk,tkl,trl,tsl->tpk",
            diff,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
        )
        leaf_logs = _params_to_leaf_logs(x, n_tensors, n_modes)
        grad_leaf_log = np.ravel(
            [_grad_leaf_log(log, grad) for log, grad in zip(leaf_logs, grad_leaf)]
        )
        grad_core = -2 * np.einsum(
            "pqrs,tpk,tqk,trl,tsl->tkl",
            diff,
            orbital_rotations,
            orbital_rotations,
            orbital_rotations,
            orbital_rotations,
        )
        grad_core[:, range(n_modes), range(n_modes)] /= 2
        param_indices = np.nonzero(diag_coulomb_mat_mask)
        grad_core = np.ravel([mat[param_indices] for mat in grad_core])
        return np.concatenate([grad_leaf_log, grad_core])

    diag_coulomb_mats = _low_rank_optimal_diag_coulomb_mats(
        two_body_tensor, orbital_rotations
    )
    x0 = _df_tensors_to_params(
        diag_coulomb_mats, orbital_rotations, diag_coulomb_mat_mask
    )
    x0 += 1e-2 * rng.standard_normal(size=x0.shape)
    result = scipy.optimize.minimize(fun, x0, method=method, jac=jac, options=options)
    diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
        result.x, n_tensors, n_modes, diag_coulomb_mat_mask
    )

    return diag_coulomb_mats, orbital_rotations


def _df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_mat_mask: np.ndarray,
):
    _, n_modes, _ = orbital_rotations.shape
    leaf_logs = [scipy.linalg.logm(mat) for mat in orbital_rotations]
    leaf_param_indices = np.triu_indices(n_modes, k=1)
    # TODO this discards the imaginary part of the logarithm, see if we can do better
    leaf_params = np.real(
        np.ravel([leaf_log[leaf_param_indices] for leaf_log in leaf_logs])
    )
    core_param_indices = np.nonzero(diag_coulomb_mat_mask)
    core_params = np.ravel(
        [diag_coulomb_mat[core_param_indices] for diag_coulomb_mat in diag_coulomb_mats]
    )
    return np.concatenate([leaf_params, core_params])


def _params_to_leaf_logs(params: np.ndarray, n_tensors: int, n_modes: int):
    leaf_logs = np.zeros((n_tensors, n_modes, n_modes))
    triu_indices = np.triu_indices(n_modes, k=1)
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_logs[i][triu_indices] = params[i * param_length : (i + 1) * param_length]
        leaf_logs[i] -= leaf_logs[i].T
    return leaf_logs


def _params_to_df_tensors(
    params: np.ndarray, n_tensors: int, n_modes: int, diag_coulomb_mat_mask: np.ndarray
):
    leaf_logs = _params_to_leaf_logs(params, n_tensors, n_modes)
    orbital_rotations = np.array([_expm_antisymmetric(mat) for mat in leaf_logs])

    n_leaf_params = n_tensors * n_modes * (n_modes - 1) // 2
    core_params = np.real(params[n_leaf_params:])
    param_indices = np.nonzero(diag_coulomb_mat_mask)
    param_length = len(param_indices[0])
    diag_coulomb_mats = np.zeros((n_tensors, n_modes, n_modes))
    for i in range(n_tensors):
        diag_coulomb_mats[i][param_indices] = core_params[
            i * param_length : (i + 1) * param_length
        ]
        diag_coulomb_mats[i] += diag_coulomb_mats[i].T
        diag_coulomb_mats[i][range(n_modes), range(n_modes)] /= 2
    return diag_coulomb_mats, orbital_rotations


def _expm_antisymmetric(mat: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mat)
    return np.real(vecs @ np.diag(np.exp(1j * eigs)) @ vecs.T.conj())


def _grad_leaf_log(mat: np.ndarray, grad_leaf: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mat)
    eig_i, eig_j = np.meshgrid(eigs, eigs, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        coeffs = -1j * (np.exp(1j * eig_i) - np.exp(1j * eig_j)) / (eig_i - eig_j)
    coeffs[eig_i == eig_j] = np.exp(1j * eig_i[eig_i == eig_j])
    grad = vecs.conj() @ (vecs.T @ grad_leaf @ vecs.conj() * coeffs) @ vecs.T
    grad -= grad.T
    n_modes, _ = mat.shape
    triu_indices = np.triu_indices(n_modes, k=1)
    return np.real(grad[triu_indices])
