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

"""Trotter Hamiltonian simulation via low rank decomposition."""

from __future__ import annotations

import abc
import itertools
from collections.abc import Iterator
from typing import Any, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.circuit.library import CPhaseGate, PhaseGate, RZZGate
from qiskit_nature.operators.second_quantization import QuadraticHamiltonian

from ffsim import DoubleFactorizedHamiltonian

from .bogoliubov_transform import BogoliubovTransformJW
from .swap_network import swap_network


class TrotterStep(abc.ABC):
    """Abstract class for a Trotter step."""

    @abc.abstractmethod
    def trotter_step(self, time: float) -> Iterator[CircuitInstruction]:
        """Perform a Trotter step."""

    @property
    @abc.abstractmethod
    def qubits(self) -> Sequence[Qubit]:
        """The qubits used by the Trotter step."""

    @property
    @abc.abstractmethod
    def control_qubits(self) -> Optional[Sequence[Qubit]]:
        """The control qubits used by the Trotter step."""


# TODO add higher-order Trotter formulas
def simulate_trotter(
    trotter_step: TrotterStep,
    time: float,
    n_steps: int = 1,
) -> Iterator[CircuitInstruction]:
    step_time = time / n_steps
    for _ in range(n_steps):
        yield from trotter_step.trotter_step(step_time)


class AsymmetricLowRankTrotterStepJW(TrotterStep):
    r"""Trotter Hamiltonian simulation via low rank decomposition.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    """

    def __init__(
        self,
        qubits: Sequence[Qubit],
        hamiltonian: DoubleFactorizedHamiltonian,
        *,
        swap_network: bool = False,
        control_qubits: Optional[Sequence[Qubit]] = None,
    ) -> None:
        """
        Args:
            hamiltonian: The Hamiltonian to simulate.
            qubits: The qubits to use.
            control_qubits: The control qubits to use, if a controlled Trotter step
                is desired.
            swap_network: Whether to use the "swap network" strategy for implementing
                the two-body terms.
        """
        if swap_network and not hamiltonian.z_representation:
            raise NotImplementedError(
                "Swap network strategy is only implemented for Z representation."
            )
        self.hamiltonian = hamiltonian
        self.swap_network = swap_network
        self._qubits = qubits
        self._control_qubits = control_qubits

    @property
    def qubits(self) -> Sequence[Qubit]:
        return self._qubits

    @property
    def control_qubits(self) -> Optional[Sequence[Qubit]]:
        return self._control_qubits

    def trotter_step(self, time: float) -> Iterator[CircuitInstruction]:
        if self.control_qubits:
            yield from self._trotter_step_controlled(time)
        else:
            yield from self._trotter_step(time)

    def _trotter_step(self, time: float) -> Iterator[CircuitInstruction]:
        yield from _simulate_one_body(
            self.qubits,
            one_body_tensor=self.hamiltonian.one_body_tensor,
            time=time,
        )
        for orbital_rotation, diag_coulomb_mat in zip(
            self.hamiltonian.orbital_rotations, self.hamiltonian.diag_coulomb_mats
        ):
            if self.swap_network:
                yield from _simulate_two_body_swap_network_z_representation(
                    self.qubits,
                    orbital_rotation=orbital_rotation,
                    diag_coulomb_mat=diag_coulomb_mat,
                    time=time,
                )
                self._qubits = self._qubits[::-1]
            else:
                if self.hamiltonian.z_representation:
                    yield from _simulate_two_body_z_representation(
                        self.qubits,
                        orbital_rotation=orbital_rotation,
                        diag_coulomb_mat=diag_coulomb_mat,
                        time=time,
                    )
                else:
                    yield from _simulate_two_body(
                        self.qubits,
                        orbital_rotation=orbital_rotation,
                        diag_coulomb_mat=diag_coulomb_mat,
                        time=time,
                    )

    def _trotter_step_controlled(self, time: float) -> Iterator[CircuitInstruction]:
        yield from _simulate_one_body_controlled(
            self.qubits,
            one_body_tensor=self.hamiltonian.one_body_tensor,
            time=time,
            control_qubits=self.control_qubits,
        )
        for orbital_rotation, diag_coulomb_mat in zip(
            self.hamiltonian.orbital_rotations, self.hamiltonian.diag_coulomb_mats
        ):
            if self.swap_network:
                yield from _simulate_two_body_controlled_swap_network(
                    self.qubits,
                    orbital_rotation=orbital_rotation,
                    diag_coulomb_mat=diag_coulomb_mat,
                    time=time,
                    control_qubits=self.control_qubits,
                )
                self._qubits = self._qubits[::-1]
            else:
                if self.hamiltonian.z_representation:
                    yield from _simulate_two_body_controlled_z_representation(
                        self.qubits,
                        orbital_rotation=orbital_rotation,
                        diag_coulomb_mat=diag_coulomb_mat,
                        time=time,
                        control_qubits=self.control_qubits,
                    )
                else:
                    yield from _simulate_two_body_controlled(
                        self.qubits,
                        orbital_rotation=orbital_rotation,
                        diag_coulomb_mat=diag_coulomb_mat,
                        time=time,
                        control_qubits=self.control_qubits,
                    )

        # apply phase from constant term
        n_control_qubits = len(self.control_qubits)
        if n_control_qubits == 1:
            yield CircuitInstruction(
                PhaseGate(-self.hamiltonian.constant * time),
                self.control_qubits,
            )
        else:
            yield CircuitInstruction(
                PhaseGate(-self.hamiltonian.constant * time).control(
                    n_control_qubits - 1
                ),
                self.control_qubits,
            )


def _simulate_one_body(
    qubits: Sequence[Qubit], one_body_tensor: np.ndarray, time: float
) -> Iterator[CircuitInstruction]:
    n_qubits = len(qubits)
    n_modes = n_qubits // 2

    transformation_matrix, orbital_energies, _ = QuadraticHamiltonian(
        one_body_tensor
    ).diagonalizing_bogoliubov_transform()
    bog_circuit = BogoliubovTransformJW(transformation_matrix.T.conj())
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])
    for i in range(n_modes):
        phase_gate = PhaseGate(-orbital_energies[i] * time)
        yield CircuitInstruction(phase_gate, (qubits[i],))
        yield CircuitInstruction(phase_gate, (qubits[n_modes + i],))
    bog_circuit = BogoliubovTransformJW(transformation_matrix)
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])


def _simulate_one_body_controlled(
    qubits: Sequence[Qubit],
    one_body_tensor: np.ndarray,
    time: float,
    control_qubits: Sequence[Qubit],
) -> Iterator[CircuitInstruction]:
    n_qubits = len(qubits)
    n_modes = n_qubits // 2
    n_control_qubits = len(control_qubits)

    transformation_matrix, orbital_energies, _ = QuadraticHamiltonian(
        one_body_tensor
    ).diagonalizing_bogoliubov_transform()
    bog_circuit = BogoliubovTransformJW(transformation_matrix.T.conj())
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])
    for i in range(n_modes):
        phase_gate = PhaseGate(-orbital_energies[i] * time).control(n_control_qubits)
        yield CircuitInstruction(phase_gate, list(control_qubits) + [qubits[i]])
        yield CircuitInstruction(
            phase_gate, list(control_qubits) + [qubits[n_modes + i]]
        )
    bog_circuit = BogoliubovTransformJW(transformation_matrix)
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])


def _simulate_two_body(
    qubits: Sequence[Qubit],
    orbital_rotation: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    time: float,
) -> Iterator[CircuitInstruction]:
    n_qubits = len(qubits)
    n_modes = n_qubits // 2

    bog_circuit = BogoliubovTransformJW(orbital_rotation.conj())
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])
    for i, j in itertools.combinations_with_replacement(range(n_modes), 2):
        coeff = 0.5 if i == j else 1
        for sigma in range(2):
            yield CircuitInstruction(
                CPhaseGate(-coeff * diag_coulomb_mat[i % n_modes, j % n_modes] * time),
                (
                    qubits[i + sigma * n_modes],
                    qubits[j + (1 - sigma) * n_modes],
                ),
            )
            if i == j:
                yield CircuitInstruction(
                    PhaseGate(
                        -coeff * diag_coulomb_mat[i % n_modes, j % n_modes] * time
                    ),
                    (qubits[i + sigma * n_modes],),
                )
            else:
                yield CircuitInstruction(
                    CPhaseGate(
                        -coeff * diag_coulomb_mat[i % n_modes, j % n_modes] * time
                    ),
                    (
                        qubits[i + sigma * n_modes],
                        qubits[j + sigma * n_modes],
                    ),
                )
    bog_circuit = BogoliubovTransformJW(orbital_rotation.T)
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])


def _simulate_two_body_z_representation(
    qubits: Sequence[Qubit],
    orbital_rotation: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    time: float,
) -> Iterator[CircuitInstruction]:
    n_qubits = len(qubits)
    n_modes = n_qubits // 2

    bog_circuit = BogoliubovTransformJW(orbital_rotation.conj())
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])
    for i, j in itertools.combinations(range(n_qubits), 2):
        yield CircuitInstruction(
            RZZGate(0.5 * diag_coulomb_mat[i % n_modes, j % n_modes] * time),
            (
                qubits[i],
                qubits[j],
            ),
        )
    bog_circuit = BogoliubovTransformJW(orbital_rotation.T)
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])


def _simulate_two_body_swap_network_z_representation(
    qubits: Sequence[Qubit],
    orbital_rotation: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    time: float,
) -> Iterator[CircuitInstruction]:
    n_qubits = len(qubits)
    n_modes = n_qubits // 2

    bog = BogoliubovTransformJW(orbital_rotation.conj())
    yield CircuitInstruction(bog, qubits[:n_modes])
    yield CircuitInstruction(bog, qubits[n_modes:])

    yield from swap_network(
        qubits,
        lambda i, j, a, b: CircuitInstruction(
            RZZGate(0.5 * diag_coulomb_mat[i % n_modes, j % n_modes] * time), (a, b)
        ),
    )
    qubits = qubits[::-1]

    bog = BogoliubovTransformJW(orbital_rotation.T)
    yield CircuitInstruction(bog, qubits[:n_modes])
    yield CircuitInstruction(bog, qubits[n_modes:])


def _simulate_two_body_controlled(
    qubits: Sequence[Qubit],
    orbital_rotation: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    time: float,
    control_qubits: Sequence[Qubit],
) -> Iterator[CircuitInstruction]:
    n_qubits = len(qubits)
    n_modes = n_qubits // 2
    n_control_qubits = len(control_qubits)

    bog_circuit = BogoliubovTransformJW(orbital_rotation.conj())
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])
    for i, j in itertools.combinations_with_replacement(range(n_modes), 2):
        coeff = 0.5 if i == j else 1
        for sigma in range(2):
            yield CircuitInstruction(
                CPhaseGate(
                    -coeff * diag_coulomb_mat[i % n_modes, j % n_modes] * time
                ).control(n_control_qubits),
                list(control_qubits)
                + [
                    qubits[i + sigma * n_modes],
                    qubits[j + (1 - sigma) * n_modes],
                ],
            )
            if i == j:
                yield CircuitInstruction(
                    PhaseGate(
                        -coeff * diag_coulomb_mat[i % n_modes, j % n_modes] * time
                    ).control(n_control_qubits),
                    list(control_qubits) + [qubits[i + sigma * n_modes]],
                )
            else:
                yield CircuitInstruction(
                    CPhaseGate(
                        -coeff * diag_coulomb_mat[i % n_modes, j % n_modes] * time
                    ).control(n_control_qubits),
                    list(control_qubits)
                    + [
                        qubits[i + sigma * n_modes],
                        qubits[j + sigma * n_modes],
                    ],
                )
    bog_circuit = BogoliubovTransformJW(orbital_rotation.T)
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])


def _simulate_two_body_controlled_z_representation(
    qubits: Sequence[Qubit],
    orbital_rotation: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    time: float,
    control_qubits: Sequence[Qubit],
) -> Iterator[CircuitInstruction]:
    n_qubits = len(qubits)
    n_modes = n_qubits // 2
    n_control_qubits = len(control_qubits)

    bog_circuit = BogoliubovTransformJW(orbital_rotation.conj())
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])
    for i, j in itertools.combinations(range(n_qubits), 2):
        yield CircuitInstruction(
            RZZGate(0.5 * diag_coulomb_mat[i % n_modes, j % n_modes] * time).control(
                n_control_qubits
            ),
            list(control_qubits)
            + [
                qubits[i],
                qubits[j],
            ],
        )
    bog_circuit = BogoliubovTransformJW(orbital_rotation.T)
    yield CircuitInstruction(bog_circuit, qubits[:n_modes])
    yield CircuitInstruction(bog_circuit, qubits[n_modes:])


def _simulate_two_body_controlled_swap_network(
    qubits: Sequence[Qubit],
    orbital_rotation: np.ndarray,
    diag_coulomb_mat: np.ndarray,
    time: float,
    control_qubits: Sequence[Qubit],
) -> Iterator[CircuitInstruction]:
    n_qubits = len(qubits)
    n_modes = n_qubits // 2
    n_control_qubits = len(control_qubits)

    bog = BogoliubovTransformJW(orbital_rotation.conj())
    yield CircuitInstruction(bog, qubits[:n_modes])
    yield CircuitInstruction(bog, qubits[n_modes:])

    yield from swap_network(
        qubits,
        lambda i, j, a, b: CircuitInstruction(
            RZZGate(0.5 * diag_coulomb_mat[i % n_modes, j % n_modes] * time).control(
                n_control_qubits
            ),
            list(control_qubits) + [a, b],
        ),
    )
    qubits = qubits[::-1]

    bog = BogoliubovTransformJW(orbital_rotation.T)
    yield CircuitInstruction(bog, qubits[:n_modes])
    yield CircuitInstruction(bog, qubits[n_modes:])
