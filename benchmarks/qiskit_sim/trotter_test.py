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

"""Test low rank Trotter step."""

import unittest

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
import qiskit_nature.settings
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from ddt import data, ddt, unpack
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import (
    Statevector,
    random_hermitian,
    random_statevector,
    state_fidelity,
)
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit_sim.low_rank import low_rank_decomposition
from qiskit_sim.random_utils import random_two_body_tensor_real
from qiskit_sim.trotter import AsymmetricLowRankTrotterStepJW, simulate_trotter

rng = np.random.default_rng(9315)
hamiltonians = {}


for n_modes, rank in [(2, 3)]:
    one_body_tensor = np.array(random_hermitian(n_modes, seed=rng))
    two_body_tensor = random_two_body_tensor_real(n_modes, rank=rank, seed=rng)
    electronic_energy = ElectronicEnergy.from_raw_integrals(
        one_body_tensor, two_body_tensor
    )
    hamiltonians[f"random_{n_modes}_{rank}"] = electronic_energy


def expectation(operator: scipy.sparse.spmatrix, state: np.ndarray) -> complex:
    """Expectation value of operator with state."""
    return np.vdot(state, operator @ state)


def variance(operator: scipy.sparse.spmatrix, state: np.ndarray) -> complex:
    """Variance of operator with state."""
    return expectation(operator @ operator, state) - expectation(operator, state) ** 2


@ddt
class TestTrotter(unittest.TestCase):
    """Tests for Trotter Hamiltonian simulation."""

    @unpack
    @data(
        ("random_2_3", False, 0.1, 20, False, 0.9999, 1e-2),
        ("random_2_3", True, 0.1, 10, False, 0.9999, 5e-3),
    )
    def test_asymmetric_low_rank_trotter_step(
        self,
        hamiltonian_name: str,
        z_representation: bool,
        time: float,
        n_steps: int,
        swap_network: bool,
        target_fidelity: float,
        atol: float,
    ):
        """Test asymmetric low rank Trotter step."""
        electronic_energy = hamiltonians[hamiltonian_name]
        hamiltonian = electronic_energy.second_q_op()
        df_hamiltonian = low_rank_decomposition(
            electronic_energy, z_representation=z_representation
        )
        n_modes = 2 * df_hamiltonian.n_orbitals

        # generate random initial state
        initial_state = random_statevector(2**n_modes, seed=1612)

        # simulate exact evolution
        hamiltonian_sparse = JordanWignerMapper().map(hamiltonian).to_spmatrix()
        exact_state = scipy.sparse.linalg.expm_multiply(
            -1j * time * hamiltonian_sparse, np.array(initial_state)
        )

        # make sure time is not too small
        self.assertLess(state_fidelity(exact_state, initial_state), 0.98)

        # simulate Trotter evolution
        qubits = QuantumRegister(n_modes)
        trotter_step = AsymmetricLowRankTrotterStepJW(
            qubits, df_hamiltonian, swap_network=swap_network
        )
        circuit = QuantumCircuit(qubits)
        circuit.set_statevector(initial_state)
        for instruction in simulate_trotter(
            trotter_step,
            time,
            n_steps=n_steps,
        ):
            circuit.append(instruction)
        circuit.save_state()
        circuit.global_phase = -time * trotter_step.hamiltonian.constant

        qubit_map = {q: i for i, q in enumerate(trotter_step.qubits)}
        # TODO should this be the inverse permutation due to Qiskit qubit ordering?
        permutation = [qubit_map[q] for q in qubits]
        simulator = AerSimulator()
        transpiled = transpile(circuit, simulator)
        final_state = simulator.run(transpiled).result().get_statevector()
        final_state = (
            np.array(final_state)
            .reshape((2,) * n_modes)
            .transpose(permutation)
            .reshape((2**n_modes,))
        )
        fidelity = state_fidelity(final_state, exact_state)
        self.assertGreater(fidelity, target_fidelity)
