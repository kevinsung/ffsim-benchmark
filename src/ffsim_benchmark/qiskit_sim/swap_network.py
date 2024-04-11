# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Swap network."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Callable, Sequence

from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    Qubit,
)
from qiskit.circuit.library import SwapGate


def swap_network(
    qubits: Sequence[Qubit],
    operation: Callable[[int, int, Qubit, Qubit], CircuitInstruction] | None = None,
    swap_gate: Gate | None = None,
    offset: bool = False,
) -> Iterator[CircuitInstruction]:
    """A swap network circuit.

    A swap network applies arbitrary pairwise interactions between qubits
    using only linear connectivity. It works by reversing the order of qubits
    with a sequence of swap gates and applying an operation when the relevant
    qubits become adjacent. Note that at the end of the operation
    the qubit ordering has been reversed; this must be kept track of.

    Reference: `arXiv:1711.04789`_

    .. _arXiv:1711.04789: https://arxiv.org/abs/1711.04789

    Args:
        register: The qubits to use.
        operation: Returns interactions to perform between qubits as
            they are swapped past each other. A call to this function takes the
            form ``operation(i, j, a, b)`` where ``i`` and ``j`` are indices
            representing the logical qubits as they were initially ordered,
            and ``a`` and ``b`` are the physical qubits containing those
            logical qubits (in that order).
            It returns the instruction to perform on the qubits.
        swap_gate: The swap gate to use. Defaults to the normal SWAP gate (an instance
            of :class:`~.SwapGate`).
        offset: If True, then qubit 0 will participate in odd-numbered layers
            instead of even-numbered layers.
    """
    swap_gate = swap_gate or SwapGate()
    n_qubits = len(qubits)
    order = list(range(n_qubits))
    for layer_num in range(n_qubits):
        lowest_active_qubit = (layer_num + offset) % 2
        active_pairs = ((i, i + 1) for i in range(lowest_active_qubit, n_qubits - 1, 2))
        for a, b in active_pairs:
            i, j = order[a], order[b]
            i_qubit, j_qubit = qubits[a], qubits[b]
            if operation:
                yield operation(i, j, i_qubit, j_qubit)
            yield CircuitInstruction(swap_gate, (i_qubit, j_qubit))
            order[a], order[b] = order[b], order[a]
