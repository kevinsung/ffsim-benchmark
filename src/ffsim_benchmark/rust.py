# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

import ffsim
from ffsim._lib import apply_givens_rotation_in_place
from ffsim._slow.gates.orbital_rotation import apply_givens_rotation_in_place_slow
from ffsim.gates.orbital_rotation import _zero_one_subspace_indices


class RustBenchmark:
    """Benchmark Rust functions."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)

        rng = np.random.default_rng()

        self.vec = ffsim.random.random_statevector(
            ffsim.dim(self.norb, self.nelec), seed=rng
        )
        dim_a, dim_b = ffsim.dims(self.norb, self.nelec)
        self.vec_as_mat = self.vec.reshape((dim_a, dim_b))

        indices = _zero_one_subspace_indices(self.norb, self.nelec[0], (1, 2))
        self.slice1 = indices[: len(indices) // 2]
        self.slice2 = indices[len(indices) // 2 :]

    def time_apply_givens_rotation_in_place_python_ffsim(self, *_):
        apply_givens_rotation_in_place_slow(
            self.vec_as_mat,
            c=0.5,
            s=(1j) ** 0.5 * np.sqrt(0.75),
            slice1=self.slice1,
            slice2=self.slice2,
        )

    def time_apply_givens_rotation_in_place_rust_ffsim(self, *_):
        apply_givens_rotation_in_place(
            self.vec_as_mat,
            c=0.5,
            s=(1j) ** 0.5 * np.sqrt(0.75),
            slice1=self.slice1,
            slice2=self.slice2,
        )
