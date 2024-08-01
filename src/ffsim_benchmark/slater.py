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


class SampleSlaterDeterminantBenchmark:
    """Benchmark sampling Slater determinant."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (32, 64, 96),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        # set benchmark parameters
        self.norb = norb
        self.nelec = int(norb * filling_fraction)
        self.shots = 1000

        # initialize test objects
        self.rng = np.random.default_rng(1889)
        orbital_rotation = ffsim.random.random_unitary(norb, seed=self.rng)
        occupied_orbitals = ffsim.testing.random_occupied_orbitals(
            self.norb, self.nelec, seed=self.rng
        )
        self.rdm = ffsim.slater_determinant_rdms(
            norb, occupied_orbitals, orbital_rotation
        )

    def time_sample_slater_determinant_ffsim(self, *_):
        _ = ffsim.sample_slater_determinant(
            self.rdm,
            self.norb,
            self.nelec,
            shots=self.shots,
            bitstring_type=ffsim.BitstringType.INT,
            seed=self.rng,
        )
