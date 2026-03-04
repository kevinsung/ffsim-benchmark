# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections.abc import Sequence

import numpy as np
from dppy.finite_dpps import FiniteDPP

import ffsim


def sample_slater_dppy(
    norb: int,
    occupied_orbitals: Sequence[int],
    orbital_rotation: np.ndarray,
    mode: str,
    shots: int = 1,
):
    rdm = ffsim.slater_determinant_rdms(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    dpp = FiniteDPP(
        kernel_type="likelihood",
        projection=False,
        L=rdm,
    )
    for _ in range(shots):
        dpp.sample_exact_k_dpp(len(occupied_orbitals), mode=mode)
    return [sum(1 << orb for orb in sample) for sample in dpp.list_of_samples]


class SampleSlaterBenchmarkReal:
    """Benchmark sampling Slater determinant."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (50, 100, 200, 400),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        # set benchmark parameters
        self.norb = norb
        self.nelec = int(norb * filling_fraction)
        self.shots = 1_000

        # initialize test objects
        self.rng = np.random.default_rng(238951493330509724797921217923308475371)
        self.orbital_rotation = ffsim.random.random_orthogonal(norb, seed=self.rng)
        self.occupied_orbitals = ffsim.testing.random_occupied_orbitals(
            self.norb, self.nelec, seed=self.rng
        )

    def time_sample_slater_real_ffsim(self, *_):
        _ = ffsim.sample_slater(
            self.norb,
            self.occupied_orbitals,
            self.orbital_rotation,
            shots=self.shots,
            bitstring_type=ffsim.BitstringType.INT,
            seed=self.rng,
        )

    def time_sample_slater_real_gs_dppy(self, *_):
        _ = sample_slater_dppy(
            self.norb,
            self.occupied_orbitals,
            self.orbital_rotation,
            mode="GS",
            shots=self.shots,
        )


class SampleSlaterBenchmarkComplex:
    """Benchmark sampling Slater determinant."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (50, 100, 200, 400),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        # set benchmark parameters
        self.norb = norb
        self.nelec = int(norb * filling_fraction)
        self.shots = 1_000

        # initialize test objects
        self.rng = np.random.default_rng(238951493330509724797921217923308475371)
        self.orbital_rotation = ffsim.random.random_unitary(norb, seed=self.rng)
        self.occupied_orbitals = ffsim.testing.random_occupied_orbitals(
            self.norb, self.nelec, seed=self.rng
        )

    def time_sample_slater_complex_ffsim(self, *_):
        _ = ffsim.sample_slater(
            self.norb,
            self.occupied_orbitals,
            self.orbital_rotation,
            shots=self.shots,
            bitstring_type=ffsim.BitstringType.INT,
            seed=self.rng,
        )
