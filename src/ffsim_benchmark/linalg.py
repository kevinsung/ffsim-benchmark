# (C) Copyright IBM 2026.
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


class GivensDecompositionBenchmark:
    """Benchmark Givens decomposition of a unitary matrix."""

    param_names = ["norb"]
    params = [(2, 8, 64, 512)]

    def setup(self, norb: int):
        rng = np.random.default_rng(143607686584498699986780850514370601378)
        self.mat = ffsim.random.random_unitary(norb, seed=rng)

    def time_givens_decomposition_ffsim(self, *_):
        _ = ffsim.linalg.givens_decomposition(self.mat)
