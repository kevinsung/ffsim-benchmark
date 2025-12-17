#!/bin/bash

# uv run asv run --bench "FermionOperatorBenchmark" v0.0.64^!
# uv run asv run --bench "OperatorActionBenchmark" v0.0.64^!
uv run asv run --bench "MolecularHamiltonianActionRealBenchmark" v0.0.64^!
uv run asv run --bench "OrbitalRotationBenchmark" v0.0.64^!
uv run asv run --bench "QuadHamEvoBenchmark" v0.0.64^!
uv run asv run --bench "TrotterBenchmark" v0.0.64^!
