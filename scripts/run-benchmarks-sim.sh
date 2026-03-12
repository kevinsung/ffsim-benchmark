#!/bin/bash

uv run asv run --config asv.conf.json --bench "QuadHamEvoBenchmark" v0.0.70^!
uv run asv run --config asv.conf.multi.json --bench "QuadHamEvoBenchmark" v0.0.70^!
uv run asv run --config asv.conf.json --bench "TrotterBenchmark" v0.0.70^!
uv run asv run --config asv.conf.multi.json --bench "TrotterBenchmark" v0.0.70^!
uv run asv run --config asv.conf.json --bench "MolecularHamiltonianActionRealBenchmark" v0.0.70^!
uv run asv run --config asv.conf.multi.json --bench "MolecularHamiltonianActionRealBenchmark" v0.0.70^!

# uv run asv run --bench "FermionOperatorBenchmark" v0.0.69^!
# uv run asv run --bench "OperatorActionBenchmark" v0.0.69^!
# uv run asv run --bench "OrbitalRotationBenchmark" v0.0.69^!