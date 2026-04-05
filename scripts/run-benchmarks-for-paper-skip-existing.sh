#!/bin/bash

uv run asv run --skip-existing --config asv.conf.json --bench "DiagCoulombEvoBenchmark" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.multi.json --bench "DiagCoulombEvoBenchmark" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.json --bench "QuadHamEvoBenchmark" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.multi.json --bench "QuadHamEvoBenchmark" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.json --bench "TrotterBenchmark" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.multi.json --bench "TrotterBenchmark" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.json --bench "MolecularHamiltonianActionRealBenchmark" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.multi.json --bench "MolecularHamiltonianActionRealBenchmark" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.json --bench "SampleSlaterBenchmarkReal" v0.0.74^!
uv run asv run --skip-existing --config asv.conf.json --bench "FermionOperatorBenchmark" v0.0.74^!
