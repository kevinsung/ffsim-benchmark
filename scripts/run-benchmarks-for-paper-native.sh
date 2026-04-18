#!/bin/bash

uv run asv run --config asv.conf.json --bench "DiagCoulombEvoBenchmark" cmake-native^!
uv run asv run --config asv.conf.multi.json --bench "DiagCoulombEvoBenchmark" cmake-native^!
uv run asv run --config asv.conf.json --bench "QuadHamEvoBenchmark" cmake-native^!
uv run asv run --config asv.conf.multi.json --bench "QuadHamEvoBenchmark" cmake-native^!
uv run asv run --config asv.conf.json --bench "TrotterBenchmark" cmake-native^!
uv run asv run --config asv.conf.multi.json --bench "TrotterBenchmark" cmake-native^!
uv run asv run --config asv.conf.json --bench "MolecularHamiltonianActionRealBenchmark" cmake-native^!
uv run asv run --config asv.conf.multi.json --bench "MolecularHamiltonianActionRealBenchmark" cmake-native^!
uv run asv run --config asv.conf.json --bench "SampleSlaterBenchmarkReal" cmake-native^!
uv run asv run --config asv.conf.json --bench "FermionOperatorBenchmark" cmake-native^!
