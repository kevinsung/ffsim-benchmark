#!/bin/bash

asv run --bench "FermionOperatorBenchmark" v0.0.47^!
asv run --bench "OperatorActionBenchmark" v0.0.47^!
asv run --bench "OrbitalRotationBenchmark" v0.0.47^!
asv run --bench "QuadHamEvoBenchmark" v0.0.47^!
asv run --bench "TrotterBenchmark" v0.0.47^!
