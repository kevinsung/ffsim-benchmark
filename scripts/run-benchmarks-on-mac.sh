#!/bin/bash

uv run asv run --config asv.conf.json --bench "diag_coulomb.DiagCoulombEvoBenchmark" v0.0.74^!
uv run asv run --config asv.conf.multi.json --bench "diag_coulomb.DiagCoulombEvoBenchmark" v0.0.74^!
uv run asv run --config asv.conf.json --bench "quad_ham_evo.QuadHamEvoBenchmark" v0.0.74^!
uv run asv run --config asv.conf.multi.json --bench "quad_ham_evo.QuadHamEvoBenchmark" v0.0.74^!
uv run asv run --config asv.conf.json --bench "trotter.TrotterBenchmark" v0.0.74^!
uv run asv run --config asv.conf.multi.json --bench "trotter.TrotterBenchmark" v0.0.74^!
uv run asv run --config asv.conf.json --bench "mol_ham_action.MolecularHamiltonianActionRealBenchmark" v0.0.74^!