#!/bin/bash

uv run asv run --config asv.conf.json --bench "diag_coulomb.DiagCoulombEvoBenchmark.*\(16, 0.5\)" v0.0.74^!
uv run asv run --config asv.conf.multi.json --bench "diag_coulomb.DiagCoulombEvoBenchmark.*\(16, 0.5\)" v0.0.74^!
uv run asv run --config asv.conf.json --bench "quad_ham_evo.QuadHamEvoBenchmark.*\(16, 0.5\)" v0.0.74^!
uv run asv run --config asv.conf.multi.json --bench "quad_ham_evo.QuadHamEvoBenchmark.*\(16, 0.5\)" v0.0.74^!
uv run asv run --config asv.conf.json --bench "trotter.TrotterBenchmark.*\(16, 0.5\)" v0.0.74^!
uv run asv run --config asv.conf.multi.json --bench "trotter.TrotterBenchmark.*\(16, 0.5\)" v0.0.74^!
uv run asv run --config asv.conf.json --bench "mol_ham_action.MolecularHamiltonianActionRealBenchmark.*\(16, 0.5\)" v0.0.74^!
uv run asv run --config asv.conf.multi.json --bench "mol_ham_action.MolecularHamiltonianActionRealBenchmark.*\(16, 0.5\)" v0.0.74^!