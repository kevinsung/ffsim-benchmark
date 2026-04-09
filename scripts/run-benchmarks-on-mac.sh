#!/bin/bash

uv run asv run --config asv.conf.json --bench "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_ffsim" v0.0.76^!
uv run asv run --config asv.conf.json --bench "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_fqe" v0.0.76^!
uv run asv run --config asv.conf.multi.json --bench "diag_coulomb.DiagCoulombEvoBenchmark.time_diag_coulomb_evolution_ffsim" v0.0.76^!

uv run asv run --config asv.conf.json --bench "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim" v0.0.76^!
uv run asv run --config asv.conf.json --bench "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_fqe" v0.0.76^!
uv run asv run --config asv.conf.multi.json --bench "quad_ham_evo.QuadHamEvoBenchmark.time_quad_ham_evolution_ffsim" v0.0.76^!

uv run asv run --config asv.conf.json --bench "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim" v0.0.76^!
uv run asv run --config asv.conf.json --bench "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_fqe" v0.0.76^!
uv run asv run --config asv.conf.multi.json --bench "trotter.TrotterBenchmark.time_simulate_trotter_double_factorized_ffsim" v0.0.76^!

uv run asv run --config asv.conf.json --bench "mol_ham_action.MolecularHamiltonianActionRealBenchmark" v0.0.76^!