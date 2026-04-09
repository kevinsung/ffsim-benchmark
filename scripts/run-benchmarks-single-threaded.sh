#!/bin/bash

uv run asv run --config asv.conf.json --bench "DiagCoulombEvoBenchmark" v0.0.76^!
uv run asv run --config asv.conf.json --bench "QuadHamEvoBenchmark" v0.0.76^!
uv run asv run --config asv.conf.json --bench "TrotterBenchmark" v0.0.76^!