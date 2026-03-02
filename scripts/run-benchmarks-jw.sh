#!/bin/bash

uv run asv run --config asv.conf.json --bench "JordanWignerBenchmark" v0.0.69^!
uv run asv run --config asv.conf.multi.json --bench "JordanWignerBenchmark" v0.0.69^!