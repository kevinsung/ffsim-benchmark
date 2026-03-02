#!/bin/bash

uv run asv run --config asv.conf.json --bench "SampleSlaterBenchmark" v0.0.69^!
