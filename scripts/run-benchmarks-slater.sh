#!/bin/bash

uv run asv run --config asv.conf.json --bench "SampleSlaterBenchmarkReal" v0.0.70^!
