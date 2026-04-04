#!/bin/bash

uv run asv run --config asv.conf.json --bench time_apply_orbital_rotation_ffsim e805660b7~2..e805660b7
uv run asv run --config asv.conf.multi.json --bench time_apply_orbital_rotation_ffsim e805660b7~2..e805660b7