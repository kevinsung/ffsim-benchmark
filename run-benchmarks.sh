#!/bin/bash

latest_tag=$(curl -s "https://api.github.com/repos/qiskit-community/ffsim/tags" | jq -r ".[0].name")
echo "Latest tag: $latest_tag"
asv run --skip-existing $latest_tag..main
asv run --skip-existing "--tags='v[0-9]*.[0-9]*.[0-9]*' --no-walk"
asv gh-pages