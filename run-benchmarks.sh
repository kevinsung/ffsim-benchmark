#!/bin/bash

latest_tag=$(curl -s "https://api.github.com/repos/qiskit-community/ffsim/tags" | jq -r ".[0].name")
echo "Latest tag: $latest_tag"
# asv run $latest_tag..main
# asv run --bench "_openfermion" main^!
# asv run --bench "_fqe" main^!
# asv run --bench "_qiskit" main^!
# asv run --bench "_ffsim" 762f435f..91ff7433
asv run --bench "_ffsim" main^!
asv run --bench "_ffsim" "--max-count=20 --tags='v[0-9]*.[0-9]*.[0-9]*' --no-walk"
asv gh-pages
