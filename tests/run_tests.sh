#!/usr/bin/env bash
set -e

pushd "$(dirname "$0")"

echo "Running all tests"
python3 test_deepFCD.py $@

popd