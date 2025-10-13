#!/bin/bash
# run_tests.sh
# Usage: ./run_tests.sh ../bytecode_3_11/election.pyc

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_tests.sh <path_to_election.pyc>"
    exit 1
fi

FILE_NAME=$(basename "$1")

if [ "$FILE_NAME" != "election.pyc" ]; then
    echo "Wrong file to test"
    exit 1
fi

echo "Running tests using: $1"
python3 src/test_runner.py "$1"