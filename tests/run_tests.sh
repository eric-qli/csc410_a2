#!/bin/bash
# Run tests for a2 project

MODULE_NAME="src.test_runner"  # or "src.test_runner" if you run inside /tests

# hardcode your class names here in order
classes=(TestRecountThreshold TestSingleCandidate TestInvaliVotes TestTie TestPartyNames TestRidingIdNameConsistency TestBadHeader TestMissMatchRowLength 
TestDuplicateCandidates TestFederalRoles)

if [ $# -eq 0 ]; then
  echo "Running all tests..."
  python -m unittest "$MODULE_NAME"
  exit 0
fi

index=$1
if (( index < 1 || index > ${#classes[@]} )); then
  echo "Invalid input. Choose from:"
  for i in "${!classes[@]}"; do
    printf "  %d) %s\n" $((i+1)) "${classes[$i]}"
  done
  exit 1
fi

class_name="${classes[$((index-1))]}"
echo "Running test class: $class_name"
python -m unittest "$MODULE_NAME.$class_name"