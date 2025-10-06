from pathlib import Path
import sys
import unittest
from pathlib import Path
import csv
import subprocess
from typing import List, Dict, Tuple


ROOT = Path(__file__).resolve().parents[2]
print(f"ROOT: {ROOT}")
pyc_path = ROOT / "bytecode_3_11" / "election.pyc"
print(f"pyc_path: {pyc_path}")

print("="*30)

TEST_DIR = Path(__file__).resolve().parents[1]   
print(f"TEST_DIR: {TEST_DIR}")
INPUT_DIR = TEST_DIR / "test_input"
print(f"INPUT_DIR: {INPUT_DIR}")

OUTPUT_DIR = TEST_DIR / "test_output"
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

print("="*30)

def run_program(input_csv: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    riding_output = OUTPUT_DIR / "riding_results.csv"
    federal_output = OUTPUT_DIR / "federal_results.csv"

    cp = subprocess.run(
        ["python3.11", str(pyc_path), str(input_csv), str(riding_output), str(federal_output)],
        cwd=OUTPUT_DIR,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(cp.stderr)

    riding_rows, federal_rows = [], []
    if riding_output.exists():
        with open(riding_output, newline="") as f:
            riding_rows = list(csv.DictReader(f))
    if federal_output.exists():
        with open(federal_output, newline="") as f:
            federal_rows = list(csv.DictReader(f))
    return riding_rows, federal_rows


def check_recount_threshold(csv_name: str, expect_recount: bool):
    """
    Runs election.pyc with the given CSV and verifies the recount threshold rule:
      - If margin ≤ 0.1%, Outcome must be 'Recount'
      - If margin >  0.1%, Outcome must be 'Plurality'

    Prints a failure message in the required format if behavior is incorrect.
    """
    input_csv = INPUT_DIR / 'recount' / csv_name
    riding_rows, _ = run_program(input_csv)

    if len(riding_rows) != 1:
        print(f"Failure: (Output shape) {csv_name}: Expected 1 riding, got {len(riding_rows)}")
        return

    row = riding_rows[0]
    try:
        margin = float(row["%margin"])
    except Exception:
        print(f"Failure: (Field parsing) {csv_name}: Missing or invalid '%margin' field")
        return

    outcome = (row.get("Outcome") or "").strip().lower()

    # Rule check
    recount_rule_holds = (margin <= 0.1)
    if expect_recount and outcome != "recount":
        print(f"Failure: (Recount rule) {csv_name}: margin={margin:.6f}% → Expected 'Recount', got '{row.get('Outcome')}'")
    if not expect_recount and outcome == "recount":
        print(f"Failure: (Recount rule) {csv_name}: margin={margin:.6f}% → Expected 'Plurality', got 'Recount'")


class TestRecountThreshold(unittest.TestCase):


    def test_recount_below(self):
        check_recount_threshold("recount_below_0p1.csv", expect_recount=True)

    def test_recount_equal(self):
        check_recount_threshold("recount_equal_0p1.csv", expect_recount=True)

    def test_recount_above(self):
        check_recount_threshold("recount_above_0p1.csv", expect_recount=False)


def check_single_candidate(csv_name: str, expect_recount: bool):
    input_csv = INPUT_DIR / 'single_candidate' / csv_name
    riding_rows, _ = run_program(input_csv)

    if len(riding_rows) != 1:
        print(f"Failure: (Output shape) {csv_name}: Expected 1 riding, got {len(riding_rows)}")
        return

    row = riding_rows[0]
    outcome = (row.get("Outcome") or "").strip().lower()

    if outcome != "plurality":
        print(f"Failure: (Single candidate rule) {csv_name}: Expected 'Plurality' for single candidate, got '{row.get('Outcome')}'")


class TestSingleCandidate(unittest.TestCase):

    def test_single_candidate(self):
        check_single_candidate("single_candidate.csv", expect_recount=False)


def check_invalid_votes(csv_name: str):
    """
    Runs election.pyc with malformed vote counts.
    Expected: the program should reject or flag invalid votes, 
    not silently produce normal results.
    """
    input_csv = INPUT_DIR / 'invalid_votes' / csv_name
    riding_rows, _ = run_program(input_csv)

    # Case 1: no output produced
    if not riding_rows:
        print(f"Failure: (Vote validation) {csv_name}: Program produced no riding output (possible crash)")
        return

    # Case 2: program silently accepted bad votes
    for row in riding_rows:
        try:
            votes = float(row.get("Votes", ""))
            if votes < 0 or not float(votes).is_integer():
                print(f"Failure: (Vote validation) {csv_name}: Invalid vote values accepted ({votes})")
        except Exception:
            # If 'Votes' field missing or non-numeric, it’s OK — means it caught the issue
            continue


class TestInvaliVotes(unittest.TestCase):
    def test_float_votes(self):
        check_invalid_votes("votes_float_equal.csv")

    def test_negatve_votes(self):
        check_invalid_votes("votes_negative_equal.csv")

    def test_nonnum_votes(self):
        check_invalid_votes("votes_nonnum_equal.csv")


def check_tie_votes(csv_name: str):
    """
    Runs election.pyc with malformed vote counts.
    Expected: the program should reject or flag invalid votes, 
    not silently produce normal results.
    """
    input_csv = INPUT_DIR / 'tie' / csv_name
    riding_rows, _ = run_program(input_csv)

    if len(riding_rows) != 1:
            print(f"Failure: (Output shape) {csv_name}: Expected 1 riding, got {len(riding_rows)}")
            return

    row = riding_rows[0]
    outcome = (row.get("Outcome") or "").strip().lower()
    if outcome != "recount":
        print(f"Failure: (Tie rule) {csv_name}: Expected 'Recount' for tied votes, got '{row.get('Outcome')}'")


class TestTie(unittest.TestCase):
    def test_tie_votes(self):
        check_tie_votes("tie_equal.csv")


def check_party_name(csv_name: str):
    """
    “Liberal”, “liberal”, and “ Liberal ” should aggregate to the same party.
    """
    input_csv = INPUT_DIR / "party_names" / csv_name
    _, federal_rows = run_program(input_csv)

    if not federal_rows:
        print(f"Failure: (Party normalization) {csv_name}: No federal results")
        return

    parties = { (r.get("Party") or "").strip().lower() for r in federal_rows }
    if len(parties) > 2:  # should be {liberal, conservative} at most; here only liberal and maybe others
        print(f"Failure: (Party normalization) {csv_name}: Party casing/spacing not normalized: {parties}")

class TestPartyNames(unittest.TestCase):

    def test_party_names(self):
        check_party_name("party_names.csv")


def check_riding_id_name_mismatch(csv_name: str):
    input_csv = INPUT_DIR / "riding_id_name_equal" / csv_name
    riding_rows, federal_rows = run_program(input_csv)

    # Many implementations abort and produce no outputs on this validation error.
    # If the program still generates normal-looking outputs, flag it.
    if riding_rows or federal_rows:
        print(f"Failure: (Riding ID-name consistency) {csv_name}: Produced outputs despite ID↔name mismatch")


class TestRidingIdNameConsistency(unittest.TestCase):
    def test_id_name_mismatch(self):
        check_riding_id_name_mismatch("different_name.csv")

    def test_id_name_mismatch(self):
        check_riding_id_name_mismatch("different_id.csv")

if __name__ == "__main__":

    print("="*10 + ' Attributes ' + "="*10)
    