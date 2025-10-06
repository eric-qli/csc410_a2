from pathlib import Path
import sys
import unittest
from pathlib import Path
import csv
import subprocess
from typing import List, Dict, Tuple

import unittest.test


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

    for path in (riding_output, federal_output):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

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
    recount_rule_holds = (margin < 0.1)
    if expect_recount and outcome != "recount":
        print(f"Failure: (Recount rule) {csv_name}: margin={margin:.6f}% → Expected 'Recount', got '{row.get('Outcome')}'")
    if not expect_recount and outcome == "recount":
        print(f"Failure: (Recount rule) {csv_name}: margin={margin:.6f}% → Expected 'Plurality', got 'Recount'")


class TestRecountThreshold(unittest.TestCase):


    def test_recount_below(self):
        check_recount_threshold("recount_below_0p1.csv", expect_recount=True)

    def test_recount_equal(self):
        check_recount_threshold("recount_equal_0p1.csv", expect_recount=False)

    def test_recount_above(self):
        check_recount_threshold("recount_above_0p1.csv", expect_recount=False)


def check_single_candidate(csv_name: str):
    input_csv = INPUT_DIR / 'single_candidate' / csv_name
    riding_rows, _ = run_program(input_csv)

    if len(riding_rows) != 1:
        print(f"Failure: (Output shape) {csv_name}: Expected 1 riding, got {len(riding_rows)}")
        return


class TestSingleCandidate(unittest.TestCase):

    def test_single_candidate(self):
        check_single_candidate("single_candidate.csv")


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
        # print(f"Failure: (Vote validation) {csv_name}: Program produced no riding output (possible crash)")
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
        return

    row = riding_rows[0]
    outcome = (row.get("Outcome") or "").strip().lower()
    if outcome != "recount":
        print(f"Failure: (Tie rule) {csv_name}: Expected 'Recount' for tied votes, got '{row.get('Outcome')}'")


class TestTie(unittest.TestCase):
    def test_tie_votes(self):
        check_tie_votes("tie_equal.csv")

    def test_tie_votes_zero(self):
        check_tie_votes("zero_tie.csv")


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
    def test_id_name_mismatch_name(self):
        check_riding_id_name_mismatch("different_name.csv")

    # there is a bug
    def test_id_name_mismatch_id(self):
        check_riding_id_name_mismatch("different_id.csv")


def check_bad_header(csv_name):
    rows,_=run_program(INPUT_DIR/"bad_headers"/csv_name)
    if rows: print(f"Failure: (Header) {csv_name}: produced output despite bad header")


class TestBadHeader(unittest.TestCase):

    def test_bad_header(self):
        check_bad_header("bad_header_equal.csv")


def check_missmatch_row_length(csv_name):
    input_csv = INPUT_DIR / "row_length_mismatch" / csv_name
    _, federal_rows = run_program(input_csv)

    # Many implementations abort and produce no outputs on this validation error.
    # If the program still generates normal-looking outputs, flag it.
    if federal_rows:
        print(f"Failure: (Missing value check) {csv_name}: Produced outputs despite there are null in cells")

class TestMissMatchRowLength(unittest.TestCase):
    def test_missmatch_row_length(self):
        check_missmatch_row_length("row_length_less.csv")


def check_duplicate_candidates(csv_name):
    input_csv = INPUT_DIR / "duplicate_candidate" / csv_name
    _, federal_rows = run_program(input_csv)

    # Many implementations abort and produce no outputs on this validation error.
    # If the program still generates normal-looking outputs, flag it.
    if federal_rows:
        print(f"Failure: (Duplicate candidates) {csv_name}: Produced outputs despite there are duplicate candidates under different parties")


class TestDuplicateCandidates(unittest.TestCase):
    def test_duplicate_candidates(self):
        check_duplicate_candidates("duplicate_candidate.csv")



def normalization(s: str) -> str:
    return (s or "").strip().lower()

def check_federal_roles(csv_name: str, expected_governing_party: str | None, expected_role: str | None, expect_opposition: str | None = None):
    input_csv = INPUT_DIR / "federal_seats" / csv_name
    _, federal_rows = run_program(input_csv)

    if not federal_rows:
        print(f"Failure: (Federal results) {csv_name}: No federal results produced")
        return
    
    no_total_lst = [p for p in federal_rows if normalization(p.get("Party")) != "total"]

    parties = {}
    for r in no_total_lst:
        p = normalization(r.get("Party"))
        seats = r.get("Seats") or r.get("Seat") or ""
        try:
            seats_i = int(str(seats).strip())
        except Exception:
            seats_i = None
        role = normalization(r.get("Role"))
        parties[p] = {"seats": seats_i, "role": role}

    # Helper to find government & opposition rows
    gov_rows = [(p, info) for p, info in parties.items() if "government" in info["role"]]
    opp_rows = [(p, info) for p, info in parties.items() if "opposition" in info["role"]]

    print()

    # tie
    if expected_governing_party is None or expected_role is None:
        if not parties or any(v["seats"] is None for v in parties.values()):
            return  
        max_seats = max(v["seats"] for v in parties.values())
        leaders = [p for p,v in parties.items() if v["seats"] == max_seats]

        # if there is a tie at top, ensure NO "government" role appears
        if len(leaders) >= 2:
            gov_labeled = [(p, v["role"]) for p, v in parties.items() if "government" in v["role"]]

            roles =  set([role for _, role in gov_labeled])
            for a in roles:
                if 'tie' not in a:
                    print(f"Failure: (Federal tie) {csv_name}: Expected 'tie' in role, got '{a[1]}'")
        else:
            print(f"Failure: (Federal tie) {csv_name}: No tie detected in seat counts")
        return


    # Government should be present and match expected party + role
    if not gov_rows:
        print(f"Failure: (Government role) {csv_name}: No party labeled as government")
        return

    gov_party, gov_info = gov_rows[0]  # assume single government row
    if gov_party != normalization(expected_governing_party):
        print(f"Failure: (Government party) {csv_name}: Expected '{expected_governing_party}', got '{gov_party}'")
    if expected_role not in gov_info["role"]:
        print(f"Failure: (Government role) {csv_name}: Expected '{expected_role}', got '{gov_info['role']}'")

    # Seat sanity: majority iff seats > total/2
    if gov_info["seats"] is not None and all(i["seats"] is not None for i in parties.values()):
        total_seats = sum(i["seats"] for i in parties.values())
        max_other = max(i["seats"] for p,i in parties.items() if p != gov_party) if len(parties)>1 else 0
        gov_seats = gov_info["seats"]
        if expected_role == "majority":
            if not (gov_seats > total_seats / 2):
                print(f"Failure: (Seat math) {csv_name}: Labeled majority but seats={gov_seats}/{total_seats}")
        elif expected_role == "minority":
            if not (gov_seats > max_other and not (gov_seats > total_seats / 2)):
                print(f"Failure: (Seat math) {csv_name}: Labeled minority but seats={gov_seats}, max_other={max_other}, total={total_seats}")

    # Optional: check official opposition
    if expect_opposition is not None:
        if not opp_rows:
            print(f"Failure: (Official opposition) {csv_name}: No opposition row found")
        else:
            opp_party, _ = opp_rows[0]
            if opp_party != normalization(expect_opposition):
                print(f"Failure: (Official opposition) {csv_name}: Expected '{expect_opposition}', got '{opp_party}'")


class TestFederalRoles(unittest.TestCase):
    def test_federal_majority(self):
        # Liberal majority; opposition should be Conservative (unique second)
        check_federal_roles("federal_majority.csv", expected_governing_party="Liberal", expected_role="majority", expect_opposition="Conservative")

    def test_federal_minority(self):
        # Liberal wins most but not > half (minority). Don't assert opposition uniqueness here.
        check_federal_roles("federal_minority.csv", expected_governing_party="Liberal", expected_role="minority")

    def test_federal_tie(self):
        # Liberal wins most but not > half (tie). Don't assert opposition uniqueness here.
        check_federal_roles("federal_tie.csv", expected_governing_party=None, expected_role=None)


if __name__ == "__main__":

    print("="*10 + ' Attributes ' + "="*10)
    