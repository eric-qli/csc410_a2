from pathlib import Path
import sys
import unittest
from pathlib import Path
import csv
import subprocess
from typing import List, Dict, Tuple

import unittest.test


from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../a2
pyc_path = (ROOT / "bytecode_3_11" / "election.pyc").resolve()

if not pyc_path.exists():
    print(f"Error: expected .pyc at {pyc_path} but it was not found")
    raise SystemExit(1)

print(f"ROOT: {ROOT}")
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


# R1
def check_input_csv(csv_name: str):
    """
    Run the program with an invalid input file.
    Passing behavior: program raises CalledProcessError (non-zero exit).
    Failing behavior: program runs successfully (returns without error).
    """
    input_csv = INPUT_DIR / "input" / csv_name

    try:
        # Run the program; if it succeeds, that’s a failure
        riding_rows, federal_rows = run_program(input_csv)
    except subprocess.CalledProcessError:
        # ✅ Expected behavior — program exited with error
        pass
    except Exception as e:
        # Any other unexpected exception → fail
        print(f"Failure: (Validation) {csv_name}: Unexpected exception type {type(e).__name__}: {e}")


class TestInputCSV(unittest.TestCase):
    def test_empty_input(self):
        check_input_csv("empty_input.csv")


# R2
def check_bad_header(csv_name):
    rows,_=run_program(INPUT_DIR/"bad_headers"/csv_name)
    if rows: print(f"Failure: (Header) {csv_name}: produced output despite bad header")


class TestBadHeader(unittest.TestCase):

    def test_bad_header(self):
        check_bad_header("bad_header_equal.csv")


# R3
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


# R4
def check_invalid_ridingnum(csv_name: str):
    """
    Spec R4:
      - Each row's RidingNum must be a positive integer.
      - Invalid values (0, negative, non-integer) must cause the program to reject the file.
    Passing behavior:
      Program exits with CalledProcessError (non-zero exit).
    Failing behavior:
      Program runs successfully (should have failed).
    """
    input_csv = INPUT_DIR / "postive_ridingNum" / csv_name

    try:
        riding_rows, federal_rows = run_program(input_csv)
    except subprocess.CalledProcessError:
        pass
    except Exception as e:
        print(f"Failure: (R4 RidingNum) {csv_name}: Unexpected exception {type(e).__name__}: {e}")


class TestInvalidRidingNum(unittest.TestCase):
    def test_zero_ridingnum(self):
        check_invalid_ridingnum("zero_ridingnum.csv")

    def test_negative_ridingnum(self):
        check_invalid_ridingnum("negative_riding_num.csv")

##
## missing R5
##

# R6 might need more code
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


# R7
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


# R8
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


# R9
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



def check_duplicate_candidate_exact(csv_name: str):
    """
    Expect the program to reject exact duplicate candidate rows in the same riding.
    Passing behavior: program aborts (no outputs) and may print a validation error.
    Failing behavior: program still produces riding/federal results.
    """
    input_csv = INPUT_DIR / "duplicate_record" / csv_name
    riding_rows, federal_rows = run_program(input_csv)

    if riding_rows or federal_rows:
        print(f"Failure: (Duplicate candidate) {csv_name}: Produced outputs despite exact duplicate candidate row")


class TestDuplicatelines(unittest.TestCase):
    def test_duplicate_candidate_exact(self):
        check_duplicate_candidate_exact("duplicated_lines.csv")


##
## missing R10
##





# R17
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


    # def test_recount_below(self):
    #     check_recount_threshold("recount_below_0p1.csv", expect_recount=True)

    def test_recount_equal(self):
        check_recount_threshold("recount_equal_0p1.csv", expect_recount=True)

    # def test_recount_above(self):
    #     check_recount_threshold("recount_above_0p1.csv", expect_recount=False)


# R17 uncontested
def check_uncontested(csv_name: str):
    """
    Spec R17 (Uncontested):
      - Input CSV has a single candidate in a riding.
      - Output riding row must have Outcome="Uncontested".
      - Winner / Votes / %vote are filled (R18–R20).
      - %margin is empty (R22).
    """
    input_csv = INPUT_DIR / "uncontested" / csv_name
    riding_rows, federal_rows = run_program(input_csv)

    # Expect exactly one riding row produced for this single-riding input
    if len(riding_rows) != 1:
        print(f"Failure: (Uncontested) {csv_name}: Expected 1 riding row, got {len(riding_rows)}")
        return

    row = riding_rows[0]

    # Outcome must be "Uncontested"
    outcome = (row.get("Outcome") or "").strip().lower()
    if outcome != "uncontested":
        print(f"Failure: (Uncontested) {csv_name}: Outcome={row.get('Outcome')!r} (expected 'Uncontested')")

    # Winner / Votes / %vote must be present (non-empty)
    winner = (row.get("Winner") or "").strip()
    votes_raw = (row.get("Votes") or "").strip()
    pct_vote = (row.get("%vote") or row.get("%Vote") or "").strip()  # tolerate header casing

    if not winner:
        print(f"Failure: (Uncontested) {csv_name}: 'Winner' is empty")
    if not votes_raw:
        print(f"Failure: (Uncontested) {csv_name}: 'Votes' is empty")
    else:
        try:
            _ = int(votes_raw.replace(",", ""))  # allow thousands separators
        except Exception:
            print(f"Failure: (Uncontested) {csv_name}: 'Votes' not an integer: {votes_raw!r}")

    if not pct_vote:
        print(f"Failure: (Uncontested) {csv_name}: '%vote' is empty")
    else:
        # Optional sanity: uncontested should be ~100%
        try:
            pv = float(pct_vote.strip("%"))
            if not (99.9 <= pv <= 100.0+1e-9):  # allow rounding tolerance
                print(f"Failure: (Uncontested) {csv_name}: '%vote' expected ≈ 100%, got {pct_vote!r}")
        except Exception:
            print(f"Failure: (Uncontested) {csv_name}: '%vote' not parseable: {pct_vote!r}")

    # %margin must be empty for Uncontested
    margin = (row.get("%margin") or "").strip()
    if margin != "":
        print(f"Failure: (Uncontested) {csv_name}: '%margin' should be empty, got {margin!r}")


class TestUncontested(unittest.TestCase):
    def test_single_candidate(self):
        # CSV located at tests/test_input/uncontested/single_candidate.csv
        check_uncontested("single_candidate.csv")


# R18
def check_single_candidate(csv_name: str):
    input_csv = INPUT_DIR / 'single_candidate' / csv_name
    riding_rows, _ = run_program(input_csv)

    if len(riding_rows) != 1:
        print(f"Failure: (Output shape) {csv_name}: Expected 1 riding, got {len(riding_rows)}")
        return


class TestSingleCandidate(unittest.TestCase):

    def test_single_candidate(self):
        check_single_candidate("single_candidate.csv")



def normalization(s: str) -> str:
    return (s or "").strip().lower()


# R29 others
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

    # def test_federal_tie(self):
    #     # Liberal wins most but not > half (tie). Don't assert opposition uniqueness here.
    #     check_federal_roles("federal_tie.csv", expected_governing_party=None, expected_role=None)


# R29 tie
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


# R29 independent
def check_independent_ignored(csv_name: str):
    """
    Expect the program to ignore independent candidates when assigning government/opposition roles.
    Passing behavior: Independents do not appear in federal results and do not affect party roles.
    Failing behavior: Independents appear in output or alter role distribution.
    """
    input_csv = INPUT_DIR / "independent" / csv_name
    riding_rows, federal_rows = run_program(input_csv)

    print(riding_rows)

    # If there's no federal output at all, that's also unexpected
    if not federal_rows:
        print(f"Failure: (Independent) {csv_name}: No federal results produced")
        return

    # Check parties in federal results
    parties = [row["Party"].strip().lower() for row in federal_rows]
    roles = [row.get("Role", "").strip().lower() for row in federal_rows]

    # Independents should not appear in results
    if "independent" in parties:
        print(f"Failure: (Independent) {csv_name}: 'Independent' incorrectly appears in federal output")

    # Role check – should only be 'government', 'opposition', or blank
    for role in roles:
        if role not in {"government", "opposition", ""}:
            print(f"Failure: (Independent) {csv_name}: Unexpected role '{role}' found")


def check_only_independent(csv_name: str):
    """
    Expect the program to ignore independent candidates when assigning government/opposition roles.
    Passing behavior: Independents do not appear in federal results and do not affect party roles.
    Failing behavior: Independents appear in output or alter role distribution.
    """
    input_csv = INPUT_DIR / "independent" / csv_name
    riding_rows, federal_rows = run_program(input_csv)

    # If there's no federal output at all, that's also unexpected
    if not riding_rows:
        print(f"Failure: (Independent) {csv_name}: No riding results produced")
        return


class TestIndependent(unittest.TestCase):
    def test_only_independent(self):
        check_only_independent("independent_case.csv")



def _to_int(s: str) -> int:
    return int((s or "").replace(",", "").strip())


def _to_float(s: str) -> float:
    return float((s or "").replace("%", "").strip())


# R30
def check_federal_total(csv_name: str):
    """
    Spec R30 (Final 'total' row).
    Assert that the LAST row of federal_results:
      - Party == 'total'
      - Seats equals the sum of Seats over all party rows (i.e., #decided ridings)
      - Votes equals the grand total of Votes over all party rows
      - %vote ~= 100%
    """
    input_csv = INPUT_DIR / "total" / csv_name
    riding_rows, federal_rows = run_program(input_csv)

    if not federal_rows:
        print(f"Failure: (Total row) {csv_name}: No federal results produced")
        return

    # The final row must be the 'total' row
    total_row = federal_rows[-1]
    party = (total_row.get("Party") or "").strip().lower()
    if party != "total":
        print(f"Failure: (Total row) {csv_name}: Last row Party={total_row.get('Party')!r}, expected 'total'")

    # Compute expected sums from all non-'total' rows
    party_rows = [r for r in federal_rows if (r.get("Party") or "").strip().lower() != "total"]

    try:
        seats_expected = sum(_to_int(r.get("Seats", "0")) for r in party_rows)
    except Exception:
        print(f"Failure: (Total row) {csv_name}: Non-integer Seats in party rows")
        seats_expected = None

    try:
        votes_expected = sum(_to_int(r.get("Votes", "0")) for r in party_rows)
    except Exception:
        print(f"Failure: (Total row) {csv_name}: Non-integer Votes in party rows")
        votes_expected = None

    # Parse total row fields
    try:
        seats_total = _to_int(total_row.get("Seats", "0"))
    except Exception:
        print(f"Failure: (Total row) {csv_name}: Total Seats not an integer: {total_row.get('Seats')!r}")
        seats_total = None

    try:
        votes_total = _to_int(total_row.get("Votes", "0"))
    except Exception:
        print(f"Failure: (Total row) {csv_name}: Total Votes not an integer: {total_row.get('Votes')!r}")
        votes_total = None

    pct_total_raw = (total_row.get("%vote") or total_row.get("%Vote") or "").strip()
    try:
        pct_total = _to_float(pct_total_raw)
    except Exception:
        print(f"Failure: (Total row) {csv_name}: Total %vote not a number: {pct_total_raw!r}")
        pct_total = None

    # Seats must equal sum of party seats (i.e., #decided ridings)
    if seats_expected is not None and seats_total is not None and seats_total != seats_expected:
        print(f"Failure: (Total row) {csv_name}: Seats={seats_total}, expected sum={seats_expected}")

    # Votes must equal grand total votes across parties
    if votes_expected is not None and votes_total is not None and votes_total != votes_expected:
        print(f"Failure: (Total row) {csv_name}: Votes={votes_total}, expected sum={votes_expected}")

    # %vote must be ~100 (allow tiny rounding tolerance)
    if pct_total is not None and not (99.9 <= pct_total <= 100.1):
        print(f"Failure: (Total row) {csv_name}: %vote={pct_total_raw} (expected ≈ 100%)")


class TestFederalTotal(unittest.TestCase):
    def test_total_row(self):
        # CSV at tests/test_input/total/multi_ridings.csv
        check_federal_total("multi_ridings.csv")






if __name__ == "__main__":

    print("="*10 + ' TESTS ' + "="*10)
    tests = [ 
        # TestInputCSV,
        # TestBadHeader,
        # TestMissMatchRowLength,
        TestInvalidRidingNum

        # TestRecountThreshold, #bug
             # TestSingleCandidate,
             # TestInvaliVotes,
             # TestTie,
             # TestPartyNames,
             # TestRidingIdNameConsistency,
             # TestBadHeader,
             # TestMissMatchRowLength,
             # TestDuplicateCandidates,
            #  TestFederalRoles, #bug
            #  TestDuplicatelines,
            #  TestIndependent #bug
            # TestUncontested
            # TestFederalTotal
             ]
    
    for test in tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    