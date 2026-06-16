"""
test_verify.py — unit tests for the dataset verification logic.

Tests the two-check scheme used in TransistorDataset._load:
  1. covered_patterns (on-coverage)
  2. check_safety     (off-safety)

Run: env/bin/python3 test_verify.py
"""
from transnet.literal import (
    ALL_VARS, parse_sop_expr, on_off_split,
    covered_patterns, check_safety,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
n_ok = 0
n_bad = 0


def check(name, transistors, nc, expr, vars_in_func,
          expect_coverage_ok, expect_safety_ok):
    global n_ok, n_bad
    on_patterns  = parse_sop_expr(expr, vars_in_func)
    _, off_patterns = on_off_split(on_patterns)

    covered = covered_patterns(transistors, nc, on_patterns)
    missing = on_patterns - covered
    safe    = check_safety(transistors, nc, off_patterns)

    cov_ok = len(missing) == 0
    result_ok = (cov_ok == expect_coverage_ok) and (safe == expect_safety_ok)

    tag = PASS if result_ok else FAIL
    print(f"[{tag}] {name}")
    if not result_ok:
        print(f"       expected cov={expect_coverage_ok} safe={expect_safety_ok}")
        print(f"       got     cov={cov_ok}            safe={safe}")
        if missing:
            print(f"       missing on-patterns: {sorted(missing)[:3]} ...")
    n_ok  += result_ok
    n_bad += not result_ok


# Patterns are N_VARS=4 tuples (a, b, c, d); SOURCE=0, SINK=1

# Test 1 — single transistor "a", correct
check(
    "single transistor 'a' — correct",
    transistors=[(0, 1, 0, 0)],
    nc=2,
    expr="a",
    vars_in_func=ALL_VARS[:3],
    expect_coverage_ok=True,
    expect_safety_ok=True,
)

# Test 2 — "a*b" in series: SOURCE→node2→SINK, correct
check(
    "series 'a*b' — correct",
    transistors=[(0, 2, 0, 0), (2, 1, 1, 0)],
    nc=3,
    expr="a*b",
    vars_in_func=ALL_VARS[:3],
    expect_coverage_ok=True,
    expect_safety_ok=True,
)

# Test 3 — "a*b" but only one transistor (just 'a').
# All on-patterns have a=1 so coverage passes, but (a=1,b=0,...) leaks → safety fail.
check(
    "series 'a*b' but only 'a' transistor — safety fail only",
    transistors=[(0, 1, 0, 0)],
    nc=2,
    expr="a*b",
    vars_in_func=ALL_VARS[:3],
    expect_coverage_ok=True,
    expect_safety_ok=False,
)

# Test 4 — "a+b" (OR) but only 'a' transistor → (b=1,a=0,...) not covered
check(
    "parallel 'a+b' but only 'a' transistor — coverage fail",
    transistors=[(0, 1, 0, 0)],
    nc=2,
    expr="a+b",
    vars_in_func=ALL_VARS[:3],
    expect_coverage_ok=False,
    expect_safety_ok=True,
)

# Test 5 — "a+b" with two parallel transistors, correct
check(
    "parallel 'a+b' — correct",
    transistors=[(0, 1, 0, 0), (0, 1, 1, 0)],
    nc=2,
    expr="a+b",
    vars_in_func=ALL_VARS[:3],
    expect_coverage_ok=True,
    expect_safety_ok=True,
)

# Test 6 — XOR a⊕b = a*!b + !a*b via 4-transistor bridge
check(
    "XOR a⊕b bridge network — correct (3-input local)",
    transistors=[
        (0, 2, 0, 0),   # a:  SOURCE → n2
        (2, 1, 1, 1),   # !b: n2     → SINK
        (0, 3, 0, 1),   # !a: SOURCE → n3
        (3, 1, 1, 0),   # b:  n3     → SINK
    ],
    nc=4,
    expr="a*!b+!a*b",
    vars_in_func=ALL_VARS[:3],
    expect_coverage_ok=True,
    expect_safety_ok=True,
)

# Test 7 — 4-input global literal indexing: "a*b", vars_in_func=ALL_VARS
check(
    "4-input global literal 'a*b' — correct",
    transistors=[(0, 2, 0, 0), (2, 1, 1, 0)],
    nc=3,
    expr="a*b",
    vars_in_func=ALL_VARS,
    expect_coverage_ok=True,
    expect_safety_ok=True,
)

# Test 8 — wrong polarity: "a*!b" implemented as "a*b" → both fail
check(
    "wrong polarity: 'a*!b' implemented as 'a*b' — both fail",
    transistors=[(0, 2, 0, 0), (2, 1, 1, 0)],
    nc=3,
    expr="a*!b",
    vars_in_func=ALL_VARS[:3],
    expect_coverage_ok=False,
    expect_safety_ok=False,
)

print(f"\n{n_ok}/{n_ok+n_bad} tests passed.")
