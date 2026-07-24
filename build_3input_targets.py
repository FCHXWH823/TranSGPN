"""
build_3input_targets.py

Build the 3-input RL target list for exp-transgpn-optimize-#t.

Reference (t_ref) derivation, per user spec:
  * aggregate.csv rows give the switching network for f(x) itself (the PDN).
  * PDN reference = smallest cg (fewest transistors) among that function's
    SAT rows (each row is one transistor-budget attempt; UNSAT rows excluded).
  * PUN reference = the same minimum, but looked up for the function
    g = !f(!x): compute g's truth table and match it to the row-set of the
    corresponding function in the sweep.

Emits one target per (function, pdn|pun) — PDN and PUN are separate targets.
"""
import csv
import glob
import os
import re

ROOT = "dataset/sweep_3input"
VARS = ["a", "b", "c"]
OUT = "results/rl_targets_3input.csv"

_LINE = re.compile(r"^\s*(\S+?)\s*\(\d+\)\s*:\s*(.+?)\s*$")


def read_expr(fid: str):
    """SOP expression of the function (identical across its t_N variants)."""
    for p in sorted(glob.glob(os.path.join(ROOT, fid, "*", "Booleans.txt"))):
        m = _LINE.match(open(p).readline())
        if m:
            return m.group(2).strip()
    return None


def eval_sop(expr: str, assign: dict) -> int:
    """Evaluate a SOP string like '!a*!b*c+a*b*c' under an assignment."""
    for term in expr.split("+"):
        term = term.strip()
        if not term:
            continue
        ok = True
        for lit in term.split("*"):
            lit = lit.strip()
            if not lit:
                continue
            neg = lit.startswith("!")
            v = lit[1:] if neg else lit
            val = assign.get(v, 0)
            if (val == 0) != neg:      # literal false
                ok = False
                break
        if ok:
            return 1
    return 0


def truth_table(expr: str):
    """Tuple of f over all 2^3 assignments, indexed by (a,b,c) bits."""
    tt = []
    for i in range(8):
        assign = {v: (i >> (len(VARS) - 1 - k)) & 1 for k, v in enumerate(VARS)}
        tt.append(eval_sop(expr, assign))
    return tuple(tt)


def dual_tt(tt):
    """g = !f(!x): g(x) = 1 - f(complement(x))."""
    n = len(tt)
    return tuple(1 - tt[(n - 1) - i] for i in range(n))


def n_tr(cg: str) -> int:
    """Transistor count = number of (u,v,lit) tuples in the cg string."""
    m = re.search(r"tr=\[(.*)\]", cg, re.S)
    if not m:
        return 10 ** 9
    return len(re.findall(r"\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", m.group(1)))


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "aggregate.csv"))))
    # minimum transistor count per function, over SAT rows only
    min_tr, exprs = {}, {}
    for r in rows:
        if r.get("status") != "SAT":
            continue                      # UNSAT = that budget is infeasible
        cg = r.get("cg") or ""
        if "tr=" not in cg:
            continue
        fid = r["function_id"]
        t = n_tr(cg)
        if fid not in min_tr or t < min_tr[fid]:
            min_tr[fid] = t
    for fid in min_tr:
        exprs[fid] = read_expr(fid)

    # truth table -> fid  (to match g = !f(!x) back to a swept function)
    tt2fid = {}
    for fid, e in exprs.items():
        if e:
            tt2fid[truth_table(e)] = fid

    out, missing_pun = [], 0
    for fid in sorted(min_tr, key=lambda s: int(s.split("_")[1])):
        e = exprs.get(fid)
        if not e:
            continue
        # PDN: reference = this function's own minimum
        out.append({"name": fid, "type": "pdn", "expr": e,
                    "t_ref": min_tr[fid], "reason": "sweep_min", "our_count": ""})
        # PUN: reference = minimum of g = !f(!x)
        g = dual_tt(truth_table(e))
        gfid = tt2fid.get(g)
        if gfid is None or gfid not in min_tr:
            missing_pun += 1
            continue
        out.append({"name": fid, "type": "pun", "expr": e,
                    "t_ref": min_tr[gfid], "reason": f"sweep_min_dual({gfid})",
                    "our_count": ""})

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "type", "expr", "t_ref",
                                          "reason", "our_count"])
        w.writeheader()
        w.writerows(out)

    n_pdn = sum(1 for r in out if r["type"] == "pdn")
    n_pun = sum(1 for r in out if r["type"] == "pun")
    print(f"functions with a SAT implementation: {len(min_tr)}")
    print(f"targets written: {len(out)}  (pdn={n_pdn}, pun={n_pun})")
    print(f"PUN skipped (dual not in sweep): {missing_pun}")
    refs = [int(r['t_ref']) for r in out]
    print(f"t_ref range: {min(refs)}..{max(refs)}   -> {OUT}")


if __name__ == "__main__":
    main()
