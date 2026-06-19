"""
generate_all_transnets.py

For every Boolean function in dataset/Booleans.txt, use the pretrained
GCPNTransNet policy to synthesize a PDN and a PUN transistor network and
compute the total transistor count.

Conventions (per user spec — the row's expression f(x) IS the PDN switching
function, i.e. f(x) = !g(x) for the original gate g):
    PDN switching function : f(x)        on-patterns = on(f),  off = off(f)
    PUN switching function : !f(!x)      on-patterns = ~off(f), off = ~on(f)
                                         (~p flips every input bit)

Total transistor count:
    total = |PDN| + |PUN| + 2 * |neg-literals used in BOTH PDN and PUN|

Incomplete generations (no correct network found in `--trials` samples) are
left blank in the CSV and flagged via the pdn_ok / pun_ok / complete columns.

Output CSV columns:
    name, expr, n_vars, n_pdn, n_pun, n_common_neg, total,
    pdn_ok, pun_ok, complete

Usage:
    python generate_all_transnets.py \
        --checkpoint checkpoints/transnet_pretrain_3_4input_v1.pt \
        --booleans   dataset/Booleans.txt \
        --out        results/transnet_counts.csv \
        --trials 64 --temperature 0.8
"""
import argparse
import csv
import os
import re
import time

import torch
from torchdrug import models

from transnet.graph import NODE_FEAT_DIM
from transnet.literal import (
    EDGE_FEAT_DIM,
    extract_vars, on_off_split, parse_sop_expr,
)
from transnet.task import GCPNTransNet


# ── Booleans.txt parsing ──────────────────────────────────────────────────────

_LINE_RE = re.compile(r"^\s*(\S+?)\s*\(\d+\)\s*:\s*(.+?)\s*$")


def parse_booleans(path):
    """Yield (name, expr) for each parseable line. Format: ' Pname(N): expr'."""
    rows = []
    with open(path) as f:
        for raw in f:
            if not raw.strip():
                continue
            m = _LINE_RE.match(raw)
            if not m:
                rows.append((raw.strip().split(":")[0].strip() or "?", None))
                continue
            rows.append((m.group(1), m.group(2)))
    return rows


# ── PDN / PUN pattern derivation ──────────────────────────────────────────────

def _comp(p):
    return tuple(1 - b for b in p)


def derive_pdn(on_f, off_f):
    """PDN = f(x): patterns unchanged."""
    return on_f, off_f


def derive_pun(on_f, off_f):
    """PUN = !f(!x): on = complement(off_f), off = complement(on_f)."""
    pun_on = frozenset(_comp(p) for p in off_f)
    pun_off = frozenset(_comp(p) for p in on_f)
    return pun_on, pun_off


# ── Generation helpers ────────────────────────────────────────────────────────

def best_network(task, vif, on_p, off_p, trials, max_steps, temperature):
    """Return (g_tran, ok). ok=True iff a correct network was found; then
    g_tran is the smallest correct one. Otherwise g_tran is the best-coverage
    partial (used only for diagnostics; caller leaves counts blank)."""
    with torch.no_grad():
        res = task.generate(
            vif, on_p, off_p,
            num_sample=trials, max_steps=max_steps,
            temperature=temperature, verbose=0,
        )
    correct = [r for r in res if r["correct"]]
    if correct:
        best = min(correct, key=lambda r: len(r["g_tran"]))
        return best["g_tran"], True
    best = max(res, key=lambda r: r["n_covered"])
    return best["g_tran"], False


def neg_var_set(g_tran):
    """Set of global variable indices used as a NEGATED literal."""
    return {gv for (_u, _v, gv, neg) in g_tran if neg == 1}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ck_path, device):
    ck = torch.load(ck_path, map_location=device)
    a = ck.get("arch_args") or ck.get("args", {})
    mpnn = models.MPNN(
        input_dim=NODE_FEAT_DIM,
        hidden_dim=a.get("hidden", 64),
        edge_input_dim=EDGE_FEAT_DIM,
        num_layer=a.get("num_layer", 3),
        batch_norm=False,
    )
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128))
    task.load_state_dict(ck["model_state"], strict=False)
    task = task.to(device)
    task.eval()
    return task, ck.get("epoch", "?")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/transnet_pretrain_3_4input_v1.pt")
    p.add_argument("--booleans",   default="dataset/Booleans.txt")
    p.add_argument("--out",        default="results/transnet_counts.csv")
    p.add_argument("--trials",      type=int,   default=64,
                   help="Trajectories per network (large = better GPU util & yield).")
    p.add_argument("--max-steps",   type=int,   default=20)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--limit",       type=int,   default=0,
                   help="Process only the first N functions (0 = all). For testing.")
    p.add_argument("--num-shards",  type=int,   default=1,
                   help="Split the function list across this many parallel processes.")
    p.add_argument("--shard-index", type=int,   default=0,
                   help="This process handles functions where idx %% num_shards == shard_index.")
    p.add_argument("--log-interval", type=int,  default=25)
    args = p.parse_args()

    # One thread per process: when many shards run on one CPU node, this keeps
    # them from oversubscribing cores (N shards x 1 thread = N cores).
    torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  shard {args.shard_index}/{args.num_shards}")

    task, epoch = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint {args.checkpoint} (epoch {epoch})")

    rows = parse_booleans(args.booleans)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Parsed {len(rows)} functions from {args.booleans}")

    # Resume: skip names already present in the output CSV.
    done = set()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    file_exists = os.path.exists(args.out)
    if file_exists:
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["name"])
        print(f"Resuming — {len(done)} functions already done.")

    fieldnames = ["name", "expr", "n_vars", "n_pdn", "n_pun",
                  "n_common_neg", "total", "pdn_ok", "pun_ok", "complete"]
    out_f = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    t0 = time.time()
    n_complete = n_pdn_ok = n_pun_ok = n_parse_err = 0
    processed = 0

    for i, (name, expr) in enumerate(rows):
        # Sharding: each parallel process handles a disjoint stripe of functions.
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_index:
            continue
        if name in done:
            continue

        row = {"name": name, "expr": expr if expr else "",
               "n_vars": "", "n_pdn": "", "n_pun": "", "n_common_neg": "",
               "total": "", "pdn_ok": False, "pun_ok": False, "complete": False}

        if not expr:
            n_parse_err += 1
            writer.writerow(row); out_f.flush()
            continue

        vif = extract_vars(expr)
        if not vif:  # constant function — no variables to gate on
            n_parse_err += 1
            row["n_vars"] = 0
            writer.writerow(row); out_f.flush()
            continue

        on_f = parse_sop_expr(expr, vif)
        on_f, off_f = on_off_split(on_f)
        row["n_vars"] = len(vif)

        pdn_on, pdn_off = derive_pdn(on_f, off_f)
        pun_on, pun_off = derive_pun(on_f, off_f)

        pdn_tran, pdn_ok = best_network(task, vif, pdn_on, pdn_off,
                                        args.trials, args.max_steps, args.temperature)
        pun_tran, pun_ok = best_network(task, vif, pun_on, pun_off,
                                        args.trials, args.max_steps, args.temperature)

        row["pdn_ok"], row["pun_ok"] = pdn_ok, pun_ok
        if pdn_ok:
            row["n_pdn"] = len(pdn_tran)
            n_pdn_ok += 1
        if pun_ok:
            row["n_pun"] = len(pun_tran)
            n_pun_ok += 1

        if pdn_ok and pun_ok:
            common = neg_var_set(pdn_tran) & neg_var_set(pun_tran)
            row["n_common_neg"] = len(common)
            row["total"] = len(pdn_tran) + len(pun_tran) + 2 * len(common)
            row["complete"] = True
            n_complete += 1

        writer.writerow(row); out_f.flush()
        processed += 1

        if processed % args.log_interval == 0:
            rate = processed / (time.time() - t0)
            print(f"  [{i+1}/{len(rows)}] processed={processed} "
                  f"complete={n_complete} pdn_ok={n_pdn_ok} pun_ok={n_pun_ok} "
                  f"({rate:.1f} fn/s)", flush=True)

    out_f.close()
    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min.")
    print(f"  processed       : {processed}")
    print(f"  complete (both) : {n_complete}")
    print(f"  pdn_ok          : {n_pdn_ok}")
    print(f"  pun_ok          : {n_pun_ok}")
    print(f"  parse/constant  : {n_parse_err}")
    print(f"  CSV → {args.out}")


if __name__ == "__main__":
    main()
