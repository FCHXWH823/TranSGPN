"""
generate_all_transnets.py

For every Boolean function f(x) in dataset/Booleans.txt, use a trained
GCPNTransNet policy to synthesize:
    PDN : switching function f(x)        (the row's expression)
    PUN : switching function !f(!x)      (SOP derived with pyeda)

For each sub-network it records the SUCCESS PROBABILITY (fraction of trials that
produced a correct network) and the BEST solution (minimum #transistors among
correct trials). The combined transistor total (when both succeed) uses:
    total = best_pdn + best_pun + 2 * |negated literals common to both|

Output CSV columns:
    name, expr, pun_expr, n_vars,
    pdn_success, pdn_best, pun_success, pun_best,
    n_common_neg, total, complete

Usage:
    python generate_all_transnets.py \
        --checkpoint checkpoints/transnet_pretrain_3_4input_v2.pt \
        --out results/transnet_counts_v2.csv --trials 128
"""
import argparse
import csv
import os
import re
import time

import torch
from torchdrug import models

from pyeda.inter import expr as pyeda_expr
from pyeda.boolalg.expr import AndOp, OrOp

from transnet.graph import NODE_FEAT_DIM
from transnet.literal import (
    EDGE_FEAT_DIM,
    extract_vars, on_off_split, parse_sop_expr,
)
from transnet.canonical import canonicalize_expr, decanon_gtran, neg_real_var_set
from transnet.task import GCPNTransNet


# ── Booleans.txt parsing ──────────────────────────────────────────────────────

_LINE_RE = re.compile(r"^\s*(\S+?)\s*\(\d+\)\s*:\s*(.+?)\s*$")


def parse_booleans(path):
    rows = []
    with open(path) as f:
        for raw in f:
            if not raw.strip():
                continue
            m = _LINE_RE.match(raw)
            rows.append((m.group(1), m.group(2)) if m else
                        (raw.strip().split(":")[0].strip() or "?", None))
    return rows


# ── PUN switching function = !f(!x), SOP via pyeda ────────────────────────────

def _to_pyeda(sop: str) -> str:
    """Our SOP ('a*!b+c') -> pyeda syntax ('a & ~b | c')."""
    return sop.replace("*", " & ").replace("+", " | ").replace("!", "~")


def pun_sop(f_expr: str):
    """Return the SOP string of !f(!x) in our format, or None if constant."""
    f = pyeda_expr(_to_pyeda(f_expr))
    g = (~(f.compose({v: ~v for v in f.inputs}))).to_dnf()
    gs = str(g)
    if gs in ("0", "1"):
        return None                      # constant — no gateable network
    lit = lambda x: str(x).replace("~", "!")
    terms = g.xs if isinstance(g, OrOp) else [g]
    out = []
    for t in terms:
        out.append("*".join(lit(x) for x in t.xs) if isinstance(t, AndOp) else lit(t))
    return "+".join(out)


# ── Generation ────────────────────────────────────────────────────────────────

def eval_network(task, vif, on_p, off_p, trials, max_steps, temperature):
    """Return (best_g_tran|None, success_prob, best_t|None).

    success_prob = fraction of trials producing a correct network.
    best_t       = minimum FUNCTIONALLY-PRUNED transistor count among correct
    trials: transistors whose removal leaves every on-pattern covered are
    redundant (removal can only reduce conduction, so safety is preserved) —
    the pruned network computes the identical function.
    """
    from transnet.graph import prune_gtran_functional
    with torch.no_grad():
        res = task.generate(vif, on_p, off_p, num_sample=trials, max_steps=max_steps,
                            temperature=temperature, verbose=0)
    correct = [r for r in res if r["correct"]]
    success = len(correct) / trials if trials else 0.0
    if correct:
        pruned = [prune_gtran_functional(r["g_tran"], r["g_num"], on_p)
                  for r in correct]
        best = min(pruned, key=len)
        return best, success, len(best)
    return None, success, None


def neg_var_set(g_tran):
    return {gv for (_u, _v, gv, neg) in g_tran if neg == 1}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ck_path, device):
    ck = torch.load(ck_path, map_location=device)
    a = ck.get("arch_args") or ck.get("args", {})
    mpnn = models.MPNN(input_dim=NODE_FEAT_DIM, hidden_dim=a.get("hidden", 64),
                       edge_input_dim=EDGE_FEAT_DIM, num_layer=a.get("num_layer", 3),
                       batch_norm=False)
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128),
                        global_attn=a.get("global_attn", False),
                        pointer_head=a.get("pointer_head", False)).to(device)
    task.load_state_dict(ck["model_state"], strict=False)
    task.eval()
    return task, ck.get("epoch", "?")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/transnet_pretrain_3_4input_v2.pt")
    p.add_argument("--booleans",   default="dataset/Booleans.txt")
    p.add_argument("--out",        default="results/transnet_counts_v2.csv")
    p.add_argument("--trials",      type=int,   default=128)
    p.add_argument("--max-steps",   type=int,   default=20)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--limit",       type=int,   default=0)
    p.add_argument("--num-shards",  type=int,   default=1)
    p.add_argument("--shard-index", type=int,   default=0)
    p.add_argument("--log-interval", type=int,  default=25)
    args = p.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, epoch = load_model(args.checkpoint, device)
    print(f"shard {args.shard_index}/{args.num_shards}  device={device}  "
          f"ckpt={args.checkpoint} (epoch {epoch})", flush=True)

    rows = parse_booleans(args.booleans)
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    file_exists = os.path.exists(args.out)
    if file_exists:
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["name"])

    fieldnames = ["name", "expr", "pun_expr", "n_vars",
                  "pdn_success", "pdn_best", "pun_success", "pun_best",
                  "n_common_neg", "total", "complete"]
    out_f = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader(); out_f.flush()

    t0 = time.time(); processed = 0; n_complete = 0
    for i, (name, expr) in enumerate(rows):
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_index:
            continue
        if name in done:
            continue

        row = {"name": name, "expr": expr or "", "pun_expr": "", "n_vars": "",
               "pdn_success": "", "pdn_best": "", "pun_success": "", "pun_best": "",
               "n_common_neg": "", "total": "", "complete": False}

        if not expr:
            writer.writerow(row); out_f.flush(); continue
        vif = extract_vars(expr)
        if not vif:
            writer.writerow(row); out_f.flush(); continue
        row["n_vars"] = len(vif)

        # PDN: f(x) — canonicalize support to low slots, generate, map back.
        c_expr, real_vars, _perm = canonicalize_expr(expr)
        cvif = extract_vars(c_expr)
        on_f, off_f = on_off_split(parse_sop_expr(c_expr, cvif))
        pdn_canon, pdn_succ, pdn_best = eval_network(
            task, cvif, on_f, off_f, args.trials, args.max_steps, args.temperature)
        pdn_tran = decanon_gtran(pdn_canon, real_vars) if pdn_canon else None
        row["pdn_success"] = f"{pdn_succ:.4f}"
        if pdn_best is not None:
            row["pdn_best"] = pdn_best

        # PUN: !f(!x) via pyeda SOP — independently canonicalized.
        pe = pun_sop(expr)
        pun_tran = None; pun_best = None
        if pe is not None:
            row["pun_expr"] = pe
            cpe, pun_real, _pp = canonicalize_expr(pe)
            pcvif = extract_vars(cpe)
            pun_on, pun_off = on_off_split(parse_sop_expr(cpe, pcvif))
            pun_canon, pun_succ, pun_best = eval_network(
                task, pcvif, pun_on, pun_off, args.trials, args.max_steps, args.temperature)
            pun_tran = decanon_gtran(pun_canon, pun_real) if pun_canon else None
            row["pun_success"] = f"{pun_succ:.4f}"
            if pun_best is not None:
                row["pun_best"] = pun_best

        # Combined total (your formula) when both sub-networks succeeded.
        # Common negated literals are intersected on REAL variable indices.
        if pdn_tran is not None and pun_tran is not None:
            common = neg_real_var_set(pdn_tran) & neg_real_var_set(pun_tran)
            row["n_common_neg"] = len(common)
            row["total"] = pdn_best + pun_best + 2 * len(common)
            row["complete"] = True
            n_complete += 1

        writer.writerow(row); out_f.flush()
        processed += 1
        if processed % args.log_interval == 0:
            rate = processed / (time.time() - t0)
            print(f"  shard{args.shard_index} [{i+1}/{len(rows)}] processed={processed} "
                  f"complete={n_complete} ({rate:.2f} fn/s)", flush=True)

    out_f.close()
    print(f"shard {args.shard_index} done: processed={processed} complete={n_complete} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
