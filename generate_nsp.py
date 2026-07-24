"""
Synthesize one switching network per function in dataset/53nsp/nsp_catalog_summary.csv
using the canonical 6-input pretrained policy. Each network implements the row's
switching function f directly (conduction == f), with variable-position
canonicalization consistent with pretraining.

Catalog format: nsp_id, expression ('.'=AND, '+'=OR, positive literals), n_variables,
n_transistors (NSP-catalog reference count). A trailing TOTAL row is skipped.

Output (results/nsp_synthesis.csv): nsp_id, expr, n_vars, success, min_t, ref_t,
meets (min_t <= ref_t), network (remapped to real variables).
"""
import os
os.environ.setdefault("TRANSGPN_VARS", "abcdef")
os.environ.setdefault("TRANSGPN_MAX_G_NODES", "70")

import argparse, csv, time
import torch

from transnet.literal import extract_vars, on_off_split, parse_sop_expr
from transnet.canonical import canonicalize_expr, decanon_gtran
from generate_all_transnets import load_model, eval_network


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/transnet_pretrain_6input_canon.pt")
    p.add_argument("--catalog",    default="dataset/53nsp/nsp_catalog_summary.csv")
    p.add_argument("--out",        default="results/nsp_synthesis.csv")
    p.add_argument("--trials",      type=int,   default=128)
    p.add_argument("--max-steps",   type=int,   default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, epoch = load_model(args.checkpoint, device)
    print(f"device={device} ckpt={args.checkpoint} (epoch {epoch})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = list(csv.DictReader(open(args.catalog)))
    fields = ["nsp_id", "expr", "n_vars", "success", "min_t", "ref_t", "meets", "network"]
    out_f = open(args.out, "w", newline=""); w = csv.DictWriter(out_f, fieldnames=fields)
    w.writeheader()

    t0 = time.time(); n_meet = 0; n_done = 0
    for r in rows:
        nid = (r.get("nsp_id") or "").strip()
        if not nid.isdigit():
            continue                                   # skip TOTAL / blank rows
        expr = (r.get("expression") or "").replace(".", "*").replace(" ", "")
        ref_t = r.get("n_transistors")
        try: ref_t = int(ref_t)
        except: ref_t = None
        if not expr:
            continue

        c_expr, real_vars, _perm = canonicalize_expr(expr)
        cvif = extract_vars(c_expr)
        on, off = on_off_split(parse_sop_expr(c_expr, cvif))
        net, succ, best = eval_network(task, cvif, on, off,
                                       args.trials, args.max_steps, args.temperature)
        net = decanon_gtran(net, real_vars) if net else None
        meets = (best is not None and ref_t is not None and best <= ref_t)
        if meets: n_meet += 1
        n_done += 1
        w.writerow({"nsp_id": nid, "expr": expr, "n_vars": len(real_vars),
                    "success": f"{succ:.4f}", "min_t": (best if best is not None else ""),
                    "ref_t": (ref_t if ref_t is not None else ""),
                    "meets": meets, "network": (net if net else "")})
        out_f.flush()
        print(f"  nsp{nid}: succ={succ:.3f} min_t={best} ref={ref_t} meets={meets}", flush=True)

    out_f.close()
    print(f"\nDone: {n_done} functions, meets-ref={n_meet}, "
          f"{(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
