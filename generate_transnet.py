"""
generate_transnet.py

Synthesize a transistor network for a given Boolean function using
the pretrained GCPNTransNet policy.

Usage:
    python generate_transnet.py --expr "a*!c+a*b+b*!c"
"""
import argparse
import torch
from torchdrug import models

from transnet.graph import NODE_FEAT_DIM
from transnet.literal import (
    ALL_VARS, EDGE_FEAT_DIM,
    extract_vars, on_off_split, parse_sop_expr,
    check_safety, covered_patterns,
)
from transnet.task import GCPNTransNet


# ─────────────────────────────────────────────────────────────────────────────
# Result display
# ─────────────────────────────────────────────────────────────────────────────

def _node_name(i):
    if i == 0:  return "SRC"
    if i == 1:  return "SNK"
    return f"I{i}"


def _lit_name(var_idx, is_neg):
    name = ALL_VARS[var_idx]
    return f"!{name}" if is_neg else name


def print_network(g_tran, g_num, vars_in_func, on_patterns, off_patterns):
    cov  = covered_patterns(g_tran, g_num, on_patterns)
    safe = check_safety(g_tran, g_num, off_patterns)
    complete = (cov == on_patterns) and safe

    print(f"\n  Transistors ({len(g_tran)}):")
    for u, v, gv, neg in g_tran:
        lit = _lit_name(gv, neg)
        print(f"    {_node_name(u)} --[{lit}]-- {_node_name(v)}")

    miss = on_patterns - cov
    print(f"\n  Coverage : {len(cov)}/{len(on_patterns)} on-patterns covered")
    if miss:
        print(f"  Missing  : {sorted(miss)}")
    print(f"  Safety   : {'✓ safe' if safe else '✗ UNSAFE'}")
    print(f"  Result   : {'✓ CORRECT' if complete else '✗ incomplete/incorrect'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--expr",        type=str, default="a*!c+a*b+b*!c")
    parser.add_argument("--expr",        type=str, default="!a*!b*c+!a*b*!c+a*!b*!c+a*b*c")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/transnet_pretrain.pt")
    parser.add_argument("--trials",      type=int, default=20)
    parser.add_argument("--max-steps",   type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature. 0=greedy.")
    args = parser.parse_args()

    # ── Parse function ─────────────────────────────────────────────────────
    expr = args.expr
    print(f"Target function : {expr}")
    vars_in_func = extract_vars(expr)
    on_patterns  = parse_sop_expr(expr, vars_in_func)
    on_p, off_p  = on_off_split(on_patterns)
    print(f"Variables       : {vars_in_func}")
    print(f"On-patterns  ({len(on_p)}): {sorted(on_p)}")
    print(f"Off-patterns ({len(off_p)}): {sorted(off_p)}")

    # ── Load model ─────────────────────────────────────────────────────────
    # Supports both pretrain checkpoints (args has hidden/num_layer/mlp_hidden)
    # and RL checkpoints (arch args stored under "arch_args" or defaulted).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device)
    a      = ckpt.get("arch_args") or ckpt.get("args", {})

    mpnn = models.MPNN(
        input_dim      = NODE_FEAT_DIM,
        hidden_dim     = a.get("hidden",     64),
        edge_input_dim = EDGE_FEAT_DIM,
        num_layer      = a.get("num_layer",  3),
        batch_norm     = False,
    ).to(device)
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128)).to(device)
    task.load_state_dict(ckpt["model_state"], strict=False)

    loss_str = (f"loss {ckpt['loss']:.4f}" if "loss" in ckpt
                else f"ppo_loss {ckpt['ppo_loss']:.4f}" if "ppo_loss" in ckpt
                else "")
    print(f"\nLoaded checkpoint (epoch {ckpt['epoch']}, {loss_str})\n")

    # ── Generate ───────────────────────────────────────────────────────────
    print(f"Running {args.trials} generation trial(s) (T={args.temperature}) …\n")

    results = task.generate(
        vars_in_func, on_p, off_p,
        num_sample=args.trials,
        max_steps=args.max_steps,
        temperature=args.temperature,
        verbose=1,
    )

    # ── Report best result ─────────────────────────────────────────────────
    n_correct = sum(r["correct"] for r in results)
    best = max(results, key=lambda r: (r["n_covered"], r["correct"], -len(r["g_tran"])))

    print(f"\n{'='*55}")
    print(f"Best network found ({n_correct}/{args.trials} correct):")
    print_network(best["g_tran"], best["g_num"], vars_in_func, on_p, off_p)

    if n_correct == 0:
        print("\n  [Note] No complete solution found in this run.")
        print("  The model is in phase 1 (NLL pretraining) and may not yet")
        print("  generate optimal networks. Phase 2 PPO fine-tuning adds a")
        print("  coverage+safety reward to guide complete synthesis.")


if __name__ == "__main__":
    main()
