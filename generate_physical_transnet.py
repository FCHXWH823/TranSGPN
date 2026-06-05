"""
generate_physical_transnet.py

Load a physical-RL-finetuned TransSyn checkpoint, synthesize paired PDN+PUN
transistor networks for a Boolean function, place them with ASTRAN, and save
the best result as a SPICE netlist (.sp).

For function f(x):
  PDN switching function : !f(x)   (NMOS, on-patterns = off-patterns of f)
  PUN switching function : f(!x)   (PMOS, on-patterns = complement(on-patterns of f))

Usage:
  python generate_physical_transnet.py \\
      --expr '!a*!b*c+!a*b*!c+a*!b*!c+a*b*c' \\
      --checkpoint checkpoints/transnet_physical_xor_v1.pt \\
      --trials 20 \\
      --out results/xor3.sp
"""
import argparse
import os
import re
import subprocess
import tempfile

import torch
from torchdrug import models

from transnet.graph import NODE_FEAT_DIM
from transnet.literal import (
    ALL_VARS, EDGE_FEAT_DIM,
    extract_vars, on_off_split, parse_sop_expr,
    covered_patterns, check_safety,
)
from transnet.task import GCPNTransNet

_ASTRAN_BIN = os.path.join(os.path.dirname(__file__),
              "astran/Astran/build/bin/Astran.app/Contents/MacOS/Astran")
_TECH_FILE  = os.path.join(os.path.dirname(__file__),
              "astran/Astran/build/Work/tech_freePDK45.rul")

# ── Place weights (must match physical_finetune_rl_transnet.py) ───────────────
_OBJ_CPP      = 3
_OBJ_MISMATCH = 4
_OBJ_ROUTING  = 1
_OBJ_DENSITY  = 4
_OBJ_GAPS     = 2

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=__doc__)
parser.add_argument("--expr",        type=str, required=True,
                    help="Boolean expression, e.g. '!a*b+a*!b'")
parser.add_argument("--checkpoint",  type=str,
                    default="checkpoints/transnet_physical_xor_v1.pt")
parser.add_argument("--trials",      type=int, default=20,
                    help="Number of PDN+PUN pairs to generate")
parser.add_argument("--max_steps",   type=int, default=20)
parser.add_argument("--temperature", type=float, default=0.6,
                    help="Sampling temperature (lower = more greedy)")
parser.add_argument("--out",         type=str, default=None,
                    help="Output path for best SPICE file (default: <cell_name>.sp)")
parser.add_argument("--astran_bin",  type=str, default=_ASTRAN_BIN)
parser.add_argument("--tech_file",   type=str, default=_TECH_FILE)
parser.add_argument("--timeout",     type=int, default=30,
                    help="ASTRAN timeout per call (seconds)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Function derivation ───────────────────────────────────────────────────────

def derive_pdn_patterns(on_p, off_p):
    """PDN: !f(x) → swap on/off."""
    return off_p, on_p

def derive_pun_patterns(on_p, off_p):
    """PUN: f(!x) → complement each pattern bit."""
    comp = lambda p: tuple(1 - b for b in p)
    return frozenset(comp(p) for p in on_p), frozenset(comp(p) for p in off_p)


# ── SPICE generator ───────────────────────────────────────────────────────────

_W_N = "0.090000U"
_W_P = "0.090000U"
_L   = "0.050000U"

def _node(n, prefix, src, snk):
    if n == 0: return src
    if n == 1: return snk
    return f"{prefix}{n}"

def _gate(var_idx, is_neg, vif):
    return vif[var_idx] if is_neg == 0 else f"{vif[var_idx]}_N"

def build_spice(cell_name, vif, pdn_tran, pdn_gnum, pun_tran, pun_gnum):
    ports = " ".join(vif) + " " + " ".join(v + "_N" for v in vif) + " ZN VCC GND"
    lines = [f".SUBCKT {cell_name} {ports}",
             f"* PDN — NMOS_VTL — !f(x)  [{len(pdn_tran)} transistors]"]
    for i, (u, v, vi, neg) in enumerate(pdn_tran):
        d = _node(v, "nd", "GND", "ZN")
        s = _node(u, "nd", "GND", "ZN")
        g = _gate(vi, neg, vif)
        lines.append(f"MN{i:<2} {d:<6} {g:<5} {s:<6} GND  NMOS_VTL  W={_W_N}  L={_L}")
    lines.append(f"* PUN — PMOS_VTL — f(!x)  [{len(pun_tran)} transistors]")
    for i, (u, v, vi, neg) in enumerate(pun_tran):
        d = _node(v, "np", "VCC", "ZN")
        s = _node(u, "np", "VCC", "ZN")
        g = _gate(vi, neg, vif)
        lines.append(f"MP{i:<2} {d:<6} {g:<5} {s:<6} VCC  PMOS_VTL  W={_W_P}  L={_L}")
    lines.append(".ENDS")
    return "\n".join(lines)


# ── ASTRAN wrapper ────────────────────────────────────────────────────────────

_COST_RE = re.compile(
    r"Final cost: Width=(\d+).*?Gate Mismatches=(\d+).*?WL=(\d+).*?Rt\. Density=(\d+).*?Nr\. Gaps=(\d+)"
)

def astran_objective(m):
    return (
        _OBJ_CPP      * m["width"]
      + _OBJ_MISMATCH * m["gate_mismatches"]
      + _OBJ_ROUTING  * m["wl"]
      + _OBJ_DENSITY  * m["rt_density"]
      + _OBJ_GAPS     * m["nr_gaps"]
    )

def run_astran(spice_text, cell_name):
    sp_path = sc_path = None
    try:
        sp_fd, sp_path = tempfile.mkstemp(suffix=".sp", prefix="transsyn_")
        os.write(sp_fd, spice_text.encode()); os.close(sp_fd)

        script = (
            f'load technology "{args.tech_file}"\n'
            f'load netlist "{sp_path}"\n'
            f'cellgen select {cell_name}\n'
            f'cellgen fold 2 0\n'
            f'cellgen place 1 1 {_OBJ_CPP} {_OBJ_MISMATCH} {_OBJ_ROUTING} {_OBJ_DENSITY} {_OBJ_GAPS}\n'
        )
        sc_fd, sc_path = tempfile.mkstemp(suffix=".run", prefix="transsyn_")
        os.write(sc_fd, script.encode()); os.close(sc_fd)

        result = subprocess.run(
            [args.astran_bin, "--shell", sc_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=args.timeout,
        )
        output = result.stdout.decode(errors="replace")
        m = _COST_RE.search(output)
        if m:
            return {
                "width":           int(m.group(1)),
                "gate_mismatches": int(m.group(2)),
                "wl":              int(m.group(3)),
                "rt_density":      int(m.group(4)),
                "nr_gaps":         int(m.group(5)),
            }
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None
    finally:
        for p in (sp_path, sc_path):
            if p and os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ck_path):
    ck  = torch.load(ck_path, map_location="cpu")
    a   = ck.get("arch_args") or ck.get("args", {})
    mpnn = models.MPNN(
        input_dim      = NODE_FEAT_DIM,
        hidden_dim     = a.get("hidden",    64),
        edge_input_dim = EDGE_FEAT_DIM,
        num_layer      = a.get("num_layer", 3),
        batch_norm     = False,
    )
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128))
    task.load_state_dict(ck["model_state"], strict=False)
    task = task.to(device)
    task.eval()
    info = {
        "epoch":   ck.get("epoch", "?"),
        "ppo_loss":ck.get("ppo_loss", float("nan")),
        "best_obj":ck.get("best_obj", {}),
    }
    return task, info


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"Expression : {args.expr}")
vif  = extract_vars(args.expr)
on_p = parse_sop_expr(args.expr, vif)
on_p, off_p = on_off_split(on_p)
N = len(vif)
print(f"Variables  : {vif}")
print(f"On-patterns  ({len(on_p)}): {sorted(on_p)}")
print(f"Off-patterns ({len(off_p)}): {sorted(off_p)}")

pdn_on, pdn_off = derive_pdn_patterns(on_p, off_p)
pun_on, pun_off = derive_pun_patterns(on_p, off_p)
print(f"\nPDN (!f(x))  on ({len(pdn_on)}): {sorted(pdn_on)}")
print(f"PUN (f(!x))  on ({len(pun_on)}): {sorted(pun_on)}")

print(f"\nLoading checkpoint: {args.checkpoint}")
task, info = load_model(args.checkpoint)
print(f"  epoch={info['epoch']}  ppo_loss={info['ppo_loss']:.4f}")

# Derive a clean cell name from the expression
_cell_name = "TRANSSYN_" + args.expr[:20].replace("!", "N").replace("*", "").replace("+", "_").replace(" ", "")
_cell_name = re.sub(r"[^A-Za-z0-9_]", "", _cell_name).upper()

print(f"\nGenerating {args.trials} PDN+PUN pairs "
      f"(T={args.temperature}, max_steps={args.max_steps}) …\n")

results = []
pdn_ok = pun_ok = both_ok = 0

with torch.no_grad():
    for trial in range(1, args.trials + 1):
        # Generate PDN
        pdn_res = task.generate(vif, pdn_on, pdn_off,
                                num_sample=1, max_steps=args.max_steps,
                                temperature=args.temperature, verbose=0)[0]
        # Generate PUN
        pun_res = task.generate(vif, pun_on, pun_off,
                                num_sample=1, max_steps=args.max_steps,
                                temperature=args.temperature, verbose=0)[0]

        pdn_c = pdn_res["correct"]
        pun_c = pun_res["correct"]
        pdn_ok += int(pdn_c)
        pun_ok += int(pun_c)

        if pdn_c and pun_c:
            both_ok += 1
            spice   = build_spice(_cell_name, vif,
                                  pdn_res["g_tran"], pdn_res["g_num"],
                                  pun_res["g_tran"], pun_res["g_num"])
            metrics = run_astran(spice, _cell_name)
            obj     = astran_objective(metrics) if metrics else None

            status = "✓✓"
            m_str  = (f"obj={obj}  cpp={metrics['width']}  "
                      f"mismatch={metrics['gate_mismatches']}  "
                      f"wl={metrics['wl']}  density={metrics['rt_density']}  "
                      f"gaps={metrics['nr_gaps']}"
                      if metrics else "ASTRAN failed")
        elif pdn_c:
            status, m_str, obj, metrics, spice = "✓·", "pun incomplete", None, None, None
        elif pun_c:
            status, m_str, obj, metrics, spice = "·✓", "pdn incomplete", None, None, None
        else:
            status, m_str, obj, metrics, spice = "··", "both incomplete", None, None, None

        print(f"  trial {trial:3d} [{status}]  "
              f"pdn={len(pdn_res['g_tran'])}T  pun={len(pun_res['g_tran'])}T  "
              f"{m_str}")

        if obj is not None:
            results.append({
                "trial":    trial,
                "obj":      obj,
                "metrics":  metrics,
                "spice":    spice,
                "pdn_tran": pdn_res["g_tran"],
                "pun_tran": pun_res["g_tran"],
            })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PDN success : {pdn_ok}/{args.trials}")
print(f"PUN success : {pun_ok}/{args.trials}")
print(f"Both complete + ASTRAN : {len(results)}/{args.trials}")

if not results:
    print("\nNo complete pairs found. Try --trials 50 or --temperature 0.8.")
else:
    best = min(results, key=lambda r: r["obj"])
    m    = best["metrics"]
    print(f"\nBest pair (trial {best['trial']}):")
    print(f"  Objective : {best['obj']}  "
          f"= {_OBJ_CPP}*{m['width']} + {_OBJ_MISMATCH}*{m['gate_mismatches']} "
          f"+ {_OBJ_ROUTING}*{m['wl']} + {_OBJ_DENSITY}*{m['rt_density']} "
          f"+ {_OBJ_GAPS}*{m['nr_gaps']}")
    print(f"  CPP (Width)      : {m['width']}")
    print(f"  Gate Mismatches  : {m['gate_mismatches']}")
    print(f"  WL               : {m['wl']}")
    print(f"  Rt. Density      : {m['rt_density']}")
    print(f"  Nr. Gaps         : {m['nr_gaps']}")
    print(f"  PDN transistors  : {len(best['pdn_tran'])}")
    print(f"  PUN transistors  : {len(best['pun_tran'])}")

    out_path = args.out or f"{_cell_name}.sp"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"* Best placement from {args.trials} trials\n")
        f.write(f"* Objective: {best['obj']}  CPP={m['width']}  "
                f"Mismatch={m['gate_mismatches']}  "
                f"WL={m['wl']}  Density={m['rt_density']}  Gaps={m['nr_gaps']}\n")
        f.write(f"* Checkpoint: {args.checkpoint}\n\n")
        f.write(best["spice"] + "\n")
    print(f"\nSaved best SPICE → {out_path}")
