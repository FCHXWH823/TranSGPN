"""
Build (1) a human-facing comparison CSV and (2) the RL-finetune target list, from
the CANONICAL generation results vs the PClassResults reference solutions.

Switching-function alignment (this is the only correct way to compare):
    f(x)      network  ==  our canon 'pdn_*'  ==  reference '#Transistors_PDN'
    !f(!x)    network  ==  our canon 'pun_*'  ==  reference '#Transistors_PUN'

User naming convention (human CSV only): PUN := f(x),  PDN := !f(!x).
RL target 'type' convention (matches finetune_rl_targets.derive_patterns and the
reference columns): 'pdn' := f(x),  'pun' := !f(!x).

A (function, network) needs RL finetuning iff we did NOT reach the reference, i.e.
our min-T is missing (failed) or strictly greater than the reference count.
"""
import csv, os, re

GEN = os.environ.get("CMP_GEN", "results/transnet_counts_canon.csv")
REF = os.environ.get("CMP_REF", "dataset/PClassResults_minitntk_nodepth_final.csv")
BOOL = os.environ.get("CMP_BOOL", "dataset/Booleans.txt")
OUT_CMP = os.environ.get("CMP_OUT_CMP", "results/transnet_vs_reference.csv")
OUT_TGT = os.environ.get("CMP_OUT_TGT", "results/rl_targets_canon.csv")

def _int(x):
    try: return int(x)
    except: return None

# expr lookup from Booleans.txt
expr = {}
_LINE = re.compile(r"^\s*(\S+?)\s*\(\d+\)\s*:\s*(.+?)\s*$")
for raw in open(BOOL):
    m = _LINE.match(raw)
    if m: expr[m.group(1)] = m.group(2)

# reference: name -> (pdn_ref=f, pun_ref=!f(!x))
# NOTE: the CG(...) columns contain unquoted commas, so any column AFTER them
# (incl. SatResult) is misaligned by csv.DictReader. The two count columns are
# BEFORE the CG fields, so they parse correctly — use only those, validated as ints.
ref = {}
for r in csv.DictReader(open(REF)):
    pdn_t, pun_t = _int(r["#Transistors_PDN"]), _int(r["#Transistors_PUN"])
    ref[r["Boolean Func"]] = (pdn_t if (pdn_t and pdn_t > 0) else None,
                              pun_t if (pun_t and pun_t > 0) else None)

gen = {r["name"]: r for r in csv.DictReader(open(GEN))}

cmp_rows, targets = [], []
stats = {"pun_meet":0, "pun_miss":0, "pun_fail":0, "pdn_meet":0, "pdn_miss":0, "pdn_fail":0, "no_ref":0}

for name, g in gen.items():
    e = g.get("expr") or expr.get(name, "")
    if name not in ref:
        stats["no_ref"] += 1
        continue
    pdn_ref, pun_ref = ref[name]                 # f, !f(!x)
    f_best   = _int(g.get("pdn_best"))           # our f network  (user PUN)
    nf_best  = _int(g.get("pun_best"))           # our !f(!x)     (user PDN)
    f_succ   = g.get("pdn_success", "")
    nf_succ  = g.get("pun_success", "")

    # ---- f(x) network  (RL type 'pdn', user-facing 'PUN'), ref=pdn_ref ----
    f_meets = (f_best is not None and pdn_ref is not None and f_best <= pdn_ref)
    if pdn_ref is not None and not f_meets:
        targets.append({"name":name, "type":"pdn", "expr":e, "t_ref":pdn_ref,
                        "reason":("failed" if f_best is None else "suboptimal"),
                        "our_count":(f_best if f_best is not None else "")})
        stats["pun_fail" if f_best is None else "pun_miss"] += 1
    elif f_meets:
        stats["pun_meet"] += 1

    # ---- !f(!x) network (RL type 'pun', user-facing 'PDN'), ref=pun_ref ----
    nf_meets = (nf_best is not None and pun_ref is not None and nf_best <= pun_ref)
    if pun_ref is not None and not nf_meets:
        targets.append({"name":name, "type":"pun", "expr":e, "t_ref":pun_ref,
                        "reason":("failed" if nf_best is None else "suboptimal"),
                        "our_count":(nf_best if nf_best is not None else "")})
        stats["pdn_fail" if nf_best is None else "pdn_miss"] += 1
    elif nf_meets:
        stats["pdn_meet"] += 1

    cmp_rows.append({
        "name":name, "expr":e, "n_vars":g.get("n_vars",""),
        # user naming: PUN = f(x)
        "PUN_success":f_succ, "PUN_minT":(f_best if f_best is not None else ""),
        "PUN_ref":(pdn_ref if pdn_ref is not None else ""), "PUN_meets":f_meets,
        # user naming: PDN = !f(!x)
        "PDN_success":nf_succ, "PDN_minT":(nf_best if nf_best is not None else ""),
        "PDN_ref":(pun_ref if pun_ref is not None else ""), "PDN_meets":nf_meets,
        "needs_rl": not (f_meets and nf_meets),
    })

# sort by P-number
def key(n):
    s = n.lstrip("P"); return (0,int(s)) if s.isdigit() else (1,n)
cmp_rows.sort(key=lambda r: key(r["name"]))
targets.sort(key=lambda r: (key(r["name"]), r["type"]))

with open(OUT_CMP,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys())); w.writeheader(); w.writerows(cmp_rows)
with open(OUT_TGT,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["name","type","expr","t_ref","reason","our_count"])
    w.writeheader(); w.writerows(targets)

n = len(cmp_rows)
print(f"functions compared: {n}   (no reference: {stats['no_ref']})")
print(f"PUN  f(x):    meet_ref={stats['pun_meet']}  suboptimal={stats['pun_miss']}  failed={stats['pun_fail']}")
print(f"PDN  !f(!x):  meet_ref={stats['pdn_meet']}  suboptimal={stats['pdn_miss']}  failed={stats['pdn_fail']}")
print(f"RL targets written: {len(targets)} -> {OUT_TGT}")
print(f"comparison written: {OUT_CMP}")
