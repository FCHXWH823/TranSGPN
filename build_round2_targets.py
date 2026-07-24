"""
Build round-2 RL targets: the 4-input sub-networks that round-1 v2 finetuning
finetuned but could NOT bring down to the SAT-optimal reference (reached_ref=False).
Round 2 warm-starts each from its round-1 checkpoint with a bigger epoch budget.

Reads round-1 shard results + the original canonical targets; writes
results/rl_targets_v2_r2.csv (same schema: name,type,expr,t_ref,reason,our_count).
"""
import csv, glob

# round-1 per-target outcomes (from all shard CSVs)
done = {}
for fp in glob.glob("results/rl_finetune_v2/rl_results.shard*.csv"):
    for r in csv.DictReader(open(fp)):
        done[(r["name"], r["type"])] = r

# original round-1 targets (carry expr / t_ref forward)
tgt = {}
for r in csv.DictReader(open("results/rl_targets_canon.csv")):
    tgt[(r["name"], r["type"])] = r

rows, n_reached, n_pending = [], 0, 0
for k, r in tgt.items():
    d = done.get(k)
    if d is None:
        n_pending += 1                      # not yet finetuned in round 1
        continue
    if d.get("reached_ref") == "True":
        n_reached += 1
        continue
    # carry the round-1 best as our_count so we can measure round-2 improvement
    out = dict(r); out["our_count"] = d.get("min_t", "")
    rows.append(out)

def key(n):
    s = n["name"].lstrip("P"); return (0, int(s)) if s.isdigit() else (1, n["name"])
rows.sort(key=lambda x: (key(x), x["type"]))

with open("results/rl_targets_v2_r2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "type", "expr", "t_ref", "reason", "our_count"])
    w.writeheader(); w.writerows(rows)

print(f"round-1 finetuned: {len(done)}   reached_ref: {n_reached}   "
      f"not-yet-done: {n_pending}")
print(f"round-2 targets (reached_ref=False): {len(rows)} -> results/rl_targets_v2_r2.csv")
