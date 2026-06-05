# Per-Function Fine-tuning Pipeline

**Script:** `run_rl_pipeline.py`

---

## Purpose

Systematically identify which of the 242 SAT 3-input functions the pretrained model (`transnet_pretrain_v6.pt`) fails on, then fine-tune each failing function individually via Phase 2 RL (`finetune_rl_transnet.py`).

---

## Failure Criteria

A function **needs fine-tuning** if EITHER of these is true:
- **Criterion A:** `success_rate < 0.50` — fewer than half of 50 generation trials produce a correct network
- **Criterion B:** `found_opt == False` — no trial found a network with ≤ `t_opt` transistors (where `t_opt` = minimum transistor count known from `aggregate.csv`)

```python
# run_rl_pipeline.py:155-157
def needs_finetuning(ev):
    return ev["success_rate"] < SUCCESS_THRESH or not ev["found_opt"]
```

---

## Evaluation Parameters

```python
EVAL_TRIALS    = 50     # generation trials per function
EVAL_TEMP      = 0.8    # sampling temperature
EVAL_STEPS     = 20     # max transistors per trial
SUCCESS_THRESH = 0.50   # minimum acceptable success rate
```

---

## Dry-Run Results (Previous Session)

Evaluated all 242 functions against `transnet_pretrain_v6.pt`:

```
Pass (both criteria)  : 206
Fail (low success <50%): 33
Fail (no opt found)   :  9
Fail (either)         : 36  → need fine-tuning
```

**Note:** The dry-run results were computed and printed but `pipeline_eval_cache.json` was NOT saved to disk (it requires a full `--skip_eval` run that starts from an existing cache, or a full run). Re-running will re-evaluate all 242 functions.

---

## Fine-tuning Parameters

```python
FT_EPOCHS   = 300
FT_NUM_TRAJ = 8
FT_TEMP     = 0.8
FT_STEPS    = 20
FT_LR       = 3e-5
FT_LOG_INT  = 10
```

Fine-tune command issued per function:
```bash
env/bin/python3 finetune_rl_transnet.py \
  --func_ids      <FID>          \
  --pretrain_ck   checkpoints/transnet_pretrain_v6.pt \
  --out_ck        checkpoints/rl_<FID>.pt \
  --epochs        300            \
  --num_traj      8              \
  --temperature   0.8            \
  --max_steps     20             \
  --lr            3e-5           \
  --log_interval  10             \
  --nll_every     0              # NLL regularization disabled
```

---

## Resumability

Two state files manage resumability:

| File | Purpose |
|------|---------|
| `pipeline_done.txt` | Append-only log of fids that completed fine-tuning |
| `pipeline_eval_cache.json` | Cached evaluation results for all functions |

```python
done = load_done()             # set of fids already fine-tuned
todo = [fid for fid in fail_any if fid not in done]
```

After each successful fine-tune: `mark_done(fid)` appends fid to `pipeline_done.txt`.

---

## Current State

| File | Status |
|------|--------|
| `pipeline_eval_cache.json` | **Does NOT exist** — will re-evaluate |
| `pipeline_done.txt` | **Does NOT exist** — no functions marked done |
| `checkpoints/rl_3_NNN.pt` | 33 checkpoints exist (from a prior pipeline run) |

The 33 existing per-function checkpoints suggest a prior pipeline run completed 33 out of 36 fine-tunes. However, without `pipeline_done.txt`, the pipeline has no memory of this and will re-run all 36.

---

## Usage Commands

```bash
# Full run (evaluate all 242, then fine-tune failing ones):
env/bin/python3 run_rl_pipeline.py

# Dry run only (identify failing functions, no training):
env/bin/python3 run_rl_pipeline.py --dry_run

# Evaluate only (no fine-tuning):
env/bin/python3 run_rl_pipeline.py --eval_only

# Skip re-evaluation (load cached results, fine-tune only):
env/bin/python3 run_rl_pipeline.py --skip_eval
# NOTE: --skip_eval requires pipeline_eval_cache.json to exist!
```

---

## Data Source

```python
_CSV_PATH = "dataset/sweep_3input/aggregate.csv"
```

`aggregate.csv` maps each `function_id` → minimum SAT transistor count `t`. The pipeline reads the minimum `t` across all SAT rows per function as `t_opt`.

Function expression read from:
```
dataset/sweep_3input/<fid>/t_<t_opt>/Booleans.txt
```

---

## Output Checkpoints

Per-function checkpoints saved to `checkpoints/rl_<fid>.pt` (e.g., `checkpoints/rl_3_101.pt`).

These are independent fine-tunes starting from `transnet_pretrain_v6.pt`. They are **not** chained — each function is fine-tuned from the same base pretrained model, not from a shared RL checkpoint.

---

## Post-Pipeline Usage

After the pipeline completes, you can load any per-function checkpoint for generation:

```bash
env/bin/python3 generate_transnet.py \
  --checkpoint checkpoints/rl_3_101.pt \
  --func_id 3_101 \
  --num_sample 50
```

Or proceed to Phase 3 physical RL using the best per-function checkpoints as starting points.
