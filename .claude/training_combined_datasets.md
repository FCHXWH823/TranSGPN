# Training on sweep_3input + sweep_4input_possani

**Datasets:** `dataset/sweep_3input/` (242 SAT functions) + `dataset/sweep_4input_possani/` (368k OK functions)  
**Combined samples:** 3,588,028 prefix training steps  
**Model:** N_VARS=4, EDGE_FEAT_DIM=6 (extended from 3-input-only N_VARS=3, EDGE_FEAT_DIM=5)

---

## Key Design Decisions

### Literal indexing difference between datasets

| Dataset | `vars_in_func` | Literal encoding |
|---------|---------------|-----------------|
| sweep_3input | `extract_vars(expr)` — only vars in expression (local) | 0..K-1 = positive, K..2K-1 = negative |
| sweep_4input_possani | `ALL_VARS = ['a','b','c','d']` (always forced) | 0=a, 1=b, 2=c, 3=d, 4=!a, 5=!b, 6=!c, 7=!d (global) |

For 3-input, `decode_literal` uses local indexing: `lit_id % K` → local var slot → looked up in `ALL_VARS`.
For 4-input, forcing `vars_in_func = ALL_VARS` makes K=4, so `lit_id % 4` = global index directly.

### Model architecture change
- `ALL_VARS` extended to `['a','b','c','d']` in `transnet/literal.py`
- `N_VARS = 4`, `EDGE_FEAT_DIM = 6`
- `mlp_var` head now outputs 4 logits (was 3)
- **Old checkpoints (N_VARS=3) are incompatible with this model**

### Graph ordering fix
`sorted_g_transistors` in `transnet/graph.py` was changed from pure snake-id ordering to **BFS-connected ordering from SOURCE/SINK**. Pure snake ordering failed for 4-input networks where SOURCE connects to high-indexed nodes, causing `Both endpoints are new` errors at prefix step k=0.

---

## Step 1: First Run (builds cache, ~56 min BFS + ~30 min graph build)

```bash
cd /Users/fch/Python/TranSGPN

# On CPU (slow — for testing only)
env/bin/python3 pretrain_transnet.py \
  --hidden 64 --num-layer 3 --mlp-hidden 128 \
  --epochs 30 --batch-size 128 --lr 1e-3 \
  --num-workers 0 \
  --log-interval 100 \
  --cache dataset/cache_3_4input.pt \
  --checkpoint checkpoints/transnet_pretrain_3_4input_v1.pt
```

**What happens on first run:**
1. BFS safety checks for all 3.5M samples (~56 min)
2. Saves raw cache to `dataset/cache_3_4input.pt` (no Graph objects — pickle-safe)
3. Rebuilds Graph objects in memory (~30 min, shown as progress bar)
4. Training begins

---

## Step 2: Subsequent Runs (loads cache, ~30 min graph rebuild then trains)

```bash
env/bin/python3 pretrain_transnet.py \
  --cache dataset/cache_3_4input.pt \
  --checkpoint checkpoints/transnet_pretrain_3_4input_v1.pt \
  [other args...]
```

Cache hit skips BFS entirely. Only graph rebuild (~30 min) before training.

---

## Recommended: Run on GPU (H100 or A100)

```bash
env/bin/python3 pretrain_transnet.py \
  --hidden 64 --num-layer 3 --mlp-hidden 128 \
  --epochs 30 --batch-size 1024 --lr 2e-3 \
  --num-workers 8 \
  --log-interval 50 \
  --cache dataset/cache_3_4input.pt \
  --checkpoint checkpoints/transnet_pretrain_3_4input_v1.pt
```

| Setting | CPU | H100/A100 |
|---------|-----|-----------|
| `--batch-size` | 128 | **1024** (GPU has 80GB) |
| `--lr` | 1e-3 | **2e-3** (linear scaling with batch) |
| `--num-workers` | 0 | **8** (prefetch batches while GPU trains) |
| Throughput | ~218 samples/s | ~5,000–8,000 samples/s |
| 30 epochs | ~137 hours | **~3–5 hours** |

---

## Profiling Results (CPU, batch=128)

```
Component       Per step (ms)      %
-------------------------------------
backward            309.9       52.8%   ← MPNN gradient (GPU target)
forward             153.1       26.1%   ← MPNN forward  (GPU target)
data_load           120.5       20.5%   ← DataLoader collation (use num-workers on GPU)
optimizer             2.9        0.5%
to_device             0.2        0.0%
-------------------------------------
TOTAL               586.5      100.0%
```

Forward+backward = 79% → GPU gives ~25-40× speedup on compute.  
DataLoader = 20.5% → drops to ~0% with `--num-workers 8` on GPU.

---

## Dataset Loading Architecture

```
First run:
  CSV files → BFS safety checks → [optional verification] → raw_samples (no Graph)
                                                                  ↓ torch.save
                                                          dataset/cache_3_4input.pt
                                                                  ↓ _build_graphs()
                                                          self.samples (with Graph objects, in RAM)

Subsequent runs (cache hit — verification skipped):
  dataset/cache_3_4input.pt → torch.load → _build_graphs() → self.samples
```

`_build_graphs()` rebuilds `torchdrug.Graph` objects from compact transistor lists — no BFS, just tensor assembly.

---

## Boolean Function Verification (first-time load only)

`TransistorDataset.__init__` accepts `verify=True` (default). On a cache miss, after
`sorted_g_transistors` orders each complete network, two checks run:

| Check | Function | What it detects |
|-------|----------|----------------|
| On-coverage | `covered_patterns(full_compact, nc, on_patterns)` | Missing SOURCE→SINK paths under on-patterns |
| Off-safety  | `check_safety(full_compact, nc, off_patterns)`    | Spurious SOURCE→SINK paths under off-patterns |

Both use BFS on the full transistor list. A failure prints:

```
[VERIFY FAIL] 3_42 (a*!b+!a*b): missing on-patterns: [(0, 1, 0, 0)]; conducts under off-pattern
```

After processing each dataset file a summary line is printed:

```
Verification sweep_3input: 374 networks checked — OK
Verification sweep_4input_possani: 368232 networks checked — OK
```

**Verification is skipped on cache hit** — the raw cache stores pre-verified data.
To re-verify explicitly, delete the cache file and reload without `--cache`, or pass
`TransistorDataset(verify=True)` directly (the default).

Tested on 5 sampled rows from each dataset — all networks verified OK. See `test_verify_dataset.py`.

---

## All CLI Arguments

```
--epochs        int     default=50       Training epochs
--batch-size    int     default=64       Samples per gradient step
--lr            float   default=1e-3     Learning rate
--hidden        int     default=64       MPNN hidden dim
--num-layer     int     default=3        MPNN message-passing layers
--mlp-hidden    int     default=128      Policy head MLP hidden dim
--num-workers   int     default=0        DataLoader workers (set 4-8 on GPU)
--cache         str     default=dataset/cache_3_4input.pt   Cache path
--checkpoint    str     default=checkpoints/transnet_pretrain_3_4input_v1.pt
--log-interval  int     default=20       Print loss every N steps
```

---

## Checkpoint Format

```python
{
    "epoch":          int,
    "model_state":    OrderedDict,
    "optimizer_state": OrderedDict,
    "loss":           float,          # best loss so far
    "args":           dict,           # {"hidden":64, "num_layer":3, "mlp_hidden":128, ...}
}
```

Checkpoint is saved whenever `avg_loss` improves. Load with:
```python
ck = torch.load("checkpoints/transnet_pretrain_3_4input_v1.pt", map_location="cpu")
a  = ck.get("arch_args") or ck.get("args", {})
mpnn = models.MPNN(input_dim=4, hidden_dim=a["hidden"], edge_input_dim=6,
                   num_layer=a["num_layer"], batch_norm=False)
task = GCPNTransNet(mpnn, hidden_dim_mlp=a["mlp_hidden"])
task.load_state_dict(ck["model_state"], strict=False)
```
