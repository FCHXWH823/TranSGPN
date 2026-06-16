"""
profile_pretrain.py

Times each component of the pretraining loop to identify the real bottleneck.
Run with: env/bin/python3 profile_pretrain.py
"""
import time
import torch
from torchdrug import data as td_data, models
from transnet import GCPNTransNet, TransistorDataset
from transnet.graph import NODE_FEAT_DIM
from transnet.literal import EDGE_FEAT_DIM

N_WARMUP = 5
N_PROFILE = 50
BATCH_SIZE = 128
CACHE_PATH = "dataset/cache_3_4input.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load dataset ─────────────────────────────────────────────────────────────
print(f"\nLoading dataset from cache …")
t0 = time.time()
dataset = TransistorDataset(cache_path=CACHE_PATH)
print(f"  {len(dataset)} samples  ({time.time()-t0:.1f}s)\n")

loader = td_data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ── Build model ───────────────────────────────────────────────────────────────
mpnn = models.MPNN(
    input_dim=NODE_FEAT_DIM, hidden_dim=64, edge_input_dim=EDGE_FEAT_DIM,
    num_layer=3, batch_norm=False,
).to(device)
task = GCPNTransNet(mpnn, hidden_dim_mlp=128).to(device)
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)

# ── Profiling loop ────────────────────────────────────────────────────────────
timings = {
    "data_load":   0.0,   # DataLoader collation (CPU)
    "to_device":   0.0,   # moving tensors to GPU
    "forward":     0.0,   # MPNN + MLP heads + loss computation
    "backward":    0.0,   # loss.backward()
    "optimizer":   0.0,   # optimizer.step()
}

task.train()
data_iter = iter(loader)
step = 0
t_batch_start = time.time()

for step in range(N_WARMUP + N_PROFILE):
    # ── Data loading ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)
    t1 = time.time()

    # ── To device ─────────────────────────────────────────────────────────────
    graph = batch["graph"].to(device)
    labels = {k: v.to(device) for k, v in batch.items() if k != "graph"}
    labels["graph"] = graph
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()

    # ── Forward ───────────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss, metric = task(labels)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.time()

    # ── Backward ──────────────────────────────────────────────────────────────
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t4 = time.time()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t5 = time.time()

    if step >= N_WARMUP:
        timings["data_load"] += t1 - t0
        timings["to_device"] += t2 - t1
        timings["forward"]   += t3 - t2
        timings["backward"]  += t4 - t3
        timings["optimizer"] += t5 - t4

    if step == N_WARMUP - 1:
        print(f"Warmup done. Profiling {N_PROFILE} steps …\n")

# ── Report ────────────────────────────────────────────────────────────────────
total = sum(timings.values())
print(f"{'Component':<15}  {'Time (s)':>10}  {'Per step (ms)':>14}  {'%':>6}")
print("-" * 52)
for name, t in timings.items():
    print(f"{name:<15}  {t:>10.3f}  {t/N_PROFILE*1000:>14.1f}  {t/total*100:>6.1f}%")
print("-" * 52)
print(f"{'TOTAL':<15}  {total:>10.3f}  {total/N_PROFILE*1000:>14.1f}  {'100.0':>6}%")

steps_per_sec = N_PROFILE / total
samples_per_sec = steps_per_sec * BATCH_SIZE
epoch_steps = len(dataset) / BATCH_SIZE
epoch_time_min = epoch_steps / steps_per_sec / 60
print(f"\nThroughput : {samples_per_sec:.0f} samples/s  ({steps_per_sec:.2f} steps/s)")
print(f"Epoch time : ~{epoch_time_min:.0f} min  ({len(dataset)/BATCH_SIZE:.0f} steps/epoch @ batch={BATCH_SIZE})")
print(f"30 epochs  : ~{30*epoch_time_min/60:.1f} hours")

if timings["data_load"] / total > 0.3:
    print("\n⚠ DataLoader is the bottleneck — increase --num-workers when on GPU.")
if (timings["forward"] + timings["backward"]) / total > 0.5:
    print("\n→ MPNN forward+backward dominates — GPU will give large speedup.")
