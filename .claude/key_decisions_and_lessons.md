# Key Engineering Decisions and Lessons Learned

---

## 1. torch.compile — REMOVED

**Decision:** Completely removed from `finetune_rl_transnet.py`.

**What happened:** Added `torch.compile(task, backend="inductor")` to speed up training. Compilation appeared to succeed, but the model crashed at the **first actual forward call** with `BackendCompilerFailed: inductor raised RuntimeError`. A `try/except` around `torch.compile(...)` didn't help because the error occurred at runtime, not at the compile call.

**Root cause:** torchdrug's `PackedGraph` uses custom Python objects (variable-length node lists, custom attributes via `with graph.node():`) that are fundamentally incompatible with the `inductor` backend's graph capture mechanism.

**Fix:** Remove `torch.compile` entirely. No alternative backend (`aot_eager`, etc.) was attempted — torchdrug's dynamic graph construction makes this infeasible without a deep rewrite.

---

## 2. Threading for Parallelism — REJECTED for MPNN, USED for ASTRAN

**Decision:** Don't use threads for MPNN parallelism; DO use threads for parallel ASTRAN.

**MPNN threading result:** 0.75× slowdown (WORSE than sequential). The Python GIL prevents true parallelism for CPU-bound torch operations. Multiple threads competing for the GIL causes more overhead than gain.

**ASTRAN threading:** Works well (`ThreadPoolExecutor(max_workers=4)`). Each ASTRAN call is a separate subprocess that releases the GIL while blocking on `subprocess.run`. True I/O-bound parallelism — 4× throughput on 4 workers.

---

## 3. Batched K-trajectory Generation — KEY SPEEDUP

**Decision:** Replace sequential `_generate_traj` loop with `_generate_K_trajs` batching.

**Profiling revealed:** MPNN = 70% of generation wall time (not BFS safety checks as initially suspected).

**Solution:** Pack all K active trajectory graphs into one `PackedGraph` per step, run one MPNN forward pass, then split results per trajectory.

```python
packed = td_data.Graph.pack(graphs).to(device)   # K graphs → 1 packed
out    = agent.model(packed, packed.node_feature.float())
# → nf_all[total_N_all, h], gf_all[K, 2h]
```

**What CANNOT be batched:** Safety BFS (per-trajectory state), node selection (variable graph sizes for n1/n2 heads).

**Speedup:** 1.65× for generation alone. Combined with batched PPO loss: 3.44× total.

---

## 4. Terminal-Only Reward — CRITICAL DESIGN

**Decision:** No per-step reward; zero reward for incomplete trajectories.

**Why:** Step costs (e.g., `-1/T` per transistor added) caused the agent to learn to STOP IMMEDIATELY to avoid accumulating costs. The GRPO baseline needs differentiable signal between trajectories — sparse terminal reward creates the necessary differentiation without inducing degenerate stopping.

**`best_T` self-tightening:** As the agent discovers a shorter solution, `best_T[fid]` updates immediately, making all future longer solutions receive lower reward. This adaptive baseline continuously raises the bar without manual curriculum design.

---

## 5. ASTRAN Power Net Names — VCC/GND NOT VDD/VSS

**Decision:** All SPICE power nets must be `VCC` and `GND`.

**What happened:** Initially used `VDD`/`VSS` (standard CMOS naming). ASTRAN crashed with SIGBUS (exit code 138, signal 10) during the `fold` step. This is a null pointer dereference inside ASTRAN's folding code when it tries to look up the power nets and finds unknown names.

**ASTRAN's internal convention:** `VCC` = positive supply, `GND` = ground. These are hardcoded in its folding/placement algorithms.

**Fix:** Changed `build_spice` to use `GND` (NMOS source/bulk) and `VCC` (PMOS source/bulk) throughout.

---

## 6. ASTRAN Model Names — NMOS_VTL/PMOS_VTL

**Decision:** Use `NMOS_VTL`/`PMOS_VTL` model names in SPICE.

**What happened:** Initial SPICE used `NMOS`/`PMOS` (generic names). ASTRAN couldn't find these in `library45.sp` (freePDK45 technology library). 

**Root cause:** freePDK45 uses threshold-voltage-qualified names: `NMOS_VTL` (Virtual Threshold Low), `PMOS_VTL`. The `VTL` suffix is the standard threshold variant for this process node.

**Where to find correct names:** `astran/Astran/build/Work/library45.sp` — look for `.model` declarations.

---

## 7. ASTRAN Invocation — `--shell` Flag Required

**Decision:** Always call `Astran --shell script.run`, never `Astran script.run`.

**What happened:** `Astran script.run` (without `--shell`) launches ASTRAN's Cocoa GUI window and waits for user interaction. On a headless server or programmatic subprocess call, it hangs forever until timeout (exit code 124).

**Fix:** `--shell` flag switches to batch/terminal mode. Combined with a script file (not inline commands), ASTRAN executes all commands and exits cleanly.

---

## 8. GRPO vs Simple Baseline

**Decision:** Use GRPO (within-group normalization) rather than a running mean baseline.

**Why:** GRPO naturally handles the sparse reward case. When most trajectories get reward=0 (incomplete), the mean is near-zero and the few complete trajectories get large positive advantage. This is exactly the right signal. A running mean baseline would slowly drift and could suppress the gradient signal from rare successes.

---

## 9. `object.__setattr__` for `_agent` and `_best_t`

**Decision:** Use `object.__setattr__` to store `_agent` and `_best_t` on `GCPNTransNet`.

**Why:** PyTorch's `nn.Module.__setattr__` intercepts all attribute assignments and registers `nn.Module` instances as submodules and tensors as parameters. `_agent` is a deepcopy of the model — if registered as a submodule, its parameters would appear in `state_dict()` (doubling checkpoint size) and receive gradients. `_best_t` is a plain dict that we don't want tracked.

```python
object.__setattr__(self, '_agent', None)   # bypass nn.Module tracking
object.__setattr__(self, '_best_t', {})
```

---

## 10. Per-Function Fine-tuning vs. All-Function Fine-tuning

**Decision:** Fine-tune failing functions individually rather than doing another all-function RL pass.

**Why all-function RL underperforms on hard functions:** With 242 functions and `funcs_per_batch=4`, any given function gets gradient updates only ~1.7% of PPO steps. Hard functions with rare success events get very sparse learning signal. The 9-transistor XOR solution is stable enough to satisfy most functions — the agent has no sustained pressure to find 8T XOR.

**Per-function fine-tuning:** 300 epochs × 8 trajectories = 2400 trajectories, all focused on one function. Gradient signal is dense and directly targeted.
