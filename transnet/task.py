"""
transnet/task.py

GCPNTransNet: NLL pretraining task for transistor-network synthesis.

Uses MPNN as backbone. Five policy heads:
  mlp_stop  : graph_feat → 2          (stop / continue)
  mlp_node1 : [node_feat, graph_feat] → 1  (source-node score)
  mlp_node2 : [node1_feat, node_feat, graph_feat] → 1  (dest-node score)
  mlp_var   : [node1_feat, node2_feat] → N_VARS   (variable index)
  mlp_sign  : [node1_feat, node2_feat] → 2         (positive / negative)

Node spaces in packed batch:
  Actual nodes  : 0 .. total_nodes-1  (real G- and C-nodes)
  Virtual nodes : total_nodes .. total_nodes+B-1  (one new-INTERNAL_G per graph)
"""
from __future__ import annotations

import copy
from collections import deque
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_max
from torch_scatter.composite import scatter_log_softmax

from torchdrug import layers

from .graph import build_gen_graph, prune_gtran, prune_gtran_functional
from .literal import N_VARS, check_safety, covered_patterns


class TrajStep:
    """One transistor-placement step in a generated trajectory.

    Stores the graph state before the step plus the action taken, the
    old log-prob from the agent, and all safety masks — so cur_logp can
    be recomputed with the current (updated) model without re-running
    safety checks.
    """
    __slots__ = (
        "graph", "g_num",
        "node1", "node2_ext", "node2_c",
        "var_idx", "is_neg",
        "old_logp",
        "n1_mask", "n2_mask", "var_valid", "sign_valid",
    )

    def __init__(
        self, graph, g_num,
        node1, node2_ext, node2_c,
        var_idx, is_neg, old_logp,
        n1_mask, n2_mask, var_valid, sign_valid,
    ):
        self.graph      = graph
        self.g_num      = g_num
        self.node1      = node1
        self.node2_ext  = node2_ext   # ext index: total_N = virtual new node
        self.node2_c    = node2_c     # compact index: g_num = virtual new node
        self.var_idx    = var_idx
        self.is_neg     = is_neg
        self.old_logp   = old_logp    # float, sum of 4 head log-probs from agent
        self.n1_mask    = n1_mask     # [total_N+1] bool, CPU
        self.n2_mask    = n2_mask     # [total_N+1] bool, CPU
        self.var_valid  = var_valid   # [N_VARS] bool, CPU
        self.sign_valid = sign_valid  # [2] bool, CPU (for chosen var_idx)


class GlobalAttnBlock(nn.Module):
    """One pre-LN self-attention + FFN block over the nodes of each graph.

    Gives every node a full-graph receptive field before the policy heads —
    the node1 head must make a *global* comparison across ~64 candidates, but
    MPNN embeddings are only num_layer-hop local and are scored independently.
    Works on packed node features via pad → attend → unpad.
    """

    def __init__(self, h: int, num_heads: int = 4):
        super().__init__()
        self.ln1  = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, num_heads, batch_first=True)
        self.ln2  = nn.LayerNorm(h)
        self.ffn  = nn.Sequential(nn.Linear(h, 2 * h), nn.ReLU(),
                                  nn.Linear(2 * h, h))

    def forward(self, node_feat, node2graph, num_graphs):
        device = node_feat.device
        counts = torch.bincount(node2graph, minlength=num_graphs)
        max_n  = int(counts.max())
        starts = torch.cumsum(counts, dim=0) - counts
        pos    = torch.arange(node_feat.size(0), device=device) - starts[node2graph]

        padded = node_feat.new_zeros(num_graphs, max_n, node_feat.size(1))
        padded[node2graph, pos] = node_feat
        pad_mask = torch.ones(num_graphs, max_n, dtype=torch.bool, device=device)
        pad_mask[node2graph, pos] = False

        x = self.ln1(padded)
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        padded = padded + a
        padded = padded + self.ffn(self.ln2(padded))
        return padded[node2graph, pos]


class GCPNTransNet(nn.Module):
    """
    NLL pretraining wrapper around an MPNN backbone.

    Args:
        model        : MPNN instance (node_output_dim=h, output_dim=2h)
        hidden_dim_mlp: hidden width of each policy MLP (default 128)
        global_attn  : add a GlobalAttnBlock between the MPNN and the heads
    """

    def __init__(self, model: nn.Module, hidden_dim_mlp: int = 128,
                 global_attn: bool = False, pointer_head: bool = False,
                 w_node1: float = 1.0, value_head: bool = False):
        super().__init__()
        self.model = model
        self.attn_block = (GlobalAttnBlock(model.node_output_dim)
                           if global_attn else None)
        self.pointer_head = pointer_head
        self.w_node1 = w_node1

        h = model.node_output_dim   # per-node feature dim  (hidden_dim)
        gh = model.output_dim       # graph feature dim      (2 * hidden_dim)

        # One learnable embedding for the virtual new-node option
        self.new_node_emb = nn.Parameter(torch.empty(h))
        nn.init.normal_(self.new_node_emb, std=0.1)

        MLP = layers.MultiLayerPerceptron
        if pointer_head:
            # Pointer-network node1 scoring: a query derived from the graph
            # state dot-scores each candidate — a set-selection architecture,
            # instead of scoring each node in isolation with a concat-MLP.
            self.n1_query = nn.Linear(gh, h)
            self.n1_key   = nn.Linear(h, h)
        else:
            self.mlp_node1 = MLP(h + gh, [hidden_dim_mlp, 1],      activation="tanh")
        self.mlp_node2 = MLP(h + h + gh, [hidden_dim_mlp, 1],      activation="tanh")
        self.mlp_var   = MLP(h + h,      [hidden_dim_mlp, N_VARS], activation="tanh")
        self.mlp_sign  = MLP(h + h,      [hidden_dim_mlp, 2],      activation="tanh")

        # Optional critic for reinforce_forward_dense: V(s) from graph feature.
        self.use_value = value_head
        if value_head:
            self.value_fn = MLP(gh, [hidden_dim_mlp, 1], activation="tanh")

        # Use object.__setattr__ so nn.Module does NOT register these as
        # submodules — prevents keys from appearing in state_dict().
        object.__setattr__(self, '_agent',  None)
        object.__setattr__(self, '_best_t', {})   # fid -> best transistor count found
        object.__setattr__(self, '_best_traj', {})  # fid -> (T_eff, steps, returns) SIL buffer
        object.__setattr__(self, '_seen_structs', {})  # fid -> set of canonical net hashes (v4 novelty)

    # ─────────────────────────────────────────────────────────────────────────
    # Training entry point (called by Engine / training loop)
    # ─────────────────────────────────────────────────────────────────────────

    def _encode(self, graph):
        """MPNN encode (+ optional global attention). → (node_feat, graph_feat)"""
        output     = self.model(graph, graph.node_feature.float())
        node_feat  = output["node_feature"]
        graph_feat = output["graph_feature"]
        if self.attn_block is not None:
            n2g = getattr(graph, "node2graph", None)
            if n2g is None:
                n2g = torch.zeros(graph.num_node, dtype=torch.long,
                                  device=node_feat.device)
            num_graphs = int(getattr(graph, "batch_size", 1))
            node_feat = self.attn_block(node_feat, n2g, num_graphs)
        return node_feat, graph_feat

    def _n1_logits(self, ext_feat, ext_gfeat):
        """node1 candidate logits: pointer (query·key) or concat-MLP scoring."""
        if self.pointer_head:
            q = self.n1_query(ext_gfeat)                     # [M, h]
            k = self.n1_key(ext_feat)                        # [M, h]
            return (q * k).sum(-1) / (k.size(-1) ** 0.5)     # [M]
        return self.mlp_node1(
            torch.cat([ext_feat, ext_gfeat], dim=1)).squeeze(-1)

    def forward(self, batch: dict):
        return self.MLE_forward(batch)

    # ─────────────────────────────────────────────────────────────────────────
    # NLL forward
    # ─────────────────────────────────────────────────────────────────────────

    def MLE_forward(self, batch: dict):
        graph            = batch["graph"]                    # PackedGraph
        node1_lb         = batch["node1"].long()             # [B]  local compact index
        node2_lb         = batch["node2"].long()             # [B]  local compact or g_num
        var_lb           = batch["var"].long()               # [B]  global var idx
        neg_lb           = batch["neg"].long()               # [B]  0=pos 1=neg
        node2_safety_m   = batch["node2_safety_mask"].bool() # [B, MAX_G_NODES]
        safety_mask      = batch["safety_mask"].bool()       # [B, N_VARS, 2]

        device = next(self.parameters()).device
        B = graph.batch_size
        total_N = graph.num_node

        # ── MPNN encoding (+ optional global attention) ────────────────────
        node_feat, graph_feat = self._encode(graph)  # [total_N, h], [B, 2h]

        # ── Extended node space (actual + one virtual per graph) ───────────
        virt_feat = self.new_node_emb.unsqueeze(0).expand(B, -1)  # [B, h]
        ext_feat  = torch.cat([node_feat, virt_feat], dim=0)      # [total_N+B, h]

        n2g      = graph.node2graph                                # [total_N]
        ext_n2g  = torch.cat([n2g, torch.arange(B, device=device)])  # [total_N+B]
        ext_gfeat = graph_feat[ext_n2g]                            # [total_N+B, 2h]

        # ── G-node mask ────────────────────────────────────────────────────
        starts      = graph.num_cum_nodes - graph.num_nodes        # [B]
        local_idx   = torch.arange(total_N, device=device) - starts[n2g]
        is_g_actual = local_idx < graph.g_num_nodes[n2g]          # [total_N]
        is_g_ext    = torch.cat([is_g_actual,
                                 torch.ones(B, dtype=torch.bool, device=device)])  # [total_N+B]

        # ── Node-1 loss ────────────────────────────────────────────────────
        n1_logits = self._n1_logits(ext_feat, ext_gfeat)           # [total_N+B]
        n1_logits = n1_logits.masked_fill(~is_g_ext, -1e9)

        node1_global = starts + node1_lb                           # [B]
        n1_logprob   = scatter_log_softmax(n1_logits, ext_n2g)
        loss_node1   = -(n1_logprob[node1_global]).mean()

        # ── Node-2 loss ────────────────────────────────────────────────────
        n1_feat_per_ext = node_feat[node1_global][ext_n2g]        # [total_N+B, h]
        n2_input  = torch.cat([n1_feat_per_ext, ext_feat, ext_gfeat], dim=1)
        n2_logits = self.mlp_node2(n2_input).squeeze(-1)
        n2_raw    = n2_logits                       # unmasked logits, for the GT guard
        n2_logits = n2_logits.masked_fill(~is_g_ext, -1e9)

        n2_exclude = torch.zeros(total_N + B, dtype=torch.bool, device=device)
        n2_exclude[node1_global] = True
        n2_logits = n2_logits.masked_fill(n2_exclude, -1e9)

        # Mask out node2 candidates that have no safe literal for any (var, neg).
        # node2_safety_m[b, j] = True iff compact-node j has ≥1 safe literal
        #   given GT node1 and the partial graph for sample b.
        # For actual G-nodes: compact index = local_idx (clamped for C-nodes, harmless).
        # For virtual slots: compact index = g_num_nodes[b].
        local_clamped   = local_idx.clamp(0, node2_safety_m.size(1) - 1)
        n2_safe_actual  = node2_safety_m[n2g, local_clamped]                         # [total_N]
        n2_safe_virtual = node2_safety_m[torch.arange(B, device=device),
                                         graph.g_num_nodes.clamp(0, node2_safety_m.size(1) - 1)]  # [B]
        n2_safe_ext     = torch.cat([n2_safe_actual, n2_safe_virtual], dim=0)         # [total_N+B]
        n2_logits       = n2_logits.masked_fill(~n2_safe_ext, -1e9)

        is_new_node  = node2_lb == graph.g_num_nodes
        node2_global = torch.where(
            is_new_node,
            total_N + torch.arange(B, device=device),
            starts + node2_lb,
        )
        # Force GT node2 always valid (mirror of the var/sign guard at l.190-194).
        # The GT candidate is safe by construction; if a mask edge-case (e.g. on the
        # larger 6-input graphs) drops it, its NLL would explode to ~1e9 and corrupt
        # the loss/checkpoint selection. Restore its unmasked logit.
        n2_logits = n2_logits.clone()
        n2_logits[node2_global] = n2_raw[node2_global]

        n2_logprob = scatter_log_softmax(n2_logits, ext_n2g)
        loss_node2 = -(n2_logprob[node2_global]).mean()

        # ── Var + sign losses (safety-masked) ─────────────────────────────
        edge_feat   = torch.cat([node_feat[node1_global],
                                 ext_feat[node2_global]], dim=1)   # [B, 2h]
        var_logits  = self.mlp_var(edge_feat)                      # [B, N_VARS]
        sign_logits = self.mlp_sign(edge_feat)                     # [B, 2]

        # Force GT always valid (GT transistors are always safe by construction,
        # but guard against floating-point / dataset edge cases).
        arange_B = torch.arange(B, device=device)
        safety_mask = safety_mask.clone()
        safety_mask[arange_B, var_lb, neg_lb] = True

        var_valid    = safety_mask.any(dim=2)                      # [B, N_VARS]
        var_logits_m = var_logits.masked_fill(~var_valid, -1e9)
        loss_var     = F.nll_loss(F.log_softmax(var_logits_m, dim=-1), var_lb)

        sign_valid    = safety_mask[arange_B, var_lb]              # [B, 2]
        sign_logits_m = sign_logits.masked_fill(~sign_valid, -1e9)
        loss_sign     = F.nll_loss(F.log_softmax(sign_logits_m, dim=-1), neg_lb)

        # ── Aggregate ──────────────────────────────────────────────────────
        total_loss = (self.w_node1 * loss_node1
                      + loss_node2 + loss_var + loss_sign)

        metric = {
            "loss/node1": loss_node1.item(),
            "loss/node2": loss_node2.item(),
            "loss/var":   loss_var.item(),
            "loss/sign":  loss_sign.item(),
            "loss/total": total_loss.item(),
        }
        metric.update(self._accuracy(
            n1_logits, n2_logits, node1_global, node2_global,
            var_lb, neg_lb, ext_n2g,
        ))
        return total_loss, metric

    # ─────────────────────────────────────────────────────────────────────────
    # RL finetuning (off-policy PPO)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def agent(self) -> "GCPNTransNet":
        """Frozen copy of this model used for trajectory generation."""
        if self._agent is None:
            self.sync_agent()
        return self._agent

    def sync_agent(self) -> None:
        """Replace frozen agent with a snapshot of current weights."""
        # Temporarily clear _agent before deepcopy so the copy does not
        # contain a nested _agent (which would itself contain _agent._agent, …).
        object.__setattr__(self, '_agent', None)
        new_agent = copy.deepcopy(self)
        for p in new_agent.parameters():
            p.requires_grad_(False)
        new_agent.eval()
        object.__setattr__(self, '_agent', new_agent)

    def reinforce_forward(
        self,
        func_list,                  # list of (fid, vars_in_func, on_patterns, off_patterns, t_opt)
        num_traj:       int   = 4,
        max_steps:      int   = 30,
        temperature:    float = 0.8,
        clip_eps:       float = 0.2,
        lambda_entropy: float = 0.01,
    ) -> "torch.Tensor | None":
        """
        Off-policy PPO with GRPO baseline and terminal-only reward.

        Reward:
          terminal_r = best_T[fid] / T   if complete  (1.0 when T matches best ever)
                     = 0.0                if partial
          No per-step reward.  All steps in a trajectory receive terminal_r.

        Baseline: GRPO — within-group mean of terminal_r across num_traj trajectories.
        Once the model discovers a shorter circuit, best_T[fid] tightens automatically
        so longer complete solutions are penalised in subsequent batches.
        """
        all_steps:      list[TrajStep] = []
        all_advantages: list[float]    = []

        with torch.no_grad():
            for fid, vars_in_func, on_patterns, off_patterns, _t_opt in func_list:

                # ── Generate all num_traj trajectories with ONE MPNN call/step
                raw_trajs = []
                for g_tran, g_num, status, steps in self._generate_K_trajs(
                    vars_in_func, on_patterns, off_patterns,
                    num_traj, max_steps, temperature
                ):
                    if not steps:
                        continue
                    T        = len(steps)
                    cov      = covered_patterns(g_tran, g_num, on_patterns)
                    complete = (cov == on_patterns)
                    # effective size: functionally redundant transistors are
                    # free to remove (removal only reduces conduction), so
                    # reward/best tracking use the functionally-pruned count.
                    T_eff = (len(prune_gtran_functional(g_tran, g_num, on_patterns))
                             if complete else T)
                    if complete:
                        prev = self._best_t.get(fid)
                        if prev is None or T_eff < prev:
                            self._best_t[fid] = T_eff
                    raw_trajs.append((T, T_eff, complete, steps))

                if not raw_trajs:
                    continue

                # ── Pass 2: assign terminal reward, compute GRPO advantage ─
                best_T = self._best_t.get(fid)
                rewards = []
                for T, T_eff, complete, steps in raw_trajs:
                    if complete:
                        r = (best_T / T_eff) if best_T else 1.0
                    else:
                        r = 0.0
                    rewards.append(r)

                group_mean = sum(rewards) / len(rewards)
                group_std  = (sum((r - group_mean) ** 2
                                  for r in rewards) / len(rewards)) ** 0.5

                for (T, T_eff, complete, steps), r in zip(raw_trajs, rewards):
                    adv = (r - group_mean) / (group_std + 1e-8)
                    # same advantage for every step in this trajectory
                    all_advantages.extend([adv] * T)
                    all_steps.extend(steps)

        if not all_steps:
            return None

        # Global normalization for stability across functions in the batch
        adv_t = torch.tensor(all_advantages, dtype=torch.float)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        all_advantages = adv_t.tolist()

        return self._ppo_loss_batched(all_steps, all_advantages, clip_eps, lambda_entropy)

    def reinforce_forward_shaped(
        self,
        func_list,                  # list of (fid, vars_in_func, on_patterns, off_patterns, t_opt)
        num_traj:       int   = 16,
        max_steps:      int   = 30,
        temperature:    float = 1.0,
        clip_eps:       float = 0.2,
        lambda_entropy: float = 0.05,
        w_partial:      float = 0.5,
        anchor_beta:    float = 0.0,
    ) -> "torch.Tensor | None":
        """
        Shaped-reward variant of reinforce_forward, designed to escape the two
        local-optimum traps of the original terminal-only `best_T/T` reward:

          Trap 1 (reward caps at 1.0 -> no pull below current best): we reward a
          complete network against the PREVIOUS best (snapshot before this batch
          updates it), so a *new shortest* circuit gets reward > 1.0 and a strong
          positive advantage, keeping pressure to shorten even after convergence.

          Trap 2 (incomplete -> reward 0 -> no signal on hard functions): we give
          partial credit proportional to on-pattern coverage, so the policy gets
          gradient toward completion instead of flatlining at 0 (the 6-input/NSP
          failure mode).

          Trap 3 (anchor_beta > 0; converged-suboptimal -> zero advantage): when
          every rollout produces the SAME suboptimal network, group-relative
          advantages are all zero and learning stops. anchor_beta adds an
          ABSOLUTE term beta*(r - 1.0) (1.0 = reward of exactly reaching t_ref),
          so a uniformly-suboptimal group receives uniformly negative advantage
          — PPO then pushes probability off the entrenched habit, re-opening
          exploration. With anchor_beta the final advantages are NOT re-centered
          (mean-preserving scaling only), otherwise the uniform signal would be
          normalized away.

        Reward:
          incomplete : r = w_partial * (covered / total_on)          in [0, w_partial)
          complete   : r = w_partial + (1 - w_partial) * (ref / T)   ( = 1.0 at T==ref,
                       > 1.0 when shorter than ref )  where ref = previous best
                       (or the shortest complete this batch on the very first hit).
        Complete networks always out-reward incomplete ones (>= w_partial).
        """
        all_steps:      list[TrajStep] = []
        all_advantages: list[float]    = []

        with torch.no_grad():
            for fid, vars_in_func, on_patterns, off_patterns, _t_opt in func_list:
                total_on  = len(on_patterns)
                prev_best = self._best_t.get(fid)        # snapshot BEFORE this batch

                raw_trajs = []
                for g_tran, g_num, status, steps in self._generate_K_trajs(
                    vars_in_func, on_patterns, off_patterns,
                    num_traj, max_steps, temperature
                ):
                    if not steps:
                        continue
                    T        = len(steps)
                    cov      = len(covered_patterns(g_tran, g_num, on_patterns))
                    complete = (cov == total_on)
                    # effective size: functionally redundant transistors are
                    # free to remove (removal only reduces conduction), so
                    # reward/best tracking use the functionally-pruned count.
                    T_eff = (len(prune_gtran_functional(g_tran, g_num, on_patterns))
                             if complete else T)
                    if complete:
                        prev = self._best_t.get(fid)
                        if prev is None or T_eff < prev:
                            self._best_t[fid] = T_eff
                    raw_trajs.append((T, T_eff, complete, cov, steps))

                if not raw_trajs:
                    continue

                # ref for the complete-reward: previous best, else shortest complete
                # found in THIS batch (so the first discovery is the baseline).
                complete_Ts = [T_eff for _T, T_eff, c, _cov, _s in raw_trajs if c]
                ref = prev_best if prev_best is not None else (
                    min(complete_Ts) if complete_Ts else None)

                rewards = []
                for T, T_eff, complete, cov, steps in raw_trajs:
                    if complete:
                        base = ref if ref else T_eff
                        r = w_partial + (1.0 - w_partial) * (base / T_eff)
                    else:
                        r = w_partial * (cov / total_on if total_on else 0.0)
                    rewards.append(r)

                group_mean = sum(rewards) / len(rewards)
                group_std  = (sum((r - group_mean) ** 2
                                  for r in rewards) / len(rewards)) ** 0.5

                for (T, T_eff, complete, cov, steps), r in zip(raw_trajs, rewards):
                    adv = (r - group_mean) / (group_std + 1e-8)
                    if anchor_beta > 0:
                        adv += anchor_beta * (r - 1.0)
                    all_advantages.extend([adv] * T)
                    all_steps.extend(steps)

        if not all_steps:
            return None

        adv_t = torch.tensor(all_advantages, dtype=torch.float)
        if anchor_beta > 0:
            # Mean-preserving: scale but do NOT re-center, so the uniform
            # negative anchor signal of a converged-suboptimal group survives.
            adv_t = (adv_t / (adv_t.std() + 1e-8).clamp(min=1.0)).clamp(-3.0, 3.0)
        else:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        all_advantages = adv_t.tolist()

        return self._ppo_loss_batched(all_steps, all_advantages, clip_eps, lambda_entropy)

    def reinforce_forward_dense(
        self,
        func_list,                  # list of (fid, vars_in_func, on_patterns, off_patterns, t_ref)
        num_traj:       int   = 16,
        max_steps:      int   = 30,
        temperature:    float = 1.0,
        clip_eps:       float = 0.2,
        lambda_entropy: float = 0.05,
        cov_bonus:      float = 1.0,
        step_cost:      float = 0.03,
        complete_bonus: float = 1.0,
        sil_weight:     float = 0.2,
        novelty_bonus:   float = 0.0,
        beat_bonus:      float = 0.0,
        sil_subopt_scale: float = 1.0,
    ) -> "torch.Tensor | None":
        """
        Core-RL upgrade over the shaped variant (requires value_head=True):

          * DENSE PER-STEP REWARD: r_t = cov_bonus * (newly covered on-patterns)
            / |on| - step_cost, plus complete_bonus * (t_ref / T_eff) at the
            final step of a complete network (T_eff = functionally pruned size).
            Every transistor placement is judged by its own contribution.
          * LEARNED CRITIC: per-step advantage = return-to-go - V(s_t), so
            credit assignment is per-state and never degenerates when all
            rollouts agree (the GRPO group-std failure mode).
          * SELF-IMITATION (SIL): the best complete trajectory per target is
            buffered and replayed each update with positive-part advantage
            max(R - V, 0) — rare good discoveries are consolidated instead of
            washed out by the next on-policy batch.
        """
        assert self.use_value, "reinforce_forward_dense requires value_head=True"
        from .graph import prune_gtran_functional as _pf

        all_steps: list[TrajStep] = []
        all_returns: list[float] = []

        with torch.no_grad():
            for fid, vars_in_func, on_patterns, off_patterns, t_ref in func_list:
                total_on = len(on_patterns)
                for g_tran, g_num, status, steps in self._generate_K_trajs(
                    vars_in_func, on_patterns, off_patterns,
                    num_traj, max_steps, temperature
                ):
                    if not steps:
                        continue
                    T = len(steps)
                    gnum_seq = [steps[t + 1].g_num for t in range(T - 1)] + [g_num]
                    rewards = []
                    prev_cov = 0
                    for t in range(T):
                        c = len(covered_patterns(g_tran[:t + 1], gnum_seq[t],
                                                 on_patterns))
                        rewards.append(cov_bonus * (c - prev_cov) / total_on
                                       - step_cost)
                        prev_cov = c
                    complete = (prev_cov == total_on)
                    if complete:
                        pruned = _pf(g_tran, g_num, on_patterns)
                        T_eff  = len(pruned)
                        prevb  = self._best_t.get(fid)
                        is_beat = (prevb is None or T_eff < prevb)
                        if is_beat:
                            self._best_t[fid] = T_eff
                        ref = t_ref if t_ref else self._best_t[fid]
                        rewards[-1] += complete_bonus * (ref / T_eff)
                        # v4: jackpot for strictly beating the running best —
                        # sharpens the vanishing ref/T_eff gradient at discovery.
                        if beat_bonus and is_beat and prevb is not None:
                            rewards[-1] += beat_bonus
                        # v4: novelty — reward a structurally-new complete net so
                        # the policy explores alternative topologies instead of
                        # re-emitting the entrenched near-optimum.
                        if novelty_bonus:
                            seen = self._seen_structs.setdefault(fid, set())
                            h = tuple(sorted(pruned))
                            if h not in seen:
                                seen.add(h)
                                rewards[-1] += novelty_bonus
                    # returns-to-go (undiscounted: episodes are short)
                    R = [0.0] * T
                    acc = 0.0
                    for t in range(T - 1, -1, -1):
                        acc += rewards[t]
                        R[t] = acc
                    all_steps.extend(steps)
                    all_returns.extend(R)
                    # SIL buffer: best (smallest T_eff) complete trajectory
                    if complete:
                        cur = self._best_traj.get(fid)
                        if cur is None or T_eff < cur[0]:
                            self._best_traj[fid] = (T_eff, list(steps), list(R))

                # replay buffered best trajectory alongside on-policy steps
                bt = self._best_traj.get(fid)
                sil_steps, sil_returns = (bt[1], bt[2]) if bt else ([], [])
                # v4: when the buffered best is still SUBOPTIMAL (> t_ref),
                # down-weight SIL so it stops cementing the local optimum and
                # leaves room for exploration to find the shorter net.
                eff_sil = sil_weight
                if bt and t_ref and bt[0] > t_ref:
                    eff_sil = sil_weight * sil_subopt_scale

        if not all_steps:
            return None
        return self._ppo_loss_dense(all_steps, all_returns,
                                    sil_steps, sil_returns,
                                    clip_eps, lambda_entropy, eff_sil)

    def _ppo_loss_dense(
        self,
        steps: "list[TrajStep]",
        returns: "list[float]",
        sil_steps: "list[TrajStep]",
        sil_returns: "list[float]",
        clip_eps: float,
        lambda_entropy: float,
        sil_weight: float,
    ) -> torch.Tensor:
        """PPO with per-step critic advantages + value loss + SIL replay.
        Single batched MPNN forward over on-policy AND SIL step graphs."""
        from torchdrug import data as td_data

        device = next(self.parameters()).device
        n_main = len(steps)
        all_s = steps + sil_steps
        all_R = torch.tensor(returns + sil_returns, dtype=torch.float,
                             device=device)

        packed = td_data.Graph.pack([s.graph for s in all_s]).to(device)
        nf_all, gf_all = self._encode(packed)
        values = self.value_fn(gf_all).squeeze(-1)          # [S_total]

        starts    = (packed.num_cum_nodes - packed.num_nodes).tolist()
        num_nodes = packed.num_nodes.tolist()

        # per-step log-probs under the current policy (same as _ppo_loss_batched)
        edge_ins = []
        for i, step in enumerate(all_s):
            nf_i  = nf_all[starts[i]: starts[i] + num_nodes[i]]
            ext_i = torch.cat([nf_i, self.new_node_emb.unsqueeze(0)], dim=0)
            edge_ins.append(torch.cat([nf_i[step.node1], ext_i[step.node2_ext]],
                                      dim=0))
        edge_in_batch = torch.stack(edge_ins, dim=0)
        var_logits_b  = self.mlp_var(edge_in_batch)
        sgn_logits_b  = self.mlp_sign(edge_in_batch)

        def _entropy(logits):
            p = F.softmax(logits, dim=0)
            return -(p * (p + 1e-10).log()).sum()

        # advantages: main = R - V (normalized); SIL = max(R - V, 0)
        adv_main = (all_R[:n_main] - values[:n_main].detach())
        if n_main > 1:
            adv_main = (adv_main - adv_main.mean()) / (adv_main.std() + 1e-8)
        adv_sil = (all_R[n_main:] - values[n_main:].detach()).clamp(min=0.0)

        pg_terms, sil_terms, entropy_terms = [], [], []
        for i, step in enumerate(all_s):
            N_i   = num_nodes[i]
            nf_i  = nf_all[starts[i]: starts[i] + N_i]
            gf_i  = gf_all[i]
            ext_i = torch.cat([nf_i, self.new_node_emb.unsqueeze(0)], dim=0)
            ext_g = gf_i.unsqueeze(0).expand(N_i + 1, -1)

            n1_logits = self._n1_logits(ext_i, ext_g).masked_fill(
                ~step.n1_mask.to(device), -1e9)
            lp = F.log_softmax(n1_logits, dim=0)[step.node1]
            n1f_exp = nf_i[step.node1].unsqueeze(0).expand(N_i + 1, -1)
            n2_logits = self.mlp_node2(
                torch.cat([n1f_exp, ext_i, ext_g], dim=1)
            ).squeeze(-1).masked_fill(~step.n2_mask.to(device), -1e9)
            lp = lp + F.log_softmax(n2_logits, dim=0)[step.node2_ext]
            var_m = var_logits_b[i].masked_fill(~step.var_valid.to(device), -1e9)
            sgn_m = sgn_logits_b[i].masked_fill(~step.sign_valid.to(device), -1e9)
            lp = lp + F.log_softmax(var_m, dim=0)[step.var_idx]
            lp = lp + F.log_softmax(sgn_m, dim=0)[step.is_neg]

            if i < n_main:
                old_logp = torch.tensor(step.old_logp, device=device)
                ratio = (lp - old_logp).exp()
                a = adv_main[i]
                pg_terms.append(-torch.min(
                    ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a))
                if lambda_entropy > 0:
                    entropy_terms.append(_entropy(n1_logits) + _entropy(n2_logits)
                                         + _entropy(var_m) + _entropy(sgn_m))
            else:
                # SIL: A2C-style with positive-part advantage
                sil_terms.append(-lp * adv_sil[i - n_main])

        loss = torch.stack(pg_terms).mean()
        loss = loss + 0.5 * F.mse_loss(values[:n_main], all_R[:n_main])
        if sil_terms:
            loss = loss + sil_weight * torch.stack(sil_terms).mean()
            loss = loss + 0.5 * sil_weight * (
                (all_R[n_main:] - values[n_main:]).clamp(min=0.0) ** 2).mean()
        if lambda_entropy > 0 and entropy_terms:
            loss = loss - lambda_entropy * torch.stack(entropy_terms).mean()
        return loss

    def _ppo_loss_batched(
        self,
        steps: "list[TrajStep]",
        advantages: "list[float]",
        clip_eps: float,
        lambda_entropy: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute PPO loss for all steps with a single batched MPNN forward pass.

        All step graphs are packed into one PackedGraph; node features are then
        sliced per step for the head MLPs.  Reduces S separate MPNN calls → 1.
        """
        from torchdrug import data as td_data

        device = next(self.parameters()).device
        S      = len(steps)

        # ── Single MPNN forward over all step graphs ───────────────────────
        packed         = td_data.Graph.pack([s.graph for s in steps]).to(device)
        nf_all, gf_all = self._encode(packed)  # [total_N_all, h], [S, 2h]

        starts    = (packed.num_cum_nodes - packed.num_nodes).tolist()  # [S]
        num_nodes = packed.num_nodes.tolist()                           # [S]

        # ── Batch mlp_var and mlp_sign: all steps share the same edge_in dim ─
        # We gather edge_in = [nf[node1], ext_feat[node2_ext]] for each step
        # then run both MLPs in one call before the per-step logit indexing.
        edge_ins = []
        for i, step in enumerate(steps):
            N_i   = num_nodes[i]
            s_i   = starts[i]
            nf_i  = nf_all[s_i : s_i + N_i]
            virt  = self.new_node_emb.unsqueeze(0)
            ext_i = torch.cat([nf_i, virt], dim=0)          # [N_i+1, h]
            edge_ins.append(torch.cat([nf_i[step.node1], ext_i[step.node2_ext]], dim=0))

        edge_in_batch = torch.stack(edge_ins, dim=0)         # [S, 2h]
        var_logits_b  = self.mlp_var(edge_in_batch)          # [S, N_VARS]
        sgn_logits_b  = self.mlp_sign(edge_in_batch)         # [S, 2]

        # ── Per-step: n1, n2 heads (variable-size, cannot batch without padding) ─
        pg_terms      = []
        entropy_terms = []

        def _entropy(logits):
            """Categorical entropy, robust to -inf masked entries."""
            p = F.softmax(logits, dim=0)
            return -(p * (p + 1e-10).log()).sum()

        for i, (step, adv) in enumerate(zip(steps, advantages)):
            N_i   = num_nodes[i]
            s_i   = starts[i]
            nf_i  = nf_all[s_i : s_i + N_i]                 # [N_i, h]
            gf_i  = gf_all[i]                                # [2h]
            virt  = self.new_node_emb.unsqueeze(0)
            ext_i = torch.cat([nf_i, virt], dim=0)           # [N_i+1, h]
            ext_g = gf_i.unsqueeze(0).expand(N_i + 1, -1)   # [N_i+1, 2h]

            n1_mask    = step.n1_mask.to(device)
            n2_mask    = step.n2_mask.to(device)
            var_valid  = step.var_valid.to(device)
            sign_valid = step.sign_valid.to(device)

            n1_logits = self._n1_logits(ext_i, ext_g).masked_fill(~n1_mask, -1e9)
            lp = F.log_softmax(n1_logits, dim=0)[step.node1]

            n1f_exp = nf_i[step.node1].unsqueeze(0).expand(N_i + 1, -1)
            n2_logits = self.mlp_node2(
                torch.cat([n1f_exp, ext_i, ext_g], dim=1)
            ).squeeze(-1).masked_fill(~n2_mask, -1e9)
            lp = lp + F.log_softmax(n2_logits, dim=0)[step.node2_ext]

            var_logits_m = var_logits_b[i].masked_fill(~var_valid, -1e9)
            sgn_logits_m = sgn_logits_b[i].masked_fill(~sign_valid, -1e9)

            lp = lp + F.log_softmax(var_logits_m, dim=0)[step.var_idx]
            lp = lp + F.log_softmax(sgn_logits_m, dim=0)[step.is_neg]

            old_logp = torch.tensor(step.old_logp, device=device)
            adv_t    = torch.tensor(adv, dtype=torch.float, device=device)
            ratio    = (lp - old_logp).exp()
            pg_terms.append(-torch.min(
                ratio * adv_t,
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t,
            ))

            if lambda_entropy > 0:
                entropy_terms.append(
                    _entropy(n1_logits) + _entropy(n2_logits) +
                    _entropy(var_logits_m) + _entropy(sgn_logits_m)
                )

        ppo_loss = torch.stack(pg_terms).mean()
        if lambda_entropy > 0 and entropy_terms:
            ppo_loss = ppo_loss - lambda_entropy * torch.stack(entropy_terms).mean()
        return ppo_loss

    @torch.no_grad()
    def _generate_traj(self, vars_in_func, on_patterns, off_patterns, max_steps, temperature):
        """Generate one trajectory using self.agent. Returns (g_tran, g_num, status, steps)."""
        agent  = self.agent
        device = next(self.parameters()).device
        g_num  = 2
        g_tran = []
        steps  = []
        added  = 0

        while added < max_steps:
            uncovered = on_patterns - covered_patterns(g_tran, g_num, on_patterns)

            graph   = build_gen_graph(g_tran, g_num, vars_in_func, on_patterns).to(device)
            nf, gf  = agent._encode(graph)
            total_N = graph.num_node

            virt_feat = agent.new_node_emb.unsqueeze(0)
            ext_feat  = torch.cat([nf, virt_feat], dim=0)
            ext_gf    = gf.expand(total_N + 1, -1)

            reach   = _reachability_mask(g_tran, g_num, uncovered, device)
            n1_mask = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
            n1_mask[:g_num] = reach
            n1_logits = agent._n1_logits(ext_feat, ext_gf).masked_fill(~n1_mask, -1e9)
            node1     = _pick(n1_logits, n1_mask, temperature, device)
            lp_n1     = F.log_softmax(n1_logits, dim=0)[node1].item()

            n2_mask = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
            n2_mask[:g_num]  = True
            n2_mask[total_N] = True
            n2_mask[node1]   = False

            n2_has_safe = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
            for n2c in range(g_num + 1):
                if n2c == node1:
                    continue
                ng2     = g_num + (1 if n2c == g_num else 0)
                ext_idx = total_N if n2c == g_num else n2c
                for vi in range(N_VARS):
                    for ni in range(2):
                        if check_safety(g_tran + [(node1, n2c, vi, ni)], ng2, off_patterns):
                            n2_has_safe[ext_idx] = True
                            break
                    if n2_has_safe[ext_idx]:
                        break

            n2_mask = n2_mask & n2_has_safe
            if not n2_mask.any():
                break

            n1f_exp   = nf[node1].unsqueeze(0).expand(total_N + 1, -1)
            n2_in     = torch.cat([n1f_exp, ext_feat, ext_gf], dim=1)
            n2_logits = agent.mlp_node2(n2_in).squeeze(-1).masked_fill(~n2_mask, -1e9)
            node2     = _pick(n2_logits, n2_mask, temperature, device)
            lp_n2     = F.log_softmax(n2_logits, dim=0)[node2].item()

            is_new_node = (node2 == total_N)
            node2_c   = g_num if is_new_node else node2
            new_g_num = g_num + 1 if is_new_node else g_num

            safety_m = torch.zeros(N_VARS, 2, dtype=torch.bool, device=device)
            for vi in range(N_VARS):
                for ni in range(2):
                    if check_safety(g_tran + [(node1, node2_c, vi, ni)], new_g_num, off_patterns):
                        safety_m[vi, ni] = True

            if not safety_m.any():
                break

            edge_in    = torch.cat([nf[node1], ext_feat[node2]], dim=0).unsqueeze(0)
            var_valid  = safety_m.any(dim=1)
            var_logits = agent.mlp_var(edge_in).squeeze(0).masked_fill(~var_valid, -1e9)
            var_idx    = _pick_simple(var_logits, temperature)
            lp_var     = F.log_softmax(var_logits, dim=0)[var_idx].item()

            sign_valid = safety_m[var_idx]
            sgn_logits = agent.mlp_sign(edge_in).squeeze(0).masked_fill(~sign_valid, -1e9)
            is_neg     = _pick_simple(sgn_logits, temperature)
            lp_sign    = F.log_softmax(sgn_logits, dim=0)[is_neg].item()

            steps.append(TrajStep(
                graph      = graph.cpu(),
                g_num      = g_num,
                node1      = node1,
                node2_ext  = node2,
                node2_c    = node2_c,
                var_idx    = var_idx,
                is_neg     = is_neg,
                old_logp   = lp_n1 + lp_n2 + lp_var + lp_sign,
                n1_mask    = n1_mask.cpu(),
                n2_mask    = n2_mask.cpu(),
                var_valid  = var_valid.cpu(),
                sign_valid = sign_valid.cpu(),
            ))

            g_tran.append((node1, node2_c, var_idx, is_neg))
            g_num  = new_g_num
            added += 1

            if covered_patterns(g_tran, g_num, on_patterns) == on_patterns:
                return g_tran, g_num, "complete", steps

        status = "complete" if covered_patterns(g_tran, g_num, on_patterns) == on_patterns else "partial"
        return g_tran, g_num, status, steps

    @torch.no_grad()
    def _generate_K_trajs(self, vars_in_func, on_patterns, off_patterns,
                          K, max_steps, temperature):
        """Generate K trajectories with a BATCHED MPNN forward at each step.

        At step t, all K still-active trajectory graphs are packed into one
        PackedGraph and processed with a SINGLE agent.model() call instead of
        K separate calls.  This reduces MPNN invocations from K×T to T (where
        T is the mean trajectory length), giving ~K× speedup on the dominant
        cost (MPNN is 70% of wall time per the profiler).

        Safety masks remain per-trajectory (sequential BFS) because they are
        only 8% of wall time and cannot be trivially batched.

        Returns: list of (g_tran, g_num, status, steps), length K.
        """
        from torchdrug import data as td_data

        agent  = self.agent
        device = next(self.parameters()).device

        # ── Per-trajectory state ───────────────────────────────────────────
        trajs = [{
            "g_tran": [],
            "g_num":  2,
            "steps":  [],
            "done":   False,
            "status": "partial",
        } for _ in range(K)]

        for _ in range(max_steps):
            active = [i for i, s in enumerate(trajs) if not s["done"]]
            if not active:
                break

            # ── Build graphs for all active trajectories ───────────────────
            graphs     = []
            uncovereds = []
            for i in active:
                s = trajs[i]
                unc = on_patterns - covered_patterns(s["g_tran"], s["g_num"],
                                                     on_patterns)
                uncovereds.append(unc)
                graphs.append(
                    build_gen_graph(s["g_tran"], s["g_num"],
                                    vars_in_func, on_patterns)
                )

            # ── ONE batched MPNN forward for all active trajectories ────────
            packed         = td_data.Graph.pack(graphs).to(device)
            nf_all, gf_all = agent._encode(packed)  # [total_N_all, h], [n_active, 2h]
            starts    = (packed.num_cum_nodes - packed.num_nodes).tolist()
            num_nodes = packed.num_nodes.tolist()

            # ── Per-trajectory: heads + safety + action ────────────────────
            for b, traj_idx in enumerate(active):
                s        = trajs[traj_idx]
                unc      = uncovereds[b]
                g_tran   = s["g_tran"]
                g_num    = s["g_num"]
                N_i      = num_nodes[b]
                s_i      = starts[b]
                nf       = nf_all[s_i : s_i + N_i]
                gf       = gf_all[b]
                total_N  = N_i

                virt_feat = agent.new_node_emb.unsqueeze(0)
                ext_feat  = torch.cat([nf, virt_feat], dim=0)
                ext_gf    = gf.expand(total_N + 1, -1)

                # Node 1
                reach   = _reachability_mask(g_tran, g_num, unc, device)
                n1_mask = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
                n1_mask[:g_num] = reach
                n1_logits = agent._n1_logits(ext_feat, ext_gf).masked_fill(~n1_mask, -1e9)
                node1     = _pick(n1_logits, n1_mask, temperature, device)
                lp_n1     = F.log_softmax(n1_logits, dim=0)[node1].item()

                # Node 2 safety mask (BFS — cheap, ~6% of total time)
                n2_mask = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
                n2_mask[:g_num]  = True
                n2_mask[total_N] = True
                n2_mask[node1]   = False
                n2_has_safe = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
                for n2c in range(g_num + 1):
                    if n2c == node1:
                        continue
                    ng2     = g_num + (1 if n2c == g_num else 0)
                    ext_idx = total_N if n2c == g_num else n2c
                    for vi in range(N_VARS):
                        for ni in range(2):
                            if check_safety(g_tran + [(node1, n2c, vi, ni)],
                                            ng2, off_patterns):
                                n2_has_safe[ext_idx] = True
                                break
                        if n2_has_safe[ext_idx]:
                            break
                n2_mask &= n2_has_safe
                if not n2_mask.any():
                    s["done"] = True
                    continue

                # Node 2
                n1f_exp   = nf[node1].unsqueeze(0).expand(total_N + 1, -1)
                n2_in     = torch.cat([n1f_exp, ext_feat, ext_gf], dim=1)
                n2_logits = agent.mlp_node2(n2_in).squeeze(-1).masked_fill(~n2_mask, -1e9)
                node2     = _pick(n2_logits, n2_mask, temperature, device)
                lp_n2     = F.log_softmax(n2_logits, dim=0)[node2].item()

                is_new_node = (node2 == total_N)
                node2_c   = g_num if is_new_node else node2
                new_g_num = g_num + 1 if is_new_node else g_num

                # Var/sign safety mask (BFS — cheap, ~2% of total time)
                safety_m = torch.zeros(N_VARS, 2, dtype=torch.bool, device=device)
                for vi in range(N_VARS):
                    for ni in range(2):
                        if check_safety(g_tran + [(node1, node2_c, vi, ni)],
                                        new_g_num, off_patterns):
                            safety_m[vi, ni] = True
                if not safety_m.any():
                    s["done"] = True
                    continue

                # Var / sign
                edge_in    = torch.cat([nf[node1], ext_feat[node2]], dim=0).unsqueeze(0)
                var_valid  = safety_m.any(dim=1)
                var_logits = agent.mlp_var(edge_in).squeeze(0).masked_fill(~var_valid, -1e9)
                var_idx    = _pick_simple(var_logits, temperature)
                lp_var     = F.log_softmax(var_logits, dim=0)[var_idx].item()
                sign_valid = safety_m[var_idx]
                sgn_logits = agent.mlp_sign(edge_in).squeeze(0).masked_fill(~sign_valid, -1e9)
                is_neg     = _pick_simple(sgn_logits, temperature)
                lp_sign    = F.log_softmax(sgn_logits, dim=0)[is_neg].item()

                s["steps"].append(TrajStep(
                    graph      = graphs[b].cpu(),
                    g_num      = g_num,
                    node1      = node1,
                    node2_ext  = node2,
                    node2_c    = node2_c,
                    var_idx    = var_idx,
                    is_neg     = is_neg,
                    old_logp   = lp_n1 + lp_n2 + lp_var + lp_sign,
                    n1_mask    = n1_mask.cpu(),
                    n2_mask    = n2_mask.cpu(),
                    var_valid  = var_valid.cpu(),
                    sign_valid = sign_valid.cpu(),
                ))
                s["g_tran"].append((node1, node2_c, var_idx, is_neg))
                s["g_num"] = new_g_num

                if covered_patterns(s["g_tran"], s["g_num"], on_patterns) == on_patterns:
                    s["done"]   = True
                    s["status"] = "complete"

        return [
            (s["g_tran"], s["g_num"], s["status"], s["steps"])
            for s in trajs
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        vars_in_func,
        on_patterns,
        off_patterns,
        num_sample: int = 20,
        max_steps: int = 30,
        temperature: float = 0.8,
        verbose: int = 0,
    ):
        """
        Generate transistor networks for a Boolean function.

        Parameters
        ----------
        vars_in_func : list[str]   variables used in the function
        on_patterns  : frozenset   input patterns where function = 1
        off_patterns : frozenset   input patterns where function = 0
        num_sample   : int         number of independent generation trials
        max_steps    : int         max transistors added AND max consecutive
                                   missed steps before declaring a dead end
        temperature  : float       sampling temperature (0 = greedy)
        verbose      : int         0=silent, 1=print per-trial line

        Returns
        -------
        list of dicts, one per trial, with keys:
            g_tran    – list of (u, v, var_idx, is_neg) in compact indices
            g_num     – number of compact G-nodes
            status    – 'complete' | 'partial'
            n_covered – number of on-patterns covered
            safe      – bool, no off-pattern path exists
            correct   – bool, fully covers on-patterns and is safe
        """
        is_training = self.training
        self.eval()

        results = []
        for i in range(num_sample):
            g_tran, g_num, status = self._generate_once(
                vars_in_func, on_patterns, off_patterns, max_steps, temperature
            )
            cov  = covered_patterns(g_tran, g_num, on_patterns)
            safe = check_safety(g_tran, g_num, off_patterns)
            correct = (cov == on_patterns) and safe
            results.append({
                "g_tran":    g_tran,
                "g_num":     g_num,
                "status":    status,
                "n_covered": len(cov),
                "safe":      safe,
                "correct":   correct,
            })
            if verbose:
                sym = "✓" if correct else "·"
                print(f"  [{sym}] trial {i+1:2d}: {len(g_tran):2d} transistors, "
                      f"coverage {len(cov)}/{len(on_patterns)}, "
                      f"safe={safe}, status={status}")

        self.train(is_training)
        return results

    def _generate_once(self, vars_in_func, on_patterns, off_patterns, max_steps, temperature):
        """Single generation trial. Returns (g_tran, g_num, status).

        max_steps counts transistors successfully added, not loop iterations.
        Stops immediately if no safe literal exists for the sampled edge.
        """
        device = next(self.parameters()).device
        g_num  = 2       # SOURCE=0, SINK=1
        g_tran = []      # (u_compact, v_compact, var_idx, is_neg)

        added = 0   # transistors successfully placed

        while added < max_steps:
            uncovered = on_patterns - covered_patterns(g_tran, g_num, on_patterns)

            graph  = build_gen_graph(g_tran, g_num, vars_in_func, on_patterns).to(device)
            nf, gf = self._encode(graph)  # [total_N, h], [1, 2h]
            total_N = graph.num_node

            virt_feat = self.new_node_emb.unsqueeze(0)
            ext_feat  = torch.cat([nf, virt_feat], dim=0)   # [total_N+1, h]
            ext_gf    = gf.expand(total_N + 1, -1)          # [total_N+1, 2h]

            # ── Node 1: reachability-masked ────────────────────────────────
            reach   = _reachability_mask(g_tran, g_num, uncovered, device)
            n1_mask = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
            n1_mask[:g_num] = reach
            n1_logits = self._n1_logits(ext_feat, ext_gf)
            n1_logits = n1_logits.masked_fill(~n1_mask, -1e9)
            node1 = _pick(n1_logits, n1_mask, temperature, device)

            # ── Node 2: mask candidates that have no safe literal ─────────
            # For each candidate node2, check if ≥1 (var, neg) is safe.
            # Only candidates with at least one safe literal are selectable.
            n2_mask = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
            n2_mask[:g_num] = True
            n2_mask[total_N] = True
            n2_mask[node1]   = False

            n2_has_safe = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
            for n2c in range(g_num + 1):          # 0..g_num-1 existing, g_num = virtual
                if n2c == node1:
                    continue
                ng2     = g_num + (1 if n2c == g_num else 0)
                ext_idx = total_N if n2c == g_num else n2c
                for vi in range(N_VARS):
                    for ni in range(2):
                        if check_safety(g_tran + [(node1, n2c, vi, ni)], ng2, off_patterns):
                            n2_has_safe[ext_idx] = True
                            break
                    if n2_has_safe[ext_idx]:
                        break

            n2_mask = n2_mask & n2_has_safe
            if not n2_mask.any():
                break  # no node2 candidate has any safe literal → dead end

            n1f_exp   = nf[node1].unsqueeze(0).expand(total_N + 1, -1)
            n2_in     = torch.cat([n1f_exp, ext_feat, ext_gf], dim=1)
            n2_logits = self.mlp_node2(n2_in).squeeze(-1)
            n2_logits = n2_logits.masked_fill(~n2_mask, -1e9)
            node2 = _pick(n2_logits, n2_mask, temperature, device)

            # ── Resolve compact node2 ──────────────────────────────────────
            is_new_node = (node2 == total_N)
            node2_c   = g_num if is_new_node else node2
            new_g_num = g_num + 1 if is_new_node else g_num

            # ── var/sign safety mask for the chosen (node1, node2) ────────
            safety_m = torch.zeros(N_VARS, 2, dtype=torch.bool, device=device)
            for vi in range(N_VARS):
                for ni in range(2):
                    if check_safety(
                        g_tran + [(node1, node2_c, vi, ni)],
                        new_g_num, off_patterns,
                    ):
                        safety_m[vi, ni] = True

            if not safety_m.any():
                break  # guard: should not trigger given n2_has_safe check above

            # ── Sample var and sign from safe set ─────────────────────────
            edge_in    = torch.cat([nf[node1], ext_feat[node2]], dim=0).unsqueeze(0)
            var_logits = self.mlp_var(edge_in).squeeze(0)
            sgn_logits = self.mlp_sign(edge_in).squeeze(0)

            var_valid = safety_m.any(dim=1)
            var_idx   = _pick_simple(var_logits.masked_fill(~var_valid, -1e9), temperature)
            is_neg    = _pick_simple(sgn_logits.masked_fill(~safety_m[var_idx], -1e9), temperature)

            # ── Add transistor (safe by construction) ─────────────────────
            g_tran.append((node1, node2_c, var_idx, is_neg))
            g_num  = new_g_num
            added += 1

            if covered_patterns(g_tran, g_num, on_patterns) == on_patterns:
                return g_tran, g_num, "complete"

        status = "complete" if covered_patterns(g_tran, g_num, on_patterns) == on_patterns else "partial"
        return g_tran, g_num, status

    # ─────────────────────────────────────────────────────────────────────────
    # Accuracy metrics
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _accuracy(
        self,
        n1_logits, n2_logits, node1_global, node2_global,
        var_lb, neg_lb, ext_n2g,
    ):
        n1_pred = _scatter_argmax(n1_logits, ext_n2g, n1_logits.shape[0])
        n2_pred = _scatter_argmax(n2_logits, ext_n2g, n2_logits.shape[0])
        return {
            "acc/node1": (n1_pred == node1_global).float().mean().item(),
            "acc/node2": (n2_pred == node2_global).float().mean().item(),
        }


def _pick(logits, mask, temperature, device):
    """Masked argmax or multinomial sampling."""
    valid = mask.nonzero(as_tuple=True)[0]
    if temperature <= 0:
        return logits.argmax().item()
    probs = F.softmax(logits[valid] / temperature, dim=-1)
    return valid[torch.multinomial(probs, 1).item()].item()


def _pick_simple(logits, temperature):
    if temperature <= 0:
        return logits.argmax().item()
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item()


def _reachability_mask(g_tran, g_num, uncovered, device):
    """
    [g_num] bool — True if a G-node is reachable from SRC (0) or SNK (1)
    via transistors that are active under at least one uncovered on-pattern.
    SRC and SNK are always True.  Used to prune node1 candidates.
    """
    mask = torch.zeros(g_num, dtype=torch.bool, device=device)
    mask[0] = True
    mask[1] = True
    if not g_tran or not uncovered:
        return mask  # only SRC/SNK exist, or nothing is uncovered

    for pat in uncovered:
        adj: list[list[int]] = [[] for _ in range(g_num)]
        for u, v, var_idx, is_neg in g_tran:
            val = pat[var_idx]
            if (is_neg == 0 and val == 1) or (is_neg == 1 and val == 0):
                adj[u].append(v)
                adj[v].append(u)
        for start in (0, 1):
            visited: set[int] = {start}
            q: deque[int] = deque([start])
            while q:
                n = q.popleft()
                for nb in adj[n]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            for n in visited:
                mask[n] = True

    return mask



def _scatter_argmax(
    src: torch.Tensor,
    index: torch.Tensor,
    total_nodes: int,
) -> torch.Tensor:
    """
    Return the global argmax position within each group defined by `index`.
    Returns tensor of shape [num_groups].
    """
    B = int(index.max().item()) + 1
    _, argmax = scatter_max(src, index, dim_size=B)
    return argmax  # [B] global position of max in src for each group
