# AAAI Paper: TransSGPN

**Title:** `TransSGPN: Transistor Network Synthesis via Switching Graph Policy Network`  
**Venue:** AAAI 2027 (anonymous submission)  
**Paper directory:** `/Users/fch/Python/TranSGPN/aaai-2026-transgpn/`  
**GitHub:** `FCHXWH823/aaai-2026-transgpn` (private repo)  
**Overleaf ZIP:** `/Users/fch/Python/GCPN/aaai-2026-transgpn.zip` (upload via Overleaf → New Project → Upload)

---

## LaTeX Setup

**Main file:** `aaai-2026-transgpn/main.tex`

```latex
\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2027}   % anonymous; remove [submission] for camera-ready
```

**Key macros (main.tex:54-56):**
```latex
\newcommand{\method}{TransSGPN}
\newcommand{\sgpn}{Switching Graph Policy Network\xspace}
\newcommand{\SGPN}{SGPN\xspace}
```

**Theorem environments (main.tex:39-46):**
```latex
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}   % added in this session
\newtheorem{definition}{Definition}
```

**Review markup (remove before submission):**
```latex
\newcommand{\TODO}[1]{{\color{red}\textbf{[TODO: #1]}}}
\newcommand{\NOTE}[1]{{\color{blue}\textit{[#1]}}}
```

---

## Paper Structure

| File | Section | Content |
|------|---------|---------|
| `0-abstract.tex` | Abstract | Problem, method overview, results teaser |
| `1-introduction.tex` | §1 Introduction | Motivation, contributions |
| `2-preliminaries.tex` | §2 Preliminaries | Background only (no contributions) |
| `3-methodology.tex` | §3 Methodology | All theoretical/technical contributions |
| `4-experiments.tex` | §4 Experiments | Results (mostly `\TODO{}` placeholders) |
| `5-conclusion.tex` | §5 Conclusion | Summary and future work |

**Page limit:** AAAI 8 pages. Currently at limit — experiments section needs real numbers without overflow.

---

## Section 2: Preliminaries (background only)

- **§2.1 Transistor Networks and Switching Graphs:** Definition: Switching Graph, Definition: Transistor Network, correctness/safety criteria, prior work (ASTRAN exhaustive search)
- **§2.2 CMOS Cell Synthesis as Dual Switching Network Problems:** Definition: PDN/PUN, Proposition 1: Dual Switching Network Equivalence (PDN=¬f(x), PUN=f(¬x)) with proof
- **§2.3 Placement and Routing:** CPP formula, ASTRAN tool description

**Structuring decision:** Proposition 1 and all theorems were moved FROM preliminaries INTO methodology. Preliminaries contains only established background knowledge, not our contributions.

---

## Section 3: Methodology (contributions)

### §3.1 Theoretical Foundations
- **Theorem 1 (Monotonicity):** Adding a transistor never reduces coverage
- **Corollary 1 (Safety):** The safety mask is consistent with all correct solutions
- **Theorem 2 (SOP Correctness):** Sum-of-products topologies are always reachable
- **Theorem 3 (Completeness):** The safety mask never prunes all valid next actions

### §3.2 MDP Formulation
- State: partial switching graph G_t
- Action: (node1, node2, variable, polarity) — factored
- Two masks: reachability mask (node1) + safety mask (node2, var, sign)
- Reward equation (terminal-only)

### §3.3 SGPN Architecture
- MPNN backbone (full name: Message Passing Neural Network, citation added)
- Five policy heads with input dimensions
- Figure 1 reference: `fig/sgpn_arch.tex`

### §3.4 Physical-Aware RL (Phase 3)
- PDN/PUN joint synthesis setup
- ASTRAN closed-loop reward
- Algorithm 2 pseudocode
- `\label{sec:physical}` for cross-references
- Figure 2 reference: `fig/training_flow.tex`

### Hyperparameter Table
- Phase 1, Phase 2, Phase 3 rows
- Includes `\TODO{}` for some values pending real experiments

---

## Figures

### `fig/sgpn_arch.tex`
TikZ diagram: `unified G∪C graph → MPNN encoder → [node1 head | node2 head | var head | sign head]` with safety mask annotations.

### `fig/training_flow.tex`
TikZ 3-phase training flow:
```
Dataset → [Phase 1: NLL] → [Phase 2: PPO count] → [Phase 3: PPO physical/ASTRAN]
                ↓                    ↓                         ↓
          pretrain_v6.pt      transnet_rl_v1.pt         transnet_physical_v1.pt
```

---

## References

**Bibliography file:** `references.bib` (primary), `aaai2027.bib` (AAAI template refs)

**MPNN citation:** Added when MPNN first appears in §3.3 — cite the original MPNN paper (Gilmer et al., 2017 or torchdrug's formulation).

---

## Build

```bash
cd aaai-2026-transgpn
make          # runs pdflatex + bibtex
# or manually:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Output: `main.pdf`

---

## Pending TODO Items

All marked with `\TODO{}` in red in the PDF:

1. **Experiments §4:** Real transistor count comparison table (TransSGPN vs. ASTRAN synthesis vs. baseline)
2. **Experiments §4:** Success rate table across all 242 3-input functions
3. **GPU name:** `\TODO{GPU model}` in experimental setup
4. **Dataset stats:** Exact counts of training/test split
5. **Physical RL results:** CPP comparison table (needs Phase 3 training to complete)
6. **Ablation study:** Phase 1 only vs. Phase 1+2 vs. Phase 1+2+3

---

## Overleaf Import Options

**Free plan:** Upload ZIP → Overleaf → New Project → Upload Project  
**Premium plan:** GitHub sync → import from `FCHXWH823/aaai-2026-transgpn`  

ZIP location: `/Users/fch/Python/GCPN/aaai-2026-transgpn.zip`  
To recreate: `cd /Users/fch/Python/TranSGPN && zip -r /Users/fch/Python/GCPN/aaai-2026-transgpn.zip aaai-2026-transgpn/`

---

## Architecture Name History

The network was renamed several times before settling:
1. GCPN (rejected: existing term in molecular graph generation literature)
2. TransUSGPN (Transistor Unified Switching Graph Policy Network) — "Unified" removed per user
3. **TransSGPN** (Transistor Switching Graph Policy Network) — **current name**
