# ASTRAN Integration

**Relevant files:** `physical_finetune_rl_transnet.py`, `place_spice.py`, `generate_physical_transnet.py`

---

## What ASTRAN Does

ASTRAN is a standard-cell transistor placement and routing tool. Given a SPICE netlist describing a CMOS cell, it:
1. **Folds** the transistor stacks into rows
2. **Places** transistors within rows to minimize the cost objective
3. Reports: Width (CPP), Gate Mismatches, Wire Length (WL), Routing Density, Nr. Gaps

---

## Binary Location

```python
_ASTRAN_BIN = os.path.join(os.path.dirname(__file__),
    "astran/Astran/build/bin/Astran.app/Contents/MacOS/Astran")
_TECH_FILE  = os.path.join(os.path.dirname(__file__),
    "astran/Astran/build/Work/tech_freePDK45.rul")
```

---

## Correct Invocation

```bash
Astran --shell script.run
```

**WRONG:** `Astran script.run` → launches Cocoa GUI → hangs forever on headless machines (exit code 124, timeout)  
**RIGHT:** `Astran --shell script.run` → batch/shell mode → executes script, then exits

```python
result = subprocess.run(
    [astran_bin, "--shell", sc_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,   # SA annealing progress dots → discard
    timeout=timeout,             # default 30s
)
```

---

## Script Format (`.run` file)

```
load technology "/path/to/tech_freePDK45.rul"
load netlist "/path/to/cell.sp"
cellgen select TRANSSYN
cellgen fold 2 0
cellgen place 1 1 3 4 1 4 2
```

**`cellgen fold 2 0`**: fold into 2 rows, 0 finger splits. Minimum folding for small cells — preserves topology without splitting transistors into multiple fingers.

**`cellgen place <SA_quality> <#attempts> <W> <Mis> <Rt> <Dens> <Gaps>`**:
- `1 1` = quality and attempts (fast, single SA run)
- `3 4 1 4 2` = weights matching `_OBJ_*` Python constants

**No explicit `exit` needed:** `--shell` mode exits after the script finishes.

---

## SPICE Format Requirements

Critical constraints for ASTRAN freePDK45:

| Requirement | Correct | WRONG (causes crash) |
|-------------|---------|---------------------|
| Power net (N-side) | `GND` | `VSS` (SIGBUS in fold) |
| Power net (P-side) | `VCC` | `VDD` (SIGBUS in fold) |
| NMOS model | `NMOS_VTL` | `NMOS` (model not found) |
| PMOS model | `PMOS_VTL` | `PMOS` (model not found) |
| Tech lib file | `library45.sp` | `lib65.sp` |

Model names `NMOS_VTL`/`PMOS_VTL` come from `library45.sp` (freePDK45). The `VTL` suffix = Virtual Threshold Low (nominal threshold voltage variant).

**Transistor widths** (minimum size for freePDK45):
```python
_W_N = "0.090000U"   # NMOS minimum width
_W_P = "0.090000U"   # PMOS minimum width (same as NMOS for now)
_L   = "0.050000U"   # gate length = 50 nm
```

---

## SPICE Netlist Structure (`build_spice`)

```spice
.SUBCKT TRANSSYN a b c a_N b_N c_N ZN VCC GND
* PDN — NMOS_VTL — implements !f(x)
MN0  ZN    a     GND   GND  NMOS_VTL  W=0.090000U  L=0.050000U
MN1  nd2   b     GND   GND  NMOS_VTL  W=0.090000U  L=0.050000U
...
* PUN — PMOS_VTL — implements f(!x)
MP0  ZN    a_N   VCC   VCC  PMOS_VTL  W=0.090000U  L=0.050000U
...
.ENDS
```

**Port order:** `<inputs> <inverted inputs> ZN VCC GND`  
- `vif` = variable names (e.g., `['a','b','c']`)  
- Inverted inputs named `<var>_N` (e.g., `a_N`) — used as gate signals for negative literals

**Node naming:**
- PDN internal: `nd{k}` (e.g., `nd2`, `nd3`)
- PUN internal: `np{k}` (e.g., `np2`, `np3`)
- Node 0 (SOURCE) → `GND` (PDN) or `VCC` (PUN)
- Node 1 (SINK) → `ZN`

---

## Output Parsing

ASTRAN prints one key line to stdout:
```
Final cost: Width=19; Gate Mismatches=0; WL=57; Rt. Density=6; Nr. Gaps=4
```

Parsed by regex:
```python
_COST_RE = re.compile(
    r"Final cost: Width=(\d+).*?Gate Mismatches=(\d+).*?WL=(\d+)"
    r".*?Rt\. Density=(\d+).*?Nr\. Gaps=(\d+)"
)
```

Returns dict: `{"width": int, "gate_mismatches": int, "wl": int, "rt_density": int, "nr_gaps": int}`  
Returns `None` on timeout or failure.

---

## CPP Formula

```
Cell Width = 2 × max(T_PDN, T_PUN) + 1 + N_gap_positions
```

Where:
- `T_PDN`, `T_PUN` = transistor count in each network
- `+1` = one mandatory poly pitch for the cell boundary
- `N_gap_positions` = number of positions where both P and N rows have a diffusion cut simultaneously

Example: XOR3 with 8+8 transistors: `2×8 + 1 + 1 = 18` (one shared gap position, reported as `Nr.Gaps=2` but contributes only +1 to width since the P and N cuts are co-located).

---

## ASTRAN Errors Fixed

| Error | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| GUI hang | Exit code 124 (timeout) | `Astran script.run` launches Cocoa GUI | Use `Astran --shell script.run` |
| SIGBUS (exit 138) | Crash during fold | VDD/VSS nets used instead of VCC/GND | Change all power nets to `VCC`/`GND` |
| Model not found | ASTRAN silent fail | `NMOS`/`PMOS` not in freePDK45 lib | Use `NMOS_VTL`/`PMOS_VTL` |

---

## `place_spice.py` — External SPICE Placement

Places an arbitrary SPICE file (e.g., from a third-party tool):

```bash
env/bin/python3 place_spice.py /path/to/NET.sp CELL_NAME
```

Preprocessing steps:
1. Remove `//` comments (→ `*`)
2. Remove `.model` lines
3. Remove continuation lines (starting with `+`)
4. Fix `.end` → `.ends`

Then runs the standard ASTRAN flow and reports all 5 metrics + objective.
