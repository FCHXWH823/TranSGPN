"""
place_spice.py

Clean and place an existing SPICE netlist with ASTRAN using the same
placement parameters as the TransSyn physical RL flow.

Handles:
  - // comments → * comments
  - .model lines → removed (ASTRAN ignores device models)
  - + continuation lines → removed (ad/pd/as/ps not needed for placement)
  - .end → .ends <cellname>

Usage:
  python place_spice.py --spice astran/Astran/build/Work/NET.sp --cell NET
  python place_spice.py --spice results/xor3_best.sp --cell XOR3_TRANSSYN
"""
import argparse
import os
import re
import subprocess
import tempfile

_ASTRAN_BIN = os.path.join(os.path.dirname(__file__),
              "astran/Astran/build/bin/Astran.app/Contents/MacOS/Astran")
_TECH_FILE  = os.path.join(os.path.dirname(__file__),
              "astran/Astran/build/Work/tech_freePDK45.rul")

_OBJ_CPP      = 3
_OBJ_MISMATCH = 4
_OBJ_ROUTING  = 1
_OBJ_DENSITY  = 4
_OBJ_GAPS     = 2

_COST_RE = re.compile(
    r"Final cost: Width=(\d+).*?Gate Mismatches=(\d+).*?WL=(\d+)"
    r".*?Rt\. Density=(\d+).*?Nr\. Gaps=(\d+)"
)
_PMOS_ORDER_RE = re.compile(r"PMOS:\s*(.+)")
_NMOS_ORDER_RE = re.compile(r"NMOS:\s*(.+)")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=__doc__)
parser.add_argument("--spice",      required=True,  help="Input SPICE file path")
parser.add_argument("--cell",       default=None,   help="Subcircuit name (auto-detected if omitted)")
parser.add_argument("--astran_bin", default=_ASTRAN_BIN)
parser.add_argument("--tech_file",  default=_TECH_FILE)
parser.add_argument("--timeout",    type=int, default=60)
parser.add_argument("--out",        default=None,   help="Save cleaned SPICE to this path")
args = parser.parse_args()


# ── SPICE cleaner ─────────────────────────────────────────────────────────────

def clean_spice(src_path, cell_name=None):
    """
    Return (cleaned_text, detected_cell_name).

    Transformations:
      //...   →  * ...        (C++ comment → SPICE comment)
      + ...   →  (removed)    (continuation lines: ad/pd/as/ps)
      .model  →  (removed)    (device model defs, not used by ASTRAN)
      .end    →  .ends CELL   (.end without name → .ends with name)
    """
    lines  = open(src_path).readlines()
    result = []
    detected = cell_name

    for raw in lines:
        s = raw.strip()

        if s.startswith("//"):
            result.append("* " + s[2:].strip() + "\n")
            continue

        if s.lower().startswith(".model"):
            continue

        if s.startswith("+"):
            continue

        # detect cell name from .subckt line
        if s.lower().startswith(".subckt") and detected is None:
            parts = s.split()
            if len(parts) >= 2:
                detected = parts[1]

        # fix bare .end → .ends CELLNAME
        if s.lower() == ".end":
            result.append(f".ends {detected or 'CELL'}\n")
            continue

        result.append(raw)

    return "".join(result), detected or "CELL"


# ── ASTRAN runner ─────────────────────────────────────────────────────────────

def run_astran(spice_text, cell_name):
    sp_path = sc_path = None
    try:
        sp_fd, sp_path = tempfile.mkstemp(suffix=".sp", prefix="place_")
        os.write(sp_fd, spice_text.encode()); os.close(sp_fd)

        script = (
            f'load technology "{args.tech_file}"\n'
            f'load netlist "{sp_path}"\n'
            f'cellgen select {cell_name}\n'
            f'cellgen fold 2 0\n'
            f'cellgen place 1 1 {_OBJ_CPP} {_OBJ_MISMATCH}'
            f' {_OBJ_ROUTING} {_OBJ_DENSITY} {_OBJ_GAPS}\n'
        )
        sc_fd, sc_path = tempfile.mkstemp(suffix=".run", prefix="place_")
        os.write(sc_fd, script.encode()); os.close(sc_fd)

        result = subprocess.run(
            [args.astran_bin, "--shell", sc_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=args.timeout,
        )
        return result.stdout.decode(errors="replace")

    except (subprocess.TimeoutExpired, Exception) as e:
        return f"ERROR: {e}"
    finally:
        for p in (sp_path, sc_path):
            if p and os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"Input SPICE : {args.spice}")
spice_text, cell_name = clean_spice(args.spice, args.cell)
print(f"Cell name   : {cell_name}")

if args.out:
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(spice_text)
    print(f"Saved cleaned SPICE → {args.out}")

print(f"\nRunning ASTRAN placement "
      f"(place 1 1 {_OBJ_CPP} {_OBJ_MISMATCH} {_OBJ_ROUTING} {_OBJ_DENSITY} {_OBJ_GAPS}) …\n")

output = run_astran(spice_text, cell_name)

# ── Parse and report ──────────────────────────────────────────────────────────
m = _COST_RE.search(output)
if not m:
    print("ASTRAN placement failed. Raw output:")
    print(output)
else:
    width    = int(m.group(1))
    mismatch = int(m.group(2))
    wl       = int(m.group(3))
    density  = int(m.group(4))
    gaps     = int(m.group(5))
    obj      = (_OBJ_CPP * width + _OBJ_MISMATCH * mismatch
              + _OBJ_ROUTING * wl + _OBJ_DENSITY * density
              + _OBJ_GAPS * gaps)

    p_order = (m2.group(1).strip() if (m2 := _PMOS_ORDER_RE.search(output)) else "?")
    n_order = (m2.group(1).strip() if (m2 := _NMOS_ORDER_RE.search(output)) else "?")

    print(f"{'='*55}")
    print(f"Placement result for cell: {cell_name}")
    print(f"{'='*55}")
    print(f"  Objective : {obj}")
    print(f"    = {_OBJ_CPP}×CPP({width}) + {_OBJ_MISMATCH}×Mismatch({mismatch})"
          f" + {_OBJ_ROUTING}×WL({wl}) + {_OBJ_DENSITY}×Density({density})"
          f" + {_OBJ_GAPS}×Gaps({gaps})")
    print(f"  CPP (Width)     : {width}")
    print(f"  Gate Mismatches : {mismatch}")
    print(f"  WL              : {wl}")
    print(f"  Rt. Density     : {density}")
    print(f"  Nr. Gaps        : {gaps}")
    print(f"\n  PMOS ordering: {p_order}")
    print(f"  NMOS ordering: {n_order}")
