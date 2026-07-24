"""
Variable-position canonicalization (shared by pretraining and generation).

Root problem: the policy operates in GLOBAL ALL_VARS coordinates (slot 0=a..3=d).
Training functions pack their support into the LOW slots, so the policy only
learned low-slot networks; a function like `c+d` (support on slots 2,3) is
out-of-distribution and fails. The fix is to relabel every function's support
to the lowest slots BEFORE it touches the model, identically at train and
inference time, then map generated gate indices back to the real variables.

Canonicalization (matches the dataset's local literal rule):
    support      = sorted variables appearing in expr      e.g. ['c','d']
    canon_expr   = expr with support[i] -> ALL_VARS[i]      'c+!d' -> 'a+!b'
    perm[g]      = canonical global slot for original slot g (support packs low)
    real_vars[i] = the original variable now living at canonical slot i
"""
from .literal import ALL_VARS, N_VARS, decode_literal, extract_vars


def canonicalize_expr(expr):
    """Return (canon_expr, real_vars, perm).

    canon_expr : expr relabeled so its support occupies ALL_VARS[:K].
    real_vars  : support list (sorted); real_vars[i] lives at canonical slot i.
    perm       : length-N_VARS list, perm[orig_global_slot] = canonical_slot.
    """
    support = extract_vars(expr)
    cmap = {support[i]: ALL_VARS[i] for i in range(len(support))}
    canon_expr = expr.translate(str.maketrans(cmap))

    perm = [None] * N_VARS
    for i, v in enumerate(support):
        perm[ALL_VARS.index(v)] = i
    free = [s for s in range(N_VARS) if s not in range(len(support))]
    fi = 0
    for g in range(N_VARS):
        if perm[g] is None:
            perm[g] = free[fi]; fi += 1
    return canon_expr, support, perm


def canon_lit(lit_id, vars_in_func, perm, K):
    """Remap a raw transistor lit_id into the canonical local scheme
    (positive 0..K-1, negative K..2K-1) so decode_literal(_, ALL_VARS[:K])
    yields the canonical low slot."""
    gv, neg = decode_literal(lit_id, vars_in_func)
    return perm[gv] + (K if neg else 0)


def real_var_index(canon_slot, real_vars):
    """Map a canonical gate slot back to the real global ALL_VARS index."""
    if canon_slot < len(real_vars):
        return ALL_VARS.index(real_vars[canon_slot])
    return canon_slot  # free slot — should not appear in a correct network


def decanon_gtran(g_tran, real_vars):
    """Map a generated g_tran (canonical gate slots) back to real ALL_VARS slots."""
    return [(u, v, real_var_index(gv, real_vars), neg) for (u, v, gv, neg) in g_tran]


def neg_real_var_set(g_tran_real):
    """Set of real global var indices that appear negated."""
    return {gv for (_u, _v, gv, neg) in g_tran_real if neg == 1}
