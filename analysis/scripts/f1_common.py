#!/usr/bin/env python3
"""
F1 shared machinery: the cosine estimators under test.

ESTIMANDS AND ESTIMATORS
========================
Ground truth (the estimand):
    cos_true = S_AB / sqrt(S_AA * S_BB)
where, over the FULL canonical k-mer support U = supp(A) u supp(B),
    S_AB = sum_{x in U} a_x b_x,  S_AA = sum a_x^2,  S_BB = sum b_x^2
and a_x, b_x are k-mer coverages (0 if absent).

Estimator 1 -- "shipped" (calculate_similarity.py:134 cosine_similarity_manual,
mirrored by fgrlib.cosine): dot product over sketch(A) n sketch(B), but each L2
norm over that sample's OWN bottom-N sketch.

    cos_shipped = sum_{x in SA n SB} a_x b_x / ( ||a|_SA|| * ||b|_SB|| )

Why this is structurally biased. Let H be the hash space, |A| = |supp(A)|.
The bottom-N sketch of A is {x in A : h(x) < t_A} with t_A ~ N*H/|A|.
  * numerator support = SA n SB = {x in A n B : h(x) < min(t_A, t_B)}, so
      E[num] ~ (N / max(|A|,|B|)) * S_AB
  * E[||a|_SA||^2] ~ (N/|A|) * S_AA ; likewise for B.
Therefore
      E[cos_shipped] ~ cos_true * sqrt(|A|*|B|) / max(|A|,|B|)
                     = cos_true / sqrt(r),  r = max(|A|,|B|)/min(|A|,|B|).
Two consequences: (i) the bias is multiplicative and purely a function of the
SUPPORT-SIZE RATIO r, not of N; (ii) it therefore does NOT vanish as N grows.
This is the abundance analogue of the direct-Jaccard bias.

Estimator 2 -- "union_bottomN" (the principled fix). Let
    S = the N globally smallest hashes of sketch(A) u sketch(B).
Because a k-mer whose hash is among the N smallest of A u B must, if present in
A, also be among the N smallest of A, S is exactly the bottom-N sketch of the
TRUE union support, and the coverages of its members in both samples are fully
recoverable from the two sketches (absent => 0). S is a uniform random size-N
sample of U, so
    num/N -> E_U[a b],  ||a|_S||^2/N -> E_U[a^2],  ||b|_S||^2/N -> E_U[b^2]
and the plug-in ratio is a consistent estimator of cos_true with only O(1/N)
ratio bias. Numerator and both denominators live on ONE common, unbiasedly
sampled support -- which is precisely what the shipped estimator violates.

    cos_union = sum_{x in S} a_x b_x / ( ||a|_S|| * ||b|_S|| )

Estimator 3 -- "intersect_only" (a second defensible candidate): take the dot
product AND both norms over SA n SB. This puts all three sums on a common
support too, but that support is the INTERSECTION, so it estimates the cosine of
the two vectors restricted to shared k-mers -- a different estimand, biased
upward relative to cos_true whenever the samples have private k-mers.

Estimator 4 -- "shipped_analytic" (a check on the derivation, not a serious
proposal): genome size is estimable from the sketch itself as |A| ~ N*H/t_A with
t_A the largest hash in sketch A, so the predicted bias factor
sqrt(|A||B|)/max(|A|,|B|) = min(t_A,t_B)/sqrt(t_A t_B) is computable, giving
    cos_shipped_analytic = cos_shipped * sqrt(t_A t_B) / min(t_A, t_B).
If the derivation is right this lands near cos_true.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- estimators
# All estimators take two dicts key -> (hash, coverage). Keys are k-mer strings
# for real fgr2 sketches, integer 2-bit codes for the synthetic track; the
# estimators never look at the key's type.

def _norm(vals):
    return math.sqrt(sum(float(v) * float(v) for v in vals))


def cos_shipped(sk_a, sk_b):
    """calculate_similarity.py cosine_similarity_manual, exactly."""
    if not sk_a or not sk_b:
        return 0.0
    na = _norm(c for _, c in sk_a.values())
    nb = _norm(c for _, c in sk_b.values())
    if na == 0 or nb == 0:
        return 0.0
    small, large = (sk_a, sk_b) if len(sk_a) <= len(sk_b) else (sk_b, sk_a)
    dot = 0.0
    for km, (_, c) in small.items():
        o = large.get(km)
        if o is not None:
            dot += float(c) * float(o[1])
    return dot / (na * nb)


def cos_shipped_analytic(sk_a, sk_b):
    """cos_shipped rescaled by the analytically predicted bias factor."""
    base = cos_shipped(sk_a, sk_b)
    if base == 0.0 or not sk_a or not sk_b:
        return base
    ta = max(h for h, _ in sk_a.values())
    tb = max(h for h, _ in sk_b.values())
    if ta <= 0 or tb <= 0:
        return base
    return base * math.sqrt(float(ta) * float(tb)) / float(min(ta, tb))


def union_support(sk_a, sk_b, n):
    """The N globally smallest hashes across the merged sketch = bottom-N of the
    true union support."""
    merged = {}
    for km, (h, _) in sk_a.items():
        merged[km] = h
    for km, (h, _) in sk_b.items():
        merged[km] = h
    keys = sorted(merged, key=merged.__getitem__)[:n]
    return keys


def cos_union(sk_a, sk_b, n):
    """Bottom-N-of-union cosine: numerator and both norms on a common support."""
    keys = union_support(sk_a, sk_b, n)
    if not keys:
        return 0.0
    dot = saa = sbb = 0.0
    for km in keys:
        ea = sk_a.get(km)
        eb = sk_b.get(km)
        a = float(ea[1]) if ea is not None else 0.0
        b = float(eb[1]) if eb is not None else 0.0
        dot += a * b
        saa += a * a
        sbb += b * b
    if saa == 0 or sbb == 0:
        return 0.0
    return dot / math.sqrt(saa * sbb)


def cos_intersect_only(sk_a, sk_b):
    """Dot AND both norms over the sketch intersection."""
    small, large = (sk_a, sk_b) if len(sk_a) <= len(sk_b) else (sk_b, sk_a)
    dot = saa = sbb = 0.0
    for km, (_, c) in small.items():
        o = large.get(km)
        if o is not None:
            a = float(c)
            b = float(o[1])
            dot += a * b
            saa += a * a
            sbb += b * b
    if saa == 0 or sbb == 0:
        return 0.0
    return dot / math.sqrt(saa * sbb)


def all_estimators(sk_a, sk_b, n):
    return {
        "shipped": cos_shipped(sk_a, sk_b),
        "union_bottomN": cos_union(sk_a, sk_b, n),
        "intersect_only": cos_intersect_only(sk_a, sk_b),
        "shipped_analytic": cos_shipped_analytic(sk_a, sk_b),
    }


EST_NAMES = ["shipped", "union_bottomN", "intersect_only", "shipped_analytic"]


# ------------------------------------------------------- ground-truth cosine
def cos_true_counts(counts_a, counts_b):
    """Exact abundance-weighted cosine over the union index, from two
    key -> count dicts (full, un-sketched)."""
    saa = sum(float(v) * v for v in counts_a.values())
    sbb = sum(float(v) * v for v in counts_b.values())
    if saa == 0 or sbb == 0:
        return 0.0
    small, large = ((counts_a, counts_b) if len(counts_a) <= len(counts_b)
                    else (counts_b, counts_a))
    dot = sum(float(v) * large[km] for km, v in small.items() if km in large)
    return dot / math.sqrt(saa * sbb)


def cos_true_arrays(codes_a, cnt_a, codes_b, cnt_b):
    """Same, for sorted uint64 code arrays with parallel count arrays."""
    import numpy as np
    saa = float(np.sum(cnt_a.astype(np.float64) ** 2))
    sbb = float(np.sum(cnt_b.astype(np.float64) ** 2))
    if saa == 0 or sbb == 0:
        return 0.0
    _, ia, ib = np.intersect1d(codes_a, codes_b, assume_unique=True,
                               return_indices=True)
    dot = float(np.dot(cnt_a[ia].astype(np.float64),
                       cnt_b[ib].astype(np.float64)))
    return dot / math.sqrt(saa * sbb)


# ----------------------------------------------------------- k-mer counting
def canonical_code_counts(seq, k):
    """Exact canonical k-mer 2-bit codes WITH multiplicities.
    Same construction as fgrlib.fast_canonical_codes, but returns counts."""
    import numpy as np
    lut = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    codes = lut[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    n = codes.size
    m = n - k + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
    valid_base = codes != 255
    cs = np.concatenate(([0], np.cumsum(valid_base, dtype=np.int64)))
    valid_win = (cs[k:] - cs[:-k]) == k
    safe = np.where(valid_base, codes, 0).astype(np.uint64)
    comp = (np.uint64(3) - safe)
    fwd = np.zeros(m, dtype=np.uint64)
    rev = np.zeros(m, dtype=np.uint64)
    for j in range(k):
        fwd = (fwd << np.uint64(2)) | safe[j:j + m]
        rev |= comp[j:j + m] << np.uint64(2 * j)
    canon = np.minimum(fwd, rev)[valid_win]
    return np.unique(canon, return_counts=True)
