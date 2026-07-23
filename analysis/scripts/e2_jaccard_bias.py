#!/usr/bin/env python3
"""
Experiment E2 -- quantify the bias of the Jaccard estimator used in
calculate_similarity.py:434 (`jaccard_direct`) against the unbiased
bottom-N MinHash / Mash construction (`jaccard_union`).

Regime A: controlled synthetic pairs, known exact ground truth, size ratio
          |B|/|A| in {1,2,5,10,50}.
Regime B: all 120 pairs among 16 real RefSeq bacterial genomes, real fgr2 sketches.

Everything is seeded. Raw per-replicate measurements are written to JSON.
"""
import json
import math
import os
import random
import sys
import time
from itertools import combinations

import numpy as np

SCRIPTS = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/scripts"
RESULTS = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/results"
FIGURES = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/figures"
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad")
GENOMES = os.path.join(SCRATCH, "genomes")
FGR2 = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/fgr2"

sys.path.insert(0, SCRIPTS)
import fgrlib  # noqa: E402

K = 31
NS = [1000, 10000]
MASTER_SEED = 20260721

# --------------------------------------------------------------------------
# Vectorised murmur3_x64_64 (validated bit-for-bit against fgrlib below)
# --------------------------------------------------------------------------
U = np.uint64


def _rotl64_vec(x, r):
    r = int(r)
    return (x << U(r)) | (x >> U(64 - r))


def mm3_vec(codes):
    """numpy vectorisation of fgrlib.murmur3_x64_64(code, seed=42)."""
    c1 = U(0x87C37B91114253D5)
    c2 = U(0x4CF5AD432745937F)
    h1 = np.full(codes.shape, U(42), dtype=np.uint64)
    h2 = np.full(codes.shape, U(42), dtype=np.uint64)

    k1 = codes.astype(np.uint64) * c1
    k1 = _rotl64_vec(k1, 31) * c2
    h1 = h1 ^ k1
    h1 = _rotl64_vec(h1, 27) + h2
    h1 = h1 * U(5) + U(0x52DCE729)
    h2 = _rotl64_vec(h2, 31) + h1
    h2 = h2 * U(5) + U(0x38495AB5)

    h1 = h1 ^ U(8)
    h2 = h2 ^ U(8)
    h1 = h1 + h2
    h2 = h2 + h1
    h1 = _fmix_vec(h1)
    h2 = _fmix_vec(h2)
    return h1 + h2


def _fmix_vec(k):
    k = k ^ (k >> U(33))
    k = k * U(0xFF51AFD7ED558CCD)
    k = k ^ (k >> U(33))
    k = k * U(0xC4CEB9FE1A85EC53)
    k = k ^ (k >> U(33))
    return k


def validate_mm3(rng):
    codes = rng.integers(0, 2**63, size=2000, dtype=np.uint64)
    codes = np.concatenate([codes, np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64)])
    fast = mm3_vec(codes)
    ok = all(int(fast[i]) == fgrlib.murmur3_x64_64(int(codes[i])) for i in range(codes.size))
    return bool(ok), int(codes.size)


# --------------------------------------------------------------------------
# Bottom-N sketching on code arrays
# --------------------------------------------------------------------------

def bottom_n(codes, n):
    """Return (sketch_codes, sketch_hashes) for the n smallest murmur3 hashes.
    `codes` must be unique (fast_canonical_codes guarantees this)."""
    h = mm3_vec(codes)
    if h.size <= n:
        order = np.argsort(h, kind="stable")
    else:
        idx = np.argpartition(h, n - 1)[:n]
        order = idx[np.argsort(h[idx], kind="stable")]
    return codes[order], h[order]


def j_direct_codes(a_codes, b_codes):
    """|A n B| / |A u B| computed directly on two independent bottom-N sketches
    -- exactly what calculate_similarity.py:434 does."""
    if a_codes.size == 0 and b_codes.size == 0:
        return 1.0
    inter = np.intersect1d(a_codes, b_codes).size
    union = a_codes.size + b_codes.size - inter
    return inter / union if union else 0.0


def j_union_codes(a_codes, a_h, b_codes, b_h, n):
    """Unbiased bottom-N MinHash estimator: bottom-N of the merged sketch,
    fraction present in both."""
    all_codes = np.concatenate([a_codes, b_codes])
    all_h = np.concatenate([a_h, b_h])
    uniq_codes, first = np.unique(all_codes, return_index=True)
    uniq_h = all_h[first]
    if uniq_codes.size == 0:
        return 1.0
    m = min(n, uniq_codes.size)
    idx = np.argsort(uniq_h, kind="stable")[:m]
    sel = uniq_codes[idx]
    in_a = np.isin(sel, a_codes)
    in_b = np.isin(sel, b_codes)
    return float(np.count_nonzero(in_a & in_b)) / m


def validate_estimators(rng):
    """Cross-check the code-space implementations against fgrlib's string-space
    reference implementations on a small random pair."""
    seq_a = "".join(rng.choice(list("ACGT"), size=3000))
    seq_b = "".join(rng.choice(list("ACGT"), size=3000))
    # make them share material
    seq_b = seq_a[:1500] + seq_b[1500:]
    k = 21
    n = 200
    ok = True
    for seq_x, seq_y in [(seq_a, seq_b)]:
        ka = fgrlib.canonical_kmers(seq_x, k)
        kb = fgrlib.canonical_kmers(seq_y, k)
        ra = fgrlib.reference_bottom_n(ka, n)
        rb = fgrlib.reference_bottom_n(kb, n)
        ref_direct = fgrlib.jaccard_direct({km for km, _ in ra}, {km for km, _ in rb})
        ref_union = fgrlib.jaccard_union(dict(ra), dict(rb), n)

        ca = fgrlib.fast_canonical_codes(seq_x, k)
        cb = fgrlib.fast_canonical_codes(seq_y, k)
        sa, ha = bottom_n(ca, n)
        sb, hb = bottom_n(cb, n)
        got_direct = j_direct_codes(sa, sb)
        got_union = j_union_codes(sa, ha, sb, hb, n)
        ok &= abs(ref_direct - got_direct) < 1e-12
        ok &= abs(ref_union - got_union) < 1e-12
    return bool(ok)


# --------------------------------------------------------------------------
# Regime A
# --------------------------------------------------------------------------
BASE_LEN = 100_000
RATIOS = [1, 2, 5, 10, 50]
RATES = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
NREP = 20


def rand_seq(rng_np, length):
    return "".join(np.array(list("ACGT"))[rng_np.integers(0, 4, size=length)])


def regime_a_cell(args):
    ratio, rate, rep = args
    seed = (MASTER_SEED * 1000003 + hash((ratio, rate, rep)) % 10**9) % (2**31)
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed + 7)

    core = rand_seq(rng_np, BASE_LEN)
    a_seq = core
    b_core = fgrlib.mutate(core, rate, rng_py)
    if ratio > 1:
        filler = rand_seq(rng_np, BASE_LEN * (ratio - 1))
        b_seq = b_core + "N" + filler
    else:
        b_seq = b_core

    ca = fgrlib.fast_canonical_codes(a_seq, K)
    cb = fgrlib.fast_canonical_codes(b_seq, K)
    j_true = fgrlib.jaccard_from_codes(ca, cb)

    rec = {"ratio": ratio, "rate": rate, "rep": rep, "seed": seed,
           "j_true": j_true, "n_kmers_a": int(ca.size), "n_kmers_b": int(cb.size),
           "size_ratio_realised": float(cb.size) / float(ca.size)}
    for n in NS:
        sa, ha = bottom_n(ca, n)
        sb, hb = bottom_n(cb, n)
        rec[f"j_direct_N{n}"] = j_direct_codes(sa, sb)
        rec[f"j_union_N{n}"] = j_union_codes(sa, ha, sb, hb, n)
    return rec


def run_regime_a():
    from multiprocessing import Pool
    jobs = [(r, m, rep) for r in RATIOS for m in RATES for rep in range(NREP)]
    t0 = time.time()
    with Pool(14) as pool:
        recs = pool.map(regime_a_cell, jobs, chunksize=1)
    sys.stderr.write(f"regime A: {len(recs)} replicates in {time.time()-t0:.1f}s\n")
    return recs


# --------------------------------------------------------------------------
# Regime B
# --------------------------------------------------------------------------

def run_regime_b():
    names = sorted(f[:-4] for f in os.listdir(GENOMES) if f.endswith(".fna"))
    codes = {}
    for nm in names:
        seq = fgrlib.load_genome(os.path.join(GENOMES, nm + ".fna"))
        codes[nm] = fgrlib.fast_canonical_codes(seq, K)
        sys.stderr.write(f"  loaded {nm}: {codes[nm].size} kmers\n")

    # real fgr2 sketches
    sk = {n: {} for n in NS}
    outdir = os.path.join(SCRATCH, "e2_sketches")
    os.makedirs(outdir, exist_ok=True)
    for n in NS:
        for nm in names:
            out = os.path.join(outdir, f"{nm}.N{n}.fgr2")
            if not os.path.exists(out):
                rc = os.system(f"{FGR2} -k {K} -N {n} -c 1 -o {out} "
                               f"{os.path.join(GENOMES, nm + '.fna')} 2>/dev/null")
                if rc != 0:
                    raise RuntimeError(f"fgr2 failed for {nm} N={n}")
            hashes, _, _ = fgrlib.parse_sketch(out)
            sk[n][nm] = hashes

    recs = []
    for a, b in combinations(names, 2):
        j_true = fgrlib.jaccard_from_codes(codes[a], codes[b])
        rec = {"a": a, "b": b, "j_true": j_true,
               "n_kmers_a": int(codes[a].size), "n_kmers_b": int(codes[b].size),
               "size_ratio": max(codes[a].size, codes[b].size) / min(codes[a].size, codes[b].size)}
        for n in NS:
            ha, hb = sk[n][a], sk[n][b]
            rec[f"sketch_size_a_N{n}"] = len(ha)
            rec[f"sketch_size_b_N{n}"] = len(hb)
            rec[f"j_direct_N{n}"] = fgrlib.jaccard_direct(set(ha), set(hb))
            rec[f"j_union_N{n}"] = fgrlib.jaccard_union(ha, hb, n)
        recs.append(rec)
    return names, recs


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def summarise_a(recs):
    cells = {}
    for r in recs:
        cells.setdefault((r["ratio"], r["rate"]), []).append(r)
    out = []
    for (ratio, rate), rs in sorted(cells.items()):
        jt = np.array([r["j_true"] for r in rs])
        row = {"ratio": ratio, "rate": rate, "n_rep": len(rs),
               "j_true_mean": float(jt.mean()), "j_true_sd": float(jt.std(ddof=1))}
        for n in NS:
            for est in ("direct", "union"):
                v = np.array([r[f"j_{est}_N{n}"] for r in rs])
                bias = v - jt
                row[f"{est}_N{n}_mean"] = float(v.mean())
                row[f"{est}_N{n}_bias"] = float(bias.mean())
                row[f"{est}_N{n}_bias_se"] = float(bias.std(ddof=1) / math.sqrt(len(rs)))
                row[f"{est}_N{n}_rmse"] = float(np.sqrt((bias ** 2).mean()))
                row[f"{est}_N{n}_sd"] = float(v.std(ddof=1))
                row[f"{est}_N{n}_binom_se"] = float(
                    math.sqrt(max(jt.mean() * (1 - jt.mean()), 0) / n))
        out.append(row)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    vrng = np.random.default_rng(MASTER_SEED)
    mm3_ok, mm3_n = validate_mm3(vrng)
    est_ok = validate_estimators(np.random.default_rng(MASTER_SEED + 1))
    sys.stderr.write(f"validation: mm3_vec=={mm3_ok} (n={mm3_n}), estimators=={est_ok}\n")
    if not (mm3_ok and est_ok):
        raise SystemExit("validation failed")

    a_recs = run_regime_a()
    a_summary = summarise_a(a_recs)
    names, b_recs = run_regime_b()

    payload = {
        "experiment": "E2_jaccard_estimator_bias",
        "date": "2026-07-21",
        "k": K, "N_values": NS, "master_seed": MASTER_SEED,
        "validation": {"vectorised_murmur3_matches_fgrlib": mm3_ok,
                       "n_hash_values_checked": mm3_n,
                       "code_space_estimators_match_fgrlib_string_space": est_ok},
        "regime_A": {
            "design": {"base_len": BASE_LEN, "ratios": RATIOS, "mutation_rates": RATES,
                       "n_replicates_per_cell": NREP,
                       "note": ("size ratio achieved by appending independent random "
                                "sequence to B; this caps J_true at 1/ratio, which is a "
                                "mathematical necessity, not a design choice")},
            "replicates": a_recs,
            "cell_summary": a_summary,
        },
        "regime_B": {"genomes": names, "pairs": b_recs},
    }
    path = os.path.join(RESULTS, "e2_jaccard_bias.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    sys.stderr.write(f"wrote {path}\n")


if __name__ == "__main__":
    main()
