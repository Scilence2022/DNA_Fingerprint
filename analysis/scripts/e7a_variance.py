"""E7(a) -- Jaccard estimator variance vs sketch size N.

RESAMPLING SCHEME (stated explicitly for the Methods section)
------------------------------------------------------------
fgr2 is deterministic: for a fixed input it always emits the same sketch,
because its MurmurHash3 selection hash uses a hard-coded seed of 42
(fgr2.c). There is therefore no sampling distribution to observe from a
single genome pair. MinHash theory's sqrt(J(1-J)/N) refers to variability
over the random draw of the hash function, so we resample exactly that:

  For replicate r = 0..R-1 we re-run bottom-N selection using
  MurmurHash3_x64_64 with seed = r instead of seed = 42. This is the same
  hash *family* fgr2 uses -- only the seed changes -- so each replicate is a
  legitimate alternative instantiation of fgr2's sketch. The k-mer universe
  of each genome is held fixed (exact canonical 31-mer sets, computed by
  fgrlib.fast_canonical_codes), so the ONLY source of variation is the hash.

For each genome pair we then have R independent Jaccard estimates at each N,
whose sample standard deviation is the empirical standard error. The true
Jaccard J is computed exactly from the full k-mer sets (no sketching).

Two estimators are evaluated on the same replicates:
  * union   -- unbiased Mash-style bottom-N-of-the-merged-sketch estimator
               (fgrlib.jaccard_union)
  * direct  -- |A_N & B_N| / |A_N | B_N| on two independently built sketches,
               which is what calculate_similarity.py actually computes
               (fgrlib.jaccard_direct)

Bottom-N of the union is always a subset of (bottom-Nmax of A) U
(bottom-Nmax of B) for N <= Nmax, so we materialise bottom-100000 once per
genome per seed and derive every smaller N as a prefix. Verified against
fgrlib.jaccard_union / jaccard_direct in _selfcheck().
"""
import itertools
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_common as C
import fgrlib

np.seterr(over="ignore")

K = 31
NS = [100, 316, 1000, 3162, 10000, 31623, 100000]
NMAX = max(NS)
R = 100          # replicates (hash seeds 0..99)
SEED_BASE = 0

_CODES = {}      # name -> uint64 array, filled in main (fork-shared)
_NAMES = []


def bottom_nmax(codes, seed):
    """Return (codes, hashes) of the NMAX smallest murmur3(seed) hashes."""
    h = C.murmur3_vec(codes, seed)
    idx = np.argpartition(h, NMAX)[:NMAX]
    hs = h[idx]
    order = np.argsort(hs, kind="stable")
    return codes[idx][order], hs[order]


def pair_estimates(botA, botB):
    """Return dict N -> (j_union, j_direct) using prefix logic."""
    cA, hA = botA
    cB, hB = botB
    # merged, unique by code, sorted by hash
    allc = np.concatenate([cA, cB])
    allh = np.concatenate([hA, hB])
    o = np.argsort(allh, kind="stable")
    allc, allh = allc[o], allh[o]
    _, first = np.unique(allc, return_index=True)
    keep = np.zeros(allc.shape[0], dtype=bool)
    keep[np.sort(first)] = True
    mc = allc[keep]                      # already hash-sorted
    inA = np.isin(mc, cA, assume_unique=True)
    inB = np.isin(mc, cB, assume_unique=True)
    both = np.cumsum(inA & inB)
    out = {}
    for N in NS:
        m = min(N, mc.shape[0])
        out[N] = (both[m - 1] / m,)
    # direct estimator: prefixes of each sketch
    for N in NS:
        a = cA[:N]
        b = cB[:N]
        inter = np.intersect1d(a, b, assume_unique=True).size
        union = a.size + b.size - inter
        out[N] = (out[N][0], inter / union if union else 0.0)
    return out


def _worker(seed):
    bots = {n: bottom_nmax(_CODES[n], seed) for n in _NAMES}
    res = {}
    for a, b in itertools.combinations(_NAMES, 2):
        est = pair_estimates(bots[a], bots[b])
        res[f"{a}|{b}"] = {str(N): [float(est[N][0]), float(est[N][1])] for N in NS}
    return seed, res


def _selfcheck():
    """Confirm the prefix/vectorised machinery reproduces fgrlib exactly."""
    a = _CODES[_NAMES[0]]
    b = _CODES[_NAMES[1]]
    seed = 3
    botA, botB = bottom_nmax(a, seed), bottom_nmax(b, seed)
    est = pair_estimates(botA, botB)
    for N in (100, 1000):
        ah = {int(c): int(h) for c, h in zip(botA[0][:N], botA[1][:N])}
        bh = {int(c): int(h) for c, h in zip(botB[0][:N], botB[1][:N])}
        ju = fgrlib.jaccard_union(ah, bh, N)
        jd = fgrlib.jaccard_direct(set(ah), set(bh))
        assert abs(ju - est[N][0]) < 1e-12, (N, ju, est[N][0])
        assert abs(jd - est[N][1]) < 1e-12, (N, jd, est[N][1])
    return True


def main():
    global _NAMES
    paths = C.genome_paths()
    for p in paths:
        n = os.path.basename(p).replace(".fna", "")
        _CODES[n] = C.codes_cached(p, K)
        _NAMES.append(n)
    real_names = list(_NAMES)

    # The 120 real pairs only reach J ~= 0.61. To probe the high-J regime we add
    # point-mutated variants of E. coli K12 MG1655 (labelled SIM_*). Pairs
    # involving a SIM genome are flagged "simulated" in the output.
    import random
    base = fgrlib.load_genome(os.path.join(C.GENOME_DIR, "Ecoli_K12_MG1655.fna"))
    for i, rate in enumerate([0.0002, 0.0005, 0.001, 0.002, 0.005]):
        nm = f"SIM_K12_mut{rate}"
        cache = os.path.join(C.SCRATCH, "e7_cache", f"{nm}.k{K}.npy")
        if os.path.exists(cache):
            _CODES[nm] = np.load(cache)
        else:
            m = fgrlib.mutate(base, rate, random.Random(1000 + i))
            _CODES[nm] = fgrlib.fast_canonical_codes(m, K)
            np.save(cache, _CODES[nm])
        _NAMES.append(nm)
    print(f"{len(real_names)} real + {len(_NAMES)-len(real_names)} simulated genomes",
          flush=True)

    assert _selfcheck()
    print("selfcheck vs fgrlib: OK", flush=True)

    # exact Jaccard for every pair
    truth = {}
    for a, b in itertools.combinations(_NAMES, 2):
        truth[f"{a}|{b}"] = float(fgrlib.jaccard_from_codes(_CODES[a], _CODES[b]))
    print(f"{len(truth)} pairs, exact J range "
          f"{min(truth.values()):.4g}..{max(truth.values()):.4g}", flush=True)

    t0 = time.time()
    ctx = mp.get_context("fork")   # fork so workers inherit _CODES
    with ctx.Pool(8) as pool:
        out = pool.map(_worker, range(SEED_BASE, SEED_BASE + R))
    print(f"{R} replicates in {time.time()-t0:.1f}s", flush=True)

    per_seed = {str(s): r for s, r in out}
    assert per_seed["0"], "workers returned no data"

    # aggregate
    agg = {}
    for pair, J in truth.items():
        rec = {"true_jaccard": J,
               "simulated": pair.split("|")[0].startswith("SIM_")
                            or pair.split("|")[1].startswith("SIM_"),
               "by_N": {}}
        for N in NS:
            u = np.array([per_seed[str(s)][pair][str(N)][0] for s in range(R)])
            d = np.array([per_seed[str(s)][pair][str(N)][1] for s in range(R)])
            theo = float(np.sqrt(J * (1 - J) / N))
            rec["by_N"][str(N)] = {
                "union_mean": float(u.mean()),
                "union_se_empirical": float(u.std(ddof=1)),
                "union_bias": float(u.mean() - J),
                "direct_mean": float(d.mean()),
                "direct_se_empirical": float(d.std(ddof=1)),
                "direct_bias": float(d.mean() - J),
                "theoretical_se": theo,
                "ratio_union": float(u.std(ddof=1) / theo) if theo > 0 else None,
                "ratio_direct": float(d.std(ddof=1) / theo) if theo > 0 else None,
                "n_replicates": R,
            }
        agg[pair] = rec

    # summary over pairs, and grouped by true-J bin
    bins = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.0)]
    real_pairs = [p for p in agg if not agg[p]["simulated"]]
    summary = {"by_N": {}, "by_N_and_Jbin": {}}
    for N in NS:
        ru = [agg[p]["by_N"][str(N)]["ratio_union"] for p in agg]
        rd = [agg[p]["by_N"][str(N)]["ratio_direct"] for p in agg]
        bu = [agg[p]["by_N"][str(N)]["union_bias"] for p in agg]
        bd = [agg[p]["by_N"][str(N)]["direct_bias"] for p in agg]
        summary["by_N"][str(N)] = {
            "n_pairs": len(agg),
            "ratio_union_median": float(np.median(ru)),
            "ratio_union_mean": float(np.mean(ru)),
            "ratio_union_p05": float(np.percentile(ru, 5)),
            "ratio_union_p95": float(np.percentile(ru, 95)),
            "ratio_direct_median": float(np.median(rd)),
            "union_bias_median": float(np.median(bu)),
            "union_bias_max_abs": float(np.max(np.abs(bu))),
            "direct_bias_median": float(np.median(bd)),
            "direct_bias_max_abs": float(np.max(np.abs(bd))),
            "n_real_pairs": len(real_pairs),
            "ratio_union_median_real_only": float(np.median(
                [agg[p]["by_N"][str(N)]["ratio_union"] for p in real_pairs])),
            "direct_bias_median_real_only": float(np.median(
                [agg[p]["by_N"][str(N)]["direct_bias"] for p in real_pairs])),
        }
        for lo, hi in bins:
            ps = [p for p in agg if lo <= agg[p]["true_jaccard"] < hi]
            if not ps:
                continue
            key = f"{lo}-{hi}"
            summary["by_N_and_Jbin"].setdefault(key, {})[str(N)] = {
                "n_pairs": len(ps),
                "mean_true_J": float(np.mean([agg[p]["true_jaccard"] for p in ps])),
                "ratio_union_median": float(np.median(
                    [agg[p]["by_N"][str(N)]["ratio_union"] for p in ps])),
                "union_se_median": float(np.median(
                    [agg[p]["by_N"][str(N)]["union_se_empirical"] for p in ps])),
                "theoretical_se_median": float(np.median(
                    [agg[p]["by_N"][str(N)]["theoretical_se"] for p in ps])),
                "direct_bias_median": float(np.median(
                    [agg[p]["by_N"][str(N)]["direct_bias"] for p in ps])),
            }

    res = {
        "experiment": "E7a_estimator_variance_vs_N",
        "k": K, "N_values": NS, "n_replicates": R,
        "resampling": ("MurmurHash3_x64_64 seed varied 0..%d in place of fgr2's "
                       "hard-coded seed 42; k-mer universes held fixed" % (R - 1)),
        "genomes": _NAMES,
        "per_pair": agg,
        "summary": summary,
        "raw_per_seed": per_seed,
    }
    out_path = os.path.join(C.RESULTS, "E7a_variance.json")
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh)
    print("wrote", out_path)
    for N in NS:
        s = summary["by_N"][str(N)]
        print(f"N={N:6d} ratio_union median={s['ratio_union_median']:.3f} "
              f"[p05 {s['ratio_union_p05']:.3f}, p95 {s['ratio_union_p95']:.3f}]  "
              f"union_bias_med={s['union_bias_median']:+.5f}  "
              f"direct_bias_med={s['direct_bias_median']:+.5f}")


if __name__ == "__main__":
    main()
