#!/usr/bin/env python3
"""
S5(b) -- MurmurHash3 vs Thomas Wang as the fgr2 selection hash, on ENCODED-POOL
k-mer populations.

Everything is evaluated on the canonical k-mer codes of synthetic encoded pools
(random payloads + shared 20 nt primer sites), NOT genomes. Four axes:

 1. Uniformity   -- chi-square on the top-16 and low-16 hash bits, per-output-bit
    balance, and a KS test on the spacings of the smallest 100 000 hashes (the
    low end is what bottom-N selection actually consumes).
 2. Avalanche    -- flip each of the 2k input bits of a real pool k-mer and
    measure how often each of the 64 output bits changes.
 3. Accuracy     -- effect on the fingerprint Jaccard estimate vs the exact
    Jaccard, over pairs of related archives (nested pools with graded overlap).
 4. Speed        -- fgr2 sketch-construction wall time, murmur vs -w (Wang),
    alternated per round with one untimed warm-up, on encoded-pool read sets.

Reuses the numpy hash ports (validated bit-for-bit vs fgr2 in e7_common) and the
uniformity/avalanche/estimator kernels from e7b, applied to pool codes.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import random

import numpy as np
from scipy import stats

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
np.seterr(over="ignore")
import fgrlib          # noqa: E402
import e7_common as C  # noqa: E402  (FGR2, SCRATCH, RESULTS, HASHES, murmur/wang vec)
import e7b_hash as e7b  # noqa: E402  (uniformity, avalanche, bottom, jaccard_* )

K = 31
OLIGO_LEN = 150
RNG_SEED = 20260722
RESULTS = os.path.join(REPO := os.path.dirname(os.path.dirname(SCRIPTS)),
                       "analysis", "results")
FIGDIR = os.path.join(REPO, "analysis", "figures")


def pool_codes(n_oligos, seed):
    rng = random.Random(seed)
    pool = fgrlib.make_oligo_pool(n_oligos, OLIGO_LEN, rng)
    return fgrlib.fast_canonical_codes("N".join(pool), K)  # sorted unique


def make_related_archives():
    """Nested encoded archives sharing a common prefix of oligos, so pairwise
    exact Jaccard spans a broad range (~ size_small / size_large)."""
    rng = random.Random(RNG_SEED)
    master = fgrlib.make_oligo_pool(6400, OLIGO_LEN, rng)
    sizes = [200, 400, 800, 1600, 3200, 6400]
    codes = {}
    for s in sizes:
        codes[f"archive_{s}"] = fgrlib.fast_canonical_codes(
            "N".join(master[:s]), K)
    return codes, [f"archive_{s}" for s in sizes]


def speed(paths, reps=9, N=10000, k=31, threads=4):
    """fgr2 sketch-construction wall time, murmur vs -w, alternated per round
    with one untimed warm-up per input (page-cache + thermal-drift control)."""
    out = {"reps": reps, "N": N, "k": k, "threads": threads,
           "protocol": "1 untimed warmup per pool; hash order alternated per round",
           "per_input": {}}
    tmp = os.path.join(C.SCRATCH, "s5b_speed.fgr2")

    def one(p, extra):
        t0 = time.perf_counter()
        subprocess.run([C.FGR2, "-k", str(k), "-N", str(N), "-c", "1",
                        "-t", str(threads)] + extra + ["-o", tmp, p],
                       check=True, capture_output=True)
        return time.perf_counter() - t0

    for p in paths:
        nm = os.path.basename(p).replace(".fa", "")
        one(p, [])
        one(p, ["-w"])
        ts = {"murmurhash3": [], "wang": []}
        for r in range(reps):
            order = [("murmurhash3", []), ("wang", ["-w"])]
            if r % 2:
                order = order[::-1]
            for label, extra in order:
                ts[label].append(one(p, extra))
        rec = {}
        for label in ("murmurhash3", "wang"):
            rec[label] = {"times_s": ts[label],
                          "median_s": statistics.median(ts[label]),
                          "min_s": min(ts[label]),
                          "iqr_s": float(np.percentile(ts[label], 75)
                                         - np.percentile(ts[label], 25))}
        rec["wang_over_murmur_median"] = (rec["wang"]["median_s"]
                                          / rec["murmurhash3"]["median_s"])
        rec["input_bytes"] = os.path.getsize(p)
        out["per_input"][nm] = rec
        print(f"  speed {nm}: murmur {rec['murmurhash3']['median_s']:.3f}s "
              f"wang {rec['wang']['median_s']:.3f}s "
              f"ratio {rec['wang_over_murmur_median']:.3f}", flush=True)
    ratios = [out["per_input"][g]["wang_over_murmur_median"]
              for g in out["per_input"]]
    mm = [out["per_input"][g]["murmurhash3"]["median_s"] for g in out["per_input"]]
    ww = [out["per_input"][g]["wang"]["median_s"] for g in out["per_input"]]
    summ = {
        "n_inputs": len(ratios),
        "wang_over_murmur_median_ratio": float(np.median(ratios)),
        "wang_over_murmur_min": float(np.min(ratios)),
        "wang_over_murmur_max": float(np.max(ratios)),
        "total_murmur_median_s": float(sum(mm)),
        "total_wang_median_s": float(sum(ww)),
    }
    if len(mm) > 1:
        w = stats.wilcoxon(mm, ww)
        summ["paired_wilcoxon_over_inputs"] = {
            "n": len(mm), "stat": float(w.statistic), "p": float(w.pvalue)}
    out["summary"] = summ
    if os.path.exists(tmp):
        os.remove(tmp)
    return out


def build_speed_inputs():
    """Encoded-pool read sets of a few sizes, written to disk for timing."""
    paths = []
    for n_oligos, copies in [(2000, 20), (8000, 20), (20000, 20)]:
        rng = random.Random(1000 + n_oligos)
        pool = fgrlib.make_oligo_pool(n_oligos, OLIGO_LEN, rng)
        cps = fgrlib.pcr_bias_copies(len(pool), copies, 0.3, rng)
        reads = fgrlib.pool_to_reads(pool, cps, 0.005, rng)
        p = os.path.join(C.SCRATCH, f"s5b_pool_{n_oligos}.fa")
        fgrlib.write_reads_fasta(p, reads)
        paths.append(p)
    return paths


def main():
    rng = np.random.default_rng(RNG_SEED)
    res = {"experiment": "S5b_hash_comparison_on_encoded_pools",
           "input_type": "encoded_oligo_pool", "k": K, "seed": RNG_SEED,
           "oligo_len": OLIGO_LEN}

    # independent large pools for uniformity / avalanche
    pools = {f"pool_{i}": pool_codes(50000, 100 + i) for i in range(3)}
    for nm, c in pools.items():
        print(f"{nm}: {c.size} distinct canonical k-mers", flush=True)

    # 1. uniformity
    res["uniformity"] = {}
    for nm, c in pools.items():
        res["uniformity"][nm] = {h: e7b.uniformity(c, h, rng)
                                 for h in ("murmurhash3", "wang")}
        print(f"uniformity {nm} done", flush=True)

    # 2. avalanche (one representative pool)
    res["avalanche"] = {h: e7b.avalanche(pools["pool_0"], h, rng)
                        for h in ("murmurhash3", "wang")}
    print("avalanche done", flush=True)

    # 3. accuracy on related archives
    codes, names = make_related_archives()
    res["accuracy"] = e7b.accuracy(codes, names)
    res["accuracy"]["archive_sizes_kmers"] = {n: int(codes[n].size)
                                              for n in names}
    print("accuracy done", flush=True)

    # 4. speed
    paths = build_speed_inputs()
    res["speed"] = speed(paths)
    print("speed done", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "s5b_hash.json")
    with open(out_path, "w") as fh:
        json.dump(res, fh)
    print("wrote", out_path)

    # console summary
    for h in ("murmurhash3", "wang"):
        u = res["uniformity"]["pool_0"][h]
        av = res["avalanche"][h]
        print(f"{h:12s} low16 chi2/dof={u['low16']['chi2_over_dof']:.4f} "
              f"z={u['low16']['z']:+.2f}  top16 z={u['top16']['z']:+.2f}  "
              f"KSp={u['low_end_spacing_KS']['ks_p']:.3f}  "
              f"maxbitdev={u['per_bit']['max_abs_dev_from_half']:.2e}  "
              f"avalanche maxdev={av['max_abs_dev_from_half']:.4f}")
    for h in ("murmurhash3", "wang"):
        s = res["accuracy"]["summary"][f"{h}_union_N1000"]
        print(f"{h:12s} union N=1000 bias={s['bias_mean_error']:+.5f} "
              f"rmse={s['rmse']:.5f}")
    print("speed wang/murmur median ratio:",
          res["speed"]["summary"]["wang_over_murmur_median_ratio"])


if __name__ == "__main__":
    main()
