"""E7(b) -- MurmurHash3 vs Thomas Wang hash inside fgr2.

Four independent characterisations, all on REAL genomic k-mer populations
(structured, non-uniform 2-bit codes) rather than random integers:

 1. Uniformity: chi-square on bucketed hash values, using the TOP 16 bits
    (65536 buckets) and, separately, the LOW 16 bits. Bottom-N selection
    depends on the LOW end of the hash range, so we additionally run a
    one-sample Kolmogorov-Smirnov test of the smallest 100000 hash values
    against the Uniform order-statistic expectation, and a chi-square on
    each individual output bit (is bit j set in half the k-mers?).
 2. Avalanche / bit independence: flip each of the 2k input bits of a real
    k-mer's 2-bit code and record which of the 64 output bits change.
 3. Practical impact on the Jaccard estimate vs exact ground truth.
 4. Sketch-construction wall time of the real fgr2 binary, with and without -w.
"""
import itertools
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_common as C
import fgrlib

np.seterr(over="ignore")
K = 31
RNG_SEED = 20260721


# ---------------------------------------------------------------- 1. uniformity
def uniformity(codes, hname, rng):
    h = C.HASHES[hname](codes)
    n = h.shape[0]
    out = {"n_kmers": int(n)}

    for label, shift, nbits in (("top16", 48, 16), ("low16", 0, 16)):
        b = ((h >> np.uint64(shift)) & np.uint64((1 << nbits) - 1)).astype(np.int64)
        counts = np.bincount(b, minlength=1 << nbits)
        exp = n / (1 << nbits)
        chi2 = float(((counts - exp) ** 2 / exp).sum())
        dof = (1 << nbits) - 1
        out[label] = {
            "n_buckets": 1 << nbits,
            "chi2": chi2,
            "dof": dof,
            "chi2_over_dof": chi2 / dof,
            # z-score of chi2 under H0 (chi2 mean=dof, var=2*dof)
            "z": float((chi2 - dof) / np.sqrt(2 * dof)),
            "p_value": float(stats.chi2.sf(chi2, dof)),
            "max_bucket_excess_ratio": float(counts.max() / exp),
            "min_bucket_ratio": float(counts.min() / exp),
        }

    # per-output-bit balance
    bit_frac = np.array([float(((h >> np.uint64(j)) & np.uint64(1)).mean())
                         for j in range(64)])
    # chi-square per bit vs 0.5
    bit_z = (bit_frac - 0.5) * 2 * np.sqrt(n)
    out["per_bit"] = {
        "fraction_set": bit_frac.tolist(),
        "max_abs_dev_from_half": float(np.abs(bit_frac - 0.5).max()),
        "worst_bit": int(np.argmax(np.abs(bit_frac - 0.5))),
        "max_abs_z": float(np.abs(bit_z).max()),
        "n_bits_with_abs_z_gt_5": int((np.abs(bit_z) > 5).sum()),
    }

    # Low-end behaviour -- this is what bottom-N selection actually consumes.
    # If the hash is uniform on [0,2^64) then the hash values form (to an
    # excellent approximation for n << 2^64) a Poisson process of rate
    # n/2^64, so the SPACINGS between consecutive order statistics at the
    # bottom are iid Exponential with mean 2^64/n. We KS-test the normalised
    # spacings of the M smallest hashes against Exp(1). (An earlier version
    # of this script KS-tested the order statistics themselves against
    # Uniform(0,1), which is not their null distribution; that statistic was
    # meaningless and has been removed.)
    M = 100000
    small = np.sort(h)[:M].astype(np.float64)
    gaps = np.diff(small)
    rate = n / 2.0**64
    ks = stats.kstest(gaps * rate, "expon")
    out["low_end_spacing_KS"] = {
        "M_smallest": M,
        "null": "Exp(1) after scaling by n/2^64",
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "mean_scaled_gap": float((gaps * rate).mean()),
        "expected_mean_scaled_gap": 1.0,
    }
    out["low_end_gaps"] = {
        "mean_gap": float(gaps.mean()),
        "expected_mean_gap": float(2.0**64 / n),
        "cv_gap": float(gaps.std() / gaps.mean()),
        "expected_cv_exponential": 1.0,
        "n_zero_gaps_collisions": int((gaps == 0).sum()),
    }
    return out


# ---------------------------------------------------------------- 2. avalanche
def avalanche(codes, hname, rng, n_sample=50000):
    idx = rng.choice(codes.shape[0], size=min(n_sample, codes.shape[0]),
                     replace=False)
    base = codes[idx]
    hf = C.HASHES[hname]
    h0 = hf(base)
    nin = 2 * K
    mat = np.zeros((nin, 64))
    for i in range(nin):
        flipped = base ^ np.uint64(1 << i)
        d = hf(flipped) ^ h0
        for j in range(64):
            mat[i, j] = float(((d >> np.uint64(j)) & np.uint64(1)).mean())
    dev = np.abs(mat - 0.5)
    # per-flip total output bits changed
    return {
        "n_sample": int(base.shape[0]),
        "n_input_bits": nin,
        "mean_flip_prob": float(mat.mean()),
        "max_abs_dev_from_half": float(dev.max()),
        "mean_abs_dev_from_half": float(dev.mean()),
        "p99_abs_dev": float(np.percentile(dev, 99)),
        "worst_cell": [int(np.unravel_index(dev.argmax(), dev.shape)[0]),
                       int(np.unravel_index(dev.argmax(), dev.shape)[1]),
                       float(mat[np.unravel_index(dev.argmax(), dev.shape)])],
        # a strict avalanche criterion violation: |p-0.5| > 5 binomial SDs
        "n_cells": int(mat.size),
        "n_cells_dev_gt_5sd": int((dev > 5 * 0.5 / np.sqrt(base.shape[0])).sum()),
        "matrix_row_means": mat.mean(axis=1).tolist(),
        "matrix_col_means": mat.mean(axis=0).tolist(),
        "matrix": mat.tolist(),
        # Wang's final step is key += key<<31, which cannot propagate any
        # information downward past bit 31; split the matrix accordingly.
        "max_dev_output_bits_0_30": float(dev[:, :31].max()),
        "max_dev_output_bits_31_63": float(dev[:, 31:].max()),
        "max_dev_highinput_to_lowoutput": float(dev[40:, :31].max()),
    }


# ------------------------------------------------------- 3. estimator accuracy
def bottom(codes, hname, N):
    h = C.HASHES[hname](codes)
    idx = np.argpartition(h, N)[:N]
    hs = h[idx]
    o = np.argsort(hs, kind="stable")
    return codes[idx][o], hs[o]


def jaccard_union_np(botA, botB, N):
    cA, hA = botA[0][:N], botA[1][:N]
    cB, hB = botB[0][:N], botB[1][:N]
    allc = np.concatenate([cA, cB]); allh = np.concatenate([hA, hB])
    o = np.argsort(allh, kind="stable"); allc, allh = allc[o], allh[o]
    _, first = np.unique(allc, return_index=True)
    keep = np.zeros(allc.shape[0], dtype=bool); keep[np.sort(first)] = True
    mc = allc[keep][:N]
    inA = np.isin(mc, cA, assume_unique=True)
    inB = np.isin(mc, cB, assume_unique=True)
    return float((inA & inB).sum() / mc.shape[0])


def jaccard_direct_np(botA, botB, N):
    a, b = botA[0][:N], botB[0][:N]
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 0.0


def accuracy(codes_by_name, names, Ns=(1000, 10000)):
    truth = {}
    for a, b in itertools.combinations(names, 2):
        truth[(a, b)] = float(fgrlib.jaccard_from_codes(codes_by_name[a],
                                                        codes_by_name[b]))
    res = {"per_pair": {}, "summary": {}}
    bots = {}
    for hname in ("murmurhash3", "wang"):
        for nm in names:
            bots[(hname, nm)] = bottom(codes_by_name[nm], hname, max(Ns))
    for (a, b), J in truth.items():
        rec = {"true_jaccard": J}
        for hname in ("murmurhash3", "wang"):
            for N in Ns:
                rec[f"{hname}_union_N{N}"] = jaccard_union_np(
                    bots[(hname, a)], bots[(hname, b)], N)
                rec[f"{hname}_direct_N{N}"] = jaccard_direct_np(
                    bots[(hname, a)], bots[(hname, b)], N)
        res["per_pair"][f"{a}|{b}"] = rec
    for hname in ("murmurhash3", "wang"):
        for est in ("union", "direct"):
            for N in Ns:
                errs = np.array([res["per_pair"][p][f"{hname}_{est}_N{N}"]
                                 - res["per_pair"][p]["true_jaccard"]
                                 for p in res["per_pair"]])
                res["summary"][f"{hname}_{est}_N{N}"] = {
                    "n_pairs": len(errs),
                    "bias_mean_error": float(errs.mean()),
                    "rmse": float(np.sqrt((errs ** 2).mean())),
                    "mae": float(np.abs(errs).mean()),
                    "max_abs_error": float(np.abs(errs).max()),
                }
    return res


# ------------------------------------------------------------------- 4. speed
def speed(paths, reps=9, N=10000, k=31, threads=4):
    """Wall-clock sketch construction, murmur3 vs -w.

    Confound control: one untimed warm-up run per genome (so the FASTA is in
    the page cache before any timing), then reps timed rounds in which the two
    hashes ALTERNATE within each round (murmur, wang, wang, murmur, ...) so
    neither systematically benefits from cache warming or thermal drift.
    """
    out = {"reps": reps, "N": N, "k": k, "threads": threads,
           "protocol": "1 untimed warmup per genome; order alternated per round",
           "per_genome": {}}
    tmp = os.path.join(C.SCRATCH, "e7b_speed.fgr2")

    def one(p, extra):
        t0 = time.perf_counter()
        subprocess.run([C.FGR2, "-k", str(k), "-N", str(N), "-c", "1",
                        "-t", str(threads)] + extra + ["-o", tmp, p],
                       check=True, capture_output=True)
        return time.perf_counter() - t0

    for p in paths:
        nm = os.path.basename(p).replace(".fna", "")
        one(p, [])          # warm the page cache, untimed
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
        out["per_genome"][nm] = rec
    ratios = [out["per_genome"][g]["wang_over_murmur_median"]
              for g in out["per_genome"]]
    out["summary"] = {
        "n_genomes": len(ratios),
        "wang_over_murmur_median_ratio": float(np.median(ratios)),
        "wang_over_murmur_min": float(np.min(ratios)),
        "wang_over_murmur_max": float(np.max(ratios)),
        "total_murmur_median_s": float(sum(
            out["per_genome"][g]["murmurhash3"]["median_s"] for g in out["per_genome"])),
        "total_wang_median_s": float(sum(
            out["per_genome"][g]["wang"]["median_s"] for g in out["per_genome"])),
    }
    mm = [out["per_genome"][g]["murmurhash3"]["median_s"] for g in out["per_genome"]]
    ww = [out["per_genome"][g]["wang"]["median_s"] for g in out["per_genome"]]
    w = stats.wilcoxon(mm, ww)
    out["summary"]["paired_wilcoxon_over_genomes"] = {
        "n": len(mm), "stat": float(w.statistic), "p": float(w.pvalue)}
    if os.path.exists(tmp):
        os.remove(tmp)
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    paths = C.genome_paths()
    names = [os.path.basename(p).replace(".fna", "") for p in paths]
    codes = {n: C.codes_cached(p, K) for n, p in zip(names, paths)}

    res = {"experiment": "E7b_hash_comparison", "k": K, "seed": RNG_SEED}

    # 1. uniformity on three real genomes (different GC / phylogeny)
    unif_genomes = ["Ecoli_K12_MG1655", "Pseudomonas_aeruginosa_PAO1",
                    "Bacillus_subtilis_168"]
    res["uniformity"] = {}
    for g in unif_genomes:
        res["uniformity"][g] = {h: uniformity(codes[g], h, rng)
                                for h in ("murmurhash3", "wang")}
        print(f"uniformity {g} done", flush=True)

    # 2. avalanche
    res["avalanche"] = {h: avalanche(codes["Ecoli_K12_MG1655"], h, rng)
                        for h in ("murmurhash3", "wang")}
    print("avalanche done", flush=True)

    # 3. accuracy
    res["accuracy"] = accuracy(codes, names)
    print("accuracy done", flush=True)

    # 4. speed -- all 16 genomes, 9 alternating timed rounds each
    res["speed"] = speed(paths)
    print("speed done", flush=True)

    os.makedirs(C.RESULTS, exist_ok=True)
    out_path = os.path.join(C.RESULTS, "E7b_hash.json")
    with open(out_path, "w") as fh:
        json.dump(res, fh)
    print("wrote", out_path)

    for h in ("murmurhash3", "wang"):
        u = res["uniformity"]["Ecoli_K12_MG1655"][h]
        print(f"{h:12s} low16 chi2/dof={u['low16']['chi2_over_dof']:.4f} "
              f"z={u['low16']['z']:+.2f}  top16 z={u['top16']['z']:+.2f}  "
              f"maxbitdev={u['per_bit']['max_abs_dev_from_half']:.2e}  "
              f"avalanche maxdev={res['avalanche'][h]['max_abs_dev_from_half']:.4f}")
    for h in ("murmurhash3", "wang"):
        s = res["accuracy"]["summary"][f"{h}_union_N1000"]
        print(f"{h:12s} union N=1000 bias={s['bias_mean_error']:+.5f} "
              f"rmse={s['rmse']:.5f}")
    print("speed wang/murmur ratio median:",
          res["speed"]["summary"]["wang_over_murmur_median_ratio"])


if __name__ == "__main__":
    main()
