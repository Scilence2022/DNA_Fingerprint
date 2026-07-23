"""E7(c) -- fgr2 coverage-counter saturation (KC_BITS=14, KC_MAX=16383).

fgr2.c:13-14 packs the k-mer count into the low 14 bits of the hash-table key
and fgr2.c:279 increments only while the count is < KC_MAX, so a counter pins
at 16383 and every further observation of that k-mer is discarded.

Two experiments, both run through the REAL fgr2 binary:

  c1  Sequencing-depth sweep. Error-free reads are simulated from a 10 kb
      E. coli K12 fragment at increasing depth. Because reads are error free
      the true (unsaturated) count of every k-mer is known exactly by direct
      counting of the read set in Python, so the saturation-induced error is
      measured, not modelled. Answers: at what depth do saturated counters
      first appear, and what fraction of the sketch is saturated.

  c2  Oligo-pool copy-number sweep with abundance (cosine) similarity. Two
      samples share a 60-oligo pool but have different, correlated copy-number
      profiles. Copy numbers are scaled by s. Because each oligo is written
      s*w_i times verbatim, the TRUE coverage of every k-mer is exactly its
      oligo's copy number -- verified against fgr2 at low s where nothing
      saturates. Cosine similarity is then computed from (i) the true
      unsaturated count vectors and (ii) the counts fgr2 actually reports, so
      the cosine error is attributable to saturation alone. Jaccard is
      computed at every scale as a control: the k-mer SET does not change, so
      Jaccard must be unaffected.
"""
import json
import os
import random
import subprocess
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_common as C
import fgrlib

KC_MAX = 16383
K = 31
SEED = 20260721
WORK = os.path.join(C.SCRATCH, "e7c")
os.makedirs(WORK, exist_ok=True)


def run_fgr2(fa, out, N=200000, k=K):
    subprocess.run([C.FGR2, "-k", str(k), "-N", str(N), "-c", "1", "-t", "4",
                    "-o", out, fa], check=True, capture_output=True)
    return fgrlib.parse_sketch(out)


def write_records(path, records):
    """records: iterable of (name, seq). Buffered writer for very large files."""
    with open(path, "w") as fh:
        buf = []
        for nm, sq in records:
            buf.append(f">{nm}\n{sq}\n")
            if len(buf) >= 20000:
                fh.write("".join(buf)); buf = []
        if buf:
            fh.write("".join(buf))


# --------------------------------------------------------------- c1: depth
def depth_sweep(depths=(1000, 4000, 8000, 12000, 16000, 20000, 26000, 32000),
                frag_len=10000, read_len=150):
    genome = fgrlib.load_genome(os.path.join(C.GENOME_DIR, "Ecoli_K12_MG1655.fna"))
    frag = genome[1000000:1000000 + frag_len]
    assert "N" not in frag
    out = {"frag_len": frag_len, "read_len": read_len, "k": K,
           "KC_MAX": KC_MAX, "seed": SEED, "per_depth": {}}
    fa = os.path.join(WORK, "reads.fa")
    sk = os.path.join(WORK, "reads.fgr2")
    for D in depths:
        rng = random.Random(SEED + D)
        reads = fgrlib.simulate_reads(frag, D, read_len, 0.0, rng, indel_frac=0.0)
        write_records(fa, ((f"r{i}", r) for i, r in enumerate(reads)))
        # exact unsaturated truth: count canonical k-mers over all reads
        true = {fgrlib.decode2bit(c, K): n
                for c, n in C.counts_from_reads(reads, K).items()}
        hashes, cov, meta = run_fgr2(fa, sk)
        n_sat = sum(1 for v in cov.values() if v >= KC_MAX)
        common = [km for km in cov if km in true]
        obs = np.array([cov[km] for km in common], dtype=float)
        tru = np.array([true[km] for km in common], dtype=float)
        rel = np.abs(obs - tru) / np.maximum(tru, 1)
        out["per_depth"][str(D)] = {
            "n_reads": len(reads),
            "total_bp": len(reads) * read_len,
            "n_sketch_kmers": len(cov),
            "n_saturated": int(n_sat),
            "frac_saturated": n_sat / len(cov) if cov else 0.0,
            "max_true_count": int(tru.max()),
            "median_true_count": float(np.median(tru)),
            "max_reported_count": int(obs.max()),
            "mean_rel_count_error": float(rel.mean()),
            "max_rel_count_error": float(rel.max()),
            "total_counts_lost": float((tru - obs).sum()),
            "frac_counts_lost": float((tru - obs).sum() / tru.sum()),
        }
        print(f"  depth {D:6d}: {len(reads)} reads, sat={n_sat}/{len(cov)} "
              f"({100*n_sat/max(1,len(cov)):.2f}%), max_true={int(tru.max())}",
              flush=True)
        del reads, true
    for p in (fa, sk):
        if os.path.exists(p):
            os.remove(p)
    return out


# ------------------------------------------------- c2: copy number + cosine
def cosine_np(a, b, keys):
    x = np.array([a.get(k, 0) for k in keys], dtype=float)
    y = np.array([b.get(k, 0) for k in keys], dtype=float)
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x.dot(y) / d) if d else 0.0


def copy_sweep(n_oligos=60, oligo_len=200,
               scales=(100, 300, 1000, 3000, 8000, 16383, 25000, 40000)):
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)
    pool = fgrlib.make_oligo_pool(n_oligos, oligo_len, rng)
    # abundance profiles: sample X lognormal, sample Y = X perturbed
    wx = nprng.lognormal(0.0, 1.0, n_oligos)
    wx /= wx.mean()
    wy = wx * nprng.lognormal(0.0, 0.3, n_oligos)
    wy /= wy.mean()

    # map oligo -> its canonical k-mers (must be disjoint across oligos)
    kmers = [sorted(fgrlib.canonical_kmers(o, K)) for o in pool]
    allk = set()
    overlap = 0
    for s in kmers:
        for km in s:
            if km in allk:
                overlap += 1
            allk.add(km)

    out = {"n_oligos": n_oligos, "oligo_len": oligo_len, "k": K,
           "KC_MAX": KC_MAX, "seed": SEED,
           "n_distinct_kmers": len(allk),
           "n_kmers_shared_between_oligos": overlap,
           "weights_x": wx.tolist(), "weights_y": wy.tolist(),
           "per_scale": {}}

    fa = os.path.join(WORK, "pool.fa")
    sk = os.path.join(WORK, "pool.fgr2")
    for s in scales:
        rec = {"scale": s}
        cov_obs = {}
        cov_true = {}
        for tag, w in (("x", wx), ("y", wy)):
            copies = [max(1, int(round(s * wi))) for wi in w]
            rec[f"copies_{tag}"] = copies
            write_records(fa, ((f"o{i}_{j}", pool[i])
                               for i in range(n_oligos)
                               for j in range(copies[i])))
            hashes, cov, meta = run_fgr2(fa, sk)
            cov_obs[tag] = cov
            t = {}
            for i in range(n_oligos):
                for km in kmers[i]:
                    t[km] = t.get(km, 0) + copies[i]
            cov_true[tag] = t
            n_sat = sum(1 for v in cov.values() if v >= KC_MAX)
            rec[f"n_sketch_kmers_{tag}"] = len(cov)
            rec[f"n_saturated_{tag}"] = n_sat
            rec[f"frac_saturated_{tag}"] = n_sat / len(cov) if cov else 0.0
            rec[f"max_true_count_{tag}"] = max(t.values())
            rec[f"max_reported_count_{tag}"] = max(cov.values())
            # agreement between fgr2 counts and analytic truth on unsaturated kmers
            uns = [km for km in cov if km in t and t[km] < KC_MAX]
            rec[f"n_unsaturated_checked_{tag}"] = len(uns)
            rec[f"unsaturated_count_exact_match_{tag}"] = bool(
                all(cov[km] == t[km] for km in uns))
        keys = sorted(set(cov_true["x"]) | set(cov_true["y"]))
        rec["cosine_true"] = cosine_np(cov_true["x"], cov_true["y"], keys)
        rec["cosine_observed"] = cosine_np(cov_obs["x"], cov_obs["y"], keys)
        rec["cosine_abs_error"] = abs(rec["cosine_observed"] - rec["cosine_true"])
        rec["cosine_signed_error"] = rec["cosine_observed"] - rec["cosine_true"]
        sx, sy = set(cov_obs["x"]), set(cov_obs["y"])
        rec["jaccard_observed"] = len(sx & sy) / len(sx | sy)
        rec["jaccard_true"] = (len(set(cov_true["x"]) & set(cov_true["y"]))
                               / len(set(cov_true["x"]) | set(cov_true["y"])))
        rec["frac_saturated_mean"] = 0.5 * (rec["frac_saturated_x"]
                                            + rec["frac_saturated_y"])
        out["per_scale"][str(s)] = rec
        print(f"  scale {s:6d}: sat_x={rec['frac_saturated_x']*100:.2f}% "
              f"cos_true={rec['cosine_true']:.6f} "
              f"cos_obs={rec['cosine_observed']:.6f} "
              f"err={rec['cosine_signed_error']:+.6f} "
              f"J={rec['jaccard_observed']:.6f}", flush=True)
    for p in (fa, sk):
        if os.path.exists(p):
            os.remove(p)
    return out


def main():
    res = {"experiment": "E7c_counter_saturation", "KC_BITS": 14,
           "KC_MAX": KC_MAX, "k": K, "seed": SEED}
    print("c1 depth sweep", flush=True)
    res["depth_sweep"] = depth_sweep()
    print("c2 copy-number sweep", flush=True)
    res["copy_sweep"] = copy_sweep()
    os.makedirs(C.RESULTS, exist_ok=True)
    p = os.path.join(C.RESULTS, "E7c_saturation.json")
    with open(p, "w") as fh:
        json.dump(res, fh)
    print("wrote", p)


if __name__ == "__main__":
    main()
