#!/usr/bin/env python3
"""
F1 part 1 -- controlled synthetic sweep of cosine estimator bias.

Two k-mer "samples" are built directly at the 2-bit-code level with known
coverage vectors, so the ground-truth abundance-weighted cosine is exact.
Bottom-N sketches are then built with fgr2's own selection hash (MurmurHash3
x64_64, seed 42 -- vectorised port validated element-for-element against
fgrlib.murmur3_x64_64), so the sketching step is bit-identical to fgr2 -c 1.

Factors: support-size ratio r = |supp(B)|/|supp(A)|, support overlap fraction,
coverage-correlation tau (spans the true-cosine axis), sketch size N.
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))

import f1_common as F                      # noqa: E402
from f1_hash_np import murmur3_vec         # noqa: E402

OUT = os.path.join(REPO, "analysis", "results", "f1_cosine_synthetic.json")

N_A = 200_000            # support size of sample A (k-mers)
RATIOS = [1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
OVERLAPS = [1.0, 0.8, 0.5]
TAUS = [0.3, 1.0]        # log-scale coverage perturbation on shared k-mers
N_SIZES = [100, 1000, 10000]
N_REPS = 25
SEED0 = 20260721
COV_SIGMA = 0.7          # lognormal coverage dispersion
COV_MEAN = 20.0


def _sketch(codes, counts, n):
    h = murmur3_vec(codes)
    idx = np.argsort(h, kind="stable")[:n]
    return {int(codes[i]): (int(h[i]), int(counts[i])) for i in idx}


def one_rep(task):
    ratio, ov, tau, rep = task
    seed = SEED0 + hash((round(ratio * 1000), round(ov * 100),
                         round(tau * 100), rep)) % 10_000_019
    rng = np.random.default_rng(seed)

    n_a = N_A
    n_b = int(round(ratio * n_a))
    s = int(round(ov * min(n_a, n_b)))
    n_priv_b = n_b - s
    total = n_a + n_priv_b
    # unique random 2-bit codes for k=21 (space 4^21 = 4.4e12 >> total)
    pool = np.unique(rng.integers(0, 4 ** 21, size=int(total * 1.02),
                                  dtype=np.uint64))
    rng.shuffle(pool)
    pool = pool[:total]
    assert pool.size == total

    codes_a = pool[:n_a]
    cnt_a = np.maximum(1, np.round(
        COV_MEAN * np.exp(rng.normal(0, COV_SIGMA, n_a)))).astype(np.int64)

    shared_idx = np.arange(s)
    cnt_b_shared = np.maximum(1, np.round(
        cnt_a[shared_idx] * np.exp(rng.normal(0, tau, s)))).astype(np.int64)
    codes_b_priv = pool[n_a:]
    cnt_b_priv = np.maximum(1, np.round(
        COV_MEAN * np.exp(rng.normal(0, COV_SIGMA, n_priv_b)))).astype(np.int64)
    codes_b = np.concatenate([codes_a[shared_idx], codes_b_priv])
    cnt_b = np.concatenate([cnt_b_shared, cnt_b_priv])

    oa = np.argsort(codes_a)
    ob = np.argsort(codes_b)
    truth = F.cos_true_arrays(codes_a[oa], cnt_a[oa], codes_b[ob], cnt_b[ob])

    rec = {"ratio": ratio, "overlap": ov, "tau": tau, "rep": rep, "seed": seed,
           "n_a": int(n_a), "n_b": int(n_b), "n_shared": int(s),
           "cos_true": truth, "est": {}}
    for n in N_SIZES:
        sk_a = _sketch(codes_a, cnt_a, n)
        sk_b = _sketch(codes_b, cnt_b, n)
        rec["est"][str(n)] = F.all_estimators(sk_a, sk_b, n)
    return rec


def main():
    tasks = [(r, ov, tau, rep)
             for r in RATIOS for ov in OVERLAPS for tau in TAUS
             for rep in range(N_REPS)]
    t0 = time.time()
    recs = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, rec in enumerate(ex.map(one_rep, tasks, chunksize=4)):
            recs.append(rec)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)

    out = {
        "experiment_id": "F1_cosine_bias_synthetic",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "n_a_support": N_A, "ratios": RATIOS, "overlaps": OVERLAPS,
            "taus": TAUS, "N_sizes": N_SIZES, "n_reps": N_REPS,
            "seed0": SEED0, "cov_lognormal_sigma": COV_SIGMA,
            "cov_mean": COV_MEAN, "k_for_code_space": 21,
            "hash": "murmurhash3 seed=42 (fgr2 selection hash)",
        },
        "estimators": F.EST_NAMES,
        "records": recs,
        "wall_seconds": time.time() - t0,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT, f"{len(recs)} records, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
