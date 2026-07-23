#!/usr/bin/env python3
"""
S2 (a,b) -- Fingerprints must be compared on a COMMON INDEX.

DNA-DATA-STORAGE framing. Every input is a simulated ENCODED OLIGONUCLEOTIDE POOL
(random payloads flanked by shared 20 nt primers -- a faithful model of encoded
digital data, which is essentially random DNA). NO genome is used anywhere.

Setup: a REFERENCE pool and a QUERY pool are cut from one master encoded library.
They share a fixed set of oligos (the truly-overlapping data) but differ in their
TOTAL distinct-oligo count -- modelling a query that is a partial retrieval or a
different-size file. The support-size ratio r = max/min distinct-k-mer count is
swept while the TRUE overlap (exact Jaccard / abundance cosine over full k-mer
sets) is held fixed by construction (fixed shared count S and fixed total extra E,
only the ref/query split of E changes).

(a) For each cell (>=20 seeds) at N in {1000,10000} we record:
      - true Jaccard & true abundance cosine (exact, full k-mer sets)
      - direct/legacy estimators (each sketch scored on its OWN bottom-N index)
      - common-index (union) estimators (both scored on the merged bottom-N index)
    -> bias & RMSE vs r; the r at which direct-Jaccard bias exceeds 0.01; the
    cosine-bias law  E[c_hat] = c_true / sqrt(r).
(b) N in {100,1000,10000}: direct bias ~flat in N, union bias shrinks; OLS slope
    of |bias| on log10 N.

All measurements are persisted to JSON before any reporting.
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

import fgrlib as FG                       # noqa: E402  validated ground truth
import f1_common as F                     # noqa: E402  cosine estimators
from f1_hash_np import murmur3_vec        # noqa: E402  fgr2 selection hash (vectorised)

OUT = os.path.join(REPO, "analysis", "results", "s2_support_sweep.json")

# ---- design -------------------------------------------------------------
# Each oligo is a pure random synthetic encoded sequence (an unambiguous stand-in
# for encoded digital data, which is essentially random DNA). Shared primer sites
# are deliberately OMITTED here: a 20 nt primer flanking every oligo injects a few
# very-low-complexity primer/payload-junction k-mers that are shared across the
# whole pool and carry enormous pooled coverage, which would contaminate the
# fixed-overlap ground truth of this controlled support-size study. The full pool
# model (primers + simulated reads + error) is exercised in part (c).
OLIGO_LEN = 150            # nt of random payload per oligo
K = 21
S_SHARED = 600            # oligos shared by reference and query (fixed overlap)
E_EXTRA = 1800           # total private oligos, split between ref and query (fixed)
# -> true Jaccard over oligos = S/(S+E) = 600/2400 = 0.25, held fixed vs r.
N_MASTER = S_SHARED + E_EXTRA
RATIOS = [1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0]
N_SIZES = [100, 1000, 10000]
N_SEEDS = 24
SEED0 = 20260722
COV_MEAN = 20.0
COV_SIGMA = 0.5           # per-oligo abundance dispersion (encoded-pool copy number)


def split_for_ratio(r):
    """Given fixed shared S and total extra E, choose the ref/query split (a,b)
    of E so that the distinct-oligo support-size ratio is exactly r.
    ref = larger pool. a=ref-only, b=query-only, a+b=E.
        (S+a)/(S+b) = r,  a+b = E  =>  b = (E - S(r-1))/(1+r)."""
    b = (E_EXTRA - S_SHARED * (r - 1.0)) / (1.0 + r)
    b = int(round(b))
    b = max(0, min(E_EXTRA, b))
    a = E_EXTRA - b
    return a, b


def oligo_codes(seq):
    return FG.fast_canonical_codes(seq, K)


def pool_maps(idx, codes_list, abund):
    """Build a pool's exact canonical-code -> coverage map from a set of oligo
    indices. Coverage of a k-mer = sum of abundances of oligos containing it."""
    parts_c = [codes_list[i] for i in idx]
    parts_w = [np.full(codes_list[i].size, abund[i], dtype=np.int64) for i in idx]
    all_c = np.concatenate(parts_c)
    all_w = np.concatenate(parts_w)
    order = np.argsort(all_c, kind="stable")
    all_c = all_c[order]
    all_w = all_w[order]
    uniq, first = np.unique(all_c, return_index=True)
    # sum weights within each unique-code group
    csum = np.concatenate(([0], np.cumsum(all_w)))
    bounds = np.append(first, all_w.size)
    cov = (csum[bounds[1:]] - csum[bounds[:-1]]).astype(np.int64)
    return uniq, cov            # uniq sorted ascending by code


def bottom_n(codes, cov, n):
    h = murmur3_vec(codes)
    if h.size <= n:
        order = np.argsort(h, kind="stable")
    else:
        part = np.argpartition(h, n - 1)[:n]
        order = part[np.argsort(h[part], kind="stable")]
    sc, sh, scov = codes[order], h[order], cov[order]
    d = {int(sc[i]): (int(sh[i]), int(scov[i])) for i in range(sc.size)}
    return sc, sh, d


def j_direct(a_codes, b_codes):
    if a_codes.size == 0 and b_codes.size == 0:
        return 1.0
    inter = np.intersect1d(a_codes, b_codes, assume_unique=True).size
    union = a_codes.size + b_codes.size - inter
    return inter / union if union else 0.0


def j_union(a_codes, a_h, b_codes, b_h, n):
    all_codes = np.concatenate([a_codes, b_codes])
    all_h = np.concatenate([a_h, b_h])
    uniq, first = np.unique(all_codes, return_index=True)
    uh = all_h[first]
    if uniq.size == 0:
        return 1.0
    m = min(n, uniq.size)
    idx = np.argsort(uh, kind="stable")[:m]
    sel = uniq[idx]
    return float(np.count_nonzero(np.isin(sel, a_codes) & np.isin(sel, b_codes))) / m


def one_seed(seed):
    rng = np.random.default_rng(seed)
    # one master encoded library; ref and query are partitions of it
    payloads = rng.choice(list("ACGT"), size=(N_MASTER, OLIGO_LEN))
    codes_list = []
    for i in range(N_MASTER):
        codes_list.append(oligo_codes("".join(payloads[i])))
    abund = np.maximum(1, np.round(
        COV_MEAN * np.exp(rng.normal(0, COV_SIGMA, N_MASTER)))).astype(np.int64)

    shared = np.arange(S_SHARED)
    extra = np.arange(S_SHARED, N_MASTER)

    out = []
    for r in RATIOS:
        a, b = split_for_ratio(r)                 # ref-only, query-only counts
        ref_idx = np.concatenate([shared, extra[:a]])
        qry_idx = np.concatenate([shared, extra[a:a + b]])
        rc, rcov = pool_maps(ref_idx, codes_list, abund)
        qc, qcov = pool_maps(qry_idx, codes_list, abund)

        j_true = FG.jaccard_from_codes(rc, qc)
        c_true = F.cos_true_arrays(rc, rcov, qc, qcov)
        r_real = max(rc.size, qc.size) / min(rc.size, qc.size)

        rec = {"seed": int(seed), "ratio": r, "ref_only": a, "qry_only": b,
               "support_ref": int(rc.size), "support_qry": int(qc.size),
               "r_realised": float(r_real),
               "j_true": float(j_true), "c_true": float(c_true), "N": {}}
        for n in N_SIZES:
            sac, sah, sad = bottom_n(rc, rcov, n)
            sbc, sbh, sbd = bottom_n(qc, qcov, n)
            jd = j_direct(sac, sbc)
            ju = j_union(sac, sah, sbc, sbh, n)
            cos = F.all_estimators(sad, sbd, n)
            rec["N"][str(n)] = {
                "j_direct": float(jd), "j_union": float(ju),
                "c_direct": float(cos["shipped"]),
                "c_union": float(cos["union_bottomN"]),
                "sketch_ref": len(sad), "sketch_qry": len(sbd),
            }
        out.append(rec)
    return out


def main():
    t0 = time.time()
    seeds = [SEED0 + i for i in range(N_SEEDS)]
    recs = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, sub in enumerate(ex.map(one_seed, seeds)):
            recs.extend(sub)
            print(f"  seed {i+1}/{N_SEEDS}  {time.time()-t0:.0f}s", flush=True)

    payload = {
        "experiment": "S2_common_index_support_sweep",
        "framing": "DNA data storage: reference vs query encoded oligo pools; no genomes",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "oligo_len": OLIGO_LEN, "primer_len": 20, "k": K,
            "shared_oligos_S": S_SHARED, "total_extra_E": E_EXTRA,
            "master_pool_oligos": N_MASTER,
            "true_jaccard_by_design": S_SHARED / (S_SHARED + E_EXTRA),
            "ratios": RATIOS, "N_sizes": N_SIZES, "n_seeds": N_SEEDS,
            "seed0": SEED0, "cov_mean": COV_MEAN, "cov_sigma": COV_SIGMA,
            "hash": "murmurhash3 seed=42 (fgr2 selection hash, vectorised)",
            "direct_estimator": "each sketch scored on its own bottom-N index "
                                "(calculate_similarity legacy)",
            "union_estimator": "both scored on the merged bottom-N common index",
        },
        "records": recs,
        "wall_seconds": time.time() - t0,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh)
    print("wrote", OUT, f"({len(recs)} cells, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
