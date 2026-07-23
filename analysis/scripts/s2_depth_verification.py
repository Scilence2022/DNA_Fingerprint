#!/usr/bin/env python3
"""
S2 (c) -- Practical impact for retrieval verification under a DEPTH gap.

DNA-DATA-STORAGE framing, full pool model. A REFERENCE archive (an encoded oligo
pool with shared primers) is sequenced once at a fixed depth. A QUERY is sequenced
at a range of depths (different per-oligo copy numbers) with per-base sequencing
error. Two query identities:
  - MATCHED  : the query is the SAME encoded file as the reference (a full,
               correct retrieval) -> the correct verification similarity is 1.0.
  - MISMATCH : the query is a DIFFERENT random encoded file -> correct ~ 0.

Mechanism: with sequencing error and -c 1 (no coverage filter), each read set spawns
distinct error k-mers whose number grows with depth. Unequal depth therefore makes
the two fingerprints' support sizes unequal (support ratio r != 1) EVEN for identical
data. The direct/legacy estimator (each fingerprint scored on its own index) then
reports a depth-dependent, biased similarity; the common-index (union) estimator scores
both on the merged bottom-N index and recovers the correct verification similarity.

Everything seeded; raw per-condition measurements persisted to JSON before reporting.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))

import fgrlib as FG                       # noqa: E402
import f1_common as F                     # noqa: E402
from f1_hash_np import murmur3_vec        # noqa: E402

OUT = os.path.join(REPO, "analysis", "results", "s2_depth_verification.json")

R_OLIGOS = 1000
OLIGO_LEN = 150
K = 21
ERR = 0.01                 # 1% per-base substitution (illumina-like for storage reads)
D_REF = 30
D_QRY = [5, 10, 20, 30, 50, 100]
N_SIZES = [1000, 10000]
N_SEEDS = 15
SEED0 = 20260722


def reads_to_maps(reads):
    """Exact canonical-code -> coverage (occurrence count across all reads),
    with 'N' separators so no k-mer spans a read boundary."""
    joined = "N".join(reads)
    codes, cnt = F.canonical_code_counts(joined, K)   # sorted ascending, counts
    return codes, cnt.astype(np.int64)


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
    import random
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed + 101)

    ref_pool = FG.make_oligo_pool(R_OLIGOS, OLIGO_LEN, pyrng)
    other_pool = FG.make_oligo_pool(R_OLIGOS, OLIGO_LEN, pyrng)  # different file

    # reference sequenced once at fixed depth
    ref_reads = FG.pool_to_reads(ref_pool, [D_REF] * R_OLIGOS, ERR, pyrng)
    rc, rcov = reads_to_maps(ref_reads)

    recs = []
    for identity, qpool in (("matched", ref_pool), ("mismatch", other_pool)):
        for dq in D_QRY:
            qreads = FG.pool_to_reads(qpool, [dq] * R_OLIGOS, ERR, pyrng)
            qc, qcov = reads_to_maps(qreads)
            r_real = max(rc.size, qc.size) / min(rc.size, qc.size)
            rec = {"seed": int(seed), "identity": identity, "d_ref": D_REF,
                   "d_qry": dq, "depth_ratio": max(D_REF, dq) / min(D_REF, dq),
                   "support_ref": int(rc.size), "support_qry": int(qc.size),
                   "support_ratio": float(r_real),
                   "target": 1.0 if identity == "matched" else 0.0, "N": {}}
            for n in N_SIZES:
                sac, sah, sad = bottom_n(rc, rcov, n)
                sbc, sbh, sbd = bottom_n(qc, qcov, n)
                cos = F.all_estimators(sad, sbd, n)
                rec["N"][str(n)] = {
                    "j_direct": float(j_direct(sac, sbc)),
                    "j_union": float(j_union(sac, sah, sbc, sbh, n)),
                    "c_direct": float(cos["shipped"]),
                    "c_union": float(cos["union_bottomN"]),
                }
            recs.append(rec)
    return recs


def main():
    t0 = time.time()
    seeds = [SEED0 + i for i in range(N_SEEDS)]
    recs = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, sub in enumerate(ex.map(one_seed, seeds)):
            recs.extend(sub)
            print(f"  seed {i+1}/{N_SEEDS}  {time.time()-t0:.0f}s", flush=True)

    payload = {
        "experiment": "S2c_depth_verification",
        "framing": "DNA data storage: reference archive vs query retrieval at a depth gap; no genomes",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "r_oligos": R_OLIGOS, "oligo_len": OLIGO_LEN, "k": K, "err_rate": ERR,
            "d_ref": D_REF, "d_qry": D_QRY, "N_sizes": N_SIZES, "n_seeds": N_SEEDS,
            "seed0": SEED0, "coverage_filter": "c=1 (none)",
            "note": "error k-mer inflation grows with depth -> unequal support -> "
                    "support-size ratio r != 1 even for identical data",
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
