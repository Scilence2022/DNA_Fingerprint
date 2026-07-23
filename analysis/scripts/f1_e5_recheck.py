#!/usr/bin/env python3
"""
F1 part 3 -- re-examine E5's headline (pure amplification skew is detectable by
abundance-weighted cosine but not by set metrics) under the CORRECTED cosine.

Uses the exact E5 control design from analysis/results/e5_pcrbias_control.json.
The fgr2 sketches from that run are still on disk and are reused verbatim; the
read sets were deleted after sketching but are exactly reproducible from the
recorded seeds, so they are regenerated here to obtain the GROUND-TRUTH
abundance-weighted cosine over the complete canonical 21-mer coverage vectors
of each read set vs the reference read set. That ground truth is an oracle: it
bounds what any sketch-based cosine could achieve.

Metrics scored: shipped cosine, union-bottom-N cosine, intersection-only
cosine, analytic-corrected shipped cosine, and the exact cosine oracle.
AUC convention is E5's: score = -cosine (lower similarity = more degraded).
"""
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))

import fgrlib                       # noqa: E402
import f1_common as F               # noqa: E402
from e5_dna_storage import (POOL_SEED, N_OLIGOS, OLIGO_LEN, BASE_COPIES,  # noqa: E402
                            BASE_ERR, K, build_pool, auc, roc_curve)
from e5_pcrbias_control import control_copies, TARGET_TOTAL, SIGMAS  # noqa: E402

E5 = os.path.join(REPO, "analysis", "results", "e5_pcrbias_control.json")
OUT = os.path.join(REPO, "analysis", "results", "f1_e5_cosine_recheck.json")
N_SIZES = [100, 1000, 10000]
BOOT = 10000
BOOT_SEED = 424242

_REF = {}


def read_counts(tag, sigma, seed):
    """Regenerate the E5 read set exactly and return (codes, counts)."""
    pool = build_pool()
    rng = random.Random(seed)
    if tag == "ctrlref":
        copies = [BASE_COPIES] * N_OLIGOS
    else:
        copies = control_copies(N_OLIGOS, sigma, rng)
    reads = fgrlib.pool_to_reads(pool, copies, BASE_ERR, rng)
    rng.shuffle(reads)
    seq = "N".join(reads)
    del reads
    codes, cnt = F.canonical_code_counts(seq, K)
    return codes, cnt.astype(np.int64), sum(copies), len(copies)


def worker(task):
    tag, sigma, seed, ref_seed = task
    if "ref" not in _REF:
        _REF["ref"] = read_counts("ctrlref", 0.0, ref_seed)
    rc, rn, _, _ = _REF["ref"]
    codes, cnt, total_copies, _ = read_counts(tag, sigma, seed)
    truth = F.cos_true_arrays(rc, rn, codes, cnt)
    return {"tag": tag, "cos_true": truth,
            "supp": int(codes.size), "supp_ref": int(rc.size),
            "supp_ratio": codes.size / rc.size,
            "total_copies": int(total_copies)}


def main():
    e5 = json.load(open(E5))
    ref = e5["reference"]
    ref_seed = ref["seed"]
    recs = [r for r in e5["records"] if r["tag"] != "ctrlref"]
    byname = {r["tag"]: r for r in recs}
    n_seeds = e5["parameters"]["n_seeds"]

    t0 = time.time()
    # ---- sketch-based estimators (reuse the E5 sketch files verbatim)
    ref_sk = {}
    for n in N_SIZES:
        h, c, _ = fgrlib.parse_sketch(ref["fgr2"][str(n)]["path"])
        ref_sk[n] = {km: (h[km], c[km]) for km in h}
    est = {}
    for r in recs:
        est[r["tag"]] = {}
        for n in N_SIZES:
            h, c, _ = fgrlib.parse_sketch(r["fgr2"][str(n)]["path"])
            sk = {km: (h[km], c[km]) for km in h}
            est[r["tag"]][str(n)] = F.all_estimators(ref_sk[n], sk, n)
    print(f"sketch estimators done {time.time()-t0:.0f}s", flush=True)

    # ---- ground-truth cosine (regenerated read sets)
    tasks = [(r["tag"], r["sigma"], r["seed"], ref_seed) for r in recs]
    CACHE = os.path.join(os.path.dirname(OUT), "f1_e5_truth_cache.json")
    truth = {}
    if os.path.exists(CACHE):
        cached = json.load(open(CACHE))
        if all(r["tag"] in cached for r in recs):
            truth = cached
            print("reusing ground-truth cache", CACHE, flush=True)
    if not truth:
        with ProcessPoolExecutor(max_workers=5) as ex:
            for i, o in enumerate(ex.map(worker, tasks)):
                truth[o["tag"]] = o
                if (i + 1) % 10 == 0:
                    print(f"  truth {i+1}/{len(tasks)} {time.time()-t0:.0f}s",
                          flush=True)
        with open(CACHE, "w") as fh:
            json.dump(truth, fh)

    # ---- assemble scores; E5 convention: score = -cosine
    METRICS = [(f"cosine_{name}", name) for name in F.EST_NAMES]

    def scores(li, name, size):
        return [-est[f"ctrl_L{li}_S{s}"][str(size)][name] for s in range(n_seeds)]

    def scores_truth(li):
        return [-truth[f"ctrl_L{li}_S{s}"]["cos_true"] for s in range(n_seeds)]

    rng = np.random.default_rng(BOOT_SEED)

    def _auc_np(neg, pos):
        d = pos[..., :, None] - neg[..., None, :]
        return ((d > 0) + 0.5 * (d == 0)).mean(axis=(-1, -2))

    def boot_auc(neg, pos_by_level):
        """Bootstrap the pooled AUC by resampling seeds within each level.
        Vectorised; identical convention to e5_dna_storage.auc (ties = 0.5)."""
        neg = np.asarray(neg, dtype=float)
        pos_levels = [np.asarray(p, dtype=float) for p in pos_by_level]
        ng = neg[rng.integers(0, neg.size, size=(BOOT, neg.size))]
        ps = np.concatenate(
            [p[rng.integers(0, p.size, size=(BOOT, p.size))] for p in pos_levels],
            axis=1)
        return _auc_np(ng, ps)

    agg = {}
    boot_store = {}
    for label, name in METRICS:
        agg[label] = {}
        for size in N_SIZES:
            neg = scores(0, name, size)
            per_level, pos_all, pos_levels = [], [], []
            for li, sg in enumerate(SIGMAS):
                pos = scores(li, name, size)
                raw = [-v for v in pos]
                if li > 0:
                    pos_all += pos
                    pos_levels.append(pos)
                mu = sum(raw) / len(raw)
                per_level.append({
                    "level_index": li, "sigma": sg, "mean_cosine": mu,
                    "sd": math.sqrt(sum((x - mu) ** 2 for x in raw) / (len(raw) - 1)),
                    "values_cosine": raw,
                    "auc_vs_level0": auc(neg, pos) if li > 0 else 0.5,
                    "n": len(raw)})
            fpr, tpr = roc_curve(neg, pos_all)
            floor = next((e["sigma"] for e in per_level[1:]
                          if e["auc_vs_level0"] >= 0.95), None)
            agg[label][str(size)] = {
                "per_level": per_level,
                "pooled_auc": auc(neg, pos_all),
                "detection_floor_auc0.95": floor,
                "n_neg": len(neg), "n_pos": len(pos_all),
                "roc": {"fpr": fpr, "tpr": tpr}}
            boot_store[(label, size)] = boot_auc(neg, pos_levels)

    # ground-truth oracle
    agg["cosine_ground_truth"] = {}
    neg_t = scores_truth(0)
    per_level, pos_all, pos_levels = [], [], []
    for li, sg in enumerate(SIGMAS):
        pos = scores_truth(li)
        raw = [-v for v in pos]
        if li > 0:
            pos_all += pos
            pos_levels.append(pos)
        mu = sum(raw) / len(raw)
        per_level.append({"level_index": li, "sigma": sg, "mean_cosine": mu,
                          "sd": math.sqrt(sum((x - mu) ** 2 for x in raw) / (len(raw) - 1)),
                          "values_cosine": raw,
                          "auc_vs_level0": auc(neg_t, pos) if li > 0 else 0.5,
                          "n": len(raw)})
    fpr, tpr = roc_curve(neg_t, pos_all)
    agg["cosine_ground_truth"]["exact"] = {
        "per_level": per_level, "pooled_auc": auc(neg_t, pos_all),
        "detection_floor_auc0.95": next((e["sigma"] for e in per_level[1:]
                                         if e["auc_vs_level0"] >= 0.95), None),
        "n_neg": len(neg_t), "n_pos": len(pos_all), "roc": {"fpr": fpr, "tpr": tpr}}
    boot_store[("cosine_ground_truth", "exact")] = boot_auc(neg_t, pos_levels)

    # bootstrap CIs and paired difference union - shipped
    ci = {}
    for (label, size), v in boot_store.items():
        ci[f"{label}|{size}"] = {"mean": float(v.mean()),
                                 "ci95": [float(np.percentile(v, 2.5)),
                                          float(np.percentile(v, 97.5))]}
    diffs = {}
    for size in N_SIZES:
        d = boot_store[("cosine_union_bottomN", size)] - boot_store[("cosine_shipped", size)]
        p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
        diffs[str(size)] = {"mean_diff_auc": float(d.mean()),
                            "ci95": [float(np.percentile(d, 2.5)),
                                     float(np.percentile(d, 97.5))],
                            "bootstrap_two_sided_p": float(p),
                            "note": "paired bootstrap over seeds, B=%d" % BOOT}

    supp = {t: {kk: truth[t][kk] for kk in ("supp", "supp_ref", "supp_ratio",
                                            "total_copies")} for t in truth}
    out = {"experiment_id": "F1_E5_cosine_recheck",
           "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source_design": E5,
           "parameters": {"sigmas": SIGMAS, "N_sizes": N_SIZES,
                          "n_seeds": n_seeds, "k": K, "bootstrap_B": BOOT,
                          "bootstrap_seed": BOOT_SEED,
                          "auc_convention": "score = -cosine"},
           "per_record_estimates": est,
           "per_record_truth": {t: truth[t]["cos_true"] for t in truth},
           "support_sizes": supp,
           "aggregate": agg,
           "bootstrap_ci": ci,
           "auc_diff_union_minus_shipped": diffs,
           "wall_seconds": time.time() - t0}
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT)
    for label in [f"cosine_{n}" for n in F.EST_NAMES]:
        print(f"{label:26}", " ".join(
            f"N{s}:AUC={agg[label][str(s)]['pooled_auc']:.3f}" for s in N_SIZES))
    print(f"{'cosine_ground_truth':26} exact:AUC="
          f"{agg['cosine_ground_truth']['exact']['pooled_auc']:.3f}")
    print("diff union-shipped:", json.dumps(diffs, indent=1))


if __name__ == "__main__":
    main()
