#!/usr/bin/env python3
"""
E5 control -- STRICTLY set-invariant, depth-matched PCR amplification bias.

Motivation: in the main E5 pcrbias series, raising the log-normal skew sigma also
raises TOTAL read depth (E[e^X] = e^{sigma^2/2}; measured 400k -> 3.0M copies from
sigma 0 -> 2.0) and, at sigma >= 1.5, drives a few percent of species to zero
copies by rounding. Either effect can move a presence/absence sketch, so the main
series does not cleanly isolate abundance.

This control removes both confounds:
  * copies are rescaled so total depth is held at ~400,000 reads at every sigma
  * every copy number is clamped to >= 1, so the set of oligo species -- and
    therefore the true k-mer SET -- is EXACTLY invariant across all sigma
Only the RELATIVE abundance distribution changes. If cosine still separates and
Jaccard/Mash do not, the abundance-weighting claim survives its strongest test.

All other parameters identical to the main run.
"""
import json
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468/"
           "scratchpad/e5c")
sys.path.insert(0, os.path.join(REPO, "analysis", "scripts"))
import fgrlib  # noqa: E402
from e5_dna_storage import (FGR2, MASH, POOL_SEED, N_OLIGOS, OLIGO_LEN,  # noqa: E402
                            BASE_COPIES, BASE_ERR, K, N_SIZES, MASH_SIZES,
                            N_SEEDS, REF_SEED, build_pool, run_fgr2,
                            run_mash_sketch, mash_dist, sketch_bytes, auc,
                            roc_curve)

SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
TARGET_TOTAL = N_OLIGOS * BASE_COPIES  # 400,000 reads at every sigma


def control_copies(n, sigma, rng):
    """Log-normal relative abundance, rescaled to TARGET_TOTAL, clamped >= 1 so
    the species set is exactly preserved."""
    if sigma <= 0:
        return [BASE_COPIES] * n
    f = [math.exp(rng.gauss(0.0, sigma)) for _ in range(n)]
    s = sum(f)
    copies = [max(1, int(round(TARGET_TOTAL * x / s))) for x in f]
    return copies


def process(task):
    li, sigma, seed, tag = task
    t0 = time.time()
    pool = build_pool()
    rng = random.Random(seed)
    if tag == "ctrlref":
        copies = [BASE_COPIES] * N_OLIGOS
    else:
        copies = control_copies(N_OLIGOS, sigma, rng)
    reads = fgrlib.pool_to_reads(pool, copies, BASE_ERR, rng)
    rng.shuffle(reads)
    fa = os.path.join(SCRATCH, f"{tag}.fa")
    fgrlib.write_reads_fasta(fa, reads)
    n_reads = len(reads)
    del reads
    rec = {"level_index": li, "sigma": sigma, "seed": seed, "tag": tag,
           "n_species_present": sum(1 for c in copies if c > 0),
           "total_copies": sum(copies), "n_reads": n_reads,
           "max_copies": max(copies), "min_copies": min(copies),
           "fgr2": {}, "mash": {}}
    for n in N_SIZES:
        out = os.path.join(SCRATCH, f"{tag}.N{n}.fgr2")
        run_fgr2(fa, out, n)
        rec["fgr2"][str(n)] = {"path": out, "bytes": sketch_bytes(out)}
    for s in MASH_SIZES:
        pre = os.path.join(SCRATCH, f"{tag}.s{s}")
        run_mash_sketch(fa, pre, s)
        rec["mash"][str(s)] = {"path": pre + ".msh",
                               "bytes": sketch_bytes(pre + ".msh")}
    os.remove(fa)
    rec["wall_seconds"] = time.time() - t0
    return rec


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    t_start = time.time()
    tasks = [(-1, 0.0, REF_SEED, "ctrlref")]
    for li, sg in enumerate(SIGMAS):
        for s in range(N_SEEDS):
            tasks.append((li, sg, 700000 + li * 100 + s, f"ctrl_L{li}_S{s}"))
    print(f"{len(tasks)} read sets", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=7) as ex:
        for i, rec in enumerate(ex.map(process, tasks)):
            records.append(rec)
            if i % 10 == 0:
                print(f"  {i+1}/{len(tasks)} ({time.time()-t_start:.0f}s)", flush=True)

    byname = {r["tag"]: r for r in records}
    ref = byname["ctrlref"]
    ref_sk = {}
    for n in N_SIZES:
        h, c, _ = fgrlib.parse_sketch(ref["fgr2"][str(n)]["path"])
        ref_sk[n] = (h, c)

    for rec in records:
        if rec["tag"] == "ctrlref":
            continue
        for n in N_SIZES:
            h, c, _ = fgrlib.parse_sketch(rec["fgr2"][str(n)]["path"])
            rh, rc = ref_sk[n]
            rec["fgr2"][str(n)]["jaccard_direct"] = fgrlib.jaccard_direct(set(rh), set(h))
            rec["fgr2"][str(n)]["jaccard_union"] = fgrlib.jaccard_union(rh, h, n)
            rec["fgr2"][str(n)]["cosine"] = fgrlib.cosine(rc, c)
        for s in MASH_SIZES:
            d, p = mash_dist(ref["mash"][str(s)]["path"], rec["mash"][str(s)]["path"])
            rec["mash"][str(s)]["distance"] = d

    METRICS = [("cosine", "fgr2", -1), ("jaccard_direct", "fgr2", -1),
               ("jaccard_union", "fgr2", -1), ("mash_distance", "mash", +1)]

    def scores(li, metric, family, size):
        out = []
        for s in range(N_SEEDS):
            r = byname[f"ctrl_L{li}_S{s}"]
            out.append(r["fgr2"][str(size)][metric] if family == "fgr2"
                       else r["mash"][str(size)]["distance"])
        return out

    agg, pooled = {}, {}
    for metric, family, sign in METRICS:
        agg[metric], pooled[metric] = {}, {}
        for size in N_SIZES:
            neg = [sign * v for v in scores(0, metric, family, size)]
            per_level, pos_all = [], []
            for li, sg in enumerate(SIGMAS):
                raw = scores(li, metric, family, size)
                pos = [sign * v for v in raw]
                if li > 0:
                    pos_all += pos
                mu = sum(raw) / len(raw)
                per_level.append({
                    "level_index": li, "sigma": sg, "mean": mu,
                    "sd": (sum((x - mu) ** 2 for x in raw) / (len(raw) - 1)) ** 0.5,
                    "values": raw,
                    "auc_vs_level0": auc(neg, pos) if li > 0 else 0.5, "n": len(raw)})
            floor = next((e["sigma"] for e in per_level[1:]
                          if e["auc_vs_level0"] >= 0.95), None)
            f, t = roc_curve(neg, pos_all)
            agg[metric][str(size)] = {"per_level": per_level,
                                      "detection_floor_auc0.95": floor}
            pooled[metric][str(size)] = {"auc": auc(neg, pos_all), "fpr": f, "tpr": t,
                                         "n_neg": len(neg), "n_pos": len(pos_all)}

    out = {"experiment_id": "E5_control_setinvariant_depthmatched_pcrbias",
           "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "parameters": {"sigmas": SIGMAS, "target_total_reads": TARGET_TOTAL,
                          "clamp_min_copies": 1, "pool_seed": POOL_SEED,
                          "n_oligos": N_OLIGOS, "oligo_len": OLIGO_LEN,
                          "baseline_error": BASE_ERR, "k": K, "N_sizes": N_SIZES,
                          "n_seeds": N_SEEDS, "ref_seed": REF_SEED,
                          "note": "species set exactly invariant; total depth matched"},
           "reference": ref, "records": records,
           "aggregate_auc": agg, "pooled_roc": pooled,
           "total_wall_seconds": time.time() - t_start}
    path = os.path.join(REPO, "analysis", "results", "e5_pcrbias_control.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)
    for metric in ["cosine", "jaccard_direct", "jaccard_union", "mash_distance"]:
        e = agg[metric]["1000"]["per_level"]
        print(f"{metric:16}", " ".join(f"{p['sigma']}:{p['mean']:.4f}"
                                       f"(AUC{p['auc_vs_level0']:.2f})" for p in e),
              " pooledAUC=", round(pooled[metric]["1000"]["auc"], 3))


if __name__ == "__main__":
    main()
