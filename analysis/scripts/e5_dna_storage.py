#!/usr/bin/env python3
"""
E5 -- DNA data storage integrity verification.

Hypothesis: because fgr2 retains per-k-mer ABUNDANCE in its bottom-N sketch, an
abundance-weighted cosine similarity detects DNA-storage failure modes that a
presence/absence sketch (Jaccard / Mash) is structurally blind to.

Design (all parameters fixed BEFORE any result was inspected):
  pool          20,000 oligos, 150 nt, shared 20 nt fwd/rev primer sites
  base copies   20 per species
  baseline err  1e-3 per base, applied to EVERY condition including the reference
  k             21
  N             {100, 1000, 10000}
  coverage      -c 1 (filtering disabled) so abundance is not truncated
  replicates    10 independent seeds per (mode, level)

Failure modes:
  subst     total per-base substitution rate: 0.001(intact) .. 0.05
  dropout   fraction of oligo SPECIES lost entirely: 0 .. 0.50
  pcrbias   log-normal amplification skew sigma: 0 .. 2.0  (SET invariant --
            the decisive condition for the abundance claim)
  combined  t in 0..1 -> subst 0.001+0.02t, dropout 0.25t, skew 1.0t

Every condition is compared against ONE intact reference read set (seed 999000).
Level 0 of each mode is an independent intact replicate, providing the negative
class for ROC.
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
           "scratchpad/e5")
sys.path.insert(0, os.path.join(REPO, "analysis", "scripts"))
import fgrlib  # noqa: E402

FGR2 = os.path.join(REPO, "fgr2")
MASH = "/opt/homebrew/bin/mash"

# ---- fixed parameters -----------------------------------------------------
POOL_SEED = 20260721
N_OLIGOS = 20000
OLIGO_LEN = 150
BASE_COPIES = 20
BASE_ERR = 0.001
K = 21
N_SIZES = [100, 1000, 10000]
MASH_SIZES = [100, 1000, 10000]
N_SEEDS = 10
REF_SEED = 999000

LEVELS = {
    "subst":    [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05],
    "dropout":  [0.0, 0.05, 0.10, 0.20, 0.35, 0.50],
    "pcrbias":  [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
    "combined": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}


def build_pool():
    return fgrlib.make_oligo_pool(N_OLIGOS, OLIGO_LEN, random.Random(POOL_SEED))


def condition_reads(pool, mode, level, seed):
    """Return (reads, n_species_present, total_copies)."""
    rng = random.Random(seed)
    n = len(pool)
    if mode == "subst":
        copies = [BASE_COPIES] * n
        err = level
    elif mode == "dropout":
        copies = fgrlib.dropout_copies([BASE_COPIES] * n, level, rng)
        err = BASE_ERR
    elif mode == "pcrbias":
        copies = fgrlib.pcr_bias_copies(n, BASE_COPIES, level, rng)
        err = BASE_ERR
    elif mode == "combined":
        copies = fgrlib.pcr_bias_copies(n, BASE_COPIES, 1.0 * level, rng)
        copies = fgrlib.dropout_copies(copies, 0.25 * level, rng)
        err = BASE_ERR + 0.02 * level
    elif mode == "reference":
        copies = [BASE_COPIES] * n
        err = BASE_ERR
    else:
        raise ValueError(mode)
    reads = fgrlib.pool_to_reads(pool, copies, err, rng)
    rng.shuffle(reads)
    return reads, sum(1 for c in copies if c > 0), sum(copies)


def run_fgr2(fasta, out, n):
    subprocess.run([FGR2, "-k", str(K), "-N", str(n), "-c", "1", "-t", "2",
                    "-o", out, fasta], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    h = out + ".hist"
    if os.path.exists(h):
        os.remove(h)


def run_mash_sketch(fasta, out_prefix, s):
    subprocess.run([MASH, "sketch", "-k", str(K), "-s", str(s), "-m", "1",
                    "-o", out_prefix, fasta], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mash_dist(a_msh, b_msh):
    r = subprocess.run([MASH, "dist", a_msh, b_msh], check=True,
                       capture_output=True, text=True)
    f = r.stdout.split("\n")[0].split("\t")
    return float(f[2]), float(f[3])  # distance, p-value


def sketch_bytes(path):
    return os.path.getsize(path)


def process(task):
    mode, li, level, seed, tag = task
    t0 = time.time()
    pool = build_pool()
    reads, n_species, total_copies = condition_reads(pool, mode, level, seed)
    fa = os.path.join(SCRATCH, f"{tag}.fa")
    fgrlib.write_reads_fasta(fa, reads)
    del reads
    rec = {"mode": mode, "level_index": li, "level": level, "seed": seed,
           "tag": tag, "n_species_present": n_species,
           "total_copies": total_copies, "fgr2": {}, "mash": {}}
    for n in N_SIZES:
        out = os.path.join(SCRATCH, f"{tag}.N{n}.fgr2")
        t = time.time()
        run_fgr2(fa, out, n)
        rec["fgr2"][str(n)] = {"path": out, "bytes": sketch_bytes(out),
                               "seconds": time.time() - t}
    for s in MASH_SIZES:
        pre = os.path.join(SCRATCH, f"{tag}.s{s}")
        t = time.time()
        run_mash_sketch(fa, pre, s)
        rec["mash"][str(s)] = {"path": pre + ".msh",
                               "bytes": sketch_bytes(pre + ".msh"),
                               "seconds": time.time() - t}
    os.remove(fa)
    rec["wall_seconds"] = time.time() - t0
    return rec


def auc(neg, pos):
    """P(pos score > neg score) with ties at 0.5. Higher score = more degraded."""
    if not neg or not pos:
        return None
    tot = 0.0
    for p in pos:
        for q in neg:
            tot += 1.0 if p > q else (0.5 if p == q else 0.0)
    return tot / (len(pos) * len(neg))


def roc_curve(neg, pos):
    """Return (fpr list, tpr list) for score threshold sweep."""
    pts = sorted(set(neg + pos), reverse=True)
    fpr, tpr = [0.0], [0.0]
    for th in pts:
        tp = sum(1 for p in pos if p >= th)
        fp = sum(1 for q in neg if q >= th)
        tpr.append(tp / len(pos))
        fpr.append(fp / len(neg))
    fpr.append(1.0)
    tpr.append(1.0)
    return fpr, tpr


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    resdir = os.path.join(REPO, "analysis", "results")
    os.makedirs(resdir, exist_ok=True)
    t_start = time.time()

    # deterministic seeds (no PYTHONHASHSEED dependence)
    mode_ids = {"subst": 1, "dropout": 2, "pcrbias": 3, "combined": 4}
    tasks = [("reference", -1, 0.0, REF_SEED, "ref")]
    for mode, levels in LEVELS.items():
        for li, lv in enumerate(levels):
            for s in range(N_SEEDS):
                seed = 100000 + mode_ids[mode] * 10000 + li * 100 + s
                tasks.append((mode, li, lv, seed, f"{mode}_L{li}_S{s}"))
    print(f"{len(tasks)} read sets to build", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=7) as ex:
        for i, rec in enumerate(ex.map(process, tasks)):
            records.append(rec)
            if i % 10 == 0:
                print(f"  built {i+1}/{len(tasks)}  ({time.time()-t_start:.0f}s)",
                      flush=True)

    byname = {r["tag"]: r for r in records}
    ref = byname["ref"]

    # ---- load reference sketches -----------------------------------------
    ref_sk = {}
    for n in N_SIZES:
        h, c, _ = fgrlib.parse_sketch(ref["fgr2"][str(n)]["path"])
        ref_sk[n] = (h, c)

    # ---- similarities -----------------------------------------------------
    print("computing similarities", flush=True)
    for rec in records:
        if rec["tag"] == "ref":
            continue
        for n in N_SIZES:
            h, c, _ = fgrlib.parse_sketch(rec["fgr2"][str(n)]["path"])
            rh, rc = ref_sk[n]
            rec["fgr2"][str(n)]["jaccard_direct"] = fgrlib.jaccard_direct(
                set(rh), set(h))
            rec["fgr2"][str(n)]["jaccard_union"] = fgrlib.jaccard_union(rh, h, n)
            rec["fgr2"][str(n)]["cosine"] = fgrlib.cosine(rc, c)
            rec["fgr2"][str(n)]["n_kmers"] = len(h)
        for s in MASH_SIZES:
            d, p = mash_dist(ref["mash"][str(s)]["path"], rec["mash"][str(s)]["path"])
            rec["mash"][str(s)]["distance"] = d
            rec["mash"][str(s)]["pvalue"] = p

    # ---- aggregate: AUC per (mode, level, metric, size) -------------------
    METRICS = [("cosine", "fgr2", -1), ("jaccard_direct", "fgr2", -1),
               ("jaccard_union", "fgr2", -1), ("mash_distance", "mash", +1)]

    def scores(mode, li, metric, family, size):
        out = []
        for s in range(N_SEEDS):
            r = byname[f"{mode}_L{li}_S{s}"]
            if family == "fgr2":
                out.append(r["fgr2"][str(size)][metric])
            else:
                out.append(r["mash"][str(size)]["distance"])
        return out

    agg = {}
    for mode, levels in LEVELS.items():
        agg[mode] = {}
        for metric, family, sign in METRICS:
            agg[mode][metric] = {}
            for size in (N_SIZES if family == "fgr2" else MASH_SIZES):
                neg = [sign * v for v in scores(mode, 0, metric, family, size)]
                per_level = []
                for li, lv in enumerate(levels):
                    raw = scores(mode, li, metric, family, size)
                    pos = [sign * v for v in raw]
                    a = auc(neg, pos) if li > 0 else 0.5
                    per_level.append({
                        "level_index": li, "level": lv,
                        "mean": sum(raw) / len(raw),
                        "sd": (sum((x - sum(raw) / len(raw)) ** 2 for x in raw)
                               / (len(raw) - 1)) ** 0.5,
                        "values": raw, "auc_vs_level0": a, "n": len(raw)})
                # detection floor: smallest level with AUC >= 0.95
                floor = None
                for e in per_level[1:]:
                    if e["auc_vs_level0"] is not None and e["auc_vs_level0"] >= 0.95:
                        floor = e["level"]
                        break
                agg[mode][metric][str(size)] = {
                    "per_level": per_level, "detection_floor_auc0.95": floor}

    # ---- pooled ROC at a stated degradation threshold ---------------------
    # "Degraded" = the lowest NON-ZERO perturbation level and above, pooled.
    pooled = {}
    for mode, levels in LEVELS.items():
        pooled[mode] = {}
        for metric, family, sign in METRICS:
            pooled[mode][metric] = {}
            for size in (N_SIZES if family == "fgr2" else MASH_SIZES):
                neg = [sign * v for v in scores(mode, 0, metric, family, size)]
                pos = []
                for li in range(1, len(levels)):
                    pos += [sign * v for v in scores(mode, li, metric, family, size)]
                f, t = roc_curve(neg, pos)
                pooled[mode][metric][str(size)] = {
                    "auc": auc(neg, pos), "n_neg": len(neg), "n_pos": len(pos),
                    "fpr": f, "tpr": t}

    out = {
        "experiment_id": "E5_dna_storage_integrity",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "pool_seed": POOL_SEED, "n_oligos": N_OLIGOS, "oligo_len": OLIGO_LEN,
            "primer_len_each_end": 20, "base_copies": BASE_COPIES,
            "baseline_error": BASE_ERR, "k": K, "N_sizes": N_SIZES,
            "mash_sizes": MASH_SIZES, "n_seeds": N_SEEDS, "ref_seed": REF_SEED,
            "levels": LEVELS, "fgr2_flags": "-c 1 -t 2 (coverage filter disabled)",
            "mash_flags": f"sketch -k {K} -s S -m 1",
            "degraded_definition": ("pooled ROC positive class = all non-zero "
                                    "perturbation levels of that mode; negative "
                                    "class = level-0 intact replicates"),
        },
        "reference": ref,
        "records": records,
        "aggregate_auc": agg,
        "pooled_roc": pooled,
        "total_wall_seconds": time.time() - t_start,
    }
    path = os.path.join(resdir, "e5_dna_storage.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path, f"({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
