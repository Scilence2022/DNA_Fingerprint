#!/usr/bin/env python3
"""
F3 -- Fragility of the E5 abundance-weighted (cosine) result to sequencing-DEPTH
      MISMATCH between the two pools being compared.

E5's control (analysis/results/e5_pcrbias_control.json) held total depth matched
to 0.9% while sweeping log-normal amplification skew sigma with the oligo SPECIES
SET EXACTLY INVARIANT. It reported pooled cosine AUC = 1.000 and Jaccard/Mash at
chance.  The smallest claimed detectable signal is sigma = 0.25, where mean cosine
moves 0.999133 -> 0.995834, i.e. only 0.0033.

Real DNA-storage comparisons are not depth matched.  This experiment reproduces
E5's control design EXACTLY (same pool seed, same 20,000 x 150 nt oligo pool, same
0.001 per-base error, k=21, -c 1, clamp copies >= 1 so the species set is exactly
invariant) and CROSSES amplification skew sigma with a DEPTH RATIO between the
query pool and the fixed 400,000-read reference.

Three stages
------------
A  main sweep, per-base error 0.001 (E5's setting).
   ratios x sigmas x 10 seeds, sketches at N = 1000 and N = 10000.
   Seeds are constructed so that the log-normal abundance weights depend ONLY on
   (sigma, seed index) and NOT on the depth ratio -- therefore every ratio column
   is a matched pair of the ratio-1 column and paired tests are valid.

B  mechanism arm 1: identical design but with ZERO sequencing error.  With no
   error k-mers every k-mer count is exactly proportional to depth, so cosine
   (scale invariant) must be exactly depth invariant.  Any depth sensitivity seen
   in stage A but absent here is attributable to error-k-mer dilution, not to
   coverage rescaling.

C  mechanism arm 2: counter saturation.  A small (100 oligo) error-free pool at
   base copy number 8000 is swept over depth ratios that cross fgr2's counter
   ceiling KC_MAX = 16383, alongside a low-depth control arm (base copies 20)
   where saturation is impossible.  The difference between the two arms isolates
   the saturation contribution.

All metric computation is done post hoc by f3_analyze.py from the sketch files
this script leaves on disk.
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
           "scratchpad/f3")
sys.path.insert(0, os.path.join(REPO, "analysis", "scripts"))
import fgrlib  # noqa: E402

FGR2 = os.path.join(REPO, "fgr2")
MASH = "/opt/homebrew/bin/mash"
KC_MAX = 16383

# ---- E5 control parameters, copied verbatim -------------------------------
POOL_SEED = 20260721
N_OLIGOS = 20000
OLIGO_LEN = 150
BASE_COPIES = 20
BASE_ERR = 0.001
K = 21
TARGET_TOTAL = N_OLIGOS * BASE_COPIES          # 400,000 reads at ratio 1.0
REF_SEED = 999000

# ---- F3 sweep -------------------------------------------------------------
RATIOS_A = [0.1, 0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0, 4.0, 10.0]
SIGMAS_A = [0.0, 0.25, 0.5, 1.0, 2.0]
N_SEEDS_A = 10
N_SIZES_A = [1000, 10000]
MASH_SIZE = 1000

RATIOS_B = [1.0, 1.5, 2.0, 4.0, 10.0]
SIGMAS_B = [0.0, 0.25, 1.0]
N_SEEDS_B = 5

# stage C
C_N_OLIGOS = 100
C_RATIOS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
C_SIGMAS = [0.0, 0.25, 1.0]
C_SEEDS = 5
C_ARMS = {"nosat": 20, "sat": 8000}   # base copies per oligo


# ---------------------------------------------------------------- utilities
def weights_for(sigma, wseed, n):
    """Log-normal relative abundance weights. Depends only on (sigma, wseed)."""
    if sigma <= 0:
        return [1.0] * n
    r = random.Random(wseed)
    return [math.exp(r.gauss(0.0, sigma)) for _ in range(n)]


def copies_from_weights(w, target_total):
    """Rescale to target_total reads and clamp >= 1 (species set invariant)."""
    s = sum(w)
    return [max(1, int(round(target_total * x / s))) for x in w]


def write_reads_streaming(path, pool, copies, err_rate, rng):
    """Stream reads to FASTA without materialising the whole list."""
    alts = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}
    rr = rng.random
    rc = rng.choice
    n = 0
    with open(path, "w") as fh:
        buf = []
        for oligo, c in zip(pool, copies):
            for _ in range(c):
                if err_rate > 0:
                    seq = "".join(rc(alts[ch]) if rr() < err_rate else ch
                                  for ch in oligo)
                else:
                    seq = oligo
                buf.append(f">r{n}\n{seq}\n")
                n += 1
                if len(buf) >= 50000:
                    fh.write("".join(buf))
                    buf = []
        if buf:
            fh.write("".join(buf))
    return n


def run_fgr2(fasta, out, n):
    subprocess.run([FGR2, "-k", str(K), "-N", str(n), "-c", "1", "-t", "2",
                    "-o", out, fasta], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    h = out + ".hist"
    if os.path.exists(h):
        os.remove(h)


def run_mash(fasta, prefix, s):
    subprocess.run([MASH, "sketch", "-k", str(K), "-s", str(s), "-m", "1",
                    "-o", prefix, fasta], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sketch_stats(path):
    _, cov, _ = fgrlib.parse_sketch(path)
    v = sorted(cov.values())
    if not v:
        return {"n_kmers": 0}
    nsat = sum(1 for x in v if x >= KC_MAX)
    return {"n_kmers": len(v), "n_saturated": nsat,
            "frac_saturated": nsat / len(v),
            "median_count": v[len(v) // 2], "max_count": v[-1],
            "min_count": v[0], "mean_count": sum(v) / len(v),
            "frac_count_eq_1": sum(1 for x in v if x == 1) / len(v),
            "total_count": sum(v)}


# ------------------------------------------------------------------ workers
def process_A(task):
    stage, ratio, sigma, si, err, tag = task
    t0 = time.time()
    pool = fgrlib.make_oligo_pool(N_OLIGOS, OLIGO_LEN, random.Random(POOL_SEED))
    if tag.startswith("ref"):
        copies = [BASE_COPIES] * N_OLIGOS
        rng = random.Random(REF_SEED)
    else:
        # weights depend ONLY on (sigma, si) -> ratio columns are matched pairs
        wseed = 800000 + int(round(sigma * 1000)) * 1000 + si
        w = weights_for(sigma, wseed, N_OLIGOS)
        copies = copies_from_weights(w, int(round(TARGET_TOTAL * ratio)))
        rng = random.Random(wseed * 31 + int(round(ratio * 1000)))
    fa = os.path.join(SCRATCH, f"{tag}.fa")
    n_reads = write_reads_streaming(fa, pool, copies, err, rng)
    rec = {"stage": stage, "tag": tag, "ratio": ratio, "sigma": sigma,
           "seed_index": si, "err_rate": err,
           "n_species_present": sum(1 for c in copies if c > 0),
           "total_copies": sum(copies), "n_reads": n_reads,
           "max_copies": max(copies), "min_copies": min(copies),
           "fgr2": {}, "mash": {}}
    sizes = N_SIZES_A if stage == "A" else [1000]
    for n in sizes:
        out = os.path.join(SCRATCH, f"{tag}.N{n}.fgr2")
        run_fgr2(fa, out, n)
        rec["fgr2"][str(n)] = dict(path=out, **sketch_stats(out))
    if stage == "A":
        pre = os.path.join(SCRATCH, f"{tag}.s{MASH_SIZE}")
        run_mash(fa, pre, MASH_SIZE)
        rec["mash"][str(MASH_SIZE)] = {"path": pre + ".msh"}
    os.remove(fa)
    rec["wall_seconds"] = time.time() - t0
    return rec


def process_C(task):
    arm, base, ratio, sigma, si, tag = task
    t0 = time.time()
    pool = fgrlib.make_oligo_pool(C_N_OLIGOS, OLIGO_LEN,
                                  random.Random(POOL_SEED + 7))
    target = int(round(C_N_OLIGOS * base * ratio))
    if tag.startswith("ref"):
        copies = [int(round(base))] * C_N_OLIGOS
        rng = random.Random(REF_SEED)
    else:
        wseed = 900000 + int(round(sigma * 1000)) * 1000 + si
        w = weights_for(sigma, wseed, C_N_OLIGOS)
        copies = copies_from_weights(w, target)
        rng = random.Random(wseed * 31 + int(round(ratio * 1000)))
    fa = os.path.join(SCRATCH, f"{tag}.fa")
    n_reads = write_reads_streaming(fa, pool, copies, 0.0, rng)
    rec = {"stage": "C", "arm": arm, "base_copies": base, "tag": tag,
           "ratio": ratio, "sigma": sigma, "seed_index": si, "err_rate": 0.0,
           "n_reads": n_reads, "total_copies": sum(copies),
           "max_copies": max(copies), "min_copies": min(copies), "fgr2": {},
           "mash": {}}
    out = os.path.join(SCRATCH, f"{tag}.N1000.fgr2")
    run_fgr2(fa, out, 1000)
    rec["fgr2"]["1000"] = dict(path=out, **sketch_stats(out))
    os.remove(fa)
    rec["wall_seconds"] = time.time() - t0
    return rec


# --------------------------------------------------------------------- main
def main():
    os.makedirs(SCRATCH, exist_ok=True)
    t0 = time.time()

    tasks_A = [("A", 1.0, 0.0, -1, BASE_ERR, "refA")]
    for r in RATIOS_A:
        for sg in SIGMAS_A:
            for si in range(N_SEEDS_A):
                tasks_A.append(("A", r, sg, si, BASE_ERR,
                                f"A_r{r}_s{sg}_S{si}"))
    tasks_B = [("B", 1.0, 0.0, -1, 0.0, "refB")]
    for r in RATIOS_B:
        for sg in SIGMAS_B:
            for si in range(N_SEEDS_B):
                tasks_B.append(("B", r, sg, si, 0.0, f"B_r{r}_s{sg}_S{si}"))

    tasks_C = []
    for arm, base in C_ARMS.items():
        tasks_C.append((arm, base, 1.0, 0.0, -1, f"refC_{arm}"))
        for r in C_RATIOS:
            for sg in C_SIGMAS:
                for si in range(C_SEEDS):
                    tasks_C.append((arm, base, r, sg, si,
                                    f"C_{arm}_r{r}_s{sg}_S{si}"))

    print(f"stage A {len(tasks_A)}, stage B {len(tasks_B)}, "
          f"stage C {len(tasks_C)}", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=7) as ex:
        for name, fn, tasks in (("A", process_A, tasks_A),
                                ("B", process_A, tasks_B),
                                ("C", process_C, tasks_C)):
            done = 0
            for rec in ex.map(fn, tasks):
                records.append(rec)
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    print(f"  stage {name}: {done}/{len(tasks)} "
                          f"({time.time()-t0:.0f}s)", flush=True)

    out = {
        "experiment_id": "F3_depth_mismatch_fragility",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "pool_seed": POOL_SEED, "n_oligos": N_OLIGOS,
            "oligo_len": OLIGO_LEN, "base_copies": BASE_COPIES,
            "baseline_error": BASE_ERR, "k": K,
            "target_total_reads_at_ratio1": TARGET_TOTAL,
            "clamp_min_copies": 1, "ref_seed": REF_SEED,
            "fgr2_flags": "-k 21 -c 1 -t 2 (coverage filter disabled)",
            "mash_flags": f"sketch -k {K} -s {MASH_SIZE} -m 1",
            "KC_MAX": KC_MAX,
            "stage_A": {"ratios": RATIOS_A, "sigmas": SIGMAS_A,
                        "n_seeds": N_SEEDS_A, "N_sizes": N_SIZES_A,
                        "err_rate": BASE_ERR,
                        "note": "exact E5 control design + depth ratio"},
            "stage_B": {"ratios": RATIOS_B, "sigmas": SIGMAS_B,
                        "n_seeds": N_SEEDS_B, "N_sizes": [1000],
                        "err_rate": 0.0,
                        "note": "zero-error mechanism control"},
            "stage_C": {"arms": C_ARMS, "n_oligos": C_N_OLIGOS,
                        "ratios": C_RATIOS, "sigmas": C_SIGMAS,
                        "n_seeds": C_SEEDS, "err_rate": 0.0,
                        "note": "counter-saturation mechanism arm"},
            "pairing": ("log-normal weights are a function of (sigma, seed "
                        "index) only, so the same seed index at different "
                        "ratios is the SAME abundance profile sequenced deeper"),
        },
        "records": records,
        "total_wall_seconds": time.time() - t0,
    }
    p = os.path.join(REPO, "analysis", "results", "f3_depth_sweep_raw.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", p, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
