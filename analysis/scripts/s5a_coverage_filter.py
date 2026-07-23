#!/usr/bin/env python3
"""
S5(a) -- Coverage-based error filtering on SEQUENCED ENCODED POOLS.

A DNA-data-storage pool is synthesised as many random-payload oligos flanked by
shared primer sites (fgrlib.make_oligo_pool). Sequencing it produces the true
oligo k-mers PLUS synthesis/sequencing-error k-mers. A coverage cutoff is the
standard way to strip the error k-mers before fingerprinting.

Ground truth: a canonical k-mer is TRUE iff it occurs in the NOISE-FREE pool
(the oligos themselves); every other observed k-mer is an ERROR k-mer. The
optimal cutoff is the c maximising F1 of the rule "coverage >= c => TRUE".

We compare, per condition:
  * fgr2's auto-detected cutoff (first-increasing-coverage valley, fgr2.c)
  * fixed cutoffs c in {1,2,3,5}
  * a smoothed-histogram valley baseline
  * the ground-truth optimal c (F1 and Youden's J)

Copy numbers are drawn under log-normal PCR amplification bias (skew=0.5) around
each target mean depth, so the coverage histogram has a realistic spread (true
k-mers from low-copy oligos genuinely blur into the error mode) rather than a
delta. This is the honest, harder test of any auto-detector.

All randomness seeded and recorded. Raw per-condition true/error coverage
histograms are persisted so every headline number can be recomputed.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import fgrlib  # noqa: E402
# Reuse VALIDATED k-mer-counting and metric machinery from E3 (import is safe:
# its heavy work is guarded by __main__).
import e3_coverage_autodetect as e3  # noqa: E402

REPO = os.path.dirname(os.path.dirname(SCRIPTS))
FGR2 = os.path.join(REPO, "fgr2")
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/"
           "2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad")

K = 31
OLIGO_LEN = 150
N_OLIGOS = 1000
PCR_SKEW = 0.5                 # log-normal amplification-bias sigma
KC_MAX = (1 << 14) - 1         # fgr2 count saturation
FIXED_C = [1, 2, 3, 5]
DEPTHS = [2, 5, 10, 20, 50, 100, 200]
ERR_RATES = [0.001, 0.005, 0.010]
SEEDS = [0, 1, 2, 3, 4]


def build_pool(seed):
    """A pool is fixed by its seed so the ground truth is shared across all
    depths / error rates evaluated on it."""
    rng = random.Random(zlib.crc32(f"pool|{seed}".encode()))
    pool = fgrlib.make_oligo_pool(N_OLIGOS, OLIGO_LEN, rng)
    true_codes = fgrlib.fast_canonical_codes("N".join(pool), K)  # sorted unique
    return pool, true_codes


def run_job(args):
    err, seed = args
    pool, true_codes = build_pool(seed)
    workdir = tempfile.mkdtemp(prefix="s5a_", dir=SCRATCH)
    out = []
    try:
        for depth in DEPTHS:
            t0 = time.time()
            rng = random.Random(zlib.crc32(
                f"reads|{seed}|{err}|{depth}".encode()))
            copies = fgrlib.pcr_bias_copies(len(pool), depth, PCR_SKEW, rng)
            reads = fgrlib.pool_to_reads(pool, copies, err, rng)
            realized_mean_copies = float(np.mean(copies))
            n_oligos_present = int(sum(1 for c in copies if c > 0))

            fa = os.path.join(workdir, "reads.fa")
            fgrlib.write_reads_fasta(fa, reads)

            sk = os.path.join(workdir, "s.fgr2")
            proc = subprocess.run(
                [FGR2, "-k", str(K), "-N", "20000", "-t", "2", "-o", sk, fa],
                capture_output=True, text=True)
            detected = None
            fallback = False
            for line in proc.stderr.splitlines():
                if "Auto-detected coverage threshold:" in line:
                    detected = int(line.rsplit(":", 1)[1])
                if "No coverage threshold detected automatically" in line:
                    fallback = True
            hdr_thr = None
            if os.path.exists(sk):
                with open(sk) as fh:
                    hline = fh.readline()
                for tok in hline.strip().split("\t"):
                    if tok.startswith("coverage_threshold="):
                        hdr_thr = int(tok.split("=")[1])
            if detected is None:
                detected = hdr_thr

            # fgr2's own histogram, for cross-validation against our ground truth
            fgr2_hist = None
            hp = sk + ".hist"
            if os.path.exists(hp):
                fh_ = {}
                with open(hp) as fh:
                    for line in fh:
                        if line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            fh_[int(parts[0])] = int(parts[1])
                if fh_:
                    fgr2_hist = np.zeros(max(fh_) + 1, dtype=np.int64)
                    for kk, vv in fh_.items():
                        fgr2_hist[kk] = vv

            # ground-truth true/error coverage histograms
            uniq, counts = e3.read_kmer_counts(reads, K)
            del reads
            counts = np.minimum(counts, KC_MAX)
            if true_codes.size and uniq.size:
                pos = np.searchsorted(true_codes, uniq)
                pos = np.clip(pos, 0, true_codes.size - 1)
                is_true = true_codes[pos] == uniq
            else:
                is_true = np.zeros(uniq.size, dtype=bool)
            nbin = int(counts.max()) + 1 if counts.size else 1
            hist_true = np.bincount(counts[is_true], minlength=nbin)
            hist_err = np.bincount(counts[~is_true], minlength=nbin)
            total_hist = hist_true + hist_err

            hist_match = None
            if fgr2_hist is not None:
                L = max(fgr2_hist.size, total_hist.size)
                a = np.zeros(L, dtype=np.int64); a[:fgr2_hist.size] = fgr2_hist
                b = np.zeros(L, dtype=np.int64); b[:total_hist.size] = total_hist
                a[0] = b[0] = 0     # fgr2 .hist starts at coverage 1
                hist_match = bool(np.array_equal(a, b))

            cmax = min(int(counts.max()) if counts.size else 1, 500)
            cmax = max(cmax, 5)
            f1, you, n_true, n_err = e3.metrics_from_hists(
                hist_true, hist_err, cmax)
            opt_f1_c = max(f1, key=lambda c: (f1[c], -c))
            opt_you_c = max(you, key=lambda c: (you[c], -c))
            base_c = e3.smoothed_valley(total_hist, cmax)

            keep = min(total_hist.size, 601)
            rec = dict(
                input_type="encoded_oligo_pool",
                n_oligos=N_OLIGOS, oligo_len=OLIGO_LEN, pcr_skew=PCR_SKEW,
                err_rate=err, seed=seed, target_depth=depth,
                realized_mean_copies=round(realized_mean_copies, 3),
                n_oligos_present=n_oligos_present, k=K,
                detected=detected, detection_fallback=fallback,
                optimal_f1_c=opt_f1_c, optimal_f1=f1[opt_f1_c],
                optimal_youden_c=opt_you_c, optimal_youden=you[opt_you_c],
                baseline_valley_c=base_c,
                f1_at_detected=f1.get(detected),
                youden_at_detected=you.get(detected),
                f1_at_baseline=f1.get(base_c) if base_c else None,
                youden_at_baseline=you.get(base_c) if base_c else None,
                f1_fixed={str(c): f1.get(c) for c in FIXED_C},
                youden_fixed={str(c): you.get(c) for c in FIXED_C},
                n_true_kmers=n_true, n_error_kmers=n_err,
                n_pool_kmers=int(true_codes.size),
                hist_true=hist_true[:keep].tolist(),
                hist_error=hist_err[:keep].tolist(),
                hist_true_tail=int(hist_true[keep:].sum()),
                hist_error_tail=int(hist_err[keep:].sum()),
                fgr2_hist_matches_ground_truth=hist_match,
                cmax_searched=cmax,
                wall_s=round(time.time() - t0, 2),
            )
            out.append(rec)
            for p in (fa, sk, hp):
                if os.path.exists(p):
                    os.remove(p)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(
        REPO, "analysis", "results", "s5a_coverage_filter.json"))
    a = ap.parse_args()

    jobs = [(er, sd) for er in ERR_RATES for sd in SEEDS]
    print(f"{len(jobs)} jobs x {len(DEPTHS)} depths = "
          f"{len(jobs)*len(DEPTHS)} conditions", flush=True)

    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            records.extend(f.result())
            done += 1
            print(f"[{done}/{len(jobs)}] err={futs[f][0]} seed={futs[f][1]} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    meta = dict(
        experiment="S5a_coverage_error_filtering_on_encoded_pools",
        k=K, oligo_len=OLIGO_LEN, n_oligos=N_OLIGOS, pcr_skew=PCR_SKEW,
        depths=DEPTHS, err_rates=ERR_RATES, seeds=SEEDS, fixed_c=FIXED_C,
        criterion="F1 of rule 'coverage >= c => TRUE k-mer' (primary); "
                  "Youden J reported alongside",
        ground_truth="k-mer TRUE iff present in the noise-free pool",
        copy_model="log-normal PCR bias (fgrlib.pcr_bias_copies, skew=0.5) "
                   "around each target mean depth",
        rng="python random.Random seeded by zlib.crc32 of a per-condition tag",
        fgr2_binary=FGR2,
        detector="fgr2 find_first_increasing_coverage (auto threshold)",
        baseline="3-point moving-average histogram valley (e3.smoothed_valley)",
        total_wall_s=round(time.time() - t0, 1),
    )
    with open(a.out, "w") as fh:
        json.dump(dict(meta=meta, records=records), fh)
    print("wrote", a.out, len(records), "records")


if __name__ == "__main__":
    main()
