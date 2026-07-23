#!/usr/bin/env python3
"""
Experiment E3 -- Coverage-threshold auto-detection accuracy.

Tests fgr2's per-sample coverage-cutoff inference (first local minimum of the
k-mer coverage histogram, implemented in fgr2.c:find_first_increasing_coverage)
against (a) the ground-truth optimal threshold, (b) fixed thresholds c in
{1,2,3,5}, and (c) a smoothed-histogram valley baseline.

Ground truth: reads are simulated from a known genome slice, so every distinct
canonical k-mer in the read set is labelled TRUE (present in the source slice)
or ERROR (absent).  The optimal threshold is the c maximising F1 of the rule
"coverage >= c  =>  TRUE".

All randomness is seeded.  Raw per-condition histograms are persisted so a
third party can recompute every headline number.
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

REPO = os.path.dirname(os.path.dirname(SCRIPTS))
FGR2 = os.path.join(REPO, "fgr2")
GENOME_DIR = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
              "-claude-worktrees-session-64bad5/"
              "2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad/genomes")
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/"
           "2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad")

K = 31
READ_LEN = 150
SLICE_LEN = 1_000_000
KC_MAX = (1 << 14) - 1          # fgr2's count saturation
FIXED_C = [1, 2, 3, 5]

GENOMES = [
    ("Bacillus_subtilis_168", "Bacillus_subtilis_168.fna"),
    ("Ecoli_K12_MG1655", "Ecoli_K12_MG1655.fna"),
    ("Klebsiella_pneumoniae", "Klebsiella_pneumoniae.fna"),
    ("Pseudomonas_aeruginosa_PAO1", "Pseudomonas_aeruginosa_PAO1.fna"),
]

PROFILES = [
    # name, err_rate, indel_frac, class
    ("illumina_0.1pct", 0.001, 0.0, "illumina"),
    ("illumina_0.5pct", 0.005, 0.0, "illumina"),
    ("illumina_1pct",   0.010, 0.0, "illumina"),
    ("longread_5pct",   0.050, 0.5, "longread"),
    ("longread_10pct",  0.100, 0.5, "longread"),
]

DEPTHS = [1, 2, 5, 10, 20, 50, 100, 200]
SEEDS = [0, 1, 2]


# ----------------------------------------------------------------------
# k-mer machinery
# ----------------------------------------------------------------------
def raw_canonical_codes(seq: str, k: int) -> np.ndarray:
    """Byte-for-byte identical to fgrlib.fast_canonical_codes EXCEPT that the
    final np.unique() is omitted, so multiplicities are preserved.  The total
    histogram derived from this is cross-checked against fgr2's own .hist."""
    lut = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    codes = lut[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    n = codes.size
    m = n - k + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    valid_base = codes != 255
    cs = np.concatenate(([0], np.cumsum(valid_base, dtype=np.int64)))
    valid_win = (cs[k:] - cs[:-k]) == k
    safe = np.where(valid_base, codes, 0).astype(np.uint64)
    comp = (np.uint64(3) - safe)
    fwd = np.zeros(m, dtype=np.uint64)
    rev = np.zeros(m, dtype=np.uint64)
    for j in range(k):
        fwd = (fwd << np.uint64(2)) | safe[j:j + m]
        rev |= comp[j:j + m] << np.uint64(2 * j)
    canon = np.minimum(fwd, rev)
    return canon[valid_win]


def read_kmer_counts(reads, k):
    """Distinct canonical k-mer codes of a read set with their multiplicities."""
    chunks = []
    buf = []
    buflen = 0
    for r in reads:
        buf.append(r)
        buflen += len(r) + 1
        if buflen > 8_000_000:
            chunks.append(raw_canonical_codes("N".join(buf), k))
            buf, buflen = [], 0
    if buf:
        chunks.append(raw_canonical_codes("N".join(buf), k))
    if not chunks:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
    allc = np.concatenate(chunks)
    del chunks
    allc.sort()
    if allc.size == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
    first = np.empty(allc.size, dtype=bool)
    first[0] = True
    np.not_equal(allc[1:], allc[:-1], out=first[1:])
    idx = np.flatnonzero(first)
    uniq = allc[idx]
    counts = np.diff(np.append(idx, allc.size))
    return uniq, counts


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def metrics_from_hists(hist_true, hist_err, cmax):
    """F1 and Youden's J of the rule 'coverage >= c => TRUE', for c=1..cmax."""
    n_true = int(hist_true.sum())
    n_err = int(hist_err.sum())
    # suffix sums: tp[c] = # TRUE k-mers with coverage >= c
    tt = np.concatenate((np.cumsum(hist_true[::-1])[::-1], [0]))
    ee = np.concatenate((np.cumsum(hist_err[::-1])[::-1], [0]))
    f1, you = {}, {}
    for c in range(1, cmax + 1):
        tp = int(tt[c]) if c < tt.size else 0
        fp = int(ee[c]) if c < ee.size else 0
        fn = n_true - tp
        denom = 2 * tp + fp + fn
        f1[c] = (2.0 * tp / denom) if denom > 0 else 0.0
        tpr = tp / n_true if n_true else 0.0
        fpr = fp / n_err if n_err else 0.0
        you[c] = tpr - fpr
    return f1, you, n_true, n_err


def smoothed_valley(hist, cmax):
    """Honest description: this is NOT a parametric mixture fit.  It is the
    classical two-mode spectrum heuristic used by genome-size estimators:

      1. Smooth the coverage histogram over c = 1..cmax with a 3-point moving
         average.
      2. Locate the SIGNAL peak as the RIGHTMOST strict local maximum of the
         smoothed curve at c >= 3 (the error mode always sits at c = 1-2, so
         the rightmost local max is the genomic mode when one exists).
      3. Return the argmin of the smoothed curve over c in [1, peak] -- the
         valley separating the error mode from the signal mode -- shifted by
         +1 so the returned value is the first coverage bin KEPT.

    Returns None when no local maximum at c >= 3 exists (i.e. the spectrum is
    monotonically decreasing and no signal mode is resolvable)."""
    h = np.zeros(cmax + 3, dtype=float)
    n = min(cmax + 1, hist.size)
    h[1:n] = hist[1:n]
    sm = h.copy()
    for c in range(2, cmax + 1):
        sm[c] = (h[c - 1] + h[c] + h[c + 1]) / 3.0
    if cmax < 4:
        return None
    peak = None
    for c in range(cmax - 1, 2, -1):
        if sm[c] > sm[c - 1] and sm[c] > sm[c + 1]:
            peak = c
            break
    if peak is None:
        return None
    valley = int(np.argmin(sm[1:peak + 1])) + 1
    return min(valley + 1, cmax)


# ----------------------------------------------------------------------
# one (genome, profile, seed) job: sweeps all depths
# ----------------------------------------------------------------------
def run_job(args):
    gname, gfile, pname, err, indel, pclass, seed = args
    genome = fgrlib.load_genome(os.path.join(GENOME_DIR, gfile))[:SLICE_LEN]
    true_codes = fgrlib.fast_canonical_codes(genome, K)   # sorted, unique
    gc = (genome.count("G") + genome.count("C")) / len(genome)

    workdir = tempfile.mkdtemp(prefix="e3_", dir=SCRATCH)
    out = []
    try:
        for depth in DEPTHS:
            t0 = time.time()
            seed_int = zlib.crc32(f"{gname}|{pname}|{seed}|{depth}".encode())
            rng = random.Random(seed_int)
            reads = fgrlib.simulate_reads(genome, depth, READ_LEN, err, rng,
                                          indel_frac=indel)
            fa = os.path.join(workdir, "reads.fa")
            fgrlib.write_reads_fasta(fa, reads)

            sk = os.path.join(workdir, "s.fgr2")
            proc = subprocess.run(
                [FGR2, "-k", str(K), "-N", "10000", "-t", "2", "-o", sk, fa],
                capture_output=True, text=True)
            detected = None
            fallback = False
            for line in proc.stderr.splitlines():
                if "Auto-detected coverage threshold:" in line:
                    detected = int(line.rsplit(":", 1)[1])
                if "No coverage threshold detected automatically" in line:
                    fallback = True
                    detected = int(line.rsplit(":", 1)[1])
            hdr_thr = None
            n_sel = None
            if os.path.exists(sk):
                with open(sk) as fh:
                    h = fh.readline()
                for tok in h.strip().split("\t"):
                    if tok.startswith("coverage_threshold="):
                        hdr_thr = int(tok.split("=")[1])
                    if tok.startswith("n_kmers="):
                        n_sel = int(tok.split("=")[1])

            # fgr2's own histogram (for cross-validation)
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

            # ground truth
            uniq, counts = read_kmer_counts(reads, K)
            del reads
            counts = np.minimum(counts, KC_MAX)
            pos = np.searchsorted(true_codes, uniq)
            pos = np.clip(pos, 0, max(true_codes.size - 1, 0))
            is_true = (true_codes.size > 0) & (true_codes[pos] == uniq)
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
            f1, you, n_true, n_err = metrics_from_hists(hist_true, hist_err, cmax)
            opt_f1_c = max(f1, key=lambda c: (f1[c], -c))
            opt_you_c = max(you, key=lambda c: (you[c], -c))
            base_c = smoothed_valley(total_hist, cmax)

            keep = min(total_hist.size, 601)
            rec = dict(
                genome=gname, genome_gc=round(gc, 4), slice_len=len(genome),
                profile=pname, err_rate=err, indel_frac=indel,
                profile_class=pclass, seed=seed, depth=depth, k=K,
                detected=detected, detected_header=hdr_thr,
                detection_fallback=fallback, n_sketch_kmers=n_sel,
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
                n_genome_kmers=int(true_codes.size),
                hist_true=hist_true[:keep].tolist(),
                hist_error=hist_err[:keep].tolist(),
                hist_true_tail=int(hist_true[keep:].sum()),
                hist_error_tail=int(hist_err[keep:].sum()),
                fgr2_hist_matches_ground_truth=hist_match,
                cmax_searched=cmax,
                wall_s=round(time.time() - t0, 2),
            )
            out.append(rec)
            for p in (fa, sk, sk + ".hist"):
                if os.path.exists(p):
                    os.remove(p)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(REPO, "analysis", "results",
                                                  "e3_coverage_autodetect.json"))
    a = ap.parse_args()

    jobs = [(gn, gf, pn, er, inf, pc, sd)
            for gn, gf in GENOMES
            for pn, er, inf, pc in PROFILES
            for sd in SEEDS]
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
            print(f"[{done}/{len(jobs)}] {futs[f][0]} {futs[f][2]} seed"
                  f"{futs[f][6]}  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    meta = dict(
        experiment="E3_coverage_threshold_autodetection",
        k=K, read_len=READ_LEN, slice_len=SLICE_LEN,
        genomes=[g[0] for g in GENOMES],
        profiles=[dict(name=p[0], err_rate=p[1], indel_frac=p[2]) for p in PROFILES],
        depths=DEPTHS, seeds=SEEDS, fixed_c=FIXED_C,
        criterion="F1 of rule 'coverage >= c => TRUE k-mer' (primary); "
                  "Youden J reported alongside",
        rng="python random.Random seeded by zlib.crc32(f'{genome}|{profile}|{seed}|{depth}')",
        fgr2_binary=FGR2,
        detector="fgr2.c find_first_increasing_coverage (first count increase "
                 "after a decreasing streak)",
        baseline="3-point moving-average histogram valley (see smoothed_valley docstring)",
        total_wall_s=round(time.time() - t0, 1),
    )
    with open(a.out, "w") as fh:
        json.dump(dict(meta=meta, records=records), fh)
    print("wrote", a.out, len(records), "records")


if __name__ == "__main__":
    main()
