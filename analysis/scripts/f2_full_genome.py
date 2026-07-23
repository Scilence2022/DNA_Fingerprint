#!/usr/bin/env python3
"""F2 -- E3's coverage-threshold auto-detection experiment re-run on FULL genomes.

Identical methodology to e3_coverage_autodetect.py (same F1 criterion on the rule
"coverage >= c => TRUE k-mer", same optimal-threshold definition, same fixed-c
baselines, same profiles/depths/seeds, same fgr2 invocation) with two changes:

  1. arm="full"    : genomes are NOT truncated (4.22-6.26 Mbp instead of 1 Mbp).
  2. arm="slice1mb": the SAME 1 Mbp truncation as E3, re-run here as a control so
     that any difference between F2-full and E3 can be attributed to genome
     length rather than to the simulator change below.

Simulator: E3 used fgrlib.simulate_reads (pure Python, ~1 Mbp/s), which cannot
produce the ~121 Gbp this design needs.  f2_fastsim implements the IDENTICAL
generative model vectorised with numpy; only the RNG stream differs.  The
slice1mb arm is the empirical check that this substitution is inconsequential.

Memory is bounded by chunked read generation; concurrency is throttled per depth
because fgr2's hash table grows with the number of distinct k-mers.
"""
import argparse
import json
import os
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
import fgrlib                       # noqa: E402
import f2_fastsim as FS             # noqa: E402

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
FGR2 = os.path.join(REPO, "fgr2")
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/"
           "2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad")
GENOME_DIR = os.path.join(SCRATCH, "genomes")

K = 31
READ_LEN = 150
KC_MAX = (1 << 14) - 1
FIXED_C = [1, 2, 3, 5]
SLICE_LEN = 1_000_000

GENOMES = [
    ("Bacillus_subtilis_168", "Bacillus_subtilis_168.fna"),
    ("Ecoli_K12_MG1655", "Ecoli_K12_MG1655.fna"),
    ("Klebsiella_pneumoniae", "Klebsiella_pneumoniae.fna"),
    ("Pseudomonas_aeruginosa_PAO1", "Pseudomonas_aeruginosa_PAO1.fna"),
]
PROFILES = [
    ("illumina_0.1pct", 0.001, 0.0, "illumina"),
    ("illumina_0.5pct", 0.005, 0.0, "illumina"),
    ("illumina_1pct",   0.010, 0.0, "illumina"),
    ("longread_5pct",   0.050, 0.5, "longread"),
    ("longread_10pct",  0.100, 0.5, "longread"),
]
DEPTHS = [1, 2, 5, 10, 20, 50, 100, 200]
SEEDS = [0, 1, 2]

# fgr2 RSS measured at 3.5 GB for 232 Mbp of 10%-error reads; workers are capped
# per depth so peak resident stays well under the 128 GB available.
WORKERS_BY_DEPTH = {1: 12, 2: 12, 5: 12, 10: 12, 20: 10, 50: 7, 100: 4, 200: 3}


# ---------------------------------------------------------------- metrics
# (verbatim from e3_coverage_autodetect.py / e3_baselines.py)
def metrics_from_hists(hist_true, hist_err, cmax):
    n_true = int(hist_true.sum())
    n_err = int(hist_err.sum())
    tt = np.concatenate((np.cumsum(hist_true[::-1])[::-1], [0]))
    ee = np.concatenate((np.cumsum(hist_err[::-1])[::-1], [0]))
    f1, you = {}, {}
    for c in range(1, cmax + 1):
        tp = int(tt[c]) if c < tt.size else 0
        fp = int(ee[c]) if c < ee.size else 0
        fn = n_true - tp
        denom = 2 * tp + fp + fn
        f1[c] = (2.0 * tp / denom) if denom > 0 else 0.0
        you[c] = (tp / n_true if n_true else 0.0) - (fp / n_err if n_err else 0.0)
    return f1, you, n_true, n_err


def smooth(hist, cmax):
    h = np.zeros(cmax + 3, dtype=float)
    n = min(cmax + 1, hist.size)
    h[1:n] = hist[1:n]
    sm = h.copy()
    for c in range(2, cmax + 1):
        sm[c] = (h[c - 1] + h[c] + h[c + 1]) / 3.0
    return sm


def valley_globalmax(hist, cmax):
    if cmax < 4:
        return None
    sm = smooth(hist, cmax)
    peak = int(np.argmax(sm[3:cmax + 1])) + 3
    if peak <= 1:
        return None
    valley = int(np.argmin(sm[1:peak + 1])) + 1
    return int(min(valley + 1, cmax))


def fgr2_rule(counts, hist_size):
    """Faithful re-implementation of fgr2.c find_first_increasing_coverage."""
    start = 1
    while start < hist_size and counts[start] == 0:
        start += 1
    if start >= hist_size:
        return 0
    prev = counts[start]
    dec = 0
    for i in range(start + 1, hist_size):
        if counts[i] == 0:
            continue
        if counts[i] < prev:
            dec += 1
            prev = counts[i]
            continue
        if dec > 0 and counts[i] > prev:
            return i
        prev = counts[i]
    return 0


def fgr2_rule_smoothed(hist, cmax):
    r = fgr2_rule(smooth(hist, cmax), cmax + 1)
    return int(r) if r > 0 else None


# ---------------------------------------------------------------- one condition
def run_condition(args):
    (gname, gfile, pname, err, indel, pclass, seed, depth, arm) = args
    t0 = time.time()
    genome = fgrlib.load_genome(os.path.join(GENOME_DIR, gfile))
    if arm == "slice1mb":
        genome = genome[:SLICE_LEN]
    true_codes = fgrlib.fast_canonical_codes(genome, K)      # sorted unique
    gc_frac = (genome.count("G") + genome.count("C")) / len(genome)
    glen = len(genome)
    gcodes = FS.genome_codes(genome)
    del genome

    seed_int = zlib.crc32(f"{arm}|{gname}|{pname}|{seed}|{depth}".encode())
    rng = np.random.default_rng(seed_int)

    workdir = tempfile.mkdtemp(prefix="f2_", dir=SCRATCH)
    fa = os.path.join(workdir, "reads.fa")
    sk = os.path.join(workdir, "s.fgr2")
    try:
        # --- simulate + write FASTA + accumulate ground-truth counts ----------
        # Per chunk the k-mer codes are sorted and collapsed to (code, count)
        # BEFORE the membership test, so the expensive binary search runs on
        # sorted, deduplicated queries.  Error k-mers are filed into NBUCK
        # prefix buckets (contiguous slices of the sorted chunk, so bucketing
        # is free) and merged bucket-by-bucket afterwards, which bounds peak
        # memory at 200x.  The resulting total histogram is checked against
        # fgr2's own .hist for every condition.
        NBUCK = 64
        SHIFT = np.uint64(2 * K - 6)          # top 6 bits of the 2K-bit code
        true_counts = np.zeros(true_codes.size, dtype=np.int64)
        bkt_codes = [[] for _ in range(NBUCK)]
        bkt_cnts = [[] for _ in range(NBUCK)]
        n_bases = n_reads = 0
        with open(fa, "wb") as fh:
            for (seq, lens), off in FS.simulate_stream(
                    gcodes, depth, READ_LEN, err, indel, rng):
                fh.write(FS._fasta_bytes(seq, lens, off).tobytes())
                n_bases += int(seq.size)
                n_reads += int(lens.size)
                codes = FS.canonical_codes_from_bytes(FS._sep_join(seq, lens), K)
                if codes.size == 0:
                    continue
                codes.sort()
                first = np.empty(codes.size, dtype=bool)
                first[0] = True
                np.not_equal(codes[1:], codes[:-1], out=first[1:])
                idx = np.flatnonzero(first)
                u = codes[idx]
                cnt = np.diff(np.append(idx, codes.size))
                del codes, first, idx
                if true_codes.size:
                    pos = np.searchsorted(true_codes, u)
                    np.clip(pos, 0, true_codes.size - 1, out=pos)
                    hit = true_codes[pos] == u
                    if hit.any():
                        true_counts += np.bincount(
                            pos[hit], weights=cnt[hit],
                            minlength=true_codes.size).astype(np.int64)
                    bad_u, bad_c = u[~hit], cnt[~hit]
                    del pos, hit
                else:
                    bad_u, bad_c = u, cnt
                del u, cnt
                if bad_u.size:
                    edges = np.searchsorted(
                        bad_u, (np.arange(NBUCK + 1, dtype=np.uint64) << SHIFT))
                    for b in range(NBUCK):
                        lo, hi = int(edges[b]), int(edges[b + 1])
                        if hi > lo:
                            bkt_codes[b].append(bad_u[lo:hi].copy())
                            bkt_cnts[b].append(bad_c[lo:hi].astype(np.int32))
                del bad_u, bad_c
        del gcodes

        # --- error-k-mer multiplicities, bucket by bucket ---------------------
        err_hist = np.zeros(KC_MAX + 2, dtype=np.int64)
        for b in range(NBUCK):
            if not bkt_codes[b]:
                continue
            cs = np.concatenate(bkt_codes[b])
            ct = np.concatenate(bkt_cnts[b]).astype(np.int64)
            bkt_codes[b] = bkt_cnts[b] = None
            order = np.argsort(cs, kind="stable")
            cs = cs[order]
            ct = ct[order]
            del order
            first = np.empty(cs.size, dtype=bool)
            first[0] = True
            np.not_equal(cs[1:], cs[:-1], out=first[1:])
            starts = np.flatnonzero(first)
            mult = np.add.reduceat(ct, starts)
            np.minimum(mult, KC_MAX, out=mult)
            err_hist += np.bincount(mult, minlength=KC_MAX + 2)
            del cs, ct, first, starts, mult
        del bkt_codes, bkt_cnts

        tc = true_counts[true_counts > 0]
        del true_counts
        tc = np.minimum(tc, KC_MAX)
        nz = np.flatnonzero(err_hist)
        nbin = int(max(tc.max() if tc.size else 1,
                       nz.max() if nz.size else 1)) + 1
        hist_true = np.bincount(tc, minlength=nbin)[:nbin]
        hist_err = err_hist[:nbin].copy()
        hist_true[0] = 0
        hist_err[0] = 0
        total_hist = hist_true + hist_err
        del tc, err_hist
        t_sim = time.time() - t0

        # --- fgr2 -------------------------------------------------------------
        t1 = time.time()
        proc = subprocess.run(
            [FGR2, "-k", str(K), "-N", "10000", "-t", "2", "-o", sk, fa],
            capture_output=True, text=True)
        t_fgr2 = time.time() - t1
        detected, fallback = None, False
        for line in proc.stderr.splitlines():
            if "Auto-detected coverage threshold:" in line:
                detected = int(line.rsplit(":", 1)[1])
            if "No coverage threshold detected automatically" in line:
                fallback = True
                detected = int(line.rsplit(":", 1)[1])
        hdr_thr = n_sel = None
        if os.path.exists(sk):
            with open(sk) as fh:
                h = fh.readline()
            for tok in h.strip().split("\t"):
                if tok.startswith("coverage_threshold="):
                    hdr_thr = int(tok.split("=")[1])
                if tok.startswith("n_kmers="):
                    n_sel = int(tok.split("=")[1])

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

        hist_match = None
        if fgr2_hist is not None:
            L = max(fgr2_hist.size, total_hist.size)
            a = np.zeros(L, dtype=np.int64); a[:fgr2_hist.size] = fgr2_hist
            b = np.zeros(L, dtype=np.int64); b[:total_hist.size] = total_hist
            a[0] = b[0] = 0
            hist_match = bool(np.array_equal(a, b))

        # --- scoring ----------------------------------------------------------
        cmax = min(int(total_hist.size) - 1, 500)
        cmax = max(cmax, 5)
        f1, you, n_true, n_err = metrics_from_hists(hist_true, hist_err, cmax)
        opt_f1_c = max(f1, key=lambda c: (f1[c], -c))
        opt_you_c = max(you, key=lambda c: (you[c], -c))
        vg = valley_globalmax(total_hist, cmax)
        srule = fgr2_rule_smoothed(total_hist, cmax)
        raw_rule = int(fgr2_rule(total_hist.astype(float), total_hist.size))

        keep = min(total_hist.size, 601)
        rec = dict(
            arm=arm, genome=gname, genome_gc=round(gc_frac, 4),
            genome_len=glen,
            profile=pname, err_rate=err, indel_frac=indel, profile_class=pclass,
            seed=seed, depth=depth, k=K, read_len=READ_LEN,
            n_reads=n_reads, n_read_bases=n_bases,
            detected=detected, detected_header=hdr_thr,
            detection_fallback=fallback, n_sketch_kmers=n_sel,
            optimal_f1_c=opt_f1_c, optimal_f1=f1[opt_f1_c],
            optimal_youden_c=opt_you_c, optimal_youden=you[opt_you_c],
            f1_at_detected=f1.get(detected), youden_at_detected=you.get(detected),
            f1_fixed={str(c): f1.get(c) for c in FIXED_C},
            youden_fixed={str(c): you.get(c) for c in FIXED_C},
            valley_globalmax_c=vg,
            f1_at_valley_globalmax=f1.get(vg) if vg else None,
            fgr2_rule_smoothed_c=srule,
            f1_at_fgr2_rule_smoothed=f1.get(srule) if srule else None,
            reimpl_raw_rule_c=raw_rule,
            reimpl_matches_fgr2=bool((raw_rule if raw_rule > 0 else 1) == detected),
            n_true_kmers=n_true, n_error_kmers=n_err,
            n_genome_kmers=int(true_codes.size),
            hist_true=hist_true[:keep].tolist(),
            hist_error=hist_err[:keep].tolist(),
            hist_true_tail=int(hist_true[keep:].sum()),
            hist_error_tail=int(hist_err[keep:].sum()),
            fgr2_hist_matches_ground_truth=hist_match,
            cmax_searched=cmax,
            wall_sim_s=round(t_sim, 1), wall_fgr2_s=round(t_fgr2, 2),
            wall_s=round(time.time() - t0, 1),
        )
        return rec
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="full,slice1mb")
    ap.add_argument("--out", default=os.path.join(
        REPO, "analysis", "results", "f2_full_genome.json"))
    ap.add_argument("--jsonl", default=os.path.join(SCRATCH, "f2_records.jsonl"))
    a = ap.parse_args()
    arms = a.arms.split(",")

    done_keys = set()
    if os.path.exists(a.jsonl):
        with open(a.jsonl) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done_keys.add((r["arm"], r["genome"], r["profile"],
                               r["seed"], r["depth"]))
    print(f"resuming with {len(done_keys)} conditions already done", flush=True)

    t0 = time.time()
    jl = open(a.jsonl, "a")
    for depth in DEPTHS:
        jobs = [(gn, gf, pn, er, inf, pc, sd, depth, arm)
                for arm in arms
                for gn, gf in GENOMES
                for pn, er, inf, pc in PROFILES
                for sd in SEEDS
                if (arm, gn, pn, sd, depth) not in done_keys]
        if not jobs:
            continue
        w = WORKERS_BY_DEPTH[depth]
        print(f"=== depth {depth}x : {len(jobs)} conditions, {w} workers "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(run_condition, j): j for j in jobs}
            n = 0
            for f in as_completed(futs):
                r = f.result()
                jl.write(json.dumps(r) + "\n")
                jl.flush()
                n += 1
                if n % 10 == 0 or n == len(jobs):
                    print(f"  [{n}/{len(jobs)}] depth {depth}x "
                          f"({time.time()-t0:.0f}s)", flush=True)
    jl.close()

    records = []
    with open(a.jsonl) as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    meta = dict(
        experiment="F2_coverage_threshold_autodetection_FULL_GENOMES",
        replicates="E3 methodology, genomes untruncated (arm=full) plus a "
                   "1 Mbp-truncation control (arm=slice1mb)",
        k=K, read_len=READ_LEN, slice_len_control=SLICE_LEN,
        genomes=[g[0] for g in GENOMES],
        profiles=[dict(name=p[0], err_rate=p[1], indel_frac=p[2],
                       cls=p[3]) for p in PROFILES],
        depths=DEPTHS, seeds=SEEDS, fixed_c=FIXED_C, arms=arms,
        criterion="F1 of rule 'coverage >= c => TRUE k-mer' (primary); "
                  "Youden J reported alongside",
        rng="numpy default_rng seeded by "
            "zlib.crc32(f'{arm}|{genome}|{profile}|{seed}|{depth}')",
        simulator="f2_fastsim (vectorised; same generative model as "
                  "fgrlib.simulate_reads, different RNG stream)",
        fgr2_binary=FGR2,
        detector="fgr2.c find_first_increasing_coverage",
        total_wall_s=round(time.time() - t0, 1),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(meta=meta, records=records), fh)
    print("wrote", a.out, len(records), "records")


if __name__ == "__main__":
    main()
