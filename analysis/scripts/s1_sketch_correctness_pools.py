#!/usr/bin/env python3
"""
S1 -- The fgr2 fingerprint generator is EXACT on simulated encoded oligo pools.

Storage-native correctness check. Every input is a simulated ENCODED OLIGO POOL
(random payloads flanked by shared 20nt primer sites) or reads derived from one.
No genome is ever loaded.

Two input types per pool:
  (A) direct  -- the oligo pool written as a FASTA of oligos, no error.
  (B) reads   -- the pool expanded to reads at KNOWN per-oligo copy numbers with
                 a small per-base substitution error rate (pool_to_reads).

Grid: pool in {(5000,120),(20000,150),(50000,200)} x input in {direct,reads}
      x k in {21,25,31} x N in {1000,10000} x hash in {murmurhash3,wang}.

Per cell, fgr2 -c 1 (coverage filter disabled), checks against a ground truth
recomputed independently from fgrlib:

  C1 exact set equality  fgr2 k-mer set == true bottom-N set over the input
  C2 hash fidelity       every emitted hash == recomputed sketch hash of k-mer
  C3 sort order          output strictly ascending by hash
  C4 provenance          every emitted k-mer occurs in the input (fwd or rc)
  C5 determinism         same config run twice -> byte-identical file
  C6 coverage annotation fgr2 coverage == true canonical occurrence count in the
                         input, for every sketch k-mer (all unsaturated, < KC_MAX)
  C7 canonical strings   every emitted k-mer string is the canonical (min) form

Plus thread invariance (-t 1/4/16 byte-identical) on a subset.

The vectorised numpy hashes and the vectorised canonical-count enumerator are
VERIFIED against fgrlib's scalar reference implementations before use.
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

SEED = 20260722

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468"
           "/scratchpad")
WORK = os.path.join(SCRATCH, "s1")
FGR2 = os.path.join(REPO, "fgr2")
RESULTS = os.path.join(REPO, "analysis/results/s1_sketch_correctness_pools.json")

sys.path.insert(0, os.path.join(REPO, "analysis/scripts"))
import fgrlib  # noqa: E402

# pool (n_oligos, oligo_len)
POOLS = [(5000, 120), (20000, 150), (50000, 200)]
KS = [21, 25, 31]
NS = [1000, 10000]
HASHES = ["murmurhash3", "wang"]
INPUTS = ["direct", "reads"]

# reads: known per-oligo copy numbers drawn uniformly in [COPY_LO, COPY_HI], and
# a small per-base substitution error rate.
COPY_LO, COPY_HI = 2, 6
READ_ERR = 0.01

KC_MAX = (1 << 14) - 1  # fgr2 coverage counter saturation (fgr2.c:14)

# thread-invariance subset
THREAD_POOL = (20000, 150)
THREAD_COUNTS = [1, 4, 16]

u64 = np.uint64


# ---------------------------------------------------------------- vectorised hashes
# (ports of fgrlib.murmur3_x64_64 / wang_hash; validated against the scalar
#  versions in validate_vectorised_hashes before use)

def _rotl(x, r):
    return (x << u64(r)) | (x >> u64(64 - r))


def _fmix64(k):
    k = k ^ (k >> u64(33))
    k = k * u64(0xFF51AFD7ED558CCD)
    k = k ^ (k >> u64(33))
    k = k * u64(0xC4CEB9FE1A85EC53)
    k = k ^ (k >> u64(33))
    return k


def vec_murmur3(codes):
    codes = codes.astype(np.uint64, copy=False)
    c1, c2 = u64(0x87C37B91114253D5), u64(0x4CF5AD432745937F)
    h1 = np.full(codes.shape, u64(42), dtype=np.uint64)
    h2 = np.full(codes.shape, u64(42), dtype=np.uint64)
    k1 = codes * c1
    k1 = _rotl(k1, 31)
    k1 = k1 * c2
    h1 = h1 ^ k1
    h1 = _rotl(h1, 27)
    h1 = h1 + h2
    h1 = h1 * u64(5) + u64(0x52DCE729)
    h2 = _rotl(h2, 31)
    h2 = h2 + h1
    h2 = h2 * u64(5) + u64(0x38495AB5)
    h1 = h1 ^ u64(8)
    h2 = h2 ^ u64(8)
    h1 = h1 + h2
    h2 = h2 + h1
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    return h1 + h2


def vec_wang(codes):
    key = codes.astype(np.uint64, copy=False)
    key = (~key) + (key << u64(21))
    key = key ^ (key >> u64(24))
    key = key + (key << u64(3)) + (key << u64(8))
    key = key ^ (key >> u64(14))
    key = key + (key << u64(2)) + (key << u64(4))
    key = key ^ (key >> u64(28))
    key = key + (key << u64(31))
    return key


VEC = {"murmurhash3": vec_murmur3, "wang": vec_wang}
SCALAR = {"murmurhash3": fgrlib.murmur3_x64_64, "wang": fgrlib.wang_hash}


def validate_vectorised_hashes(rng, n_probe=5000):
    probes = rng.integers(0, 1 << 62, size=n_probe, dtype=np.uint64)
    probes = np.concatenate([np.array([0, 1, 2, (1 << 62) - 1], dtype=np.uint64),
                             probes])
    out = {}
    for name in HASHES:
        vec = VEC[name](probes)
        ref = np.array([SCALAR[name](int(p)) for p in probes], dtype=np.uint64)
        nm = int(np.count_nonzero(vec != ref))
        out[name] = {"n_probes": int(probes.size), "n_mismatch": nm,
                     "agrees": nm == 0}
    return out


# ------------------------------------------------ vectorised canonical code counts

def fast_canonical_code_counts(seq, k):
    """Canonical 2-bit codes and their occurrence COUNTS over `seq`, as a sorted
    unique uint64 array + parallel int64 count array. Windows with non-ACGT are
    excluded (window reset), matching fgr2. Verified against
    fgrlib.canonical_kmer_counts in validate_vectorised_counts."""
    lut = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    codes = lut[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    n = codes.size
    m = n - k + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
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
    canon = np.minimum(fwd, rev)[valid_win]
    uniq, cnt = np.unique(canon, return_counts=True)
    return uniq, cnt.astype(np.int64)


def validate_vectorised_counts(rng, n_trials=6):
    """Verify fast_canonical_code_counts against fgrlib.canonical_kmer_counts
    (the slow string reference) on small random sequences, including N-runs."""
    out = {"n_trials": 0, "n_mismatch_trials": 0, "details": []}
    for t in range(n_trials):
        L = int(rng.integers(400, 1200))
        seq = "".join(random.Random(int(rng.integers(0, 1 << 30))).choice("ACGT")
                      for _ in range(L))
        if t % 2 == 1:  # inject ambiguous runs
            p = L // 3
            seq = seq[:p] + "NNN" + seq[p:2 * p] + "N" + seq[2 * p:]
        k = int(rng.choice(KS))
        ref = fgrlib.canonical_kmer_counts(seq, k)  # canonical string -> count
        ref_codes = {fgrlib.encode2bit(km): c for km, c in ref.items()}
        uniq, cnt = fast_canonical_code_counts(seq, k)
        got = {int(u): int(c) for u, c in zip(uniq, cnt)}
        agree = got == ref_codes
        out["n_trials"] += 1
        out["n_mismatch_trials"] += int(not agree)
        out["details"].append({"len": L, "k": k, "n_kmers_ref": len(ref_codes),
                               "n_kmers_got": len(got), "agrees": agree})
    out["agrees"] = out["n_mismatch_trials"] == 0
    return out


# ---------------------------------------------------------------- helpers

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_fgr2(fasta, out, k, n, hashname, threads=None):
    cmd = [FGR2, "-k", str(k), "-N", str(n), "-c", "1"]
    if hashname == "wang":
        cmd.append("-w")
    if threads is not None:
        cmd += ["-t", str(threads)]
    cmd += ["-o", out, fasta]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"fgr2 failed: {' '.join(cmd)}\n{p.stderr}")
    return time.time() - t0


def read_sketch_ordered(path):
    """Return (kmers, hashes, coverages) in FILE ORDER (for the sort check)."""
    kmers, hashes, covs = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            kmers.append(parts[0])
            hashes.append(int(parts[1]))
            covs.append(int(parts[2]))
    return kmers, hashes, covs


def decode(code, k):
    return fgrlib.decode2bit(int(code), k)


def canonical_code_of(kmer):
    fwd = fgrlib.encode2bit(kmer)
    rev = fgrlib.encode2bit(fgrlib.revcomp(kmer))
    return min(fwd, rev)


def true_bottom_n(codes, hashname, n, k):
    """Ground-truth bottom-N over `codes`: (kmer strings, hashes ascending),
    plus a boundary-tie flag that would make the SET ambiguous."""
    hv = VEC[hashname](codes)
    if hv.size <= n:
        order = np.argsort(hv, kind="stable")
        boundary_tie = False
    else:
        part = np.argpartition(hv, n - 1)
        sel = part[:n]
        sel = sel[np.argsort(hv[sel], kind="stable")]
        nth = hv[sel[-1]]
        rest = part[n:]
        boundary_tie = bool(np.any(hv[rest] == nth))
        order = sel
    kmers = [decode(codes[i], k) for i in order]
    return kmers, hv[order], boundary_tie, int(hv.size)


# ---------------------------------------------------------------- main

def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    rng = np.random.default_rng(SEED)

    hash_validation = validate_vectorised_hashes(rng)
    for name, v in hash_validation.items():
        if not v["agrees"]:
            raise SystemExit(f"vectorised {name} disagrees with fgrlib -- aborting")
    print("vectorised hash validation:", hash_validation, flush=True)

    count_validation = validate_vectorised_counts(rng)
    if not count_validation["agrees"]:
        raise SystemExit("vectorised canonical-count enumerator disagrees with "
                         "fgrlib.canonical_kmer_counts -- aborting")
    print("vectorised count validation: agrees on",
          count_validation["n_trials"], "trials", flush=True)

    results = {
        "experiment_id": "S1",
        "description": "fgr2 sketch exactness on simulated encoded oligo pools "
                       "(no genome; storage-native).",
        "seed": SEED,
        "fgr2_binary": FGR2,
        "fgr2_sha256": sha256(FGR2),
        "pools": [{"n_oligos": n, "oligo_len": L} for n, L in POOLS],
        "grid": {"k": KS, "N": NS, "hash": HASHES, "input": INPUTS},
        "read_params": {"copy_lo": COPY_LO, "copy_hi": COPY_HI,
                        "err_rate": READ_ERR},
        "kc_max": KC_MAX,
        "hash_validation": hash_validation,
        "count_validation": count_validation,
        "cells": [],
        "thread_invariance": [],
        "pool_stats": {},
    }

    n_cells = 0
    n_cells_pass = 0

    for (n_oligos, oligo_len) in POOLS:
        pool_tag = f"n{n_oligos}_L{oligo_len}"
        # Deterministic RNG per pool so results are reproducible & independent.
        prng = random.Random(f"{SEED}:{pool_tag}")
        pool = fgrlib.make_oligo_pool(n_oligos, oligo_len, prng)

        # known per-oligo copy numbers + reads (small error)
        copies = [prng.randint(COPY_LO, COPY_HI) for _ in range(n_oligos)]
        reads = fgrlib.pool_to_reads(pool, copies, READ_ERR, prng)

        # write both input FASTAs
        direct_fa = os.path.join(WORK, pool_tag + ".oligos.fa")
        reads_fa = os.path.join(WORK, pool_tag + ".reads.fa")
        fgrlib.write_fasta(direct_fa, [(f"oligo{i}", s) for i, s in enumerate(pool)])
        fgrlib.write_reads_fasta(reads_fa, reads)

        input_seq = {
            "direct": "N".join(pool),   # no k-mer spans a record boundary
            "reads": "N".join(reads),
        }
        input_fa = {"direct": direct_fa, "reads": reads_fa}

        results["pool_stats"][pool_tag] = {
            "n_oligos": n_oligos, "oligo_len": oligo_len,
            "n_reads": len(reads), "total_copies": int(sum(copies)),
            "copy_min": min(copies), "copy_max": max(copies),
            "copy_mean": round(sum(copies) / len(copies), 4),
        }
        print(f"\n=== pool {pool_tag}: {n_oligos} oligos x {oligo_len}nt, "
              f"{len(reads):,} reads (copies {COPY_LO}-{COPY_HI}, "
              f"err={READ_ERR}) ===", flush=True)

        for inp in INPUTS:
            seq = input_seq[inp]
            fasta = input_fa[inp]
            for k in KS:
                codes, counts = fast_canonical_code_counts(seq, k)
                count_of = dict(zip(codes.tolist(), counts.tolist()))
                n_sat = int(np.count_nonzero(counts >= KC_MAX))
                results["pool_stats"][pool_tag][f"{inp}_n_canon_k{k}"] = int(codes.size)
                results["pool_stats"][pool_tag][f"{inp}_max_cov_k{k}"] = int(counts.max()) if counts.size else 0

                for hashname in HASHES:
                    for n in NS:
                        n_cells += 1
                        tag = f"{pool_tag}_{inp}_k{k}_N{n}_{hashname}"
                        out1 = os.path.join(WORK, tag + ".r1.fgr2")
                        out2 = os.path.join(WORK, tag + ".r2.fgr2")
                        wall1 = run_fgr2(fasta, out1, k, n, hashname)
                        wall2 = run_fgr2(fasta, out2, k, n, hashname)

                        e_kmers, e_hashes, e_covs = read_sketch_ordered(out1)
                        _, _, meta = fgrlib.parse_sketch(out1)

                        true_kmers, _, boundary_tie, n_total = \
                            true_bottom_n(codes, hashname, n, k)

                        # C1 exact set equality
                        set_e, set_t = set(e_kmers), set(true_kmers)
                        only_f = sorted(set_e - set_t)
                        only_t = sorted(set_t - set_e)
                        c1 = (not only_f) and (not only_t)

                        # C2 hash fidelity
                        bad_hash = [km for km, hv in zip(e_kmers, e_hashes)
                                    if int(SCALAR[hashname](canonical_code_of(km))) != hv]
                        c2 = not bad_hash

                        # C3 sorted ascending
                        n_inv = sum(1 for a, b in zip(e_hashes, e_hashes[1:]) if b < a)
                        c3 = n_inv == 0

                        # C4 provenance (present in input codes)
                        ecodes = np.array([canonical_code_of(km) for km in e_kmers],
                                          dtype=np.uint64)
                        pos = np.searchsorted(codes, ecodes)
                        ok = pos < codes.size
                        present = np.zeros(ecodes.size, dtype=bool)
                        present[ok] = codes[pos[ok]] == ecodes[ok]
                        n_absent = int(np.count_nonzero(~present))
                        c4 = n_absent == 0

                        # C5 determinism
                        h1, h2 = sha256(out1), sha256(out2)
                        c5 = h1 == h2

                        # C6 coverage annotation == true occurrence count
                        n_cov_checked = 0
                        n_cov_bad = 0
                        cov_examples = []
                        for km, cov in zip(e_kmers, e_covs):
                            tc = count_of.get(canonical_code_of(km))
                            if tc is None:
                                continue  # absent -> already flagged by C4
                            if tc >= KC_MAX:
                                continue  # saturated; excluded per spec
                            n_cov_checked += 1
                            if cov != tc:
                                n_cov_bad += 1
                                if len(cov_examples) < 5:
                                    cov_examples.append(
                                        {"kmer": km, "fgr2_cov": cov, "true_count": tc})
                        c6 = n_cov_bad == 0

                        # C7 canonical strings
                        n_noncanon = sum(1 for km in e_kmers
                                         if fgrlib.encode2bit(km) != canonical_code_of(km))
                        c7 = n_noncanon == 0

                        cell_pass = c1 and c2 and c3 and c4 and c5 and c6 and c7
                        n_cells_pass += int(cell_pass)

                        results["cells"].append({
                            "pool": pool_tag, "n_oligos": n_oligos,
                            "oligo_len": oligo_len, "input": inp,
                            "k": k, "N": n, "hash": hashname,
                            "n_emitted": len(e_kmers),
                            "n_distinct_canonical_kmers": n_total,
                            "n_expected": min(n, n_total),
                            "boundary_hash_tie": boundary_tie,
                            "C1_set_equality": c1,
                            "n_only_in_fgr2": len(only_f),
                            "n_only_in_truth": len(only_t),
                            "example_only_in_fgr2": only_f[:5],
                            "example_only_in_truth": only_t[:5],
                            "C2_hash_fidelity": c2,
                            "n_bad_hash": len(bad_hash),
                            "example_bad_hash": bad_hash[:5],
                            "C3_sorted_ascending": c3, "n_hash_inversions": n_inv,
                            "C4_provenance": c4, "n_kmers_absent_from_input": n_absent,
                            "C5_determinism": c5,
                            "sha256_run1": h1, "sha256_run2": h2,
                            "C6_coverage_correct": c6,
                            "n_cov_checked": n_cov_checked,
                            "n_cov_mismatch": n_cov_bad,
                            "n_cov_saturated_excluded": n_sat,
                            "example_cov_mismatch": cov_examples,
                            "C7_canonical_strings": c7,
                            "n_noncanonical_strings": n_noncanon,
                            "cell_pass": cell_pass,
                            "wall_s_run1": round(wall1, 4),
                            "wall_s_run2": round(wall2, 4),
                            "header_meta": meta,
                        })
                        status = "PASS" if cell_pass else "FAIL"
                        print(f"  [{status}] {inp:6s} k={k} N={n} {hashname:12s} "
                              f"emit={len(e_kmers)} onlyF={len(only_f)} "
                              f"onlyT={len(only_t)} badh={len(bad_hash)} "
                              f"inv={n_inv} absent={n_absent} "
                              f"cov(chk={n_cov_checked},bad={n_cov_bad}) "
                              f"det={c5}", flush=True)

    # ---------------- thread invariance
    print("\n=== thread invariance ===", flush=True)
    n_thread_pass = 0
    tp_tag = f"n{THREAD_POOL[0]}_L{THREAD_POOL[1]}"
    for inp in INPUTS:
        fasta = os.path.join(WORK, tp_tag + (".oligos.fa" if inp == "direct"
                                             else ".reads.fa"))
        for k in KS:
            for hashname in HASHES:
                n = 10000
                digests = {}
                for t in THREAD_COUNTS:
                    out = os.path.join(WORK, f"{tp_tag}_{inp}_k{k}_{hashname}_t{t}.fgr2")
                    run_fgr2(fasta, out, k, n, hashname, threads=t)
                    digests[t] = sha256(out)
                identical = len(set(digests.values())) == 1
                n_thread_pass += int(identical)
                results["thread_invariance"].append({
                    "pool": tp_tag, "input": inp, "k": k, "N": n, "hash": hashname,
                    "threads": THREAD_COUNTS,
                    "sha256_by_threads": {str(t): d for t, d in digests.items()},
                    "byte_identical": identical,
                })
                print(f"  [{'PASS' if identical else 'FAIL'}] {inp:6s} k={k} "
                      f"{hashname} t={THREAD_COUNTS}", flush=True)

    checks = ["C1_set_equality", "C2_hash_fidelity", "C3_sorted_ascending",
              "C4_provenance", "C5_determinism", "C6_coverage_correct",
              "C7_canonical_strings"]
    results["summary"] = {
        "n_cells": n_cells,
        "n_cells_pass": n_cells_pass,
        "n_cells_fail": n_cells - n_cells_pass,
        "n_fgr2_runs_grid": n_cells * 2,
        "checks_per_cell": checks,
        "per_check_pass": {c: sum(1 for x in results["cells"] if x[c])
                           for c in checks},
        "n_cells_with_boundary_hash_tie":
            sum(1 for x in results["cells"] if x["boundary_hash_tie"]),
        "total_cov_kmers_checked":
            sum(x["n_cov_checked"] for x in results["cells"]),
        "total_cov_mismatches":
            sum(x["n_cov_mismatch"] for x in results["cells"]),
        "n_thread_configs": len(results["thread_invariance"]),
        "n_thread_configs_pass": n_thread_pass,
        "n_fgr2_runs_thread": len(results["thread_invariance"]) * len(THREAD_COUNTS),
        "all_pass": (n_cells_pass == n_cells
                     and n_thread_pass == len(results["thread_invariance"])),
    }

    with open(RESULTS, "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    print("\n" + json.dumps(results["summary"], indent=1))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
