#!/usr/bin/env python3
"""
E1 -- Is the fgr2 sketch a correct bottom-N MinHash at genome scale?

For 6 real RefSeq genomes x k in {21,25,31} x N in {1000,10000} x
hash in {murmurhash3, wang}, run fgr2 -c 1 and check six independent
properties against a ground truth recomputed from fgrlib.

  C1 exact set equality  fgr2 k-mer set == true bottom-N k-mer set
  C2 hash fidelity       every emitted hash == recomputed hash of that k-mer
  C3 sort order          output strictly ascending by hash
  C4 provenance          every emitted k-mer occurs in the genome (fwd or rc)
  C5 determinism         same config run twice -> byte-identical file
  C6 thread invariance   -t 1/4/16 -> byte-identical file (3 genomes)

Everything is deterministic; the only RNG is the sampling of k-mers for the
literal substring provenance spot-check, seeded at SEED below.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

SEED = 20260721

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468"
           "/scratchpad")
GENOME_DIR = os.path.join(SCRATCH, "genomes")
WORK = os.path.join(SCRATCH, "e1")
FGR2 = os.path.join(REPO, "fgr2")
RESULTS = os.path.join(REPO, "analysis/results/e1_sketch_correctness.json")

sys.path.insert(0, os.path.join(REPO, "analysis/scripts"))
import fgrlib  # noqa: E402

GENOMES = [
    "Ecoli_K12_MG1655",
    "Bacillus_subtilis_168",
    "Pseudomonas_aeruginosa_PAO1",
    "Salmonella_Typhimurium_LT2",
    "Klebsiella_pneumoniae",
    "Yersinia_pestis_CO92",
]
KS = [21, 25, 31]
NS = [1000, 10000]
HASHES = ["murmurhash3", "wang"]
THREAD_GENOMES = GENOMES[:3]
THREAD_COUNTS = [1, 4, 16]

u64 = np.uint64


# ---------------------------------------------------------------- vectorised hashes

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
    """Vectorised port of fgrlib.murmur3_x64_64 (seed=42, len=8)."""
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
    """Vectorised port of fgrlib.wang_hash."""
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
    """Verify the numpy hashes against fgrlib's scalar versions before use."""
    probes = rng.integers(0, 1 << 62, size=n_probe, dtype=np.uint64)
    probes = np.concatenate([np.array([0, 1, 2, (1 << 62) - 1], dtype=np.uint64),
                             probes])
    out = {}
    for name in HASHES:
        vec = VEC[name](probes)
        ref = np.array([SCALAR[name](int(p)) for p in probes], dtype=np.uint64)
        n_mismatch = int(np.count_nonzero(vec != ref))
        out[name] = {"n_probes": int(probes.size), "n_mismatch": n_mismatch,
                     "agrees": n_mismatch == 0}
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


def decode(code, k):
    return fgrlib.decode2bit(int(code), k)


def true_bottom_n(codes, hashname, n, k):
    """Ground-truth bottom-N: (kmer strings, hashes) ascending by hash.

    Also reports whether the N-th/N+1-th hash values tie, which would make the
    bottom-N k-mer SET ambiguous (and any set-equality check unfair)."""
    hv = VEC[hashname](codes)
    if hv.size <= n:
        order = np.argsort(hv, kind="stable")
        boundary_tie = False
    else:
        part = np.argpartition(hv, n - 1)
        sel = part[:n]
        sel = sel[np.argsort(hv[sel], kind="stable")]
        nth = hv[sel[-1]]
        # ambiguity iff some non-selected k-mer has hash == the N-th hash
        rest = part[n:]
        boundary_tie = bool(np.any(hv[rest] == nth))
        order = sel
    kmers = [decode(codes[i], k) for i in order]
    return kmers, hv[order], boundary_tie, int(hv.size)


def canonical_code_of(kmer):
    k = len(kmer)
    fwd = fgrlib.encode2bit(kmer)
    rev = fgrlib.encode2bit(fgrlib.revcomp(kmer))
    return min(fwd, rev), fwd, rev, k


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

    results = {
        "experiment_id": "E1",
        "seed": SEED,
        "fgr2_binary": FGR2,
        "fgr2_sha256": sha256(FGR2),
        "genomes": GENOMES,
        "grid": {"k": KS, "N": NS, "hash": HASHES},
        "hash_validation": hash_validation,
        "cells": [],
        "thread_invariance": [],
        "genome_stats": {},
    }

    n_cells = 0
    n_cells_pass = 0

    for gname in GENOMES:
        fasta = os.path.join(GENOME_DIR, gname + ".fna")
        seq = fgrlib.load_genome(fasta)
        results["genome_stats"].setdefault(gname, {})["length_bp"] = len(seq)
        print(f"\n=== {gname} ({len(seq):,} bp) ===", flush=True)

        for k in KS:
            t0 = time.time()
            codes = fgrlib.fast_canonical_codes(seq, k)
            enum_s = time.time() - t0
            results["genome_stats"][gname][f"n_canonical_kmers_k{k}"] = int(codes.size)
            print(f"  k={k}: {codes.size:,} distinct canonical k-mers "
                  f"({enum_s:.1f}s)", flush=True)

            for hashname in HASHES:
                for n in NS:
                    n_cells += 1
                    tag = f"{gname}_k{k}_N{n}_{hashname}"
                    out1 = os.path.join(WORK, tag + ".r1.fgr2")
                    out2 = os.path.join(WORK, tag + ".r2.fgr2")
                    wall1 = run_fgr2(fasta, out1, k, n, hashname)
                    wall2 = run_fgr2(fasta, out2, k, n, hashname)

                    emitted_h, emitted_cov, meta = fgrlib.parse_sketch(out1)
                    # preserve file order for the sort check
                    order_kmers, order_hashes = [], []
                    with open(out1) as fh:
                        for line in fh:
                            if line.startswith("#") or not line.strip():
                                continue
                            parts = line.rstrip("\n").split("\t")
                            if len(parts) < 3:
                                continue
                            order_kmers.append(parts[0])
                            order_hashes.append(int(parts[1]))

                    true_kmers, true_hashes, boundary_tie, n_total = \
                        true_bottom_n(codes, hashname, n, k)

                    # --- C1 exact set equality
                    set_emit, set_true = set(order_kmers), set(true_kmers)
                    only_fgr2 = sorted(set_emit - set_true)
                    only_true = sorted(set_true - set_emit)
                    c1 = (not only_fgr2) and (not only_true)

                    # --- C2 hash fidelity
                    bad_hash = []
                    for km, hv in zip(order_kmers, order_hashes):
                        code, _, _, _ = canonical_code_of(km)
                        if int(SCALAR[hashname](code)) != hv:
                            bad_hash.append(km)
                    c2 = not bad_hash

                    # --- C3 sorted ascending by hash
                    n_inversions = sum(1 for a, b in zip(order_hashes,
                                                         order_hashes[1:]) if b < a)
                    c3 = n_inversions == 0

                    # --- C4 provenance: emitted k-mer present in genome
                    ecodes = np.array([canonical_code_of(km)[0]
                                       for km in order_kmers], dtype=np.uint64)
                    pos = np.searchsorted(codes, ecodes)
                    pos_ok = (pos < codes.size)
                    present = np.zeros(ecodes.size, dtype=bool)
                    present[pos_ok] = codes[pos[pos_ok]] == ecodes[pos_ok]
                    n_absent = int(np.count_nonzero(~present))
                    # literal substring spot-check on a random sample
                    sample_idx = rng.choice(len(order_kmers),
                                            size=min(15, len(order_kmers)),
                                            replace=False)
                    n_substr_fail = 0
                    for i in sample_idx:
                        km = order_kmers[int(i)]
                        if km not in seq and fgrlib.revcomp(km) not in seq:
                            n_substr_fail += 1
                    c4 = (n_absent == 0) and (n_substr_fail == 0)

                    # --- C5 determinism
                    h1, h2 = sha256(out1), sha256(out2)
                    c5 = h1 == h2

                    # --- k-mer strings must be canonical (min of fwd/rev)
                    n_noncanon = 0
                    for km in order_kmers:
                        c, f, r, _ = canonical_code_of(km)
                        if fgrlib.encode2bit(km) != c:
                            n_noncanon += 1

                    cell_pass = c1 and c2 and c3 and c4 and c5
                    n_cells_pass += int(cell_pass)

                    results["cells"].append({
                        "genome": gname, "k": k, "N": n, "hash": hashname,
                        "n_emitted": len(order_kmers),
                        "n_distinct_canonical_kmers": n_total,
                        "n_expected": min(n, n_total),
                        "boundary_hash_tie": boundary_tie,
                        "C1_set_equality": c1,
                        "n_only_in_fgr2": len(only_fgr2),
                        "n_only_in_truth": len(only_true),
                        "example_only_in_fgr2": only_fgr2[:5],
                        "example_only_in_truth": only_true[:5],
                        "C2_hash_fidelity": c2,
                        "n_bad_hash": len(bad_hash),
                        "example_bad_hash": bad_hash[:5],
                        "C3_sorted_ascending": c3,
                        "n_hash_inversions": n_inversions,
                        "C4_provenance": c4,
                        "n_kmers_absent_from_genome": n_absent,
                        "n_substring_spotcheck_fail": n_substr_fail,
                        "n_substring_spotchecked": int(len(sample_idx)),
                        "C5_determinism": c5,
                        "sha256_run1": h1, "sha256_run2": h2,
                        "n_noncanonical_kmer_strings": n_noncanon,
                        "cell_pass": cell_pass,
                        "wall_s_run1": round(wall1, 4),
                        "wall_s_run2": round(wall2, 4),
                        "header_meta": meta,
                    })
                    status = "PASS" if cell_pass else "FAIL"
                    print(f"    [{status}] k={k} N={n} {hashname:12s} "
                          f"emitted={len(order_kmers)} "
                          f"onlyF={len(only_fgr2)} onlyT={len(only_true)} "
                          f"badhash={len(bad_hash)} inv={n_inversions} "
                          f"absent={n_absent} det={c5}", flush=True)

    # ---------------- thread invariance
    print("\n=== thread invariance ===", flush=True)
    n_thread_pass = 0
    for gname in THREAD_GENOMES:
        fasta = os.path.join(GENOME_DIR, gname + ".fna")
        for k in KS:
            for hashname in HASHES:
                n = 10000
                digests = {}
                for t in THREAD_COUNTS:
                    out = os.path.join(WORK, f"{gname}_k{k}_N{n}_{hashname}_t{t}.fgr2")
                    run_fgr2(fasta, out, k, n, hashname, threads=t)
                    digests[t] = sha256(out)
                identical = len(set(digests.values())) == 1
                n_thread_pass += int(identical)
                results["thread_invariance"].append({
                    "genome": gname, "k": k, "N": n, "hash": hashname,
                    "threads": THREAD_COUNTS,
                    "sha256_by_threads": {str(t): d for t, d in digests.items()},
                    "byte_identical": identical,
                })
                print(f"  [{'PASS' if identical else 'FAIL'}] {gname} k={k} "
                      f"{hashname} t={THREAD_COUNTS}", flush=True)

    results["summary"] = {
        "n_cells": n_cells,
        "n_cells_pass": n_cells_pass,
        "n_cells_fail": n_cells - n_cells_pass,
        "n_fgr2_runs_grid": n_cells * 2,
        "n_thread_configs": len(results["thread_invariance"]),
        "n_thread_configs_pass": n_thread_pass,
        "n_fgr2_runs_thread": len(results["thread_invariance"]) * len(THREAD_COUNTS),
        "checks_per_cell": ["C1_set_equality", "C2_hash_fidelity",
                            "C3_sorted_ascending", "C4_provenance",
                            "C5_determinism"],
        "per_check_pass": {
            c: sum(1 for x in results["cells"] if x[c])
            for c in ["C1_set_equality", "C2_hash_fidelity", "C3_sorted_ascending",
                      "C4_provenance", "C5_determinism"]
        },
        "n_cells_with_boundary_hash_tie":
            sum(1 for x in results["cells"] if x["boundary_hash_tie"]),
        "n_cells_with_noncanonical_strings":
            sum(1 for x in results["cells"] if x["n_noncanonical_kmer_strings"] > 0),
    }

    with open(RESULTS, "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    print("\n" + json.dumps(results["summary"], indent=1))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
