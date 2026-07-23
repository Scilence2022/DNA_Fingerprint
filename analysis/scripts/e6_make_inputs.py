#!/usr/bin/env python3
"""
E6 input construction.

Builds:
  1. A size ladder of concatenated FASTA files spanning 1 Mb .. 500 Mb.
     Sequence is drawn from the 16 real RefSeq genomes. The 16 genomes total
     ~83 Mb, so ladder points above that reuse the genomes with independent
     random substitutions applied (5% per base) so that DISTINCT k-mer content
     keeps growing rather than the input becoming trivially compressible.
  2. Simulated read sets from a subset of the panel, for the read-mode
     benchmark (this is where fgr2's coverage filtering is meant to matter).

All randomness is seeded. Nothing here is a measured quantity; this script only
produces benchmark inputs.
"""
import os
import sys
import json
import glob
import random
import argparse

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import fgrlib  # noqa: E402

SEED = 20260721


def fast_mutate(seq_bytes, rate, rng):
    """Vectorised substitution mutator (input construction only, not measured).

    seq_bytes : np.uint8 array of ASCII ACGT
    Substitutes a `rate` fraction of positions with a random different base.
    """
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    n = seq_bytes.size
    n_mut = int(n * rate)
    idx = rng.choice(n, size=n_mut, replace=False)
    # random base; if it equals the original, shift by one so a mutation always
    # changes the base (keeps the effective divergence equal to `rate`).
    newb = bases[rng.integers(0, 4, size=n_mut)]
    same = newb == seq_bytes[idx]
    if same.any():
        # map each colliding base to the "next" base in ACGT order
        order = {65: 67, 67: 71, 71: 84, 84: 65}
        for a, b in order.items():
            m = same & (newb == a)
            newb[m] = b
    out = seq_bytes.copy()
    out[idx] = newb
    return out


def write_fasta_bytes(path, name, arr, width=70):
    """Stream a uint8 ASCII array out as FASTA without building a giant str."""
    with open(path, "wb") as fh:
        fh.write(b">" + name.encode() + b"\n")
        mv = memoryview(arr)
        for i in range(0, arr.size, width):
            fh.write(mv[i:i + width].tobytes())
            fh.write(b"\n")


def build_ladder(genome_dir, outdir, targets_mb):
    os.makedirs(outdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(genome_dir, "*.fna")))
    assert len(paths) == 16, f"expected 16 genomes, found {len(paths)}"

    # Load all genomes once as uint8 arrays.
    seqs = []
    for p in paths:
        s = fgrlib.load_genome(p)
        seqs.append(np.frombuffer(s.encode(), dtype=np.uint8))
    total_real = sum(s.size for s in seqs)
    print(f"[inputs] 16 real genomes total {total_real/1e6:.2f} Mb", flush=True)

    rng = np.random.default_rng(SEED)
    manifest = []

    for mb in targets_mb:
        target = int(mb * 1_000_000)
        out = os.path.join(outdir, f"ladder_{mb}mb.fa")
        if os.path.exists(out) and os.path.getsize(out) > target:
            print(f"[inputs] reuse {out}", flush=True)
        else:
            chunks = []
            got = 0
            rep = 0
            while got < target:
                for s in seqs:
                    if got >= target:
                        break
                    piece = s if rep == 0 else fast_mutate(s, 0.05, rng)
                    need = target - got
                    if piece.size > need:
                        piece = piece[:need]
                    chunks.append(piece)
                    got += piece.size
                rep += 1
            arr = np.concatenate(chunks)
            write_fasta_bytes(out, f"ladder_{mb}mb", arr)
            del arr, chunks
            print(f"[inputs] wrote {out} ({mb} Mb, {rep} passes)", flush=True)
        manifest.append({
            "target_mb": mb,
            "path": out,
            "bases": target,
            "file_bytes": os.path.getsize(out),
            "contains_mutated_copies": target > total_real,
        })
    return manifest, total_real


def build_reads(genome_dir, outdir, names, depth, read_len, err_rate):
    os.makedirs(outdir, exist_ok=True)
    out_paths = []
    for i, nm in enumerate(names):
        gpath = os.path.join(genome_dir, nm + ".fna")
        out = os.path.join(outdir, f"reads_{nm}.fa")
        if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
            print(f"[reads] reuse {out}", flush=True)
            out_paths.append(out)
            continue
        genome = fgrlib.load_genome(gpath)
        rng = random.Random(SEED + i)
        reads = fgrlib.simulate_reads(genome, depth=depth, read_len=read_len,
                                      err_rate=err_rate, rng=rng)
        fgrlib.write_reads_fasta(out, reads)
        print(f"[reads] wrote {out}: {len(reads)} reads, "
              f"{len(reads)*read_len/1e6:.1f} Mb", flush=True)
        out_paths.append(out)
    return out_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomes", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    targets = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ladder, total_real = build_ladder(args.genomes, args.outdir, targets)

    read_names = ["Ecoli_K12_MG1655", "Bacillus_subtilis_168",
                  "Salmonella_Typhimurium_LT2", "Klebsiella_pneumoniae"]
    reads = build_reads(args.genomes, args.outdir, read_names,
                        depth=30.0, read_len=150, err_rate=0.01)

    man = {
        "seed": SEED,
        "ladder": ladder,
        "total_real_bases": total_real,
        "read_sets": reads,
        "read_params": {"depth": 30.0, "read_len": 150, "err_rate": 0.01,
                        "indel_frac": "fgrlib default"},
        "mutation_rate_above_real": 0.05,
    }
    with open(args.manifest, "w") as fh:
        json.dump(man, fh, indent=2)
    print(f"[inputs] manifest -> {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
