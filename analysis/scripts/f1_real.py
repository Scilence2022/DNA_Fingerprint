#!/usr/bin/env python3
"""
F1 part 2 -- cosine estimator bias on REAL genomes, with the real fgr2 binary.

Two tracks:
  (A) natural pairs: all 120 unordered pairs of the 16 RefSeq genomes. Ground
      truth = exact abundance-weighted cosine over the complete canonical
      21-mer coverage vectors; sketches = fgr2 -c 1 output. Size ratios are
      whatever nature gives (~1.0-1.5).
  (B) controlled size ratio: A = base genome; B = (optionally mutated copy of
      the base genome) ++ filler drawn from the other 15 genomes, trimmed so
      that |B| / |A| hits a target ratio. This is the "pool contains extra
      material" regime, i.e. the same driver E2 used for the Jaccard bias.
      Realised support ratios are measured, not assumed.

Sketches are built ONCE per input at N = 10000; the N = 1000 and N = 100
sketches are the 1000 / 100 smallest-hash entries of that sketch (bottom-N is
nested, so this is exact).
"""
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468/"
           "scratchpad")
GENOMES = os.path.join(SCRATCH, "genomes")
WORK = os.path.join(SCRATCH, "f1real")
FGR2 = os.path.join(REPO, "fgr2")
OUT = os.path.join(REPO, "analysis", "results", "f1_cosine_real.json")

import fgrlib                              # noqa: E402
import f1_common as F                      # noqa: E402

K = 21
N_BIG = 10000
N_SIZES = [100, 1000, 10000]
RATIOS = [1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
MUTS = [0.0, 0.01]
N_REPS = 20
SEED0 = 771000


def genome_paths():
    return sorted(os.path.join(GENOMES, f)
                  for f in os.listdir(GENOMES) if f.endswith(".fna"))


def sketch_file(fa, out, n=N_BIG):
    subprocess.run([FGR2, "-k", str(K), "-N", str(n), "-c", "1",
                    "-o", out, fa], check=True, capture_output=True)


def load_sk(path):
    h, cov, _ = fgrlib.parse_sketch(path)
    return {km: (h[km], cov[km]) for km in h}


def truncate(sk, n):
    if len(sk) <= n:
        return dict(sk)
    keys = sorted(sk, key=lambda km: sk[km][0])[:n]
    return {km: sk[km] for km in keys}


def counts_of(seq):
    codes, cnts = F.canonical_code_counts(seq, K)
    return codes, cnts.astype(np.int64)


# ------------------------------------------------------------------ track A
def track_a():
    paths = genome_paths()
    names = [os.path.basename(p)[:-4] for p in paths]
    os.makedirs(WORK, exist_ok=True)
    seqs, counts, sks = {}, {}, {}
    for p, nm in zip(paths, names):
        s = fgrlib.load_genome(p)
        seqs[nm] = s
        counts[nm] = counts_of(s)
        sp = os.path.join(WORK, f"{nm}.N{N_BIG}.fgr2")
        if not os.path.exists(sp):
            sketch_file(p, sp)
        sks[nm] = load_sk(sp)
        print(f"  [A] {nm}: {len(s)} bp, {counts[nm][0].size} distinct 21-mers",
              flush=True)
    recs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ca, na_ = counts[a]
            cb, nb_ = counts[b]
            truth = F.cos_true_arrays(ca, na_, cb, nb_)
            rec = {"a": a, "b": b,
                   "supp_a": int(ca.size), "supp_b": int(cb.size),
                   "ratio": max(ca.size, cb.size) / min(ca.size, cb.size),
                   "cos_true": truth, "est": {}}
            for n in N_SIZES:
                rec["est"][str(n)] = F.all_estimators(
                    truncate(sks[a], n), truncate(sks[b], n), n)
            recs.append(rec)
    return recs


# ------------------------------------------------------------------ track B
def _rep_worker(rep):
    paths = genome_paths()
    names = [os.path.basename(p)[:-4] for p in paths]
    base_i = rep % len(paths)
    base = fgrlib.load_genome(paths[base_i])
    rng = random.Random(SEED0 + rep)
    others = [i for i in range(len(paths)) if i != base_i]
    rng.shuffle(others)
    # filler stream: enough extra sequence for ratio 3.0
    need = int(len(base) * (max(RATIOS) - 1.0) * 1.3)
    chunks, got = [], 0
    for i in others:
        s = fgrlib.load_genome(paths[i])
        chunks.append(s)
        got += len(s)
        if got >= need:
            break
    filler = "N".join(chunks)

    wdir = os.path.join(WORK, f"rep{rep}")
    os.makedirs(wdir, exist_ok=True)
    fa_a = os.path.join(wdir, "A.fa")
    fgrlib.write_fasta(fa_a, [("A", base)])
    sk_a_path = os.path.join(wdir, "A.fgr2")
    sketch_file(fa_a, sk_a_path)
    sk_a_full = load_sk(sk_a_path)
    codes_a, cnt_a = counts_of(base)
    supp_a = codes_a.size

    out = []
    for mut in MUTS:
        core = base if mut == 0.0 else fgrlib.mutate(
            base, mut, random.Random(SEED0 + 7919 * rep + int(mut * 10000)))
        for ratio in RATIOS:
            # one secant refinement to land |supp(B)| near ratio*supp_a;
            # the REALISED ratio is what gets recorded and analysed.
            target = ratio * supp_a
            mid = int(round((ratio - 1.0) * len(base)))
            seq = core if mid == 0 else core + "N" + filler[:mid]
            codes_b, cnt_b = counts_of(seq)
            if mid > 0 and codes_b.size > supp_a:
                scale = (target - supp_a) / (codes_b.size - supp_a)
                mid2 = int(min(len(filler), max(1, round(mid * scale))))
                if abs(mid2 - mid) > 0.01 * mid:
                    mid = mid2
                    seq = core + "N" + filler[:mid]
                    codes_b, cnt_b = counts_of(seq)
            fa_b = os.path.join(wdir, "B.fa")
            fgrlib.write_fasta(fa_b, [("B", seq)])
            sk_b_path = os.path.join(wdir, "B.fgr2")
            sketch_file(fa_b, sk_b_path)
            sk_b_full = load_sk(sk_b_path)
            truth = F.cos_true_arrays(codes_a, cnt_a, codes_b, cnt_b)
            rec = {"rep": rep, "base": names[base_i], "mut": mut,
                   "target_ratio": ratio,
                   "supp_a": int(supp_a), "supp_b": int(codes_b.size),
                   "ratio": codes_b.size / supp_a,
                   "filler_bp": int(mid), "cos_true": truth, "est": {}}
            for n in N_SIZES:
                rec["est"][str(n)] = F.all_estimators(
                    truncate(sk_a_full, n), truncate(sk_b_full, n), n)
            out.append(rec)
            os.remove(fa_b)
            os.remove(sk_b_path)
    os.remove(fa_a)
    print(f"  [B] rep {rep} ({names[base_i]}) done", flush=True)
    return out


def track_b(workers=6):
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sub in ex.map(_rep_worker, range(N_REPS)):
            recs.extend(sub)
    return recs


def main():
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()
    a = track_a()
    print(f"track A done {time.time()-t0:.0f}s", flush=True)
    b = track_b()
    out = {
        "experiment_id": "F1_cosine_bias_real_genomes",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {"k": K, "N_sizes": N_SIZES, "ratios": RATIOS,
                       "muts": MUTS, "n_reps": N_REPS, "seed0": SEED0,
                       "coverage_threshold": 1,
                       "sketch_source": "fgr2 binary v2.1.0, nested truncation"},
        "estimators": F.EST_NAMES,
        "natural_pairs": a,
        "controlled_ratio": b,
        "wall_seconds": time.time() - t0,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT, len(a), "pairs +", len(b), "controlled records",
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
