#!/usr/bin/env python3
"""
S6 -- Tracking individual PCR replicates and distinct physical copies of the SAME
stored data.

Storage-native. Two aliquots / amplifications of the same encoded file carry
IDENTICAL sequence content, so any presence/absence signature reports them as
indistinguishable (Jaccard ~ 1). But each independent PCR amplification imprints
its own idiosyncratic copy-number pattern on the pool. The coverage-annotated
fingerprint therefore carries a *molecular provenance signature* of that specific
physical instance.

Design:
  one encoded file  ->  R independent PCR replicates (independent copy-number draws
  at the same nominal skew)  ->  T independent sequencing runs per replicate
  (same amplified material, independent read sampling + error).

  within-replicate pair  = two sequencing runs of the SAME amplification
  between-replicate pair = runs of DIFFERENT amplifications of the SAME file

If abundance carries provenance, cosine(within) > cosine(between) while Jaccard is
~1 for both. sigma = 0 (uniform copies) is the negative control: with no PCR skew
there is no provenance signature to read.

No genome is used anywhere.
"""
import json, os, subprocess, sys, itertools
import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import fgrlib

REPO = os.path.dirname(os.path.dirname(SCRIPTS))
FGR2 = os.path.join(REPO, "fgr2")
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/"
           "2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad/s6")
os.makedirs(SCRATCH, exist_ok=True)

K = 21
N = 10000
OLIGO_LEN = 150
N_OLIGOS = 4000
BASE_COPIES = 20
ERR = 0.005
CFILTER = 3
R_REPLICATES = 6          # independent PCR amplifications of the same file
T_RUNS = 3                # independent sequencing runs per amplification
SIGMAS = [0.0, 0.25, 0.5, 1.0]
SEED0 = 20260722


def fingerprint(reads, tag):
    fa = os.path.join(SCRATCH, tag + ".fa")
    fgrlib.write_reads_fasta(fa, reads)
    out = os.path.join(SCRATCH, tag + ".fgr2")
    subprocess.run([FGR2, "-k", str(K), "-N", str(N), "-c", str(CFILTER),
                    "-o", out, fa], check=True, capture_output=True)
    h, cov, _ = fgrlib.parse_sketch(out)
    return h, cov


def common_index(hA, hB):
    m = {}
    m.update(hA)
    m.update(hB)
    return [k for k, _ in sorted(m.items(), key=lambda kv: kv[1])[:N]]


def jaccard_ci(hA, hB):
    S = common_index(hA, hB)
    if not S:
        return 1.0
    return sum(1 for k in S if k in hA and k in hB) / len(S)


def cosine_ci(hA, hB, cA, cB):
    S = common_index(hA, hB)
    if not S:
        return 0.0
    a = np.array([cA.get(k, 0) for k in S], float)
    b = np.array([cB.get(k, 0) for k in S], float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a.dot(b) / (na * nb)) if na and nb else 0.0


def auc(neg, pos):
    if not neg or not pos:
        return None
    t = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return t / (len(pos) * len(neg))


def run_sigma(sigma, pool, rng_master):
    import random
    fps = {}          # (rep, run) -> (hashes, cov)
    copies_by_rep = {}
    for rep in range(R_REPLICATES):
        # independent PCR amplification of the SAME file
        rng_pcr = random.Random(SEED0 + 7919 * rep + int(sigma * 1000))
        copies = fgrlib.pcr_bias_copies(len(pool), BASE_COPIES, sigma, rng_pcr)
        copies_by_rep[rep] = copies
        for run in range(T_RUNS):
            # independent sequencing of that SAME amplified material
            rng_seq = random.Random(SEED0 + 104729 * rep + 31 * run + int(sigma * 1000))
            reads = fgrlib.pool_to_reads(pool, copies, ERR, rng_seq)
            fps[(rep, run)] = fingerprint(reads, f"s{sigma}_r{rep}_t{run}")
    # pairwise
    within_cos, between_cos, within_jac, between_jac = [], [], [], []
    pair_rows = []
    keys = sorted(fps)
    for a, b in itertools.combinations(keys, 2):
        ha, ca = fps[a]
        hb, cb = fps[b]
        c = cosine_ci(ha, hb, ca, cb)
        j = jaccard_ci(ha, hb)
        same = (a[0] == b[0])
        (within_cos if same else between_cos).append(c)
        (within_jac if same else between_jac).append(j)
        pair_rows.append({"a_rep": a[0], "a_run": a[1], "b_rep": b[0], "b_run": b[1],
                          "same_replicate": same, "cosine": c, "jaccard": j})

    # replicate identification: for each run, is its nearest neighbour (by cosine)
    # another run of the SAME amplification?
    correct = 0
    for q in keys:
        hq, cq = fps[q]
        best, best_s = None, -1
        for o in keys:
            if o == q:
                continue
            ho, co = fps[o]
            s = cosine_ci(hq, ho, cq, co)
            if s > best_s:
                best_s, best = s, o
        correct += (best[0] == q[0])
    top1 = correct / len(keys)

    # same, but using Jaccard (presence/absence) -- the comparison that matters
    correct_j = 0
    for q in keys:
        hq, cq = fps[q]
        best, best_s = None, -1
        for o in keys:
            if o == q:
                continue
            ho, co = fps[o]
            s = jaccard_ci(hq, ho)
            if s > best_s:
                best_s, best = s, o
        correct_j += (best[0] == q[0])
    top1_j = correct_j / len(keys)

    return {
        "sigma": sigma,
        "n_fingerprints": len(keys),
        "n_within_pairs": len(within_cos),
        "n_between_pairs": len(between_cos),
        "cosine_within_mean": float(np.mean(within_cos)),
        "cosine_within_sd": float(np.std(within_cos, ddof=1)),
        "cosine_between_mean": float(np.mean(between_cos)),
        "cosine_between_sd": float(np.std(between_cos, ddof=1)),
        "cosine_separation": float(np.mean(within_cos) - np.mean(between_cos)),
        "cosine_auc_same_vs_diff": auc(between_cos, within_cos),
        "jaccard_within_mean": float(np.mean(within_jac)),
        "jaccard_between_mean": float(np.mean(between_jac)),
        "jaccard_separation": float(np.mean(within_jac) - np.mean(between_jac)),
        "jaccard_auc_same_vs_diff": auc(between_jac, within_jac),
        "replicate_id_top1_cosine": top1,
        "replicate_id_top1_jaccard": top1_j,
        "chance_top1": (T_RUNS - 1) / (R_REPLICATES * T_RUNS - 1),
        "pairs": pair_rows,
    }


def main():
    import random
    rng = random.Random(SEED0)
    pool = fgrlib.make_oligo_pool(N_OLIGOS, OLIGO_LEN, rng)
    out = {
        "experiment_id": "S6_pcr_replicate_and_copy_tracking",
        "application": ("DNA data storage: tracking individual PCR amplifications and "
                        "distinct physical copies of the SAME encoded data"),
        "inputs_are_storage_only": True,
        "input_definition": ("one simulated encoded file (random payloads + shared 20nt "
                             "primers); no genome"),
        "params": {"k": K, "N": N, "oligo_len": OLIGO_LEN, "n_oligos": N_OLIGOS,
                   "base_copies": BASE_COPIES, "err": ERR, "cfilter": CFILTER,
                   "R_replicates": R_REPLICATES, "T_runs_per_replicate": T_RUNS,
                   "sigmas": SIGMAS, "seed0": SEED0,
                   "estimator": "common-index (merged bottom-N)"},
        "by_sigma": [],
    }
    for s in SIGMAS:
        print(f"running sigma={s} ...", flush=True)
        out["by_sigma"].append(run_sigma(s, pool, rng))

    p = os.path.join(REPO, "analysis", "results", "s6_replicate_tracking.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", p)
    print(f"\n{'sigma':>6s} {'cos_within':>11s} {'cos_between':>12s} {'sep':>8s} "
          f"{'cosAUC':>7s} {'jacAUC':>7s} {'top1_cos':>9s} {'top1_jac':>9s} {'chance':>7s}")
    for r in out["by_sigma"]:
        print(f"{r['sigma']:6.2f} {r['cosine_within_mean']:11.5f} "
              f"{r['cosine_between_mean']:12.5f} {r['cosine_separation']:8.5f} "
              f"{r['cosine_auc_same_vs_diff']:7.3f} {r['jaccard_auc_same_vs_diff']:7.3f} "
              f"{r['replicate_id_top1_cosine']:9.3f} {r['replicate_id_top1_jaccard']:9.3f} "
              f"{r['chance_top1']:7.3f}")
    print(f"\nJaccard within/between (should be ~equal -> cannot distinguish copies):")
    for r in out["by_sigma"]:
        print(f"  sigma={r['sigma']:.2f}: within={r['jaccard_within_mean']:.5f} "
              f"between={r['jaccard_between_mean']:.5f} sep={r['jaccard_separation']:+.5f}")


if __name__ == "__main__":
    main()
