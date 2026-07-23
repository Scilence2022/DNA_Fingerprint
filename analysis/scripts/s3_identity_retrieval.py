#!/usr/bin/env python3
"""
S3 -- Identity, retrieval, and deduplication of stored DNA data pools.

Storage-native replacement for "does the fingerprint preserve similarity". Every
input is a simulated ENCODED OLIGO POOL (random payload + shared primer sites);
no genome is used. Demonstrates that fingerprint similarity lets you:

  (i)   identify which stored file a sequenced pool is (retrieval, top-1),
  (ii)  detect that two pools are related (dedup / relatedness AUC),
  (iii) tell apart unrelated pools,

and that fingerprint similarity tracks the ground-truth shared-oligo fraction.

Comparison uses the corrected COMMON-INDEX estimators (bottom-N of the merged
sketch), consistent with the rest of the campaign.
"""
import json, os, subprocess, sys, tempfile, math
import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import fgrlib

REPO = os.path.dirname(os.path.dirname(SCRIPTS))
FGR2 = os.path.join(REPO, "fgr2")
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/"
           "2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad/s3")
os.makedirs(SCRATCH, exist_ok=True)

K = 21
N = 10000
OLIGO_LEN = 150
N_OLIGOS = 8000          # oligos per "file"
M_FILES = 20             # library size
BASE_COPIES = 20
SEED0 = 20260722

# --------------------------------------------------------------------------
# fingerprint helpers
# --------------------------------------------------------------------------

def fgr2_fingerprint(records, tag, cov_c):
    """records: list of (name, seq). Write FASTA, run fgr2, return (hashes, cov)."""
    fa = os.path.join(SCRATCH, tag + ".fa")
    fgrlib.write_fasta(fa, records)
    out = os.path.join(SCRATCH, tag + ".fgr2")
    subprocess.run([FGR2, "-k", str(K), "-N", str(N), "-c", str(cov_c),
                    "-o", out, fa], check=True, capture_output=True)
    h, cov, _ = fgrlib.parse_sketch(out)
    return h, cov


def reads_records(reads):
    return [("r%d" % i, s) for i, s in enumerate(reads)]


def common_index(hA, hB):
    merged = {}
    merged.update(hA)
    merged.update(hB)
    if not merged:
        return []
    return [k for k, _ in sorted(merged.items(), key=lambda kv: kv[1])[:N]]


def jaccard_ci(hA, hB):
    S = common_index(hA, hB)
    if not S:
        return 1.0
    shared = sum(1 for k in S if k in hA and k in hB)
    return shared / len(S)


def cosine_ci(hA, hB, cA, cB):
    S = common_index(hA, hB)
    if not S:
        return 0.0
    a = np.array([cA.get(k, 0) for k in S], float)
    b = np.array([cB.get(k, 0) for k in S], float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a.dot(b) / (na * nb))


def auc(neg, pos):
    if not neg or not pos:
        return None
    t = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return t / (len(pos) * len(neg))


# --------------------------------------------------------------------------
# Part 1: a library of files; retrieval of degraded re-reads
# --------------------------------------------------------------------------

def build_library(rng):
    """M distinct files, each a set of oligos. Shared primer sites across all
    files (realistic library architecture); distinct random payloads."""
    fwd = "".join(rng.choice(list("ACGT")) for _ in range(20))
    rev = "".join(rng.choice(list("ACGT")) for _ in range(20))
    files = []
    for _ in range(M_FILES):
        oligos = [fwd + "".join(rng.choice(list("ACGT"))
                                for _ in range(OLIGO_LEN - 40)) + rev
                  for _ in range(N_OLIGOS)]
        files.append(oligos)
    return files, (fwd, rev)


def part1_retrieval(seeds, degradations):
    import random
    results = {"per_seed": [], "shared_fraction_curve": None}
    top1_by_deg = {d["name"]: [] for d in degradations}

    for sd in seeds:
        rng = random.Random(SEED0 + sd)
        nprng = np.random.default_rng(SEED0 + sd)
        files, _ = build_library(rng)

        # reference fingerprint per file: clean oligos, no error, -c 1
        refs = []
        for i, oligos in enumerate(files):
            h, c = fgr2_fingerprint([("o%d" % j, s) for j, s in enumerate(oligos)],
                                    "ref_s%d_f%d" % (sd, i), cov_c=1)
            refs.append((h, c))

        for deg in degradations:
            correct = 0
            simvals_true, simvals_other = [], []
            for i, oligos in enumerate(files):
                # query: degraded re-read of file i
                copies = fgrlib.pcr_bias_copies(len(oligos), BASE_COPIES,
                                                deg["skew"], rng)
                copies = fgrlib.dropout_copies(copies, deg["dropout"], rng)
                reads = fgrlib.pool_to_reads(oligos, copies, deg["err"], rng)
                if not reads:
                    continue
                qh, qc = fgr2_fingerprint(reads_records(reads),
                                          "qry_s%d_f%d_%s" % (sd, i, deg["name"]),
                                          cov_c=deg["cfilter"])
                sims = [jaccard_ci(qh, rh) for (rh, rc) in refs]
                pred = int(np.argmax(sims))
                correct += (pred == i)
                simvals_true.append(sims[i])
                simvals_other.extend(sims[:i] + sims[i + 1:])
            acc = correct / len(files)
            top1_by_deg[deg["name"]].append(acc)
            results["per_seed"].append({
                "seed": sd, "degradation": deg["name"],
                "top1_accuracy": acc,
                "mean_sim_true": float(np.mean(simvals_true)),
                "mean_sim_other": float(np.mean(simvals_other)),
                "max_sim_other": float(np.max(simvals_other)),
                "retrieval_margin": float(np.mean(simvals_true) - np.max(simvals_other)),
                "auc_true_vs_other": auc(simvals_other, simvals_true),
            })
    results["top1_summary"] = {
        name: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1) if len(v) > 1 else 0.0),
               "n": len(v), "min": float(np.min(v))}
        for name, v in top1_by_deg.items()}
    return results


# --------------------------------------------------------------------------
# Part 2: similarity tracks shared-oligo fraction (relatedness / dedup)
# --------------------------------------------------------------------------

def part2_shared_fraction(seeds, fractions):
    import random
    rows = []
    for sd in seeds:
        rng = random.Random(SEED0 + 1000 + sd)
        fwd = "".join(rng.choice(list("ACGT")) for _ in range(20))
        rev = "".join(rng.choice(list("ACGT")) for _ in range(20))

        def mk(n):
            return [fwd + "".join(rng.choice(list("ACGT"))
                                  for _ in range(OLIGO_LEN - 40)) + rev
                    for _ in range(n)]

        base = mk(N_OLIGOS)
        hbase, cbase = fgr2_fingerprint([("o%d" % j, s) for j, s in enumerate(base)],
                                        "frac_base_s%d" % sd, cov_c=1)
        for f in fractions:
            n_shared = int(round(f * N_OLIGOS))
            shared = base[:n_shared]
            distinct = mk(N_OLIGOS - n_shared)
            other = shared + distinct
            rng.shuffle(other)
            hoth, coth = fgr2_fingerprint([("o%d" % j, s) for j, s in enumerate(other)],
                                          "frac_oth_s%d_f%.2f" % (sd, f), cov_c=1)
            # true Jaccard of the oligo k-mer sets
            true_j = fgrlib.jaccard_from_codes(
                fgrlib.fast_canonical_codes("N".join(base), K),
                fgrlib.fast_canonical_codes("N".join(other), K))
            rows.append({"seed": sd, "shared_fraction": f,
                         "true_jaccard": float(true_j),
                         "fp_jaccard_ci": jaccard_ci(hbase, hoth),
                         "fp_cosine_ci": cosine_ci(hbase, hoth, cbase, coth)})
    # correlation + dedup AUC (related f>=0.3 vs unrelated f<=0.1 as a stated cut)
    fj = np.array([r["fp_jaccard_ci"] for r in rows])
    tj = np.array([r["true_jaccard"] for r in rows])
    frac = np.array([r["shared_fraction"] for r in rows])
    pear = float(np.corrcoef(fj, tj)[0, 1])
    from scipy.stats import spearmanr
    spear = float(spearmanr(fj, frac).correlation)
    related = fj[frac >= 0.3].tolist()
    unrelated = fj[frac <= 0.1].tolist()
    dedup_auc = auc(unrelated, related)
    return {"rows": rows, "pearson_fp_vs_true": pear,
            "spearman_fp_vs_fraction": spear,
            "dedup_related_ge0.3_vs_unrelated_le0.1_auc": dedup_auc,
            "by_fraction": {
                str(f): {
                    "mean_true_jaccard": float(np.mean([r["true_jaccard"] for r in rows if r["shared_fraction"] == f])),
                    "mean_fp_jaccard": float(np.mean([r["fp_jaccard_ci"] for r in rows if r["shared_fraction"] == f])),
                    "mean_fp_cosine": float(np.mean([r["fp_cosine_ci"] for r in rows if r["shared_fraction"] == f])),
                    "n": sum(1 for r in rows if r["shared_fraction"] == f),
                } for f in fractions}}


def main():
    seeds = [0, 1, 2, 3, 4]
    degradations = [
        {"name": "mild",     "err": 0.005, "skew": 0.3, "dropout": 0.05, "cfilter": 3},
        {"name": "moderate", "err": 0.01,  "skew": 0.7, "dropout": 0.15, "cfilter": 3},
        {"name": "severe",   "err": 0.02,  "skew": 1.2, "dropout": 0.30, "cfilter": 3},
    ]
    fractions = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]

    out = {
        "experiment_id": "S3_identity_retrieval_dedup",
        "application": "DNA data storage: identity, retrieval, deduplication of stored pools",
        "inputs_are_storage_only": True,
        "input_definition": "every pool is simulated encoded oligos (random payload + shared 20nt primers); no genome",
        "params": {"k": K, "N": N, "oligo_len": OLIGO_LEN, "n_oligos": N_OLIGOS,
                   "m_files": M_FILES, "base_copies": BASE_COPIES, "seeds": seeds,
                   "estimator": "common-index (merged bottom-N)", "seed0": SEED0},
        "part1_retrieval": part1_retrieval(seeds, degradations),
        "part2_shared_fraction": part2_shared_fraction(seeds, fractions),
    }
    p = os.path.join(REPO, "analysis", "results", "s3_identity_retrieval.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", p)
    # console summary
    t = out["part1_retrieval"]["top1_summary"]
    print("\nRetrieval top-1 accuracy (M=%d files):" % M_FILES)
    for name, v in t.items():
        print("  %-9s mean=%.3f  min=%.3f  (n=%d seeds)" % (name, v["mean"], v["min"], v["n"]))
    p2 = out["part2_shared_fraction"]
    print("\nSimilarity vs shared fraction:  Pearson(fp,true)=%.4f  Spearman(fp,frac)=%.4f  dedup AUC=%.3f"
          % (p2["pearson_fp_vs_true"], p2["spearman_fp_vs_fraction"],
             p2["dedup_related_ge0.3_vs_unrelated_le0.1_auc"]))
    for f, v in p2["by_fraction"].items():
        print("  f=%s  true_J=%.3f  fp_J=%.3f  fp_cos=%.3f" %
              (f, v["mean_true_jaccard"], v["mean_fp_jaccard"], v["mean_fp_cosine"]))


if __name__ == "__main__":
    main()
