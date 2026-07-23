#!/usr/bin/env python3
"""
S6b -- Copy/replicate tracking with a CORRECTED PCR model (supersedes S6).

S6 modelled error only at sequencing time, re-drawn per run. That was wrong: a PCR
error is introduced during amplification and is then FIXED in the amplicon
population, so every sequencing run of that amplification sees it, while a
different amplification carries different errors. Likewise strand loss removes
oligos from a particular amplification. Both change WHICH k-mers exist, so a
presence/absence fingerprint is NOT structurally blind to physical-copy identity --
it has genuine discriminative power. Coverage weighting is an ENHANCEMENT in
sensitivity, not an enabler of an otherwise impossible comparison.

Corrected model, per amplification r:
  1. per-oligo copy numbers with PCR skew sigma
  2. strand loss / dropout (an amplification-specific subset is lost)
  3. PCR errors introduced into the amplicon population and INHERITED:
     with probability p_early an oligo acquires a substitution early in
     amplification, carried by a fraction of that oligo's copies (so the variant
     k-mers reach coverage well above the error filter and are specific to r)
  4. per sequencing run t: reads sampled from that fixed amplicon population,
     plus independent per-run sequencing error

within-amplification pair  = two runs of the SAME amplicon population
between-amplification pair = runs of DIFFERENT amplifications of the SAME file

Both fingerprints are compared; the question is how much sensitivity the coverage
annotation adds, and where each metric's limit lies. No genome is used.
"""
import json, os, subprocess, sys, itertools, random
import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import fgrlib

REPO = os.path.dirname(os.path.dirname(SCRIPTS))
FGR2 = os.path.join(REPO, "fgr2")
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/"
           "2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad/s6b")
os.makedirs(SCRATCH, exist_ok=True)

K = 21
N = 10000
OLIGO_LEN = 150
N_OLIGOS = 2000
BASE_COPIES = 15
SEQ_ERR = 0.005          # per-base sequencing error, independent per run
CFILTER = 3
R_REPLICATES = 6
T_RUNS = 3
SEED0 = 20260722
ALTS = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}


def amplify(pool, sigma, dropout_rate, p_early, rng):
    """Return the amplicon population as a list of (sequence, multiplicity).
    PCR errors are introduced HERE and are therefore inherited by every
    sequencing run of this amplification."""
    copies = fgrlib.pcr_bias_copies(len(pool), BASE_COPIES, sigma, rng)
    if dropout_rate > 0:
        copies = fgrlib.dropout_copies(copies, dropout_rate, rng)
    population = []
    for oligo, c in zip(pool, copies):
        if c <= 0:
            continue
        if p_early > 0 and rng.random() < p_early:
            # early-cycle substitution, inherited by a fraction of this oligo's copies
            pos = rng.randrange(len(oligo))
            base = oligo[pos]
            if base in ALTS:
                variant = oligo[:pos] + rng.choice(ALTS[base]) + oligo[pos + 1:]
                n_var = max(1, int(round(c * rng.uniform(0.2, 0.6))))
                population.append((variant, n_var))
                if c - n_var > 0:
                    population.append((oligo, c - n_var))
                continue
        population.append((oligo, c))
    return population


def sequence_population(population, rng):
    """Sample reads from a fixed amplicon population, adding per-run sequencing error."""
    reads = []
    for seq, mult in population:
        for _ in range(mult):
            if SEQ_ERR > 0:
                reads.append("".join(
                    rng.choice(ALTS.get(ch, "ACGT")) if rng.random() < SEQ_ERR else ch
                    for ch in seq))
            else:
                reads.append(seq)
    return reads


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
    return sum(1 for k in S if k in hA and k in hB) / len(S) if S else 1.0


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


def top1(fps, keys, simfn):
    correct = 0
    for q in keys:
        best, best_s = None, -1
        for o in keys:
            if o == q:
                continue
            s = simfn(fps[q], fps[o])
            if s > best_s:
                best_s, best = s, o
        correct += (best[0] == q[0])
    return correct / len(keys)


def run_condition(pool, sigma, dropout_rate, p_early, tag):
    fps = {}
    for rep in range(R_REPLICATES):
        rng_amp = random.Random(hash((SEED0, rep, tag, "amp")) & 0xFFFFFFFF)
        population = amplify(pool, sigma, dropout_rate, p_early, rng_amp)
        for run in range(T_RUNS):
            rng_seq = random.Random(hash((SEED0, rep, run, tag, "seq")) & 0xFFFFFFFF)
            reads = sequence_population(population, rng_seq)
            fps[(rep, run)] = fingerprint(reads, f"{tag}_r{rep}_t{run}")

    keys = sorted(fps)
    wj, bj, wc, bc = [], [], [], []
    for a, b in itertools.combinations(keys, 2):
        ha, ca = fps[a]
        hb, cb = fps[b]
        j = jaccard_ci(ha, hb)
        c = cosine_ci(ha, hb, ca, cb)
        (wj if a[0] == b[0] else bj).append(j)
        (wc if a[0] == b[0] else bc).append(c)
    t1_j = top1(fps, keys, lambda x, y: jaccard_ci(x[0], y[0]))
    t1_c = top1(fps, keys, lambda x, y: cosine_ci(x[0], y[0], x[1], y[1]))
    return {
        "sigma": sigma, "dropout": dropout_rate, "p_early_pcr_error": p_early,
        "jaccard_within_mean": float(np.mean(wj)), "jaccard_between_mean": float(np.mean(bj)),
        "jaccard_within_sd": float(np.std(wj, ddof=1)),
        "jaccard_auc": auc(bj, wj), "jaccard_top1": t1_j,
        "cosine_within_mean": float(np.mean(wc)), "cosine_between_mean": float(np.mean(bc)),
        "cosine_within_sd": float(np.std(wc, ddof=1)),
        "cosine_auc": auc(bc, wc), "cosine_top1": t1_c,
        "chance_top1": (T_RUNS - 1) / (R_REPLICATES * T_RUNS - 1),
        "n_within": len(wj), "n_between": len(bj),
    }


def main():
    rng = random.Random(SEED0)
    pool = fgrlib.make_oligo_pool(N_OLIGOS, OLIGO_LEN, rng)

    # Factorial over the two mechanisms that change WHICH k-mers exist
    # (inherited PCR error, strand loss) and the one that changes abundance (skew).
    conditions = []
    for p_early in (0.0, 0.02, 0.05):
        for sigma, dropout in ((0.0, 0.0), (0.25, 0.0), (0.25, 0.05), (0.5, 0.05)):
            conditions.append((sigma, dropout, p_early))

    out = {
        "experiment_id": "S6b_pcr_inheritance_replicate_tracking",
        "supersedes": "S6 (which modelled error only at sequencing time, not inherited)",
        "application": "DNA data storage: tracking PCR amplifications / physical copies",
        "inputs_are_storage_only": True,
        "model_note": ("PCR errors are introduced into the amplicon population and inherited "
                       "by all sequencing runs of that amplification; strand loss is "
                       "amplification-specific. Both alter the k-mer SET, so presence/absence "
                       "has genuine discriminative power; coverage weighting is tested as a "
                       "sensitivity enhancement."),
        "params": {"k": K, "N": N, "n_oligos": N_OLIGOS, "oligo_len": OLIGO_LEN,
                   "base_copies": BASE_COPIES, "seq_err": SEQ_ERR, "cfilter": CFILTER,
                   "R": R_REPLICATES, "T": T_RUNS, "seed0": SEED0},
        "conditions": [],
    }
    for sigma, dropout, p_early in conditions:
        tag = f"s{sigma}_d{dropout}_e{p_early}"
        print(f"running sigma={sigma} dropout={dropout} p_early={p_early} ...", flush=True)
        out["conditions"].append(run_condition(pool, sigma, dropout, p_early, tag))

    p = os.path.join(REPO, "analysis", "results", "s6b_pcr_inheritance.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", p)
    print(f"\n{'sigma':>6s} {'drop':>5s} {'pPCRerr':>8s} | {'jacAUC':>7s} {'jacTop1':>8s} | "
          f"{'cosAUC':>7s} {'cosTop1':>8s} | {'chance':>6s}")
    for r in out["conditions"]:
        print(f"{r['sigma']:6.2f} {r['dropout']:5.2f} {r['p_early_pcr_error']:8.2f} | "
              f"{r['jaccard_auc']:7.3f} {r['jaccard_top1']:8.3f} | "
              f"{r['cosine_auc']:7.3f} {r['cosine_top1']:8.3f} | {r['chance_top1']:6.3f}")


if __name__ == "__main__":
    main()
