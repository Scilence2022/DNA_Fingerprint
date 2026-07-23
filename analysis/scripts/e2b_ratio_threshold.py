#!/usr/bin/env python3
"""E2 supplement: fine size-ratio sweep to locate the ratio at which the
direct estimator's absolute bias crosses 0.01, plus an N-sweep for the
variance-vs-theory comparison. Reuses the validated machinery in e2_jaccard_bias."""
import json, math, os, random, sys, time
import numpy as np

SCRIPTS = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/scripts"
RESULTS = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/results"
sys.path.insert(0, SCRIPTS)
import fgrlib
import e2_jaccard_bias as E

K = 31
BASE_LEN = 100_000
FINE_RATIOS = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
FINE_RATES = [0.0, 0.002, 0.01]
NREP = 30
NS = [1000, 10000]
N_SWEEP = [100, 1000, 10000, 100000]
SEED = 20260721


def cell(args):
    ratio, rate, rep, Ns = args
    seed = (SEED * 7919 + hash((round(ratio, 4), rate, rep, tuple(Ns))) % 10**9) % (2**31)
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed + 11)
    core = E.rand_seq(rng_np, BASE_LEN)
    b_core = fgrlib.mutate(core, rate, rng_py)
    extra = int(round(BASE_LEN * (ratio - 1.0)))
    b_seq = b_core + ("N" + E.rand_seq(rng_np, extra) if extra > 0 else "")
    ca = fgrlib.fast_canonical_codes(core, K)
    cb = fgrlib.fast_canonical_codes(b_seq, K)
    jt = fgrlib.jaccard_from_codes(ca, cb)
    rec = {"ratio": ratio, "rate": rate, "rep": rep, "seed": seed, "j_true": jt,
           "n_kmers_a": int(ca.size), "n_kmers_b": int(cb.size),
           "size_ratio_realised": float(cb.size) / float(ca.size)}
    for n in Ns:
        sa, ha = E.bottom_n(ca, n)
        sb, hb = E.bottom_n(cb, n)
        rec[f"j_direct_N{n}"] = E.j_direct_codes(sa, sb)
        rec[f"j_union_N{n}"] = E.j_union_codes(sa, ha, sb, hb, n)
    return rec


def summarise(recs, Ns):
    cells = {}
    for r in recs:
        cells.setdefault((r["ratio"], r["rate"]), []).append(r)
    out = []
    for (ra, rt), rs in sorted(cells.items()):
        jt = np.array([r["j_true"] for r in rs])
        row = {"ratio": ra, "rate": rt, "n_rep": len(rs),
               "j_true_mean": float(jt.mean()), "j_true_sd": float(jt.std(ddof=1))}
        for n in Ns:
            for est in ("direct", "union"):
                e = np.array([r[f"j_{est}_N{n}"] for r in rs]) - jt
                row[f"{est}_N{n}_bias"] = float(e.mean())
                row[f"{est}_N{n}_bias_se"] = float(e.std(ddof=1) / math.sqrt(len(rs)))
                row[f"{est}_N{n}_rmse"] = float(np.sqrt((e ** 2).mean()))
                row[f"{est}_N{n}_resid_sd"] = float(e.std(ddof=1))
                row[f"{est}_N{n}_binom_sd"] = float(
                    math.sqrt(max(jt.mean() * (1 - jt.mean()), 0) / n))
        out.append(row)
    return out


def main():
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(r, m, rep, NS) for r in FINE_RATIOS for m in FINE_RATES for rep in range(NREP)]
    with Pool(14) as p:
        fine = p.map(cell, jobs, chunksize=1)
    sys.stderr.write(f"fine sweep {len(fine)} reps in {time.time()-t0:.1f}s\n")

    t0 = time.time()
    jobs2 = [(r, 0.005, rep, N_SWEEP) for r in (1.0, 1.2, 2.0) for rep in range(NREP)]
    with Pool(14) as p:
        nsw = p.map(cell, jobs2, chunksize=1)
    sys.stderr.write(f"N sweep {len(nsw)} reps in {time.time()-t0:.1f}s\n")

    payload = {"experiment": "E2b_size_ratio_threshold_and_N_sweep",
               "k": K, "base_len": BASE_LEN, "master_seed": SEED,
               "fine_ratio_sweep": {"ratios": FINE_RATIOS, "rates": FINE_RATES,
                                    "n_replicates": NREP, "N_values": NS,
                                    "replicates": fine, "cell_summary": summarise(fine, NS)},
               "N_sweep": {"ratios": [1.0, 1.2, 2.0], "rate": 0.005, "n_replicates": NREP,
                           "N_values": N_SWEEP, "replicates": nsw,
                           "cell_summary": summarise(nsw, N_SWEEP)}}
    path = os.path.join(RESULTS, "e2b_ratio_threshold.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    sys.stderr.write(f"wrote {path}\n")


if __name__ == "__main__":
    main()
