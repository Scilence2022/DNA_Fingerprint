#!/usr/bin/env python3
"""
S5(c) -- Fingerprinting throughput and memory footprint on ENCODED POOLS.

We build encoded-pool read sets spanning >2 orders of magnitude of input bases
by scaling the number of oligos in the pool (a bigger archive = more distinct
encoded data), at a fixed per-oligo copy depth. Each input is fingerprinted with
fgr2 (-c 1, no coverage filtering, -t 4) and we record:
  * wall-clock (median of >=3 timed reps after a warm-up), via /usr/bin/time -l
  * peak RSS in BYTES (macOS), via /usr/bin/time -l
  * distinct canonical k-mers (sum of the .hist bins)

Two questions:
  (i)  runtime vs input bases -- fit a log-log slope; slope ~1 => linear.
  (ii) peak RSS vs distinct k-mers -- the memory model of the sketch build.

Inputs are generated with numpy for speed. Reads are error-free encoded copies:
fingerprinting cost depends on sequence volume and distinct-k-mer count, not on
whether a k-mer is a "true" or "error" k-mer, so error-free copies are a clean,
faithful way to sweep both axes. RSS unit (BYTES on macOS) is verified in E6.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import e6_common as C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(SCRIPTS))
FGR2 = os.path.join(REPO, "fgr2")
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/"
           "2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad")

K = 31
N = 10000
OLIGO_LEN = 200
COPIES = 4
PRIMER = 20
# n_oligos ladder: >2 orders of magnitude of input bases
LADDER = [1000, 3000, 10000, 30000, 100000, 300000]
_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)


def write_pool_reads_fasta(path, n_oligos, oligo_len, copies, seed):
    """Fast numpy generator: random payloads with shared primer sites, each
    oligo emitted `copies` times, error-free. Returns total sequence bases."""
    rng = np.random.default_rng(seed)
    payload_len = oligo_len - 2 * PRIMER
    fwd = _BASES[rng.integers(0, 4, PRIMER)]
    rev = _BASES[rng.integers(0, 4, PRIMER)]
    total_bases = 0
    with open(path, "wb") as fh:
        ridx = 0
        BATCH = 5000
        for start in range(0, n_oligos, BATCH):
            m = min(BATCH, n_oligos - start)
            payloads = _BASES[rng.integers(0, 4, size=(m, payload_len))]
            chunk = []
            for j in range(m):
                oligo = np.concatenate([fwd, payloads[j], rev]).tobytes()
                for _ in range(copies):
                    chunk.append(b">r%d\n" % ridx)
                    chunk.append(oligo)
                    chunk.append(b"\n")
                    ridx += 1
                    total_bases += oligo_len
            fh.write(b"".join(chunk))
    return total_bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nreps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        REPO, "analysis", "results", "s5c_throughput.json"))
    a = ap.parse_args()

    workdir = os.path.join(SCRATCH, "s5c")
    os.makedirs(workdir, exist_ok=True)

    ladder = []
    for n_oligos in LADDER:
        fa = os.path.join(workdir, f"pool_{n_oligos}.fa")
        t0 = time.time()
        bases = write_pool_reads_fasta(fa, n_oligos, OLIGO_LEN, COPIES,
                                       seed=42 + n_oligos)
        fbytes = os.path.getsize(fa)
        sk = os.path.join(workdir, f"pool_{n_oligos}.fgr2")
        cmd = [FGR2, "-k", str(K), "-N", str(N), "-c", "1",
               "-t", str(a.threads), "-o", sk, fa]
        warm = 1 if n_oligos <= 100000 else 0
        res = C.repeat_timed(cmd, n=a.nreps, warmup=warm,
                             label=f"s5c_{n_oligos}")
        distinct, bins = C.hist_distinct_kmers(sk + ".hist")
        res.update({
            "n_oligos": n_oligos, "oligo_len": OLIGO_LEN, "copies": COPIES,
            "bases": bases, "file_bytes": fbytes,
            "distinct_canonical_kmers": distinct,
            "n_hist_bins": len(bins),
            "sketch_bytes": os.path.getsize(sk),
            "threads": a.threads,
            "gen_wall_s": round(time.time() - t0, 1),
        })
        res["throughput_mbp_per_s"] = (bases / 1e6) / res["wall_s_median"]
        ladder.append(res)
        print(f"[size] {n_oligos:>7} oligos  {bases/1e6:8.1f} Mbp  "
              f"median {res['wall_s_median']:8.3f}s  "
              f"({res['throughput_mbp_per_s']:6.1f} Mbp/s)  "
              f"RSS {res['max_rss_bytes_median']/1e9:6.3f} GB  "
              f"distinct {distinct:,}", flush=True)
        os.remove(fa)
        if os.path.exists(sk):
            os.remove(sk)
        if os.path.exists(sk + ".hist"):
            os.remove(sk + ".hist")

    # ---- fits ----
    bases = np.array([r["bases"] for r in ladder], dtype=float)
    walls = np.array([r["wall_s_median"] for r in ladder], dtype=float)
    rss = np.array([r["max_rss_bytes_median"] for r in ladder], dtype=float)
    distinct = np.array([r["distinct_canonical_kmers"] for r in ladder],
                        dtype=float)

    # runtime vs bases: log-log slope (== power-law exponent)
    lb, lw = np.log10(bases), np.log10(walls)
    A = np.vstack([lb, np.ones_like(lb)]).T
    slope, intercept = np.linalg.lstsq(A, lw, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_res = float(((lw - pred) ** 2).sum())
    ss_tot = float(((lw - lw.mean()) ** 2).sum())
    r2_runtime = 1 - ss_res / ss_tot if ss_tot else float("nan")

    # linear runtime vs bases (throughput constancy)
    tput = bases / walls / 1e6
    # memory vs distinct k-mers: linear fit RSS = a*distinct + b
    Am = np.vstack([distinct, np.ones_like(distinct)]).T
    (m_slope, m_int), _, _, _ = np.linalg.lstsq(Am, rss, rcond=None)
    mpred = Am @ np.array([m_slope, m_int])
    mss_res = float(((rss - mpred) ** 2).sum())
    mss_tot = float(((rss - rss.mean()) ** 2).sum())
    r2_mem = 1 - mss_res / mss_tot if mss_tot else float("nan")

    fits = {
        "runtime_vs_bases_loglog": {
            "slope": float(slope), "intercept_log10": float(intercept),
            "r2": r2_runtime,
            "interpretation": "slope ~1 => runtime linear in input bases",
        },
        "throughput_mbp_per_s": {
            "median": float(np.median(tput)), "min": float(tput.min()),
            "max": float(tput.max()), "per_point": tput.tolist(),
        },
        "memory_vs_distinct_kmers_linear": {
            "bytes_per_distinct_kmer": float(m_slope),
            "intercept_bytes": float(m_int), "r2": r2_mem,
        },
        "span": {
            "bases_min": float(bases.min()), "bases_max": float(bases.max()),
            "bases_orders_of_magnitude": float(np.log10(bases.max()
                                                        / bases.min())),
            "distinct_min": float(distinct.min()),
            "distinct_max": float(distinct.max()),
        },
    }

    results = {
        "experiment": "S5c_throughput_and_footprint_on_encoded_pools",
        "input_type": "encoded_oligo_pool_reads_errorfree",
        "k": K, "N": N, "oligo_len": OLIGO_LEN, "copies_per_oligo": COPIES,
        "threads": a.threads, "nreps": a.nreps,
        "rss_unit": "BYTES (macOS /usr/bin/time -l), verified in E6",
        "fgr2_binary": FGR2,
        "n_oligos_ladder": LADDER,
        "ladder": ladder,
        "fits": fits,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[done] -> {a.out}", flush=True)
    print(f"runtime log-log slope={slope:.3f} (r2={r2_runtime:.4f}); "
          f"mem {m_slope:.1f} bytes/distinct-kmer (r2={r2_mem:.4f}); "
          f"throughput median {np.median(tput):.1f} Mbp/s", flush=True)


if __name__ == "__main__":
    main()
