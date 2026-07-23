#!/usr/bin/env python3
"""
E6 part 1: fgr2 throughput, size scaling, thread scaling, memory behaviour.

Every number written by this script comes from an actual /usr/bin/time -l
measurement of an actual fgr2 invocation. Peak RSS is in BYTES (macOS).
"""
import os
import sys
import json
import glob
import argparse
import statistics

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import e6_common as C  # noqa: E402

K = 31
N = 10000


def bench_size_ladder(fgr2, manifest, workdir, nreps):
    """Wall-clock + peak RSS vs input size, at fixed -t 4 (fgr2 default)."""
    out = []
    for entry in manifest["ladder"]:
        path = entry["path"]
        sk = os.path.join(workdir, f"ladder_{entry['target_mb']}mb.fgr2")
        cmd = [fgr2, "-k", str(K), "-N", str(N), "-c", "1", "-t", "4",
               "-o", sk, path]
        # A larger input takes longer; keep the warmup for the small points
        # where cache state dominates, drop it for the very large ones.
        warm = 1 if entry["target_mb"] <= 200 else 0
        res = C.repeat_timed(cmd, n=nreps, warmup=warm,
                             label=f"fgr2_size_{entry['target_mb']}mb")
        distinct, bins = C.hist_distinct_kmers(sk + ".hist")
        res.update({
            "target_mb": entry["target_mb"],
            "bases": entry["bases"],
            "file_bytes": entry["file_bytes"],
            "contains_mutated_copies": entry["contains_mutated_copies"],
            "distinct_canonical_kmers": distinct,
            "hist_bins": bins,
            "sketch_bytes": os.path.getsize(sk),
            "threads": 4,
        })
        res["throughput_mbp_per_s"] = (entry["bases"] / 1e6) / res["wall_s_median"]
        out.append(res)
        print(f"[size] {entry['target_mb']:>4} Mb  "
              f"median {res['wall_s_median']:7.3f} s  "
              f"({res['throughput_mbp_per_s']:6.1f} Mbp/s)  "
              f"RSS {res['max_rss_bytes_median']/1e9:6.3f} GB  "
              f"distinct {distinct:,}", flush=True)
    return out


def bench_cold_vs_warm(fgr2, path, workdir, nreps):
    """First-touch (cold-ish) vs repeated (warm) runs on the same input.

    macOS `purge` needs root, so a true cold page cache is not achievable here.
    Instead we compare run #1 of a freshly-written copy of the input against
    subsequent runs. This bounds, rather than isolates, the I/O contribution.
    """
    import shutil
    sk = os.path.join(workdir, "coldwarm.fgr2")
    cold, warm = [], []
    for rep in range(nreps):
        cp = os.path.join(workdir, f"coldcopy_{rep}.fa")
        shutil.copyfile(path, cp)
        cmd = [fgr2, "-k", str(K), "-N", str(N), "-c", "1", "-t", "4",
               "-o", sk, cp]
        cold.append(C.run_timed(cmd)["wall_s"])
        warm.append(C.run_timed(cmd)["wall_s"])
        os.remove(cp)
    return {
        "input": path,
        "note": ("macOS purge requires root; 'cold' = first run on a "
                 "freshly-copied file, which is still likely to be in the "
                 "unified buffer cache. Treat as a weak upper bound on I/O cost."),
        "cold_wall_s_all": cold,
        "warm_wall_s_all": warm,
        "cold_median": statistics.median(cold),
        "warm_median": statistics.median(warm),
    }


def bench_threads(fgr2, path, workdir, nreps, thread_list):
    out = []
    for t in thread_list:
        sk = os.path.join(workdir, f"threads_t{t}.fgr2")
        cmd = [fgr2, "-k", str(K), "-N", str(N), "-c", "1", "-t", str(t),
               "-o", sk, path]
        res = C.repeat_timed(cmd, n=nreps, warmup=1, label=f"fgr2_t{t}")
        res["threads"] = t
        res["input"] = path
        # user+sys / real is a direct measure of achieved parallelism
        res["cpu_utilisation"] = statistics.median(
            [(r["user_s"] + r["sys_s"]) / r["wall_s"] for r in res["runs"]])
        out.append(res)
        print(f"[threads] -t {t:>2}  median {res['wall_s_median']:7.3f} s  "
              f"cpu-util {res['cpu_utilisation']:5.2f}x  "
              f"RSS {res['max_rss_bytes_median']/1e9:6.3f} GB", flush=True)
    base = out[0]["wall_s_median"]
    for r in out:
        r["speedup_vs_t1"] = base / r["wall_s_median"]
        r["parallel_efficiency"] = r["speedup_vs_t1"] / r["threads"]
    return out


def bench_prefix(fgr2, path, workdir, nreps, plist):
    out = []
    for p in plist:
        sk = os.path.join(workdir, f"prefix_p{p}.fgr2")
        cmd = [fgr2, "-k", str(K), "-N", str(N), "-c", "1", "-t", "4",
               "-p", str(p), "-o", sk, path]
        try:
            res = C.repeat_timed(cmd, n=nreps, warmup=1, label=f"fgr2_p{p}")
        except RuntimeError as e:
            print(f"[prefix] -p {p} FAILED: {e}", flush=True)
            out.append({"prefix": p, "failed": True, "error": str(e)[:500]})
            continue
        res["prefix"] = p
        res["input"] = path
        out.append(res)
        print(f"[prefix] -p {p:>2}  median {res['wall_s_median']:7.3f} s  "
              f"RSS {res['max_rss_bytes_median']/1e9:6.3f} GB", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fgr2", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nreps", type=int, default=5)
    ap.add_argument("--gate-max-wait", type=float, default=2700.0)
    ap.add_argument("--load-threshold", type=float, default=3.0,
                    help="wait until 1-min load average is below this before "
                         "each timed phase (shared machine)")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    manifest = json.load(open(args.manifest))

    results = {
        "experiment": "E6_scaling",
        "k": K, "N": N,
        "rss_unit": "BYTES (macOS /usr/bin/time -l), verified in-session",
        "nreps": args.nreps,
        "fgr2_binary": args.fgr2,
        "input_manifest": args.manifest,
        "input_seed": manifest["seed"],
    }

    results["idle_gate"] = {}
    results["idle_gate"]["size_ladder"] = C.wait_for_idle(args.load_threshold, max_wait_s=args.gate_max_wait)
    print("== size ladder ==", flush=True)
    results["size_ladder"] = bench_size_ladder(args.fgr2, manifest,
                                               args.workdir, args.nreps)

    # thread scaling on a fixed large input (200 Mb keeps 6 runs x 5 thread
    # settings tractable while still being memory-bandwidth relevant)
    big = [e for e in manifest["ladder"] if e["target_mb"] == 200][0]["path"]
    results["idle_gate"]["threads"] = C.wait_for_idle(args.load_threshold, max_wait_s=args.gate_max_wait)
    print("== thread scaling (200 Mb) ==", flush=True)
    results["thread_scaling"] = bench_threads(args.fgr2, big, args.workdir,
                                              args.nreps, [1, 2, 4, 8, 16])

    results["idle_gate"]["prefix"] = C.wait_for_idle(args.load_threshold, max_wait_s=args.gate_max_wait)
    print("== prefix length -p (200 Mb) ==", flush=True)
    results["prefix_scaling"] = bench_prefix(args.fgr2, big, args.workdir,
                                             args.nreps, [10, 12, 14, 16, 18])

    med = [e for e in manifest["ladder"] if e["target_mb"] == 50][0]["path"]
    results["idle_gate"]["cold_warm"] = C.wait_for_idle(args.load_threshold, max_wait_s=args.gate_max_wait)
    print("== cold vs warm (50 Mb) ==", flush=True)
    results["cold_vs_warm"] = bench_cold_vs_warm(args.fgr2, med,
                                                 args.workdir, 5)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
