#!/usr/bin/env python3
"""
E6 part 3: practical ceiling of the downstream comparison step.

calculate_similarity.py does all-vs-all similarity (O(n^2) pairs) and then
optionally Neighbor-Joining (O(n^3)). We measure wall-clock as a function of
sample count n, separately for:
   * pairwise only          (--no-plot_trees --no-nj_tree)
   * pairwise + NJ, 0 bootstrap replicates
   * pairwise + NJ, 100 bootstrap replicates (the tool's default)
so the O(n^2) and O(n^3) terms can be separated, and the bootstrap multiplier
on the NJ term is visible.

Samples are produced by replicating real fgr2 sketches from the 16-genome panel
with distinct filenames. NOTE: beyond n=16 the sketches repeat, so the *content*
is not 16 independent genomes. That does not affect the cost model being
measured (the work per pair is identical regardless of sketch content), but it
is recorded here and in the caveats.
"""
import os
import sys
import glob
import json
import shutil
import argparse

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import e6_common as C  # noqa: E402


def build_panel(src_dir, dst_dir, n):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    srcs = sorted(glob.glob(os.path.join(src_dir, "*.fgr2")))
    assert srcs, f"no .fgr2 sketches in {src_dir}"
    for i in range(n):
        s = srcs[i % len(srcs)]
        base = os.path.splitext(os.path.basename(s))[0]
        shutil.copyfile(s, os.path.join(dst_dir, f"s{i:04d}_{base}.fgr2"))
    return len(srcs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sketchdir", required=True,
                    help="dir of real fgr2 sketches to replicate")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nreps", type=int, default=5)
    ap.add_argument("--gate-max-wait", type=float, default=2700.0)
    ap.add_argument("--load-threshold", type=float, default=3.0)
    ap.add_argument("--per-run-timeout", type=float, default=900.0,
                    help="per-invocation wall-clock budget; exceeding it "
                         "censors that point and stops that mode")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    cs = os.path.join(args.repo, "calculate_similarity.py")

    counts = [4, 8, 16, 32, 48, 64, 96, 128, 160, 200]
    modes = {
        "pairwise_only": ["--no-plot_trees", "--no-nj_tree"],
        "nj_boot0": ["--no-plot_trees", "--bootstrap_replicates", "0"],
        "nj_boot100": ["--no-plot_trees", "--bootstrap_replicates", "100"],
    }

    res = {
        "experiment": "E6_downstream",
        "script": cs,
        "nreps": args.nreps,
        "counts": counts,
        "note": ("samples above n=16 are byte-identical replicates of the "
                 "16-genome panel sketches; cost per pair is independent of "
                 "sketch content so the scaling exponent is unaffected"),
        "n_unique_sketches": None,
        "modes": {},
    }

    gate = C.wait_for_idle(args.load_threshold, max_wait_s=args.gate_max_wait)
    res["idle_gate"] = gate

    for mode, flags in modes.items():
        res["modes"][mode] = []
        for n in counts:
            d = os.path.join(args.workdir, f"panel_{n}")
            nuniq = build_panel(args.sketchdir, d, n)
            res["n_unique_sketches"] = nuniq
            cmd = [sys.executable, cs, d, "-o", os.path.join(d, "out"),
                   "--seed", "1"] + flags
            # NJ with 100 bootstraps at n=200 is the expensive corner; use
            # fewer reps there but never fewer than 3, and record n_runs.
            reps = args.nreps
            if mode == "nj_boot100" and n >= 96:
                reps = 3
            try:
                r = C.repeat_timed(cmd, n=reps, warmup=0,
                                   label=f"{mode}_n{n}",
                                   timeout=args.per_run_timeout)
            except C.RunTimeout as e:
                print(f"[{mode}] n={n} TIMEOUT (>{args.per_run_timeout}s) -- "
                      f"recording as censored, larger n not attempted",
                      flush=True)
                res["modes"][mode].append({
                    "n_samples": n, "timed_out": True,
                    "timeout_s": args.per_run_timeout, "error": str(e)[:500]})
                shutil.rmtree(d, ignore_errors=True)
                break  # monotonic in n: everything larger will also time out
            except RuntimeError as e:
                print(f"[{mode}] n={n} FAILED: {str(e)[:300]}", flush=True)
                res["modes"][mode].append({"n_samples": n, "failed": True,
                                           "error": str(e)[:1500]})
                shutil.rmtree(d, ignore_errors=True)
                continue
            r["n_samples"] = n
            r["n_pairs"] = n * (n - 1) // 2
            res["modes"][mode].append(r)
            print(f"[{mode}] n={n:>4}  median {r['wall_s_median']:8.3f} s  "
                  f"RSS {r['max_rss_bytes_median']/1e6:8.1f} MB", flush=True)
            shutil.rmtree(d, ignore_errors=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
