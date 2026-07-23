#!/usr/bin/env python3
"""
E6 part 2: fgr2 vs mash 2.3 vs sourmash 4.9.4, head to head.

Fairness rules applied here, all stated verbatim in the output JSON:
  * identical input files for all three tools
  * identical k (31)
  * matched sketch size: fgr2 -N 10000, mash -s 10000, sourmash num=10000
    (all three are bottom-N / MinHash sketches of the same cardinality)
  * thread count matched at 1 for the primary comparison, since sourmash's
    sketch command is single-threaded; fgr2 and mash are additionally reported
    at 4 threads as a secondary row
  * no coverage filtering in the primary comparison (fgr2 -c 1) so that the
    three tools compute the same mathematical object; fgr2's auto coverage
    filtering is measured separately on the read sets, which is the setting
    where it is actually meant to help

Known ways this is NOT apples-to-apples (also recorded in the JSON):
  * fgr2 writes a plain-text sketch (k-mer string + hash + coverage per line);
    mash writes a compact binary .msh; sourmash writes gzipped JSON. Disk sizes
    therefore differ by a large constant factor that reflects serialisation
    format, not sketch content.
  * fgr2 additionally computes and writes a full coverage histogram and
    per-k-mer coverage, which mash -s and sourmash num sketches do not carry.
    Some of fgr2's runtime and memory buys that extra information.
  * mash and sourmash apply their own default filters; flags used are recorded.
"""
import os
import re
import sys
import glob
import json
import shutil
import argparse
import itertools
import statistics

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import e6_common as C  # noqa: E402
import fgrlib  # noqa: E402

K = 31
SKETCH_N = 10000
MASH = "/opt/homebrew/bin/mash"
SOURMASH = shutil.which("sourmash") or "/opt/anaconda3/bin/sourmash"


# ---------------------------------------------------------------- sketching

def sketch_fgr2(fgr2, inp, out, threads, cov):
    cmd = [fgr2, "-k", str(K), "-N", str(SKETCH_N), "-t", str(threads)]
    if cov is not None:
        cmd += ["-c", str(cov)]
    cmd += ["-o", out, inp]
    return cmd


def sketch_mash(inp, out, threads, mincopy=None, reads=False):
    cmd = [MASH, "sketch", "-k", str(K), "-s", str(SKETCH_N),
           "-p", str(threads), "-o", out]
    if reads:
        # -r tells mash the input is reads (enables its own error filtering
        # model); -m sets the minimum k-mer copy count.
        cmd += ["-r"]
        if mincopy is not None:
            cmd += ["-m", str(mincopy)]
    cmd += [inp]
    return cmd


def sketch_sourmash(inp, out, abund=False):
    param = f"k={K},num={SKETCH_N}," + ("abund" if abund else "noabund")
    return [SOURMASH, "sketch", "dna", "-p", param, "-o", out, inp]


def bench_sketching(fgr2, inputs, workdir, nreps, mode):
    """mode: 'genome' or 'reads'. Returns list of per-input, per-tool records."""
    recs = []
    for inp in inputs:
        base = os.path.splitext(os.path.basename(inp))[0]
        insize = os.path.getsize(inp)
        row = {"input": inp, "input_bytes": insize, "mode": mode, "tools": {}}

        variants = []
        # --- primary, matched, single threaded, no coverage filtering -------
        o1 = os.path.join(workdir, f"{base}.t1.fgr2")
        variants.append(("fgr2_t1_c1", sketch_fgr2(fgr2, inp, o1, 1, 1), o1))
        o4 = os.path.join(workdir, f"{base}.t4.fgr2")
        variants.append(("fgr2_t4_c1", sketch_fgr2(fgr2, inp, o4, 4, 1), o4))

        m1 = os.path.join(workdir, f"{base}.t1.mash")
        variants.append(("mash_t1", sketch_mash(inp, m1, 1), m1 + ".msh"))
        m4 = os.path.join(workdir, f"{base}.t4.mash")
        variants.append(("mash_t4", sketch_mash(inp, m4, 4), m4 + ".msh"))

        s1 = os.path.join(workdir, f"{base}.sourmash.sig")
        variants.append(("sourmash_t1", sketch_sourmash(inp, s1), s1))

        if mode == "reads":
            # fgr2's headline capability: automatic coverage-threshold detection
            oa = os.path.join(workdir, f"{base}.t1.auto.fgr2")
            variants.append(("fgr2_t1_autocov",
                             sketch_fgr2(fgr2, inp, oa, 1, None), oa))
            # mash's comparable read-mode filter
            mr = os.path.join(workdir, f"{base}.t1.mashreads")
            variants.append(("mash_t1_reads_m2",
                             sketch_mash(inp, mr, 1, mincopy=2, reads=True),
                             mr + ".msh"))
            # sourmash with abundance tracking (the nearest analogue to
            # fgr2 retaining per-k-mer coverage)
            sa = os.path.join(workdir, f"{base}.sourmash.abund.sig")
            variants.append(("sourmash_t1_abund",
                             sketch_sourmash(inp, sa, abund=True), sa))

        # Interleave the reps across tools so background-load drift is
        # common-mode rather than favouring whichever tool ran first.
        summaries = C.repeat_timed_interleaved(
            [(name, cmd) for name, cmd, _ in variants], n=nreps, warmup=1)
        for name, cmd, outpath in variants:
            res = summaries[name]
            if res.get("failed"):
                print(f"  {name:22s} FAILED: {res['error'][:200]}", flush=True)
                row["tools"][name] = res
                continue
            res["sketch_bytes_on_disk"] = (os.path.getsize(outpath)
                                           if os.path.exists(outpath) else None)
            res["sketch_path"] = outpath
            row["tools"][name] = res
            db = res["sketch_bytes_on_disk"]
            print(f"  {name:22s} {res['wall_s_median']:7.3f} s  "
                  f"RSS {res['max_rss_bytes_median']/1e6:8.1f} MB  "
                  f"disk {db if db is not None else -1:>10,} B  "
                  f"load {res['loadavg_1min_median_during']:6.1f}", flush=True)
        recs.append(row)
        print(f"[{mode}] {base} done", flush=True)
    return recs


# ---------------------------------------------------------------- accuracy

def exact_jaccard_matrix(genome_paths):
    """Ground truth: exact Jaccard over the full canonical k-mer sets."""
    codes = {}
    for p in genome_paths:
        codes[p] = fgrlib.fast_canonical_codes(fgrlib.load_genome(p), K)
        print(f"  exact codes {os.path.basename(p)}: {len(codes[p]):,}",
              flush=True)
    out = {}
    for a, b in itertools.combinations(genome_paths, 2):
        out[f"{os.path.basename(a)}|{os.path.basename(b)}"] = \
            fgrlib.jaccard_from_codes(codes[a], codes[b])
    return out


def parse_mash_dist(text):
    out = {}
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        ref, qry, dist, pv, shared = parts[:5]
        num, den = shared.split("/")
        j = int(num) / int(den)
        out[(os.path.basename(ref), os.path.basename(qry))] = {
            "mash_distance": float(dist), "jaccard": j, "shared": shared}
    return out


def bench_allvsall(fgr2, genome_paths, workdir, nreps, repo):
    """All-vs-all distance computation time on the 16-genome panel."""
    import subprocess
    res = {}

    # ---- mash: paste sketches then dist -------------------------------
    msh_dir = os.path.join(workdir, "avaa_mash")
    os.makedirs(msh_dir, exist_ok=True)
    sk = []
    for p in genome_paths:
        b = os.path.splitext(os.path.basename(p))[0]
        o = os.path.join(msh_dir, b)
        subprocess.run(sketch_mash(p, o, 4), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sk.append(o + ".msh")
    combined = os.path.join(msh_dir, "combined")
    subprocess.run([MASH, "paste", combined] + sk, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dist_cmd = [MASH, "dist", "-p", "1", combined + ".msh", combined + ".msh"]

    # ---- sourmash compare ---------------------------------------------
    sm_dir = os.path.join(workdir, "avaa_sourmash")
    os.makedirs(sm_dir, exist_ok=True)
    sigs = []
    for p in genome_paths:
        b = os.path.splitext(os.path.basename(p))[0]
        o = os.path.join(sm_dir, b + ".sig")
        subprocess.run(sketch_sourmash(p, o), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sigs.append(o)
    csv = os.path.join(sm_dir, "cmp.csv")
    cmp_cmd = [SOURMASH, "compare", "-k", str(K), "--csv", csv] + sigs

    # ---- fgr2 sketches + calculate_similarity.py ----------------------
    fg_dir = os.path.join(workdir, "avaa_fgr2")
    os.makedirs(fg_dir, exist_ok=True)
    for p in genome_paths:
        b = os.path.splitext(os.path.basename(p))[0]
        subprocess.run(sketch_fgr2(fgr2, p, os.path.join(fg_dir, b + ".fgr2"), 4, 1),
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    cs = os.path.join(repo, "calculate_similarity.py")
    cs_cmd = [sys.executable, cs, fg_dir, "-o", os.path.join(fg_dir, "sim"),
              "--no-plot_trees", "--no-nj_tree", "--seed", "1"]

    # time all three distance steps interleaved
    summ = C.repeat_timed_interleaved(
        [("mash_dist", dist_cmd),
         ("sourmash_compare", cmp_cmd),
         ("calculate_similarity", cs_cmd)], n=nreps, warmup=1)
    res.update(summ)

    # capture the actual distance values (untimed)
    mash_out = subprocess.run(dist_cmd, capture_output=True, check=True)
    res["mash_dist"]["parsed"] = {f"{a}|{b}": v for (a, b), v in
                                  parse_mash_dist(mash_out.stdout.decode()).items()}
    subprocess.run(cmp_cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    with open(csv) as fh:
        hdr = fh.readline().strip().split(",")
        mat = [[float(x) for x in ln.strip().split(",")] for ln in fh if ln.strip()]
    res["sourmash_compare"]["labels"] = hdr
    res["sourmash_compare"]["matrix"] = mat
    res["fgr2_sketch_dir"] = fg_dir
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fgr2", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--genomes", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nreps", type=int, default=5)
    ap.add_argument("--gate-max-wait", type=float, default=2700.0)
    ap.add_argument("--load-threshold", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    manifest = json.load(open(args.manifest))
    genome_paths = sorted(glob.glob(os.path.join(args.genomes, "*.fna")))

    out = {
        "experiment": "E6_tools",
        "k": K, "sketch_size": SKETCH_N,
        "rss_unit": "BYTES (macOS /usr/bin/time -l)",
        "nreps": args.nreps,
        "versions": {"fgr2": "2.1.0", "mash": "2.3", "sourmash": "4.9.4"},
        "fairness_notes": __doc__,
    }

    # Genome-mode sketching: use a 4-genome subset to keep runtime sane, plus
    # record which ones.
    gsub = genome_paths[:4]
    out["idle_gate"] = {"genomes": C.wait_for_idle(args.load_threshold, max_wait_s=args.gate_max_wait)}
    print("== sketching: assembled genomes ==", flush=True)
    out["sketch_genomes"] = bench_sketching(args.fgr2, gsub, args.workdir,
                                            args.nreps, "genome")

    out["idle_gate"]["reads"] = C.wait_for_idle(args.load_threshold, max_wait_s=args.gate_max_wait)
    print("== sketching: read sets ==", flush=True)
    out["sketch_reads"] = bench_sketching(args.fgr2, manifest["read_sets"],
                                          args.workdir, args.nreps, "reads")

    out["idle_gate"]["all_vs_all"] = C.wait_for_idle(args.load_threshold, max_wait_s=args.gate_max_wait)
    print("== all-vs-all on 16-genome panel ==", flush=True)
    out["all_vs_all"] = bench_allvsall(args.fgr2, genome_paths, args.workdir,
                                       args.nreps, args.repo)

    print("== exact Jaccard ground truth ==", flush=True)
    out["exact_jaccard"] = exact_jaccard_matrix(genome_paths)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
