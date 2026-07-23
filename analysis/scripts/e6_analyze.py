#!/usr/bin/env python3
"""
E6 part 4: analysis + figures from the three raw benchmark JSONs.

Produces:
  fig_e6_runtime_vs_size.png     runtime vs input size (log-log) + fitted slope
  fig_e6_thread_speedup.png      speedup vs thread count + saturation point
  fig_e6_rss_vs_kmers.png        peak RSS vs distinct k-mers
  fig_e6_tool_comparison.png     fgr2 vs mash vs sourmash: time / RSS / disk
  fig_e6_downstream_scaling.png  calculate_similarity.py runtime vs sample count

and writes e6_summary.json containing every derived quantity (fitted exponents,
accuracy statistics, comparison table) so nothing in the manuscript text is
computed anywhere but here.
"""
import os
import sys
import json
import glob
import argparse
import itertools
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import fgrlib  # noqa: E402

DPI = 200
K = 31


def recompute_contended(p):
    """Recompute the contention flag from the stored idle-CPU samples.

    The scaling run was written by a version of e6_common whose `contended`
    field was overwritten by a stale load-average criterion
    (loadavg > n_cores/2). Load average is unusable on this host -- it read
    ~200 while the machine was simultaneously ~70% idle -- so that criterion
    marked every condition contended regardless of the truth. The raw
    cpu_idle_pct_before/after samples ARE stored in the JSON, so the flag is
    recomputed here from those instead of re-running the benchmark.
    Criterion: contended if either idle sample fell below 50%.
    """
    vals = [p.get("cpu_idle_pct_before"), p.get("cpu_idle_pct_after")]
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return p.get("contended")
    return min(vals) < 50.0


def loglog_slope(x, y):
    """Fit log10(y) = a*log10(x) + b. Returns slope, intercept, R^2."""
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


# ----------------------------------------------------------- figures

def fig_runtime_vs_size(scaling, figdir, summary):
    pts = scaling["size_ladder"]
    x = [p["bases"] / 1e6 for p in pts]
    y = [p["wall_s_median"] for p in pts]
    ymin = [p["wall_s_min"] for p in pts]
    ymax = [p["wall_s_max"] for p in pts]
    slope, icept, r2 = loglog_slope(x, y)
    summary["runtime_vs_size"] = {
        "loglog_slope": slope, "loglog_intercept": icept, "r2": r2,
        "interpretation": ("slope 1.0 == perfectly linear in input size; "
                           ">1 means superlinear"),
        "points": [{"mb": p["target_mb"], "bases": p["bases"],
                    "wall_s_median": p["wall_s_median"],
                    "wall_s_all": p["wall_s_all"],
                    "throughput_mbp_per_s": p["throughput_mbp_per_s"],
                    "cpu_s_median": p.get("cpu_s_median"),
                    "max_rss_bytes_median": p["max_rss_bytes_median"],
                    "distinct_kmers": p["distinct_canonical_kmers"],
                    "n_runs": p["n_runs"],
                    "loadavg_median": p.get("loadavg_1min_median_during"),
                    "contended": recompute_contended(p),
                    "cpu_idle_pct_before": p.get("cpu_idle_pct_before"),
                    "cpu_idle_pct_after": p.get("cpu_idle_pct_after")} for p in pts],
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    err = [np.array(y) - np.array(ymin), np.array(ymax) - np.array(y)]
    ax.errorbar(x, y, yerr=err, fmt="o-", capsize=3, color="#1f77b4",
                label="fgr2 (-t 4), median of n=%d" % pts[0]["n_runs"])
    xf = np.logspace(np.log10(min(x)), np.log10(max(x)), 50)
    ax.plot(xf, 10 ** icept * xf ** slope, "--", color="#d62728",
            label=f"fit: slope={slope:.3f}, $R^2$={r2:.4f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Input size (Mbp)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("fgr2 sketch construction: runtime vs input size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_e6_runtime_vs_size.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def fig_thread_speedup(scaling, figdir, summary):
    pts = scaling["thread_scaling"]
    t = [p["threads"] for p in pts]
    sp = [p["speedup_vs_t1"] for p in pts]
    eff = [p["parallel_efficiency"] for p in pts]
    # saturation = first thread count where adding threads gains < 10%
    sat = None
    for i in range(1, len(pts)):
        if sp[i] / sp[i - 1] < 1.10:
            sat = t[i - 1]
            break
    if sat is None:
        sat = t[-1]
    summary["thread_scaling"] = {
        "saturation_threads": sat,
        "criterion": "first -t at which doubling threads yields <10% further speedup",
        "points": [{"threads": p["threads"], "wall_s_median": p["wall_s_median"],
                    "wall_s_all": p["wall_s_all"],
                    "speedup_vs_t1": p["speedup_vs_t1"],
                    "parallel_efficiency": p["parallel_efficiency"],
                    "cpu_utilisation": p["cpu_utilisation"],
                    "max_rss_bytes_median": p["max_rss_bytes_median"],
                    "n_runs": p["n_runs"],
                    "loadavg_median": p.get("loadavg_1min_median_during"),
                    "contended": recompute_contended(p),
                    "cpu_idle_pct_before": p.get("cpu_idle_pct_before"),
                    "cpu_idle_pct_after": p.get("cpu_idle_pct_after")} for p in pts],
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    a1.plot(t, sp, "o-", color="#1f77b4", label="measured speedup")
    a1.plot(t, t, "--", color="gray", label="ideal (linear)")
    a1.axvline(sat, color="#d62728", ls=":", lw=2,
               label=f"saturation @ -t {sat}")
    a1.set_xscale("log", base=2); a1.set_yscale("log", base=2)
    a1.set_xticks(t); a1.set_xticklabels(t)
    a1.set_xlabel("Threads (-t)"); a1.set_ylabel("Speedup vs -t 1")
    a1.set_title("Thread scaling (200 Mbp input)")
    a1.grid(True, which="both", alpha=0.3); a1.legend(fontsize=8)

    a2.plot(t, eff, "s-", color="#2ca02c")
    a2.axhline(1.0, ls="--", color="gray")
    a2.axvline(sat, color="#d62728", ls=":", lw=2)
    a2.set_xscale("log", base=2); a2.set_xticks(t); a2.set_xticklabels(t)
    a2.set_xlabel("Threads (-t)"); a2.set_ylabel("Parallel efficiency")
    a2.set_ylim(0, 1.15)
    a2.set_title("Parallel efficiency")
    a2.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_e6_thread_speedup.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def fig_rss_vs_kmers(scaling, figdir, summary):
    pts = scaling["size_ladder"]
    x = [p["distinct_canonical_kmers"] for p in pts]
    y = [p["max_rss_bytes_median"] / 1e9 for p in pts]
    slope, icept, r2 = loglog_slope(x, y)
    bytes_per = [(p["max_rss_bytes_median"] / p["distinct_canonical_kmers"])
                 for p in pts]
    summary["rss_vs_kmers"] = {
        "loglog_slope": slope, "r2": r2,
        "bytes_per_distinct_kmer": bytes_per,
        "bytes_per_distinct_kmer_at_largest": bytes_per[-1],
        "note": "RSS in BYTES (macOS /usr/bin/time -l)",
        "points": [{"distinct_kmers": a, "rss_gb": b} for a, b in zip(x, y)],
    }
    pref = scaling.get("prefix_scaling", [])
    ok = [p for p in pref if not p.get("failed")]
    summary["prefix_effect"] = [
        {"prefix": p["prefix"], "wall_s_median": p["wall_s_median"],
         "wall_s_all": p["wall_s_all"],
         "max_rss_bytes_median": p["max_rss_bytes_median"],
         "n_runs": p["n_runs"]} for p in ok]
    summary["prefix_failed"] = [p for p in pref if p.get("failed")]

    ncol = 2 if ok else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.4), squeeze=False)
    a1 = axes[0][0]
    a1.plot(x, y, "o-", color="#9467bd")
    a1.plot(x, 10 ** icept * np.array(x, float) ** slope, "--",
            color="#d62728", label=f"slope={slope:.3f}, $R^2$={r2:.4f}")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Distinct canonical 31-mers")
    a1.set_ylabel("Peak RSS (GB)")
    a1.set_title("Peak memory vs distinct k-mers")
    a1.grid(True, which="both", alpha=0.3); a1.legend(fontsize=8)

    if ok:
        a2 = axes[0][1]
        pp = [p["prefix"] for p in ok]
        pr = [p["max_rss_bytes_median"] / 1e9 for p in ok]
        pt = [p["wall_s_median"] for p in ok]
        a2.plot(pp, pr, "o-", color="#9467bd", label="peak RSS (GB)")
        a2.set_xlabel("Prefix length (-p)"); a2.set_ylabel("Peak RSS (GB)")
        a2b = a2.twinx()
        a2b.plot(pp, pt, "s--", color="#ff7f0e", label="wall time (s)")
        a2b.set_ylabel("Wall-clock (s)")
        a2.set_title("Effect of -p (200 Mbp input)")
        a2.grid(True, alpha=0.3)
        h1, l1 = a2.get_legend_handles_labels()
        h2, l2 = a2b.get_legend_handles_labels()
        a2.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_e6_rss_vs_kmers.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


# ----------------------------------------------------------- accuracy

def accuracy_table(tools, genome_dir, summary):
    """Compare each tool's pairwise Jaccard to exact Jaccard ground truth."""
    exact = tools["exact_jaccard"]
    ava = tools["all_vs_all"]

    # ---- fgr2: recompute from its sketches with BOTH estimators -------
    fg_dir = ava["fgr2_sketch_dir"]
    sk = {}
    for f in sorted(glob.glob(os.path.join(fg_dir, "*.fgr2"))):
        h, cov, meta = fgrlib.parse_sketch(f)
        sk[os.path.basename(f).replace(".fgr2", "")] = (set(h.keys()), h)

    rows = {"fgr2_direct": {}, "fgr2_union": {}, "mash": {}, "sourmash": {}}
    names = sorted(sk)
    for a, b in itertools.combinations(names, 2):
        key = f"{a}.fna|{b}.fna"
        if key not in exact:
            key = f"{b}.fna|{a}.fna"
        if key not in exact:
            continue
        rows["fgr2_direct"][key] = fgrlib.jaccard_direct(sk[a][0], sk[b][0])
        rows["fgr2_union"][key] = fgrlib.jaccard_union(sk[a][1], sk[b][1],
                                                       len(sk[a][1]))

    md = ava["mash_dist"].get("parsed", {})
    for kk, v in md.items():
        a, b = kk.split("|")
        a, b = a.replace(".fna", ""), b.replace(".fna", "")
        if a == b:
            continue
        key = f"{a}.fna|{b}.fna"
        if key not in exact:
            key = f"{b}.fna|{a}.fna"
        if key in exact:
            rows["mash"][key] = v["jaccard"]

    # sourmash names each signature after the sequence record, not the file, so
    # the CSV header cannot be joined to genome names by equality. Resolve each
    # label to a genome by finding which genome basename it contains; refuse to
    # guess if the match is not unique, rather than silently mis-pairing.
    raw_labels = ava["sourmash_compare"]["labels"]
    gnames = sorted({kk.split("|")[0].replace(".fna", "") for kk in exact} |
                    {kk.split("|")[1].replace(".fna", "") for kk in exact})
    resolved, unresolved = [], []
    for lab in raw_labels:
        hits = [g for g in gnames if g in lab]
        if len(hits) == 1:
            resolved.append(hits[0])
        else:
            resolved.append(None)
            unresolved.append({"label": lab, "candidate_hits": hits})
    summary["sourmash_label_resolution"] = {
        "raw_labels": raw_labels, "resolved": resolved,
        "unresolved": unresolved,
        "note": ("sourmash signature names derive from FASTA record headers; "
                 "labels that could not be uniquely matched to a genome are "
                 "excluded from the sourmash accuracy statistics"),
    }
    mat = ava["sourmash_compare"]["matrix"]
    for i, j in itertools.combinations(range(len(resolved)), 2):
        a, b = resolved[i], resolved[j]
        if a is None or b is None or a == b:
            continue
        key = f"{a}.fna|{b}.fna"
        if key not in exact:
            key = f"{b}.fna|{a}.fna"
        if key in exact:
            rows["sourmash"][key] = mat[i][j]

    stats = {}
    for tool, est in rows.items():
        keys = [kk for kk in est if kk in exact]
        if not keys:
            stats[tool] = {"n_pairs": 0, "note": "no overlapping pairs"}
            continue
        e = np.array([exact[kk] for kk in keys])
        p = np.array([est[kk] for kk in keys])
        d = p - e
        stats[tool] = {
            "n_pairs": len(keys),
            "mean_abs_error": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "mean_signed_error_bias": float(np.mean(d)),
            "max_abs_error": float(np.max(np.abs(d))),
            "pearson_r_vs_exact": float(np.corrcoef(e, p)[0, 1]),
            "per_pair": {kk: {"exact": float(exact[kk]), "estimate": float(est[kk])}
                         for kk in keys},
        }
    # Are mash and sourmash actually independent measurements? Both are
    # bottom-N MinHash over canonical k-mers using MurmurHash3 at matched k and
    # sketch size, so they may be computing a bit-identical sketch. Check it
    # rather than presenting them as two independent corroborating tools.
    mp, sp = rows["mash"], rows["sourmash"]
    shared = [kk for kk in mp if kk in sp]
    identical = sum(1 for kk in shared if mp[kk] == sp[kk])
    summary["mash_vs_sourmash_identity"] = {
        "n_shared_pairs": len(shared),
        "n_bit_identical_estimates": identical,
        "all_identical": bool(shared) and identical == len(shared),
        "interpretation": (
            "mash and sourmash are NOT independent corroboration here. At "
            "matched k and sketch size both compute a bottom-N MinHash with "
            "MurmurHash3 over canonical k-mers, so they select the same "
            "k-mers and return the same Jaccard. Their identical accuracy "
            "statistics are therefore one measurement reported twice, and the "
            "agreement should not be read as independent validation."),
    }
    summary["accuracy_vs_exact_jaccard"] = stats
    return stats


def fig_tool_comparison(tools, figdir, summary):
    def collect(section, tool_keys):
        out = {}
        for tk in tool_keys:
            ts, rs, ds, cs, ld = [], [], [], [], []
            for row in section:
                r = row["tools"].get(tk)
                if not r or r.get("failed"):
                    continue
                ts.append(r["wall_s_median"])
                rs.append(r["max_rss_bytes_median"])
                ds.append(r["sketch_bytes_on_disk"])
                if r.get("cpu_s_median") is not None:
                    cs.append(r["cpu_s_median"])
                ld.append(r.get("loadavg_1min_median_during"))
            if ts:
                out[tk] = {
                    "n_inputs": len(ts),
                    "wall_s_median_of_inputs": statistics.median(ts),
                    "wall_s_per_input": ts,
                    # CPU time is the contention-robust metric; prefer it when
                    # `contended` is true for any tool in the comparison.
                    "cpu_s_median_of_inputs": statistics.median(cs) if cs else None,
                    "cpu_s_per_input": cs,
                    "max_rss_bytes_median_of_inputs": statistics.median(rs),
                    "sketch_bytes_median": statistics.median(ds),
                    "loadavg_median_per_input": ld,
                }
        return out

    gkeys = ["fgr2_t1_c1", "mash_t1", "sourmash_t1", "fgr2_t4_c1", "mash_t4"]
    rkeys = gkeys + ["fgr2_t1_autocov", "mash_t1_reads_m2", "sourmash_t1_abund"]
    g = collect(tools["sketch_genomes"], gkeys)
    r = collect(tools["sketch_reads"], rkeys)
    summary["tool_comparison"] = {"genomes": g, "reads": r}

    ava = tools["all_vs_all"]
    summary["all_vs_all_16_genomes"] = {
        kk: {"wall_s_median": ava[kk]["wall_s_median"],
             "wall_s_all": ava[kk]["wall_s_all"],
             "max_rss_bytes_median": ava[kk]["max_rss_bytes_median"],
             "n_runs": ava[kk]["n_runs"], "cmd": ava[kk]["cmd"],
             "loadavg_median": ava[kk].get("loadavg_1min_median_during")}
        for kk in ("mash_dist", "sourmash_compare", "calculate_similarity")
        if kk in ava and not ava[kk].get("failed")}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (data, title) in zip(axes[:2],
                                 [(g, "Assembled genomes"), (r, "Read sets")]):
        ks = [kk for kk in (gkeys if data is g else rkeys) if kk in data]
        vals = [data[kk]["wall_s_median_of_inputs"] for kk in ks]
        cols = ["#1f77b4" if kk.startswith("fgr2") else
                "#ff7f0e" if kk.startswith("mash") else "#2ca02c" for kk in ks]
        ax.barh(range(len(ks)), vals, color=cols)
        ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Median sketch time (s)")
        ax.set_title(f"Sketch construction: {title}")
        ax.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=7)

    ax = axes[2]
    ks = [kk for kk in gkeys if kk in g]
    vals = [g[kk]["sketch_bytes_median"] / 1e3 for kk in ks]
    cols = ["#1f77b4" if kk.startswith("fgr2") else
            "#ff7f0e" if kk.startswith("mash") else "#2ca02c" for kk in ks]
    ax.barh(range(len(ks)), vals, color=cols)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Sketch size on disk (kB)")
    ax.set_title("Sketch file size (genomes)\n(text vs binary vs gzip-JSON)")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=7)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_e6_tool_comparison.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def fig_downstream(down, figdir, summary):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    cols = {"pairwise_only": "#1f77b4", "nj_boot0": "#ff7f0e",
            "nj_boot100": "#2ca02c"}
    summary["downstream"] = {
        "n_unique_sketches": down.get("n_unique_sketches"),
        "primary_statistic": "wall_s_min",
        "why_min_not_median": (
            "This machine ran other users' jobs intermittently during phase 3. "
            "Interference can only ADD wall-clock, never subtract it, so the "
            "per-condition MINIMUM is the robust estimator of the tool's own "
            "cost; the median is contaminated wherever a majority of the 5 "
            "reps overlapped external load. Observed contamination: "
            "pairwise_only n=8 had one run of 441.30 s against ~1.8 s for the "
            "other four, and pairwise_only n=160 spanned 15.65-48.46 s. Both "
            "medians and minima are reported per point so either can be used."),
        "model": (
            "Runtime is fitted as t = a + b*n_pairs rather than a pure power "
            "law. A log-log slope over the whole range is misleading here "
            "because a fixed ~1.6-1.8 s interpreter-startup and sketch-loading "
            "cost dominates at small n, which drags the apparent exponent well "
            "below 2 even though the pairwise work is genuinely O(n^2)."),
        "modes": {}}
    for mode, pts in down["modes"].items():
        good = [p for p in pts if not p.get("failed") and not p.get("timed_out")]
        cens = [p for p in pts if p.get("timed_out")]
        if len(good) < 2:
            summary["downstream"]["modes"][mode] = {
                "points": good, "censored": cens,
                "note": "too few successful points to fit"}
            continue
        x = np.array([p["n_samples"] for p in good], float)
        npair = np.array([p["n_pairs"] for p in good], float)
        y = np.array([min(p["wall_s_all"]) for p in good], float)
        ymed = np.array([p["wall_s_median"] for p in good], float)

        # t = a + b * n_pairs   (tests the O(n^2) pairwise term)
        A = np.vstack([np.ones_like(npair), npair]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_quad = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # t = a + b * n^3       (tests the O(n^3) NJ term)
        A3 = np.vstack([np.ones_like(x), x ** 3]).T
        coef3, *_ = np.linalg.lstsq(A3, y, rcond=None)
        pred3 = A3 @ coef3
        r2_cub = 1 - float(np.sum((y - pred3) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        slope, icept, r2 = loglog_slope(x, y)

        summary["downstream"]["modes"][mode] = {
            "loglog_slope_on_min": slope, "loglog_r2": r2,
            "fit_a_plus_b_npairs": {"a_s": float(coef[0]),
                                    "b_s_per_pair": float(coef[1]),
                                    "r2": r2_quad},
            "fit_a_plus_b_ncubed": {"a_s": float(coef3[0]),
                                    "b_s_per_n3": float(coef3[1]),
                                    "r2": r2_cub},
            "points": [{"n": p["n_samples"], "n_pairs": p["n_pairs"],
                        "wall_s_min": min(p["wall_s_all"]),
                        "wall_s_median": p["wall_s_median"],
                        "wall_s_all": p["wall_s_all"],
                        "n_runs": p["n_runs"],
                        "max_rss_bytes_median": p["max_rss_bytes_median"]}
                       for p in good],
            "censored": [{"n": p["n_samples"], "timeout_s": p.get("timeout_s")}
                         for p in cens],
        }
        ax.plot(x, y, "o-", color=cols.get(mode, "gray"),
                label=f"{mode} (min): $R^2_{{n^2}}$={r2_quad:.3f}, "
                      f"$R^2_{{n^3}}$={r2_cub:.3f}")
        ax.plot(x, ymed, "x", color=cols.get(mode, "gray"), alpha=0.45,
                markersize=5)
        for p in cens:
            ax.plot([p["n_samples"]], [p["timeout_s"]], "v", ms=9,
                    color=cols.get(mode, "gray"))
            ax.annotate("censored\n(>900 s)", (p["n_samples"], p["timeout_s"]),
                        textcoords="offset points", xytext=(-12, 8),
                        fontsize=7, color=cols.get(mode, "gray"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of samples (n)")
    ax.set_ylabel("calculate_similarity.py wall-clock (s)")
    ax.set_title("Downstream comparison cost vs sample count\n"
                 "(circles = min of reps, faint x = median)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, "fig_e6_downstream_scaling.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--figures", required=True)
    ap.add_argument("--genomes", required=True)
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)

    summary = {"experiment": "E6_summary",
               "rss_unit": "BYTES (macOS /usr/bin/time -l, verified in-session)"}
    figs = []

    sp = os.path.join(args.results, "e6_scaling.json")
    if os.path.exists(sp):
        scaling = json.load(open(sp))
        summary["machine"] = {"n_cores": os.cpu_count()}
        figs.append(fig_runtime_vs_size(scaling, args.figures, summary))
        figs.append(fig_thread_speedup(scaling, args.figures, summary))
        figs.append(fig_rss_vs_kmers(scaling, args.figures, summary))
        summary["cold_vs_warm"] = scaling.get("cold_vs_warm")
    else:
        summary["scaling"] = "MISSING"

    tp = os.path.join(args.results, "e6_tools.json")
    if os.path.exists(tp):
        tools = json.load(open(tp))
        figs.append(fig_tool_comparison(tools, args.figures, summary))
        accuracy_table(tools, args.genomes, summary)
    else:
        summary["tools"] = "MISSING"

    pb = os.path.join(args.results, "e6_prefix_bound.json")
    if os.path.exists(pb):
        b = json.load(open(pb))
        summary["prefix_accepted_range"] = {
            "min_accepted": b["min_accepted"], "max_accepted": b["max_accepted"],
            "accepted": b["accepted_prefixes"], "conclusion": b["conclusion"],
            "rows": b["rows"]}

    dp = os.path.join(args.results, "e6_downstream.json")
    if os.path.exists(dp):
        down = json.load(open(dp))
        figs.append(fig_downstream(down, args.figures, summary))
    else:
        summary["downstream"] = "MISSING"

    summary["figures"] = figs
    out = os.path.join(args.results, "e6_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("accuracy_vs_exact_jaccard",)}, indent=2)[:6000])
    print(f"[done] -> {out}")
    for f in figs:
        print("  fig:", f)


if __name__ == "__main__":
    main()
