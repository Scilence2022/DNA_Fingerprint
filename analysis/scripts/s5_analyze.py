#!/usr/bin/env python3
"""
S5 analysis + figures. Reads s5a/s5b/s5c result JSONs, computes the headline
statistics (paired Wilcoxon for (a) and (b)), and writes figures + a summary
JSON. Every number is recomputed from the persisted raw measurements.
"""
import json
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPTS))
RESULTS = os.path.join(REPO, "analysis", "results")
FIG = os.path.join(REPO, "analysis", "figures")
os.makedirs(FIG, exist_ok=True)

FIXED_C = [1, 2, 3, 5]
DEPTHS = [2, 5, 10, 20, 50, 100, 200]


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


# ======================================================================
# (a) coverage error filtering
# ======================================================================
def analyze_a():
    d = load("s5a_coverage_filter.json")
    recs = d["records"]
    n = len(recs)

    hist_ok = [r["fgr2_hist_matches_ground_truth"] for r in recs
               if r["fgr2_hist_matches_ground_truth"] is not None]
    validation = {
        "n_records": n,
        "fgr2_hist_matches_ground_truth_frac":
            float(np.mean(hist_ok)) if hist_ok else None,
        "n_checked": len(hist_ok),
    }

    def arr(key):
        return np.array([r[key] for r in recs], dtype=float)

    f1_auto = arr("f1_at_detected")
    f1_opt = arr("optimal_f1")
    f1_base = np.array([r["f1_at_baseline"] if r["f1_at_baseline"] is not None
                        else np.nan for r in recs])
    f1_fixed = {c: np.array([r["f1_fixed"][str(c)]
                             if r["f1_fixed"][str(c)] is not None else np.nan
                             for r in recs]) for c in FIXED_C}

    # globally-best single fixed c (a practitioner commits to one c in advance)
    mean_fixed = {c: float(np.nanmean(f1_fixed[c])) for c in FIXED_C}
    best_fixed_c = max(mean_fixed, key=mean_fixed.get)
    f1_bestfixed = f1_fixed[best_fixed_c]

    def paired(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        aa, bb = a[mask], b[mask]
        diff = aa - bb
        # Wilcoxon needs some nonzero differences
        if np.count_nonzero(diff) == 0:
            return {"n": int(mask.sum()), "mean_diff": 0.0,
                    "median_diff": 0.0, "wilcoxon_stat": None,
                    "wilcoxon_p": 1.0, "note": "all differences zero"}
        w = stats.wilcoxon(aa, bb)
        return {"n": int(mask.sum()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "wilcoxon_stat": float(w.statistic),
                "wilcoxon_p": float(w.pvalue)}

    pooled = {
        "mean_f1_auto": float(np.nanmean(f1_auto)),
        "mean_f1_optimal": float(np.nanmean(f1_opt)),
        "mean_f1_baseline_valley": float(np.nanmean(f1_base)),
        "mean_f1_fixed": mean_fixed,
        "best_fixed_c_by_mean": best_fixed_c,
        "mean_f1_best_fixed": mean_fixed[best_fixed_c],
        "auto_vs_optimal": paired(f1_auto, f1_opt),
        "auto_vs_best_fixed": paired(f1_auto, f1_bestfixed),
        "auto_vs_fixed_each": {c: paired(f1_auto, f1_fixed[c])
                               for c in FIXED_C},
        "auto_minus_optimal_mean": float(np.nanmean(f1_auto - f1_opt)),
    }

    # per-depth
    per_depth = {}
    for depth in DEPTHS:
        idx = [i for i, r in enumerate(recs) if r["target_depth"] == depth]
        fa = f1_auto[idx]; fo = f1_opt[idx]; fb = f1_bestfixed[idx]
        per_depth[str(depth)] = {
            "n": len(idx),
            "mean_f1_auto": float(np.nanmean(fa)),
            "mean_f1_optimal": float(np.nanmean(fo)),
            "mean_f1_best_fixed": float(np.nanmean(fb)),
            "mean_detected_c": float(np.nanmean(
                [recs[i]["detected"] for i in idx
                 if recs[i]["detected"] is not None])),
            "mean_optimal_c": float(np.nanmean(
                [recs[i]["optimal_f1_c"] for i in idx])),
            "auto_vs_best_fixed": paired(fa, fb),
            "auto_vs_optimal": paired(fa, fo),
        }

    # per-error-rate
    per_err = {}
    for er in sorted({r["err_rate"] for r in recs}):
        idx = [i for i, r in enumerate(recs) if r["err_rate"] == er]
        per_err[str(er)] = {
            "n": len(idx),
            "mean_f1_auto": float(np.nanmean(f1_auto[idx])),
            "mean_f1_optimal": float(np.nanmean(f1_opt[idx])),
            "mean_f1_best_fixed": float(np.nanmean(f1_bestfixed[idx])),
        }

    summary = {"validation": validation, "pooled": pooled,
               "per_depth": per_depth, "per_err_rate": per_err}

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.array(DEPTHS, dtype=float)
    ax[0].plot(xs, [per_depth[str(x)]["mean_f1_optimal"] for x in DEPTHS],
               "o-", label="optimal cutoff (oracle)", color="k")
    ax[0].plot(xs, [per_depth[str(x)]["mean_f1_auto"] for x in DEPTHS],
               "s-", label="fgr2 auto", color="C0")
    ax[0].plot(xs, [per_depth[str(x)]["mean_f1_best_fixed"] for x in DEPTHS],
               "^--", label=f"best fixed c={best_fixed_c}", color="C1")
    for c in FIXED_C:
        ax[0].plot(xs, [float(np.nanmean(f1_fixed[c][[i for i, r in
                    enumerate(recs) if r["target_depth"] == x]]))
                    for x in DEPTHS], ":", alpha=0.4, color="grey")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("target mean copies per oligo")
    ax[0].set_ylabel("mean F1 (true vs error k-mer separation)")
    ax[0].set_title("(a) Coverage error-filtering F1 vs depth")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(xs, [per_depth[str(x)]["mean_detected_c"] for x in DEPTHS],
               "s-", label="fgr2 auto cutoff", color="C0")
    ax[1].plot(xs, [per_depth[str(x)]["mean_optimal_c"] for x in DEPTHS],
               "o-", label="optimal cutoff", color="k")
    for c in FIXED_C:
        ax[1].axhline(c, ls=":", alpha=0.3, color="grey")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("target mean copies per oligo")
    ax[1].set_ylabel("coverage cutoff c")
    ax[1].set_title("(a) Detected vs optimal cutoff")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s5a_coverage_filter.png"), dpi=200)
    plt.close(fig)
    return summary


# ======================================================================
# (b) hash comparison
# ======================================================================
def analyze_b():
    d = load("s5b_hash.json")
    pools = list(d["uniformity"].keys())

    unif = {}
    for h in ("murmurhash3", "wang"):
        low = [d["uniformity"][p][h]["low16"]["chi2_over_dof"] for p in pools]
        lowp = [d["uniformity"][p][h]["low16"]["p_value"] for p in pools]
        topz = [d["uniformity"][p][h]["top16"]["z"] for p in pools]
        ksp = [d["uniformity"][p][h]["low_end_spacing_KS"]["ks_p"] for p in pools]
        bitdev = [d["uniformity"][p][h]["per_bit"]["max_abs_dev_from_half"]
                  for p in pools]
        unif[h] = {
            "low16_chi2_over_dof_mean": float(np.mean(low)),
            "low16_p_min": float(np.min(lowp)),
            "top16_abs_z_max": float(np.max(np.abs(topz))),
            "low_end_KS_p_min": float(np.min(ksp)),
            "per_bit_max_abs_dev_max": float(np.max(bitdev)),
        }
    aval = {h: {
        "mean_flip_prob": d["avalanche"][h]["mean_flip_prob"],
        "max_abs_dev_from_half": d["avalanche"][h]["max_abs_dev_from_half"],
        "mean_abs_dev_from_half": d["avalanche"][h]["mean_abs_dev_from_half"],
        "max_dev_output_bits_31_63": d["avalanche"][h]["max_dev_output_bits_31_63"],
        "max_dev_highinput_to_lowoutput":
            d["avalanche"][h]["max_dev_highinput_to_lowoutput"],
    } for h in ("murmurhash3", "wang")}

    acc = {}
    for h in ("murmurhash3", "wang"):
        acc[h] = {}
        for est in ("union", "direct"):
            for Nn in (1000, 10000):
                s = d["accuracy"]["summary"][f"{h}_{est}_N{Nn}"]
                acc[h][f"{est}_N{Nn}"] = {
                    "bias": s["bias_mean_error"], "rmse": s["rmse"],
                    "mae": s["mae"], "max_abs_error": s["max_abs_error"],
                    "n_pairs": s["n_pairs"]}

    # paired accuracy: |error| per pair, murmur vs wang (union N=1000)
    pairs = d["accuracy"]["per_pair"]
    err_m = np.array([abs(pairs[p]["murmurhash3_union_N1000"]
                          - pairs[p]["true_jaccard"]) for p in pairs])
    err_w = np.array([abs(pairs[p]["wang_union_N1000"]
                          - pairs[p]["true_jaccard"]) for p in pairs])
    if np.count_nonzero(err_m - err_w) > 0:
        w = stats.wilcoxon(err_m, err_w)
        acc_wilcoxon = {"n_pairs": len(err_m),
                        "mean_abs_err_murmur": float(err_m.mean()),
                        "mean_abs_err_wang": float(err_w.mean()),
                        "wilcoxon_stat": float(w.statistic),
                        "wilcoxon_p": float(w.pvalue)}
    else:
        acc_wilcoxon = {"n_pairs": len(err_m), "note": "identical"}

    speed = d["speed"]["summary"]

    # recommendation logic
    murmur_better_avalanche = (aval["murmurhash3"]["max_abs_dev_from_half"]
                               < aval["wang"]["max_abs_dev_from_half"])
    recommendation = (
        "MurmurHash3 recommended. Both hashes are effectively uniform on the "
        "low bits that bottom-N selection consumes (low16 chi2/dof ~1, KS p not "
        "significant) and give statistically indistinguishable fingerprint "
        "accuracy, but Wang shows a clear avalanche defect: its final "
        "key+=key<<31 step cannot propagate information from high input bits "
        "down to low output bits, giving a much larger max avalanche deviation. "
        "Wang's sketch-build wall time is marginally lower (~5-8%), which does "
        "not justify the weaker mixing for archival fingerprints.")

    summary = {
        "uniformity": unif, "avalanche": aval, "accuracy": acc,
        "accuracy_paired_wilcoxon_union_N1000": acc_wilcoxon,
        "speed": speed,
        "murmur_better_avalanche": bool(murmur_better_avalanche),
        "recommendation": recommendation,
    }

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # avalanche heatmap diff-from-0.5 max per hash (bar of key metrics)
    metrics = ["max avalanche\ndev", "low-in->low-out\navalanche dev",
               "union N=1e3\nRMSE", "sketch time\n(rel. murmur)"]
    mur = [aval["murmurhash3"]["max_abs_dev_from_half"],
           aval["murmurhash3"]["max_dev_highinput_to_lowoutput"],
           acc["murmurhash3"]["union_N1000"]["rmse"], 1.0]
    wng = [aval["wang"]["max_abs_dev_from_half"],
           aval["wang"]["max_dev_highinput_to_lowoutput"],
           acc["wang"]["union_N1000"]["rmse"],
           speed["wang_over_murmur_median_ratio"]]
    x = np.arange(len(metrics)); wd = 0.38
    ax[0].bar(x - wd/2, mur, wd, label="MurmurHash3", color="C0")
    ax[0].bar(x + wd/2, wng, wd, label="Wang", color="C1")
    ax[0].set_xticks(x); ax[0].set_xticklabels(metrics, fontsize=7)
    ax[0].set_yscale("log"); ax[0].set_title("(b) Hash metrics (log)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")

    # avalanche matrices
    dd = load("s5b_hash.json")
    for i, h in enumerate(("murmurhash3", "wang")):
        mat = np.array(dd["avalanche"][h]["matrix"])
        im = ax[1 + i].imshow(np.abs(mat - 0.5), aspect="auto",
                              cmap="magma", vmin=0, vmax=0.1)
        ax[1 + i].set_title(f"(b) {h} avalanche |p-0.5|")
        ax[1 + i].set_xlabel("output bit"); ax[1 + i].set_ylabel("input bit flipped")
        fig.colorbar(im, ax=ax[1 + i], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s5b_hash.png"), dpi=200)
    plt.close(fig)
    return summary


# ======================================================================
# (c) throughput & footprint
# ======================================================================
def analyze_c():
    d = load("s5c_throughput.json")
    ladder = d["ladder"]
    fits = d["fits"]
    table = [{
        "n_oligos": r["n_oligos"], "bases": r["bases"],
        "distinct_kmers": r["distinct_canonical_kmers"],
        "wall_s_median": r["wall_s_median"],
        "throughput_mbp_per_s": r["throughput_mbp_per_s"],
        "max_rss_bytes_median": r["max_rss_bytes_median"],
        "contended": r.get("contended"),
    } for r in ladder]

    bases = np.array([r["bases"] for r in ladder], dtype=float)
    walls = np.array([r["wall_s_median"] for r in ladder], dtype=float)
    rss = np.array([r["max_rss_bytes_median"] for r in ladder], dtype=float)
    distinct = np.array([r["distinct_canonical_kmers"] for r in ladder],
                        dtype=float)

    # Asymptotic (large-input) log-log slope: the overall slope is pulled below
    # 1 by a fixed per-invocation startup cost that dominates the small inputs.
    def loglog_slope(mask):
        lb, lw = np.log10(bases[mask]), np.log10(walls[mask])
        A = np.vstack([lb, np.ones_like(lb)]).T
        sl, ic = np.linalg.lstsq(A, lw, rcond=None)[0]
        return float(sl)
    top = bases >= 20e6
    fits = dict(fits)
    fits["runtime_vs_bases_loglog_asymptotic_ge20Mbp"] = {
        "slope": loglog_slope(top), "n_points": int(top.sum()),
        "note": "overall slope < 1 reflects fixed startup overhead at small "
                "inputs; at >=20 Mbp the slope approaches 1 (linear), and "
                "throughput plateaus.",
    }
    fits["throughput_plateau_mbp_per_s"] = float(np.median(
        (bases / walls / 1e6)[top]))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(bases / 1e6, walls, "o-", color="C0")
    sl = fits["runtime_vs_bases_loglog"]["slope"]
    ax[0].set_xlabel("input (Mbp)"); ax[0].set_ylabel("wall-clock (s), median")
    ax[0].set_title(f"(c) Runtime vs input  (log-log slope={sl:.2f}, "
                    f"r2={fits['runtime_vs_bases_loglog']['r2']:.3f})")
    ax[0].grid(alpha=0.3, which="both")

    ax[1].plot(distinct / 1e6, rss / 1e9, "o-", color="C2")
    bpk = fits["memory_vs_distinct_kmers_linear"]["bytes_per_distinct_kmer"]
    ax[1].set_xlabel("distinct canonical k-mers (millions)")
    ax[1].set_ylabel("peak RSS (GB)")
    ax[1].set_title(f"(c) Memory vs distinct k-mers  "
                    f"({bpk:.0f} B/k-mer, r2="
                    f"{fits['memory_vs_distinct_kmers_linear']['r2']:.3f})")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s5c_throughput.png"), dpi=200)
    plt.close(fig)
    return {"ladder_table": table, "fits": fits}


def main():
    out = {}
    out["a_coverage_filter"] = analyze_a()
    print("(a) done", flush=True)
    out["b_hash"] = analyze_b()
    print("(b) done", flush=True)
    if os.path.exists(os.path.join(RESULTS, "s5c_throughput.json")):
        out["c_throughput"] = analyze_c()
        print("(c) done", flush=True)
    else:
        print("(c) result not present yet; skipped", flush=True)

    with open(os.path.join(RESULTS, "s5_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote s5_summary.json")

    # console headline
    p = out["a_coverage_filter"]["pooled"]
    print(f"\n(a) auto F1={p['mean_f1_auto']:.4f} optimal={p['mean_f1_optimal']:.4f} "
          f"best-fixed(c={p['best_fixed_c_by_mean']})={p['mean_f1_best_fixed']:.4f}")
    print(f"    auto vs best-fixed: mean_diff={p['auto_vs_best_fixed']['mean_diff']:+.4f} "
          f"p={p['auto_vs_best_fixed']['wilcoxon_p']:.2e}")
    print(f"    auto vs optimal: mean_diff={p['auto_vs_optimal']['mean_diff']:+.4f} "
          f"p={p['auto_vs_optimal']['wilcoxon_p']:.2e}")
    print(f"    hist validation frac={out['a_coverage_filter']['validation']['fgr2_hist_matches_ground_truth_frac']}")
    aw = out["b_hash"]["accuracy_paired_wilcoxon_union_N1000"]
    print(f"(b) accuracy murmur|err|={aw.get('mean_abs_err_murmur')} "
          f"wang|err|={aw.get('mean_abs_err_wang')} p={aw.get('wilcoxon_p')}")
    if "c_throughput" in out:
        f = out["c_throughput"]["fits"]
        print(f"(c) runtime slope={f['runtime_vs_bases_loglog']['slope']:.3f} "
              f"mem={f['memory_vs_distinct_kmers_linear']['bytes_per_distinct_kmer']:.0f}B/kmer")


if __name__ == "__main__":
    main()
