#!/usr/bin/env python3
"""
F1 analysis + figures. Consumes:
  analysis/results/f1_cosine_synthetic.json
  analysis/results/f1_cosine_real.json
  analysis/results/f1_e5_cosine_recheck.json
Writes analysis/results/f1_cosine_summary.json and four figures.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "analysis", "results")
FIG = os.path.join(REPO, "analysis", "figures")
N_SIZES = ["100", "1000", "10000"]
ESTS = ["shipped", "union_bottomN", "intersect_only", "shipped_analytic"]
COL = {"shipped": "#d62728", "union_bottomN": "#1f77b4",
       "intersect_only": "#7f7f7f", "shipped_analytic": "#2ca02c"}
LBL = {"shipped": "shipped (calculate_similarity.py)",
       "union_bottomN": "union bottom-N (corrected)",
       "intersect_only": "intersection-only",
       "shipped_analytic": "shipped x analytic correction"}


def cell_stats(recs, est, n):
    b = np.array([r["est"][n][est] - r["cos_true"] for r in recs])
    t = np.array([r["cos_true"] for r in recs])
    e = np.array([r["est"][n][est] for r in recs])
    return {"n": int(b.size), "mean_bias": float(b.mean()),
            "sd_bias": float(b.std(ddof=1)) if b.size > 1 else None,
            "rmse": float(np.sqrt((b ** 2).mean())),
            "mean_true": float(t.mean()), "mean_est": float(e.mean()),
            "mean_relative_bias": float((e / t - 1).mean()),
            "mean_ratio_est_over_true": float((e / t).mean())}


def paired_tests(recs, n):
    """Paired |bias| comparison, shipped vs union, over matched replicates."""
    a = np.array([abs(r["est"][n]["shipped"] - r["cos_true"]) for r in recs])
    b = np.array([abs(r["est"][n]["union_bottomN"] - r["cos_true"]) for r in recs])
    if np.allclose(a, b):
        return {"n_pairs": int(a.size), "wilcoxon_p": 1.0, "note": "identical"}
    w = stats.wilcoxon(a, b)
    t = stats.ttest_rel(a, b)
    return {"n_pairs": int(a.size),
            "mean_abs_bias_shipped": float(a.mean()),
            "mean_abs_bias_union": float(b.mean()),
            "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "paired_t": float(t.statistic), "paired_t_p": float(t.pvalue)}


def main():
    syn = json.load(open(os.path.join(RES, "f1_cosine_synthetic.json")))
    real = json.load(open(os.path.join(RES, "f1_cosine_real.json")))
    e5 = json.load(open(os.path.join(RES, "f1_e5_cosine_recheck.json")))
    S = syn["records"]
    summary = {"experiment_id": "F1_cosine_structural_bias_summary",
               "inputs": ["f1_cosine_synthetic.json", "f1_cosine_real.json",
                          "f1_e5_cosine_recheck.json"]}

    # ---------------- synthetic: bias by ratio x N x estimator
    ratios = sorted({r["ratio"] for r in S})
    syn_tbl = {}
    for n in N_SIZES:
        syn_tbl[n] = {}
        for rr in ratios:
            sub = [r for r in S if r["ratio"] == rr]
            syn_tbl[n][str(rr)] = {e: cell_stats(sub, e, n) for e in ESTS}
            syn_tbl[n][str(rr)]["predicted_shipped_over_true_1_over_sqrt_r"] = \
                1.0 / math.sqrt(rr)
            syn_tbl[n][str(rr)]["paired_shipped_vs_union"] = paired_tests(sub, n)
    summary["synthetic_by_ratio"] = syn_tbl

    # derivation check: observed shipped/true vs 1/sqrt(r), N=10000
    obs = np.array([syn_tbl["10000"][str(rr)]["shipped"]["mean_ratio_est_over_true"]
                    for rr in ratios])
    pred = np.array([1 / math.sqrt(rr) for rr in ratios])
    summary["derivation_check_N10000"] = {
        "ratios": ratios, "observed_shipped_over_true": obs.tolist(),
        "predicted_1_over_sqrt_r": pred.tolist(),
        "max_abs_deviation": float(np.max(np.abs(obs - pred))),
        "pearson_r": float(stats.pearsonr(obs, pred)[0])}

    # ---------------- N-scaling of bias at each ratio (decisive question)
    nscale = {}
    for rr in ratios:
        sub = [r for r in S if r["ratio"] == rr]
        nscale[str(rr)] = {e: {n: cell_stats(sub, e, n)["mean_bias"]
                               for n in N_SIZES} for e in ESTS}
        # regression of |mean bias| on log10 N
        for e in ESTS:
            y = np.array([abs(nscale[str(rr)][e][n]) for n in N_SIZES])
            x = np.log10([float(n) for n in N_SIZES])
            sl = stats.linregress(x, y)
            nscale[str(rr)][e + "_slope_per_decade"] = float(sl.slope)
    summary["N_scaling_of_mean_bias"] = nscale

    # ---------------- synthetic: bias vs true cosine (ratio == 1 subset & all)
    bins = [(0.25, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    by_true = {}
    for n in N_SIZES:
        by_true[n] = {}
        for lo, hi in bins:
            sub = [r for r in S if lo <= r["cos_true"] < hi]
            if len(sub) < 5:
                continue
            by_true[n][f"{lo}-{hi}"] = {e: cell_stats(sub, e, n) for e in ESTS}
    summary["synthetic_by_true_cosine"] = by_true

    # overall synthetic (full set, no selection)
    summary["synthetic_overall"] = {
        n: {e: cell_stats(S, e, n) for e in ESTS} |
           {"paired_shipped_vs_union": paired_tests(S, n)} for n in N_SIZES}

    # ---------------- real genomes
    NP = real["natural_pairs"]
    CR = real["controlled_ratio"]
    summary["real_natural_pairs"] = {
        "n_pairs": len(NP),
        "ratio_range": [min(r["ratio"] for r in NP), max(r["ratio"] for r in NP)],
        "cos_true_range": [min(r["cos_true"] for r in NP),
                           max(r["cos_true"] for r in NP)],
        "by_N": {n: {e: cell_stats(NP, e, n) for e in ESTS} |
                    {"paired_shipped_vs_union": paired_tests(NP, n)}
                 for n in N_SIZES}}
    tgts = sorted({r["target_ratio"] for r in CR})
    ctl = {}
    for n in N_SIZES:
        ctl[n] = {}
        for tg in tgts:
            sub = [r for r in CR if r["target_ratio"] == tg]
            ctl[n][str(tg)] = {e: cell_stats(sub, e, n) for e in ESTS}
            ctl[n][str(tg)]["realised_ratio_mean"] = float(
                np.mean([r["ratio"] for r in sub]))
            ctl[n][str(tg)]["predicted_shipped_over_true"] = float(
                np.mean([1 / math.sqrt(r["ratio"]) for r in sub]))
            ctl[n][str(tg)]["paired_shipped_vs_union"] = paired_tests(sub, n)
    summary["real_controlled_ratio"] = {"n_records": len(CR), "by_N_by_ratio": ctl,
                                        "overall_by_N": {
        n: {e: cell_stats(CR, e, n) for e in ESTS} |
           {"paired_shipped_vs_union": paired_tests(CR, n)} for n in N_SIZES}}

    # ---------------- E5
    ag = e5["aggregate"]
    summary["e5_recheck"] = {
        "pooled_auc": {lab: ({n: ag[lab][n]["pooled_auc"] for n in N_SIZES}
                             if lab != "cosine_ground_truth"
                             else {"exact": ag[lab]["exact"]["pooled_auc"]})
                       for lab in ag},
        "bootstrap_ci": e5["bootstrap_ci"],
        "auc_diff_union_minus_shipped": e5["auc_diff_union_minus_shipped"],
        "support_ratio_range": [
            min(v["supp_ratio"] for v in e5["support_sizes"].values()),
            max(v["supp_ratio"] for v in e5["support_sizes"].values())],
        "mean_cosine_by_level": {
            lab: ({n: [p["mean_cosine"] for p in ag[lab][n]["per_level"]]
                   for n in N_SIZES} if lab != "cosine_ground_truth"
                  else {"exact": [p["mean_cosine"]
                                  for p in ag[lab]["exact"]["per_level"]]})
            for lab in ag},
        "sigmas": e5["parameters"]["sigmas"]}

    with open(os.path.join(RES, "f1_cosine_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # ============================ FIGURES ============================
    # Fig 1: bias vs support-size ratio, synthetic, one panel per N
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, n in zip(axes, N_SIZES):
        for e in ESTS:
            if e == "intersect_only":
                continue
            m = [syn_tbl[n][str(rr)][e]["mean_bias"] for rr in ratios]
            sd = [syn_tbl[n][str(rr)][e]["sd_bias"] / math.sqrt(
                syn_tbl[n][str(rr)][e]["n"]) for rr in ratios]
            ax.errorbar(ratios, m, yerr=sd, marker="o", ms=4, capsize=2,
                        color=COL[e], label=LBL[e])
        # analytic prediction curve for shipped
        tr = [syn_tbl[n][str(rr)]["shipped"]["mean_true"] for rr in ratios]
        ax.plot(ratios, [t * (1 / math.sqrt(rr) - 1) for t, rr in zip(tr, ratios)],
                "k--", lw=1, label=r"predicted: $c(1/\sqrt{r}-1)$")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"N = {n}")
        ax.set_xlabel("support-size ratio  r = |B|/|A|")
    axes[0].set_ylabel("mean bias  (estimate - true cosine)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("F1: cosine estimator bias vs support-size ratio "
                 f"(synthetic, {syn['parameters']['n_reps']} reps/cell)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_bias_vs_ratio.png"), dpi=200)
    plt.close(fig)

    # Fig 2: N-scaling of |mean bias|
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for e in ["shipped", "union_bottomN"]:
        for rr, mk in zip([1.0, 1.5, 2.0, 3.0], ["o", "s", "^", "d"]):
            y = [abs(nscale[str(rr)][e][n]) for n in N_SIZES]
            axes[0].plot([int(n) for n in N_SIZES], y, marker=mk, color=COL[e],
                         alpha=0.35 + 0.2 * [1.0, 1.5, 2.0, 3.0].index(rr),
                         label=f"{e}, r={rr}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylim(3e-4, 0.5)
    axes[0].set_xlabel("sketch size N")
    axes[0].set_ylabel("|mean bias|")
    axes[0].set_title("Does the bias vanish with N?")
    axes[0].legend(fontsize=6, ncol=2)
    for n, mk in zip(N_SIZES, ["o", "s", "^"]):
        obs = [syn_tbl[n][str(rr)]["shipped"]["mean_ratio_est_over_true"]
               for rr in ratios]
        axes[1].plot(ratios, obs, marker=mk, ls="-", color=COL["shipped"],
                     alpha=0.4 + 0.25 * N_SIZES.index(n), label=f"observed N={n}")
    axes[1].plot(ratios, [1 / math.sqrt(rr) for rr in ratios], "k--",
                 label=r"prediction $1/\sqrt{r}$")
    axes[1].set_xlabel("support-size ratio r")
    axes[1].set_ylabel("mean(shipped / true)")
    axes[1].set_title("Shipped cosine attenuation factor")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_N_scaling.png"), dpi=200)
    plt.close(fig)

    # Fig 3: real genomes
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    for e in ["shipped", "union_bottomN"]:
        ax.scatter([r["cos_true"] for r in NP],
                   [r["est"]["10000"][e] for r in NP], s=10, alpha=0.6,
                   color=COL[e], label=LBL[e])
    lim = [0, max(r["cos_true"] for r in NP) * 1.1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("true cosine (full k-mer vectors)")
    ax.set_ylabel("estimate")
    ax.set_title(f"120 natural genome pairs, N=10000\n"
                 f"ratio {summary['real_natural_pairs']['ratio_range'][0]:.2f}"
                 f"-{summary['real_natural_pairs']['ratio_range'][1]:.2f}")
    ax.legend(fontsize=7)
    ax = axes[1]
    for e in ["shipped", "union_bottomN"]:
        xs = [ctl["10000"][str(tg)]["realised_ratio_mean"] for tg in tgts]
        ys = [ctl["10000"][str(tg)][e]["mean_bias"] for tg in tgts]
        es = [ctl["10000"][str(tg)][e]["sd_bias"] /
              math.sqrt(ctl["10000"][str(tg)][e]["n"]) for tg in tgts]
        ax.errorbar(xs, ys, yerr=es, marker="o", color=COL[e], capsize=2,
                    label=LBL[e])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("realised support-size ratio r")
    ax.set_ylabel("mean bias")
    ax.set_title("Real genomes, controlled ratio (N=10000)\n"
                 f"n={len(CR)//len(tgts)} per ratio")
    ax.legend(fontsize=7)
    ax = axes[2]
    for e in ["shipped", "union_bottomN"]:
        ys = [summary["real_controlled_ratio"]["overall_by_N"][n][e]["rmse"]
              for n in N_SIZES]
        ax.plot([int(n) for n in N_SIZES], ys, marker="o", color=COL[e],
                label=LBL[e] + " (controlled)")
        ys = [summary["real_natural_pairs"]["by_N"][n][e]["rmse"] for n in N_SIZES]
        ax.plot([int(n) for n in N_SIZES], ys, marker="s", ls="--",
                color=COL[e], alpha=0.6, label=LBL[e] + " (natural)")
    ax.set_xscale("log")
    ax.set_xlabel("sketch size N")
    ax.set_ylabel("RMSE vs true cosine")
    ax.set_title("RMSE, real genomes")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_real_genomes.png"), dpi=200)
    plt.close(fig)

    # Fig 4: E5 recheck
    sig = e5["parameters"]["sigmas"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for lab, c, ls in [("cosine_shipped", COL["shipped"], "-"),
                       ("cosine_union_bottomN", COL["union_bottomN"], "-"),
                       ("cosine_ground_truth", "k", "--")]:
        key = "1000" if lab != "cosine_ground_truth" else "exact"
        pl = ag[lab][key]["per_level"]
        m = [p["mean_cosine"] for p in pl]
        s = [p["sd"] / math.sqrt(p["n"]) for p in pl]
        ax.errorbar(sig, m, yerr=s, marker="o", color=c, ls=ls, capsize=2,
                    label=lab + (f" (N={key})" if key != "exact" else " (exact)"))
    ax.set_xlabel(r"PCR amplification skew $\sigma$")
    ax.set_ylabel("mean cosine vs reference")
    ax.set_title("E5 control (set-invariant, depth-matched)\nn=10 seeds per level")
    ax.legend(fontsize=7)
    ax = axes[1]
    w = 0.35
    xs = np.arange(len(N_SIZES))
    for i, (lab, c) in enumerate([("cosine_shipped", COL["shipped"]),
                                  ("cosine_union_bottomN", COL["union_bottomN"])]):
        v = [ag[lab][n]["pooled_auc"] for n in N_SIZES]
        lo = [e5["bootstrap_ci"][f"{lab}|{n}"]["ci95"][0] for n in N_SIZES]
        hi = [e5["bootstrap_ci"][f"{lab}|{n}"]["ci95"][1] for n in N_SIZES]
        ax.bar(xs + (i - 0.5) * w, v, w, color=c, label=lab,
               yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)],
               capsize=3)
    gt = ag["cosine_ground_truth"]["exact"]["pooled_auc"]
    ax.axhline(gt, color="k", ls="--", lw=1, label=f"exact-cosine oracle ({gt:.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"N={n}" for n in N_SIZES])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("pooled AUC (skew vs no skew)")
    ax.set_title("E5 headline under shipped vs corrected cosine\n"
                 "error bars = 95% bootstrap CI (B=10000)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_e5_recheck.png"), dpi=200)
    plt.close(fig)

    print("wrote summary + 4 figures")
    for n in N_SIZES:
        o = summary["synthetic_overall"][n]
        print(f"N={n:>5}  synthetic all cells: shipped bias {o['shipped']['mean_bias']:+.4f} "
              f"rmse {o['shipped']['rmse']:.4f} | union bias "
              f"{o['union_bottomN']['mean_bias']:+.4f} rmse {o['union_bottomN']['rmse']:.4f} "
              f"| wilcoxon p={o['paired_shipped_vs_union']['wilcoxon_p']:.3g}")


if __name__ == "__main__":
    main()
