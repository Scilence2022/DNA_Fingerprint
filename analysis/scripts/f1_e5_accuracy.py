#!/usr/bin/env python3
"""
F1 addendum -- how ACCURATE (not just how discriminative) are the sketch cosines
in E5's DNA-storage regime? Compares every sketch estimator against the exact
full-coverage-vector cosine computed in f1_e5_recheck.py, per condition.

Motivation: E5 only ever used the cosine as a RANKING score, so its AUC can be
perfect even if the absolute values are far from the estimand. This script
separates the two questions and reports both.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "analysis", "results")
FIG = os.path.join(REPO, "analysis", "figures")
N_SIZES = ["100", "1000", "10000"]
ESTS = ["shipped", "union_bottomN", "intersect_only", "shipped_analytic"]


def main():
    e5 = json.load(open(os.path.join(RES, "f1_e5_cosine_recheck.json")))
    est = e5["per_record_estimates"]
    truth = e5["per_record_truth"]
    supp = e5["support_sizes"]
    sig = e5["parameters"]["sigmas"]
    tags = sorted(truth)

    out = {"experiment_id": "F1_E5_cosine_accuracy_vs_truth",
           "n_conditions": len(tags),
           "support_ratio": {
               "min": min(v["supp_ratio"] for v in supp.values()),
               "max": max(v["supp_ratio"] for v in supp.values()),
               "mean": float(np.mean([v["supp_ratio"] for v in supp.values()])),
               "note": "|supp(sample)| / |supp(reference)| over full read-set "
                       "k-mer sets; r ~ 1 means the shipped estimator's "
                       "structural ratio bias is not engaged here"},
           "by_N": {}, "by_N_by_level": {}}

    for n in N_SIZES:
        out["by_N"][n] = {}
        for e in ESTS:
            b = np.array([est[t][n][e] - truth[t] for t in tags])
            out["by_N"][n][e] = {
                "n": int(b.size), "mean_bias": float(b.mean()),
                "sd_bias": float(b.std(ddof=1)),
                "rmse": float(np.sqrt((b ** 2).mean())),
                "max_abs_bias": float(np.abs(b).max()),
                "spearman_with_truth": float(stats.spearmanr(
                    [est[t][n][e] for t in tags], [truth[t] for t in tags])[0])}
        a = np.abs([est[t][n]["shipped"] - truth[t] for t in tags])
        c = np.abs([est[t][n]["union_bottomN"] - truth[t] for t in tags])
        w = stats.wilcoxon(a, c)
        out["by_N"][n]["paired_shipped_vs_union_abs_bias"] = {
            "n_pairs": len(tags), "mean_abs_shipped": float(np.mean(a)),
            "mean_abs_union": float(np.mean(c)),
            "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue)}

    for n in N_SIZES:
        out["by_N_by_level"][n] = {}
        for li, sg in enumerate(sig):
            sub = [t for t in tags if t.startswith(f"ctrl_L{li}_")]
            out["by_N_by_level"][n][str(sg)] = {
                "n": len(sub),
                "mean_true": float(np.mean([truth[t] for t in sub])),
                **{e: float(np.mean([est[t][n][e] for t in sub])) for e in ESTS}}

    p = os.path.join(RES, "f1_e5_cosine_accuracy.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", p)
    print("support ratio", out["support_ratio"]["min"], "-",
          out["support_ratio"]["max"])
    for n in N_SIZES:
        d = out["by_N"][n]
        print(f"N={n:>5}: shipped bias {d['shipped']['mean_bias']:+.4f} "
              f"rmse {d['shipped']['rmse']:.4f} rho={d['shipped']['spearman_with_truth']:.3f}"
              f" | union bias {d['union_bottomN']['mean_bias']:+.4f} "
              f"rmse {d['union_bottomN']['rmse']:.4f} "
              f"rho={d['union_bottomN']['spearman_with_truth']:.3f}"
              f" | wilcoxon p={d['paired_shipped_vs_union_abs_bias']['wilcoxon_p']:.3g}")


def figure():
    """Companion figure: E5 accuracy (values) vs discrimination (ranking)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    acc = json.load(open(os.path.join(RES, "f1_e5_cosine_accuracy.json")))
    rec = json.load(open(os.path.join(RES, "f1_e5_cosine_recheck.json")))
    sig = rec["parameters"]["sigmas"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    lv = acc["by_N_by_level"]["10000"]
    ax.plot(sig, [lv[str(s)]["mean_true"] for s in sig], "k--o",
            label="exact cosine (full coverage vectors)")
    for e, c in [("shipped", "#d62728"), ("union_bottomN", "#1f77b4")]:
        ax.plot(sig, [lv[str(s)][e] for s in sig], marker="o", color=c,
                label=f"{e} (N=10000)")
    ax.set_xlabel(r"amplification skew $\sigma$")
    ax.set_ylabel("cosine")
    ax.set_title("E5 regime: BOTH sketch cosines are far below\n"
                 "the true cosine (support ~3.3M 21-mers, N=10000)")
    ax.legend(fontsize=7)
    ax = axes[1]
    xs = np.arange(len(N_SIZES))
    w = 0.35
    for i, (e, c) in enumerate([("shipped", "#d62728"),
                                ("union_bottomN", "#1f77b4")]):
        ax.bar(xs + (i - 0.5) * w, [acc["by_N"][n][e]["rmse"] for n in N_SIZES],
               w, color=c, label=e)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"N={n}" for n in N_SIZES])
    ax.set_ylabel("RMSE vs exact cosine")
    ax.set_title("E5 regime: the 'corrected' estimator is NOT more\n"
                 "accurate here (r~1; heavy-tailed read coverage)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_e5_accuracy.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
    figure()
