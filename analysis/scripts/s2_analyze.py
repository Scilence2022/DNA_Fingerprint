#!/usr/bin/env python3
"""
S2 analysis + figures. Reads the two raw JSONs, computes bias/RMSE, the
direct-Jaccard bias>0.01 crossover, the cosine sqrt(r) law fit, the bias-vs-N
OLS slopes, and paired Wilcoxon tests. Writes s2_summary.json and figures.
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

SWEEP = os.path.join(RES, "s2_support_sweep.json")
DEPTH = os.path.join(RES, "s2_depth_verification.json")
SUMMARY = os.path.join(RES, "s2_summary.json")


def load(p):
    with open(p) as fh:
        return json.load(fh)


def wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return float("nan"), 1.0
    try:
        s, p = stats.wilcoxon(a, b)
        return float(s), float(p)
    except ValueError:
        return float("nan"), 1.0


# ------------------------------------------------------------------ part (a,b)
def analyze_sweep(sw):
    recs = sw["records"]
    ratios = sw["parameters"]["ratios"]
    Ns = [str(n) for n in sw["parameters"]["N_sizes"]]

    by = {}
    for r in recs:
        by.setdefault((r["ratio"],), []).append(r)

    rows = []
    for ratio in ratios:
        cell = by[(ratio,)]
        jt = np.array([c["j_true"] for c in cell])
        ct = np.array([c["c_true"] for c in cell])
        rr = np.array([c["r_realised"] for c in cell])
        row = {"ratio": ratio, "n_seeds": len(cell),
               "r_realised_mean": float(rr.mean()),
               "j_true_mean": float(jt.mean()), "j_true_sd": float(jt.std(ddof=1)),
               "c_true_mean": float(ct.mean()), "c_true_sd": float(ct.std(ddof=1)),
               "N": {}}
        for n in Ns:
            jd = np.array([c["N"][n]["j_direct"] for c in cell])
            ju = np.array([c["N"][n]["j_union"] for c in cell])
            cd = np.array([c["N"][n]["c_direct"] for c in cell])
            cu = np.array([c["N"][n]["c_union"] for c in cell])
            def stat(est, truth):
                bias = est - truth
                return {"mean": float(est.mean()),
                        "bias": float(bias.mean()),
                        "bias_se": float(bias.std(ddof=1) / math.sqrt(len(est))),
                        "rmse": float(np.sqrt((bias ** 2).mean()))}
            _, pj = wilcoxon(np.abs(jd - jt), np.abs(ju - jt))
            _, pc = wilcoxon(np.abs(cd - ct), np.abs(cu - ct))
            row["N"][n] = {
                "j_direct": stat(jd, jt), "j_union": stat(ju, jt),
                "c_direct": stat(cd, ct), "c_union": stat(cu, ct),
                "wilcoxon_p_absbias_direct_vs_union_jaccard": pj,
                "wilcoxon_p_absbias_direct_vs_union_cosine": pc,
                # cosine sqrt(r) law: mean of (c_direct/c_true) vs 1/sqrt(r)
                "cosine_ratio_mean": float(np.mean(cd / ct)),
                "cosine_ratio_pred_1_over_sqrt_r": float(np.mean(1.0 / np.sqrt(rr))),
            }
        rows.append(row)

    # crossover: smallest r at which |direct jaccard bias| exceeds 0.01 (N=10000)
    cross = None
    for row in rows:
        if abs(row["N"]["10000"]["j_direct"]["bias"]) > 0.01:
            cross = row["ratio"]
            break

    # sqrt(r) law fit across all cells at N=10000: c_direct vs c_true/sqrt(r)
    rr_all, cd_all, ct_all = [], [], []
    for c in recs:
        rr_all.append(c["r_realised"]); cd_all.append(c["N"]["10000"]["c_direct"])
        ct_all.append(c["c_true"])
    rr_all = np.array(rr_all); cd_all = np.array(cd_all); ct_all = np.array(ct_all)
    pred = ct_all / np.sqrt(rr_all)
    lr = stats.linregress(pred, cd_all)
    sqrt_law = {"slope": float(lr.slope), "intercept": float(lr.intercept),
                "r2": float(lr.rvalue ** 2), "n": int(rr_all.size),
                "mean_abs_resid": float(np.mean(np.abs(cd_all - pred)))}

    # part (b): OLS slope of |bias| on log10 N, per estimator, pooled over r>1 cells
    def slope_logN(estkey, truthkey):
        xs, ys = [], []
        for c in recs:
            if c["r_realised"] < 1.01:
                continue
            t = c[truthkey]
            for n in Ns:
                xs.append(math.log10(int(n)))
                ys.append(abs(c["N"][n][estkey] - t))
        lr = stats.linregress(xs, ys)
        return {"slope_per_decade": float(lr.slope), "r2": float(lr.rvalue ** 2),
                "p": float(lr.pvalue), "n": len(xs)}
    partb = {
        "j_direct_absbias_vs_log10N": slope_logN("j_direct", "j_true"),
        "j_union_absbias_vs_log10N": slope_logN("j_union", "j_true"),
        "c_direct_absbias_vs_log10N": slope_logN("c_direct", "c_true"),
        "c_union_absbias_vs_log10N": slope_logN("c_union", "c_true"),
        "note": "pooled over r>1 cells and all seeds; direct ~flat, union shrinks",
    }
    return rows, cross, sqrt_law, partb


# -------------------------------------------------------------------- part (c)
def analyze_depth(dp):
    recs = dp["records"]
    dqs = dp["parameters"]["d_qry"]
    d_ref = dp["parameters"]["d_ref"]
    Ns = [str(n) for n in dp["parameters"]["N_sizes"]]
    out = {}
    for identity in ("matched", "mismatch"):
        rows = []
        for dq in dqs:
            cell = [r for r in recs if r["identity"] == identity and r["d_qry"] == dq]
            sr = np.array([c["support_ratio"] for c in cell])
            row = {"d_qry": dq, "depth_ratio": max(d_ref, dq) / min(d_ref, dq),
                   "support_ratio_mean": float(sr.mean()), "n_seeds": len(cell),
                   "target": cell[0]["target"], "N": {}}
            for n in Ns:
                cd = np.array([c["N"][n]["c_direct"] for c in cell])
                cu = np.array([c["N"][n]["c_union"] for c in cell])
                jd = np.array([c["N"][n]["j_direct"] for c in cell])
                ju = np.array([c["N"][n]["j_union"] for c in cell])
                row["N"][n] = {
                    "c_direct_mean": float(cd.mean()), "c_direct_sd": float(cd.std(ddof=1)),
                    "c_union_mean": float(cu.mean()), "c_union_sd": float(cu.std(ddof=1)),
                    "j_direct_mean": float(jd.mean()), "j_union_mean": float(ju.mean()),
                }
            rows.append(row)
        out[identity] = rows

    # matched-cosine: depth-robustness. reference value = matched-depth (dq=d_ref) union cosine.
    matched = out["matched"]
    n = "10000"
    base_un = [r for r in matched if r["d_qry"] == d_ref][0]["N"][n]["c_union_mean"]
    dev_direct = max(abs(r["N"][n]["c_direct_mean"] - base_un) for r in matched)
    dev_union = max(abs(r["N"][n]["c_union_mean"] - base_un) for r in matched)
    # paired wilcoxon at the largest depth gap (dq=5 vs matched): direct deviation vs union deviation
    worst_dq = min(dp["parameters"]["d_qry"], key=lambda d: min(d, d_ref) / max(d, d_ref))
    cd = [c["N"][n]["c_direct"] for c in recs if c["identity"] == "matched" and c["d_qry"] == worst_dq]
    cu = [c["N"][n]["c_union"] for c in recs if c["identity"] == "matched" and c["d_qry"] == worst_dq]
    _, p_worst = wilcoxon(np.abs(np.array(cd) - 1.0), np.abs(np.array(cu) - 1.0))
    robustness = {
        "matched_depth_union_cosine": base_un,
        "max_dev_direct_cosine_from_matched_depth": dev_direct,
        "max_dev_union_cosine_from_matched_depth": dev_union,
        "worst_gap_d_qry": worst_dq,
        "wilcoxon_p_dist_to_1_direct_vs_union_worst_gap": p_worst,
    }
    return out, robustness


# --------------------------------------------------------------------- figures
def figures(rows, sqrt_law, depth_rows, sw, dp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = [r["ratio"] for r in rows]

    # Fig 1: bias vs r (Jaccard + cosine), N=10000
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for est, c, lbl in [("j_direct", "#d1495b", "direct (own index)"),
                        ("j_union", "#2e7d32", "common index (union)")]:
        b = [r["N"]["10000"][est]["bias"] for r in rows]
        se = [r["N"]["10000"][est]["bias_se"] for r in rows]
        ax[0].errorbar(ratios, b, yerr=se, marker="o", color=c, label=lbl, capsize=3)
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].axhline(0.01, color="gray", ls=":", lw=0.8); ax[0].axhline(-0.01, color="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("support-size ratio r"); ax[0].set_ylabel("bias  (estimate - truth)")
    ax[0].set_title("Jaccard bias vs r  (N=10000, true J=0.25 fixed)")
    ax[0].legend(fontsize=8)

    for est, c, lbl in [("c_direct", "#d1495b", "direct (own index)"),
                        ("c_union", "#2e7d32", "common index (union)")]:
        b = [r["N"]["10000"][est]["bias"] for r in rows]
        se = [r["N"]["10000"][est]["bias_se"] for r in rows]
        ax[1].errorbar(ratios, b, yerr=se, marker="s", color=c, label=lbl, capsize=3)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xlabel("support-size ratio r"); ax[1].set_ylabel("bias  (estimate - truth)")
    ax[1].set_title("Cosine bias vs r  (N=10000)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_bias_vs_ratio.png"), dpi=200)
    plt.close(fig)

    # Fig 2: cosine sqrt(r) law -- c_direct vs c_true/sqrt(r)
    recs = sw["records"]
    pred = np.array([c["c_true"] / math.sqrt(c["r_realised"]) for c in recs])
    obs = np.array([c["N"]["10000"]["c_direct"] for c in recs])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(pred, obs, s=12, alpha=0.5, color="#3b6ea5")
    lo, hi = min(pred.min(), obs.min()), max(pred.max(), obs.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(r"predicted  $c_{true}/\sqrt{r}$")
    ax.set_ylabel(r"observed direct cosine $\hat c$")
    ax.set_title(f"Cosine sqrt(r) bias law\nslope={sqrt_law['slope']:.3f}, "
                 f"R^2={sqrt_law['r2']:.4f}, n={sqrt_law['n']}")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "s2_cosine_sqrt_law.png"), dpi=200)
    plt.close(fig)

    # Fig 3: bias vs N (part b), pooled per estimator over r>1
    Ns = [int(n) for n in sw["parameters"]["N_sizes"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for est, c, lbl in [("j_direct", "#d1495b", "direct"), ("j_union", "#2e7d32", "common index")]:
        means = []
        for n in Ns:
            vals = [abs(c2["N"][str(n)][est] - c2["j_true"]) for c2 in recs if c2["r_realised"] > 1.01]
            means.append(np.mean(vals))
        ax[0].plot(Ns, means, marker="o", color=c, label=lbl)
    ax[0].set_xscale("log"); ax[0].set_xlabel("sketch size N"); ax[0].set_ylabel("mean |bias|")
    ax[0].set_title("Jaccard |bias| vs N (r>1 cells)"); ax[0].legend(fontsize=8)
    for est, c, lbl in [("c_direct", "#d1495b", "direct"), ("c_union", "#2e7d32", "common index")]:
        means = []
        for n in Ns:
            vals = [abs(c2["N"][str(n)][est] - c2["c_true"]) for c2 in recs if c2["r_realised"] > 1.01]
            means.append(np.mean(vals))
        ax[1].plot(Ns, means, marker="s", color=c, label=lbl)
    ax[1].set_xscale("log"); ax[1].set_xlabel("sketch size N"); ax[1].set_ylabel("mean |bias|")
    ax[1].set_title("Cosine |bias| vs N (r>1 cells)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_bias_vs_N.png"), dpi=200)
    plt.close(fig)

    # Fig 4: part (c) verification under depth gap
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    m = depth_rows["matched"]
    dq = [r["d_qry"] for r in m]
    for est, c, lbl in [("c_direct_mean", "#d1495b", "direct (own index)"),
                        ("c_union_mean", "#2e7d32", "common index (union)")]:
        vals = [r["N"]["10000"][est] for r in m]
        sd = [r["N"]["10000"][est.replace("_mean", "_sd")] for r in m]
        ax[0].errorbar(dq, vals, yerr=sd, marker="o", color=c, label=lbl, capsize=3)
    ax[0].axhline(1.0, color="k", ls=":", lw=0.8, label="correct = 1.0 (same file)")
    ax[0].axvline(dp["parameters"]["d_ref"], color="gray", ls="--", lw=0.7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("query depth (ref=30x)")
    ax[0].set_ylabel("verification cosine")
    ax[0].set_title("Matched retrieval: same file, depth gap"); ax[0].legend(fontsize=8)

    # cosine vs support ratio, matched
    sr = [r["support_ratio_mean"] for r in m]
    order = np.argsort(sr)
    for est, c, lbl in [("c_direct_mean", "#d1495b", "direct"),
                        ("c_union_mean", "#2e7d32", "common index")]:
        vals = np.array([r["N"]["10000"][est] for r in m])
        ax[1].plot(np.array(sr)[order], vals[order], marker="o", color=c, label=lbl)
    ax[1].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("support-size ratio r (from error-k-mer inflation)")
    ax[1].set_ylabel("verification cosine")
    ax[1].set_title("Same mechanism: cosine vs support ratio"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_depth_verification.png"), dpi=200)
    plt.close(fig)

    return ["s2_bias_vs_ratio.png", "s2_cosine_sqrt_law.png",
            "s2_bias_vs_N.png", "s2_depth_verification.png"]


def main():
    sw = load(SWEEP)
    dp = load(DEPTH)
    rows, cross, sqrt_law, partb = analyze_sweep(sw)
    depth_rows, robustness = analyze_depth(dp)
    figs = figures(rows, sqrt_law, depth_rows, sw, dp)

    summary = {
        "experiment": "S2_common_index",
        "part_a_support_sweep": {
            "true_jaccard_fixed": sw["parameters"]["true_jaccard_by_design"],
            "cells": rows,
            "direct_jaccard_bias_exceeds_0.01_at_r": cross,
            "cosine_sqrt_r_law_fit_N10000": sqrt_law,
        },
        "part_b_bias_vs_N": partb,
        "part_c_depth_verification": {
            "rows": depth_rows,
            "robustness": robustness,
        },
        "figures": [os.path.join(FIG, f) for f in figs],
    }
    with open(SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", SUMMARY)

    # console recap
    print("\n== part (a) N=10000, true J=0.25 fixed ==")
    for r in rows:
        j = r["N"]["10000"]
        print(f"  r={r['ratio']:<5} Jdir bias={j['j_direct']['bias']:+.4f} "
              f"Jun bias={j['j_union']['bias']:+.4f} | "
              f"Cdir bias={j['c_direct']['bias']:+.4f} Cun bias={j['c_union']['bias']:+.4f} "
              f"| Cdir/Ctrue={j['cosine_ratio_mean']:.3f} 1/sqrt(r)={j['cosine_ratio_pred_1_over_sqrt_r']:.3f} "
              f"| pJ={j['wilcoxon_p_absbias_direct_vs_union_jaccard']:.1e} pC={j['wilcoxon_p_absbias_direct_vs_union_cosine']:.1e}")
    print(f"  direct Jaccard |bias|>0.01 first at r = {cross}")
    print(f"  sqrt(r) law: slope={sqrt_law['slope']:.3f} R2={sqrt_law['r2']:.4f} "
          f"mean|resid|={sqrt_law['mean_abs_resid']:.4f}")
    print("\n== part (b) |bias| vs log10 N slope ==")
    for k, v in partb.items():
        if isinstance(v, dict):
            print(f"  {k:40s} slope/decade={v['slope_per_decade']:+.5f} R2={v['r2']:.3f} p={v['p']:.1e}")
    print("\n== part (c) matched retrieval, N=10000 ==")
    for r in depth_rows["matched"]:
        j = r["N"]["10000"]
        print(f"  dq={r['d_qry']:>3} rSupp={r['support_ratio_mean']:.2f} "
              f"Cdir={j['c_direct_mean']:.3f} Cun={j['c_union_mean']:.3f}")
    print(f"  union max dev from matched-depth value = {robustness['max_dev_union_cosine_from_matched_depth']:.3f}; "
          f"direct max dev = {robustness['max_dev_direct_cosine_from_matched_depth']:.3f}; "
          f"p(worst gap)={robustness['wilcoxon_p_dist_to_1_direct_vs_union_worst_gap']:.1e}")


if __name__ == "__main__":
    main()
