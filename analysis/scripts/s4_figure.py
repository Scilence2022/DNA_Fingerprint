#!/usr/bin/env python3
"""S4 consolidated figure: integrity verification of stored DNA-data pools.

Four panels, all from analysis/results/s4_integrity.json:
 A  per-degradation-mode detection AUC (abundance-aware vs presence/absence)
 B  decisive amplification-bias control (set-invariant, depth-matched)
 C  depth robustness: naive vs common-index tolerance window
 D  mechanism: error-k-mer inflation drives naive depth sensitivity
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
S = json.load(open(os.path.join(REPO, "analysis", "results", "s4_integrity.json")))
FIG = os.path.join(REPO, "analysis", "figures", "s4_integrity.png")

C_COS = "#1b7837"     # abundance-aware
C_JD = "#f1a340"      # jaccard direct
C_JU = "#998ec3"      # jaccard common-index
C_MASH = "#d6604d"    # mash presence/absence

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
fig.suptitle("S4  Integrity verification of stored DNA-data pools "
             "(simulated encoded oligo pools; k=21, N=1000 unless noted)",
             fontsize=13, fontweight="bold")

# ---- Panel A: per-mode pooled detection AUC, N=1000 --------------------
axA = axes[0, 0]
modes = ["subst", "dropout", "pcrbias", "combined"]
mode_lbl = ["substitution", "oligo dropout", "amplification\nbias", "combined"]
metrics = ["cosine", "jaccard_union", "mash_distance"]
mcolor = {"cosine": C_COS, "jaccard_union": C_JU, "mash_distance": C_MASH}
mname = {"cosine": "cosine (abundance)", "jaccard_union": "Jaccard common-index",
         "mash_distance": "mash (presence/absence)"}
x = np.arange(len(modes))
w = 0.26
for i, met in enumerate(metrics):
    vals = [S["part1_per_mode_detection"][m]["metrics"][met]["N"]["1000"]["pooled_auc"]
            for m in modes]
    axA.bar(x + (i - 1) * w, vals, w, color=mcolor[met], label=mname[met])
axA.axhline(0.5, ls=":", color="gray", lw=1)
axA.set_xticks(x)
axA.set_xticklabels(mode_lbl, fontsize=9)
axA.set_ylabel("pooled detection AUC")
axA.set_ylim(0.4, 1.03)
axA.set_title("A  Detection of each degradation mode (N=1000)", fontsize=11,
              loc="left", fontweight="bold")
axA.legend(fontsize=8, loc="lower left")
axA.grid(axis="y", alpha=0.3)

# ---- Panel B: amplification-bias control -------------------------------
axB = axes[0, 1]
ctrl = S["part2_amplification_bias_control"]["metrics"]
order = ["cosine", "jaccard_direct", "jaccard_union", "mash_distance"]
labels = ["cosine\n(abundance)", "Jaccard\ndirect", "Jaccard\ncommon-idx",
          "mash\n(pres/abs)"]
cols = [C_COS, C_JD, C_JU, C_MASH]
Nsel = "10000"
aucs = [ctrl[m]["N"][Nsel]["pooled_auc"] for m in order]
ps = [ctrl[m]["N"][Nsel]["mannwhitney_p"] for m in order]
bars = axB.bar(range(len(order)), aucs, color=cols)
axB.axhline(0.5, ls=":", color="gray", lw=1)
axB.text(3.3, 0.52, "chance", fontsize=8, color="gray")
for b, a, p in zip(bars, aucs, ps):
    axB.text(b.get_x() + b.get_width() / 2, a + 0.02 if a < 0.9 else a - 0.08,
             f"{a:.3f}\np={p:.1e}", ha="center", fontsize=7.5)
axB.set_xticks(range(len(order)))
axB.set_xticklabels(labels, fontsize=8.5)
axB.set_ylabel("detection AUC")
axB.set_ylim(0, 1.08)
axB.set_title("B  Pure amplification bias, set-invariant & depth-matched (N=10000)",
              fontsize=10.5, loc="left", fontweight="bold")
axB.grid(axis="y", alpha=0.3)

# ---- Panel C: depth robustness -----------------------------------------
axC = axes[1, 0]
dp = S["part3_depth_robustness"]["metrics"]
ratios = S["part3_depth_robustness"]["ratios"]
# fixed-ratio detection AUC across depth ratio at sigma=0.25, N=1000
for met, col, lbl in [("cosine", C_COS, "naive cosine"),
                      ("cosine_common", "#08519c", "common-index cosine")]:
    fr = dp[met]["1000"]["fixed_ratio_auc"]
    ys = [fr[str(r)] for r in ratios]
    axC.plot(ratios, ys, "o-", color=col, label=lbl, lw=2)
    tol = dp[met]["1000"]["tolerance_window_ratio"]
    if tol:
        axC.axvline(tol, ls="--", color=col, alpha=0.5)
axC.axhline(0.95, ls=":", color="gray", lw=1)
axC.set_xscale("log")
axC.set_xlabel("query:reference depth ratio")
axC.set_ylabel("detection AUC (sigma=0.25 vs intact)")
axC.set_title("C  Depth robustness: common-index removes depth sensitivity",
              fontsize=10.5, loc="left", fontweight="bold")
axC.set_ylim(0.4, 1.05)
axC.legend(fontsize=8.5, loc="lower left")
axC.grid(alpha=0.3, which="both")

# ---- Panel D: mechanism ------------------------------------------------
axD = axes[1, 1]
mech = S["part3_depth_robustness"]["mechanism"]["error_kmer_inflation_N1000_sigma0"]
rs = [float(r) for r in mech]
fr = [mech[r] for r in mech]
axD.plot(rs, fr, "s-", color="#b2182b", lw=2)
axD.set_xscale("log")
axD.set_xlabel("query:reference depth ratio")
axD.set_ylabel("singleton (error) k-mer fraction in sketch")
axD.set_title("D  Mechanism: deeper reads inflate error k-mers in the sketch",
              fontsize=10.5, loc="left", fontweight="bold")
axD.grid(alpha=0.3, which="both")
zero = S["part3_depth_robustness"]["mechanism"][
    "zero_error_cosine_depth_invariance_stageB"]["distinct_cosine_values_over_all_ratios_sigma0"]
axD.text(0.12, 0.72, "zero-error control (stage B):\nnaive cosine = %s at every depth"
         % ", ".join(f"{v:g}" for v in zero),
         fontsize=8, color="#08519c",
         bbox=dict(boxstyle="round", fc="#eef4fb", ec="#08519c", alpha=0.9))

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG, dpi=200)
print("wrote", FIG)
