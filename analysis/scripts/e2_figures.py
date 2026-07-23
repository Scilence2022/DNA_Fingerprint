#!/usr/bin/env python3
"""Figures for Experiment E2."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/results"
FIGURES = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/figures"
os.makedirs(FIGURES, exist_ok=True)

A = json.load(open(os.path.join(RESULTS, "e2_jaccard_bias.json")))
Bf = json.load(open(os.path.join(RESULTS, "e2b_ratio_threshold.json")))
C = json.load(open(os.path.join(RESULTS, "e2c_threshold_zoom.json")))

CD, CU = "#c0392b", "#1f6fb4"

# --- Fig 1: bias vs J_true, one panel per size ratio -----------------------
ratios = A["regime_A"]["design"]["ratios"]
fig, axes = plt.subplots(2, len(ratios), figsize=(3.0 * len(ratios), 6.2), sharey="row")
for col, ra in enumerate(ratios):
    rows = [r for r in A["regime_A"]["cell_summary"] if r["ratio"] == ra]
    rows.sort(key=lambda r: r["j_true_mean"])
    jt = np.array([r["j_true_mean"] for r in rows])
    for row_i, n in enumerate([1000, 10000]):
        ax = axes[row_i, col]
        for est, c, lab in (("direct", CD, "J_direct (current code)"),
                            ("union", CU, "J_union (Mash, unbiased)")):
            b = np.array([r[f"{est}_N{n}_bias"] for r in rows])
            se = np.array([r[f"{est}_N{n}_bias_se"] for r in rows])
            ax.errorbar(jt, b, yerr=1.96 * se, marker="o", ms=4, lw=1.4,
                        color=c, capsize=2, label=lab)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.axhspan(-0.01, 0.01, color="0.85", zorder=0)
        ax.set_xlabel("true Jaccard")
        if col == 0:
            ax.set_ylabel(f"bias  E[J_hat] - J_true\n(N = {n})")
        if row_i == 0:
            ax.set_title(f"size ratio |B|/|A| = {ra}", fontsize=10)
        ax.grid(alpha=0.25)
axes[0, 0].legend(fontsize=7.5, loc="lower left")
fig.suptitle("E2 Fig 1 -- Jaccard estimator bias vs true Jaccard, by genome-size ratio\n"
             "(k=31, n=20 replicates/cell, error bars 95% CI; grey band = +-0.01 materiality)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIGURES, "e2_fig1_bias_vs_jtrue_by_ratio.png"), dpi=200)
plt.close(fig)

# --- Fig 2: J_direct vs J_union on real genome pairs ----------------------
pairs = A["regime_B"]["pairs"]
jt = np.array([p["j_true"] for p in pairs])
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
for ax, n in zip(axes[:2], [1000, 10000]):
    jd = np.array([p[f"j_direct_N{n}"] for p in pairs])
    ju = np.array([p[f"j_union_N{n}"] for p in pairs])
    sr = np.array([p["size_ratio"] for p in pairs])
    s = ax.scatter(ju, jd, c=sr, cmap="viridis", s=26, edgecolor="k", lw=0.3)
    lim = [0, max(jd.max(), ju.max()) * 1.08]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("J_union (unbiased)")
    ax.set_ylabel("J_direct (current code)")
    ax.set_title(f"16 RefSeq genomes, 120 pairs, N={n}")
    plt.colorbar(s, ax=ax, label="genome size ratio")
    ax.grid(alpha=0.25)
ax = axes[2]
for n, c, m in ((1000, "#888888", "o"), (10000, "#000000", "^")):
    jd = np.array([p[f"j_direct_N{n}"] for p in pairs])
    ju = np.array([p[f"j_union_N{n}"] for p in pairs])
    ax.scatter(jt, jd - jt, c=CD, marker=m, s=22, alpha=0.75,
               label=f"J_direct, N={n}")
    ax.scatter(jt, ju - jt, c=CU, marker=m, s=22, alpha=0.75,
               label=f"J_union, N={n}")
ax.axhline(0, color="0.4", lw=0.8, ls="--")
ax.axhspan(-0.01, 0.01, color="0.85", zorder=0)
ax.set_xlabel("true Jaccard (exact, k=31)")
ax.set_ylabel("estimate - truth")
ax.set_title("error vs exact truth, real genomes")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)
fig.suptitle("E2 Fig 2 -- direct vs unbiased Jaccard on real bacterial genome pairs", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIGURES, "e2_fig2_real_genomes_direct_vs_union.png"), dpi=200)
plt.close(fig)

# --- Fig 3: bias vs size ratio (threshold) + bias vs N --------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
allrows = C["cell_summary"] + [r for r in Bf["fine_ratio_sweep"]["cell_summary"]
                               if r["ratio"] > 1.05]
for rate, mk in ((0.0, "o"), (0.002, "s"), (0.01, "^")):
    rows = sorted([r for r in allrows if r["rate"] == rate], key=lambda r: r["ratio"])
    x = [r["ratio"] for r in rows]
    ax.errorbar(x, [abs(r["direct_N10000_bias"]) for r in rows],
                yerr=[1.96 * r["direct_N10000_bias_se"] for r in rows],
                marker=mk, color=CD, lw=1.3, ms=4, capsize=2,
                label=f"J_direct, sub. rate {rate}")
    ax.errorbar(x, [abs(r["union_N10000_bias"]) for r in rows],
                yerr=[1.96 * r["union_N10000_bias_se"] for r in rows],
                marker=mk, color=CU, lw=1.3, ms=4, capsize=2, ls=":",
                label=f"J_union, sub. rate {rate}")
ax.axhline(0.01, color="k", ls="--", lw=1)
ax.text(1.6, 0.0115, "materiality threshold |bias| = 0.01", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("genome size ratio |B|/|A|")
ax.set_ylabel("|bias| at N = 10000")
ax.set_title("Where the direct estimator breaks (n=30/cell)")
ax.legend(fontsize=7)
ax.grid(alpha=0.25, which="both")

ax = axes[1]
for r in Bf["N_sweep"]["cell_summary"]:
    Ns = [100, 1000, 10000, 100000]
    ax.plot(Ns, [r[f"direct_N{n}_bias"] for n in Ns], "o-", color=CD, lw=1.3, ms=4)
    ax.plot(Ns, [r[f"union_N{n}_bias"] for n in Ns], "s:", color=CU, lw=1.3, ms=4)
    ax.annotate(f"ratio {r['ratio']}", (100, r["direct_N100_bias"]),
                fontsize=8, color=CD, xytext=(4, -10), textcoords="offset points")
ax.axhline(0, color="0.4", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xlabel("sketch size N")
ax.set_ylabel("bias")
ax.set_title("Bias vs N (red = direct, blue = union)\nsub. rate 0.005, n=30")
ax.grid(alpha=0.25)

ax = axes[2]
Ns = [100, 1000, 10000, 100000]
for r in Bf["N_sweep"]["cell_summary"]:
    ax.plot(Ns, [r[f"direct_N{n}_resid_sd"] for n in Ns], "o-", color=CD, lw=1.2, ms=4)
    ax.plot(Ns, [r[f"union_N{n}_resid_sd"] for n in Ns], "s:", color=CU, lw=1.2, ms=4)
    ax.plot(Ns, [r[f"union_N{n}_binom_sd"] for n in Ns], "k--", lw=1.0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("sketch size N")
ax.set_ylabel("SD of (J_hat - J_true)")
ax.set_title("Empirical SE vs binomial sqrt(J(1-J)/N)\n(black dashed = theory)")
ax.grid(alpha=0.25, which="both")
fig.suptitle("E2 Fig 3 -- size-ratio threshold, bias-vs-N, and variance vs theory", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIGURES, "e2_fig3_threshold_and_variance.png"), dpi=200)
plt.close(fig)
print("figures written")
