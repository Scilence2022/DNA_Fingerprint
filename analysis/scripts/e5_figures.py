#!/usr/bin/env python3
"""Figures + detection-floor table for E5 (DNA data storage integrity)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
RES = os.path.join(REPO, "analysis", "results", "e5_dna_storage.json")
FIGD = os.path.join(REPO, "analysis", "figures")
os.makedirs(FIGD, exist_ok=True)

D = json.load(open(RES))
LEVELS = D["parameters"]["levels"]
MODES = ["subst", "dropout", "pcrbias", "combined"]
TITLE = {"subst": "A  Substitution error (per-base rate)",
         "dropout": "B  Strand dropout (fraction of species lost)",
         "pcrbias": "C  PCR amplification bias (log-normal $\\sigma$)",
         "combined": "D  Combined failure (severity $t$)"}
XLAB = {"subst": "per-base substitution rate", "dropout": "species dropout fraction",
        "pcrbias": "amplification skew $\\sigma$", "combined": "combined severity $t$"}
SIZE = "1000"

# ---------------- Figure 1: similarity decay --------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, mode in zip(axes.ravel(), MODES):
    for metric, col, lab in [("cosine", "#c1272d", "fgr2 cosine (abundance)"),
                             ("jaccard_direct", "#0072b2", "fgr2 Jaccard (direct)"),
                             ("jaccard_union", "#56b4e9", "fgr2 Jaccard (union est.)")]:
        e = D["aggregate_auc"][mode][metric][SIZE]["per_level"]
        x = [p["level"] for p in e]
        y = [p["mean"] for p in e]
        s = [p["sd"] for p in e]
        ax.errorbar(x, y, yerr=s, marker="o", ms=4, capsize=3, color=col, label=lab)
    ax2 = ax.twinx()
    e = D["aggregate_auc"][mode]["mash_distance"][SIZE]["per_level"]
    ax2.errorbar([p["level"] for p in e], [p["mean"] for p in e],
                 yerr=[p["sd"] for p in e], marker="s", ms=4, ls="--",
                 color="#666666", capsize=3, label="mash distance (right axis)")
    ax2.set_ylabel("mash distance", color="#666666", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#666666", labelsize=7)
    ax.set_title(TITLE[mode], fontsize=10, loc="left")
    ax.set_xlabel(XLAB[mode], fontsize=9)
    ax.set_ylabel("similarity to intact reference", fontsize=9)
    ax.grid(alpha=0.3)
    if mode == "subst":
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
fig.suptitle(f"E5  Similarity decay vs DNA-storage failure mode "
             f"(k={D['parameters']['k']}, N={SIZE}, n={D['parameters']['n_seeds']} seeds, "
             f"mean$\\pm$SD)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
f1 = os.path.join(FIGD, "e5_decay_curves.png")
fig.savefig(f1, dpi=200)
plt.close(fig)

# ---------------- Figure 2: ROC ---------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, mode in zip(axes, MODES):
    for metric, col, lab in [("cosine", "#c1272d", "fgr2 cosine"),
                             ("jaccard_direct", "#0072b2", "fgr2 Jaccard"),
                             ("mash_distance", "#666666", "mash distance")]:
        p = D["pooled_roc"][mode][metric][SIZE]
        ax.plot(p["fpr"], p["tpr"], color=col, lw=2,
                label=f"{lab}  AUC={p['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_title(TITLE[mode].split("  ")[1], fontsize=9)
    ax.set_xlabel("false positive rate", fontsize=9)
    ax.set_ylabel("true positive rate", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
p = D["pooled_roc"]["subst"]["cosine"][SIZE]
fig.suptitle(f"E5  Intact vs degraded ROC, pooled over all non-zero levels "
             f"(N={SIZE}; n={p['n_neg']} intact vs {p['n_pos']} degraded per panel)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
f2 = os.path.join(FIGD, "e5_roc.png")
fig.savefig(f2, dpi=200)
plt.close(fig)

# ---------------- Figure 3: AUC vs sketch bytes ------------------------------
recs = D["records"]


def mean_bytes(family, size):
    key = "fgr2" if family == "fgr2" else "mash"
    v = [r[key][str(size)]["bytes"] for r in recs if r["tag"] != "ref"]
    return sum(v) / len(v)


fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, mode in zip(axes, MODES):
    for metric, col, lab, fam in [("cosine", "#c1272d", "fgr2 cosine", "fgr2"),
                                  ("jaccard_direct", "#0072b2", "fgr2 Jaccard", "fgr2"),
                                  ("mash_distance", "#666666", "mash distance", "mash")]:
        xs, ys = [], []
        for size in D["parameters"]["N_sizes"]:
            xs.append(mean_bytes(fam, size) / 1024.0)
            ys.append(D["pooled_roc"][mode][metric][str(size)]["auc"])
        ax.plot(xs, ys, marker="o", color=col, label=lab)
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0.35, 1.03)
    ax.set_xlabel("sketch size (kB, on-disk)", fontsize=9)
    ax.set_ylabel("pooled AUC (intact vs degraded)", fontsize=9)
    ax.set_title(TITLE[mode].split("  ")[1], fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3, which="both")
fig.suptitle("E5  Detection power vs sketch size", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
f3 = os.path.join(FIGD, "e5_auc_vs_size.png")
fig.savefig(f3, dpi=200)
plt.close(fig)

# ---------------- Figure 4: per-level AUC (detection floor) ------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, mode in zip(axes, MODES):
    for metric, col, lab in [("cosine", "#c1272d", "fgr2 cosine"),
                             ("jaccard_direct", "#0072b2", "fgr2 Jaccard"),
                             ("mash_distance", "#666666", "mash distance")]:
        e = D["aggregate_auc"][mode][metric][SIZE]["per_level"]
        ax.plot([p["level"] for p in e], [p["auc_vs_level0"] for p in e],
                marker="o", color=col, label=lab)
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.set_ylim(0.3, 1.03)
    ax.set_xlabel(XLAB[mode], fontsize=9)
    ax.set_ylabel("AUC vs intact (level 0)", fontsize=9)
    ax.set_title(TITLE[mode].split("  ")[1], fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
fig.suptitle(f"E5  Per-level detection power (N={SIZE}, "
             f"{D['parameters']['n_seeds']} vs {D['parameters']['n_seeds']} replicates; "
             f"dotted line = AUC 0.95 detection floor)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
f4 = os.path.join(FIGD, "e5_per_level_auc.png")
fig.savefig(f4, dpi=200)
plt.close(fig)

# ---------------- Table ------------------------------------------------------
rows = []
hdr = f"{'mode':<10}{'metric':<18}{'floor(AUC>=0.95)':>18}{'pooled AUC':>12}"
rows.append(hdr)
rows.append("-" * len(hdr))
for mode in MODES:
    for metric in ["cosine", "jaccard_direct", "jaccard_union", "mash_distance"]:
        fl = D["aggregate_auc"][mode][metric][SIZE]["detection_floor_auc0.95"]
        au = D["pooled_roc"][mode][metric][SIZE]["auc"]
        rows.append(f"{mode:<10}{metric:<18}{str(fl):>18}{au:>12.3f}")
table = "\n".join(rows)
print(table)
with open(os.path.join(REPO, "analysis", "results", "e5_detection_floor_table.txt"),
          "w") as fh:
    fh.write(f"E5 detection floors, N={SIZE}, k={D['parameters']['k']}, "
             f"n={D['parameters']['n_seeds']} seeds\n"
             f"'floor' = smallest perturbation level with AUC>=0.95 vs intact; "
             f"None = never reached\n\n" + table + "\n")
print("\n".join([f1, f2, f3, f4]))

# ---------------- Figure 5: set-invariant depth-matched control -------------
CPATH = os.path.join(REPO, "analysis", "results", "e5_pcrbias_control.json")
if os.path.exists(CPATH):
    C = json.load(open(CPATH))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    for metric, col, lab in [("cosine", "#c1272d", "fgr2 cosine"),
                             ("jaccard_direct", "#0072b2", "fgr2 Jaccard"),
                             ("jaccard_union", "#56b4e9", "fgr2 Jaccard (union)")]:
        e = C["aggregate_auc"][metric]["1000"]["per_level"]
        ax.errorbar([p["sigma"] for p in e], [p["mean"] for p in e],
                    yerr=[p["sd"] for p in e], marker="o", ms=4, capsize=3,
                    color=col, label=lab)
    ax2 = ax.twinx()
    e = C["aggregate_auc"]["mash_distance"]["1000"]["per_level"]
    ax2.errorbar([p["sigma"] for p in e], [p["mean"] for p in e],
                 yerr=[p["sd"] for p in e], marker="s", ms=4, ls="--",
                 color="#666666", capsize=3, label="mash distance (right)")
    ax2.set_ylabel("mash distance", color="#666666", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#666666", labelsize=7)
    ax.set_xlabel("amplification skew $\\sigma$ (set-invariant, depth-matched)", fontsize=9)
    ax.set_ylabel("similarity to intact reference", fontsize=9)
    ax.set_title("A  Similarity vs pure abundance skew", fontsize=10, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for metric, col, lab in [("cosine", "#c1272d", "fgr2 cosine"),
                             ("jaccard_direct", "#0072b2", "fgr2 Jaccard"),
                             ("jaccard_union", "#56b4e9", "fgr2 Jaccard (union)"),
                             ("mash_distance", "#666666", "mash distance")]:
        e = C["aggregate_auc"][metric]["1000"]["per_level"]
        ax.plot([p["sigma"] for p in e], [p["auc_vs_level0"] for p in e],
                marker="o", color=col, label=lab)
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.axhline(0.5, color="k", ls="-", lw=0.6, alpha=0.5)
    ax.set_ylim(0.2, 1.03)
    ax.set_xlabel("amplification skew $\\sigma$", fontsize=9)
    ax.set_ylabel("AUC vs intact", fontsize=9)
    ax.set_title("B  Detection power (N=1000)", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for metric, col, lab, fam in [("cosine", "#c1272d", "fgr2 cosine", "fgr2"),
                                  ("jaccard_direct", "#0072b2", "fgr2 Jaccard", "fgr2"),
                                  ("mash_distance", "#666666", "mash distance", "mash")]:
        xs = [mean_bytes(fam, s) / 1024.0 for s in D["parameters"]["N_sizes"]]
        ys = [C["pooled_roc"][metric][str(s)]["auc"] for s in D["parameters"]["N_sizes"]]
        ax.plot(xs, ys, marker="o", color=col, label=lab)
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.axhline(0.5, color="k", ls="-", lw=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.set_ylim(0.2, 1.03)
    ax.set_xlabel("sketch size (kB, on-disk)", fontsize=9)
    ax.set_ylabel("pooled AUC", fontsize=9)
    ax.set_title("C  Power vs sketch size", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("E5 control  Pure amplification bias: species set EXACTLY invariant "
                 "(20,000 species) and depth matched (400k reads) at every $\\sigma$",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f5 = os.path.join(FIGD, "e5_pcrbias_control.png")
    fig.savefig(f5, dpi=200)
    plt.close(fig)
    print(f5)
