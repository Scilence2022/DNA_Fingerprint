#!/usr/bin/env python3
"""F2 figure: coverage-threshold auto-detection on FULL genomes vs 1 Mb slices.
Panel A: F1 of auto vs fixed thresholds vs depth (full genomes, Illumina).
Panel B: auto vs c=2 by stratum, full vs slice (the negative result is robust).
Panel C: detected minus optimal threshold vs depth (the overshoot)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# scripts/ lives at <repo>/analysis/scripts, so the repo root is three levels up.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
d = json.load(open(os.path.join(REPO, "analysis/results/f2_full_genome.json")))
recs = d["records"]
FIG = os.path.join(REPO, "analysis/figures/f2_full_genome.png")

DEPTHS = [1, 2, 5, 10, 20, 50, 100, 200]


def fx(r, c):
    return r["f1_fixed"][str(c)]


def mean_over(arm, depth, cls, key):
    rs = [r for r in recs if r["arm"] == arm and r["depth"] == depth
          and r["profile_class"] == cls]
    if not rs:
        return np.nan
    if key == "auto":
        return np.mean([r["f1_at_detected"] for r in rs])
    if key == "optimal":
        return np.mean([r["optimal_f1"] for r in rs])
    return np.mean([fx(r, key) for r in rs])


fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# Panel A: full-genome Illumina, F1 vs depth
ax = axes[0]
for key, lab, style in [("auto", "auto-detect", dict(color="C3", lw=2.5, marker="o")),
                        (2, "fixed c=2", dict(color="C0", lw=1.8, marker="s", ls="--")),
                        (3, "fixed c=3", dict(color="C1", lw=1.8, marker="^", ls="--")),
                        ("optimal", "optimal (oracle)", dict(color="black", lw=2.5, marker="D"))]:
    y = [mean_over("full", dp, "illumina", key) for dp in DEPTHS]
    ax.plot(DEPTHS, y, label=lab, **style)
ax.set_xscale("log")
ax.set_xticks(DEPTHS)
ax.set_xticklabels(DEPTHS)
ax.set_xlabel("sequencing depth (×)")
ax.set_ylabel("F1 of coverage-threshold rule")
ax.set_title("A  Full genomes, Illumina profiles")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# Panel B: auto vs c=2 by stratum, full vs slice
ax = axes[1]
strata = [("all", lambda r: True),
          ("depth\n≤20×", lambda r: r["depth"] <= 20),
          ("depth\n≥50×", lambda r: r["depth"] >= 50),
          ("Illumina\nonly", lambda r: r["profile_class"] == "illumina"),
          ("Illumina\n≥10×", lambda r: r["profile_class"] == "illumina" and r["depth"] >= 10),
          ("long-read\nonly", lambda r: r["profile_class"] == "longread")]
x = np.arange(len(strata))
w = 0.35
for i, arm in enumerate(["full", "slice1mb"]):
    deltas = []
    for _, filt in strata:
        rs = [r for r in recs if r["arm"] == arm and filt(r)]
        auto = np.mean([r["f1_at_detected"] for r in rs])
        c2 = np.mean([fx(r, 2) for r in rs])
        deltas.append(auto - c2)
    ax.bar(x + (i - 0.5) * w, deltas, w,
           label=("full genome" if arm == "full" else "1 Mb slice"),
           color=("C3" if arm == "full" else "C7"), alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in strata], fontsize=8)
ax.set_ylabel("mean F1(auto) − F1(c=2)")
ax.set_title("B  Auto minus fixed c=2 (negative = worse)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# Panel C: detected - optimal threshold vs depth (overshoot), full genomes
ax = axes[2]
for cls, col in [("illumina", "C0"), ("longread", "C1")]:
    med, lo, hi = [], [], []
    for dp in DEPTHS:
        rs = [r for r in recs if r["arm"] == "full" and r["depth"] == dp
              and r["profile_class"] == cls and r["detected"] and r["detected"] > 0]
        errs = np.array([r["detected"] - r["optimal_f1_c"] for r in rs]) if rs else np.array([np.nan])
        med.append(np.nanmedian(errs))
        lo.append(np.nanpercentile(errs, 25) if rs else np.nan)
        hi.append(np.nanpercentile(errs, 75) if rs else np.nan)
    ax.plot(DEPTHS, med, marker="o", color=col, label=cls, lw=2)
    ax.fill_between(DEPTHS, lo, hi, color=col, alpha=0.15)
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xticks(DEPTHS)
ax.set_xticklabels(DEPTHS)
ax.set_xlabel("sequencing depth (×)")
ax.set_ylabel("detected − optimal threshold")
ax.set_title("C  Threshold overshoot (full genomes)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG, dpi=200, bbox_inches="tight")
print("wrote", FIG)
