#!/usr/bin/env python3
"""E3 analysis + figures.  Consumes e3_coverage_autodetect.json, writes
e3_summary.json and three PNG figures at 200 dpi."""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
RES = os.path.join(REPO, "analysis", "results")
FIG = os.path.join(REPO, "analysis", "figures")
os.makedirs(FIG, exist_ok=True)

IN = os.path.join(RES, "e3_coverage_autodetect.json")
D = json.load(open(IN))
recs = D["records"]
meta = D["meta"]
DEPTHS = meta["depths"]
PROFS = [p["name"] for p in meta["profiles"]]
FIXED = ["1", "2", "3", "5"]

# corrected valley baseline (see e3_baselines.py); the version computed inline
# during the sweep used a broken peak-finder and is superseded.
BLP = os.path.join(RES, "e3_baselines.json")
if os.path.exists(BLP):
    _bl = json.load(open(BLP))["records"]
    _key = {(b["genome"], b["profile"], b["seed"], b["depth"]): b for b in _bl}
    for r in recs:
        b = _key.get((r["genome"], r["profile"], r["seed"], r["depth"]))
        if b:
            r["baseline_valley_c"] = b["valley_globalmax_c"]
            r["f1_at_baseline"] = b["f1_at_valley_globalmax"]
            r["youden_at_baseline"] = b["youden_at_valley_globalmax"]
            r["f1_at_fgr2_rule_smoothed"] = b["f1_at_fgr2_rule_smoothed"]

by = defaultdict(list)
for r in recs:
    by[(r["profile"], r["depth"])].append(r)


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    a = np.array(v, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


# ---------------------------------------------------------------- summary
summary = {"per_profile_depth": [], "per_depth_pooled": [], "headline": {}}

for p in PROFS:
    for d in DEPTHS:
        rs = by[(p, d)]
        if not rs:
            continue
        rs = [r for r in rs if r["detected"] is not None
              and r["f1_at_detected"] is not None]
        if not rs:
            continue
        absdiff = [abs(r["detected"] - r["optimal_f1_c"]) for r in rs]
        row = dict(
            profile=p, depth=d, n=len(rs),
            detected_mean=ms([r["detected"] for r in rs])[0],
            detected_sd=ms([r["detected"] for r in rs])[1],
            detected_values=[r["detected"] for r in rs],
            optimal_mean=ms([r["optimal_f1_c"] for r in rs])[0],
            optimal_values=[r["optimal_f1_c"] for r in rs],
            abs_err_mean=float(np.mean(absdiff)) if absdiff else None,
            abs_err_median=float(np.median(absdiff)) if absdiff else None,
            frac_exact=float(np.mean([a == 0 for a in absdiff])) if absdiff else None,
            frac_within1=float(np.mean([a <= 1 for a in absdiff])) if absdiff else None,
            f1_detected_mean=ms([r["f1_at_detected"] for r in rs])[0],
            f1_detected_sd=ms([r["f1_at_detected"] for r in rs])[1],
            f1_optimal_mean=ms([r["optimal_f1"] for r in rs])[0],
            f1_baseline_mean=ms([r["f1_at_baseline"] for r in rs])[0],
            baseline_n_defined=sum(1 for r in rs if r["baseline_valley_c"] is not None),
            youden_detected_mean=ms([r["youden_at_detected"] for r in rs])[0],
            youden_optimal_mean=ms([r["optimal_youden"] for r in rs])[0],
        )
        for c in FIXED:
            row[f"f1_fixed_{c}_mean"] = ms([r["f1_fixed"][c] for r in rs])[0]
            row[f"youden_fixed_{c}_mean"] = ms([r["youden_fixed"][c] for r in rs])[0]
        summary["per_profile_depth"].append(row)

for d in DEPTHS:
    rs = [r for r in recs if r["depth"] == d
          and r["f1_at_detected"] is not None and r["detected"] is not None]
    absdiff = [abs(r["detected"] - r["optimal_f1_c"]) for r in rs]
    win = [r["f1_at_detected"] > r["f1_fixed"]["2"] for r in rs]
    tie = [abs(r["f1_at_detected"] - r["f1_fixed"]["2"]) < 1e-12 for r in rs]
    delta = [r["f1_at_detected"] - r["f1_fixed"]["2"] for r in rs]
    row = dict(
        depth=d, n=len(rs),
        abs_err_mean=float(np.mean(absdiff)), abs_err_median=float(np.median(absdiff)),
        frac_exact=float(np.mean([a == 0 for a in absdiff])),
        frac_within1=float(np.mean([a <= 1 for a in absdiff])),
        f1_detected_mean=float(np.mean([r["f1_at_detected"] for r in rs])),
        f1_detected_sd=float(np.std([r["f1_at_detected"] for r in rs], ddof=1)),
        f1_optimal_mean=float(np.mean([r["optimal_f1"] for r in rs])),
        f1_c2_mean=float(np.mean([r["f1_fixed"]["2"] for r in rs])),
        f1_c2_sd=float(np.std([r["f1_fixed"]["2"] for r in rs], ddof=1)),
        frac_auto_beats_c2=float(np.mean(win)),
        frac_auto_ties_c2=float(np.mean(tie)),
        mean_delta_f1_auto_minus_c2=float(np.mean(delta)),
        median_delta_f1_auto_minus_c2=float(np.median(delta)),
    )
    for c in FIXED:
        row[f"f1_fixed_{c}_mean"] = float(np.mean([r["f1_fixed"][c] for r in rs]))
    bl = [r["f1_at_baseline"] for r in rs if r["f1_at_baseline"] is not None]
    row["f1_baseline_mean"] = float(np.mean(bl)) if bl else None
    row["baseline_n_defined"] = len(bl)
    summary["per_depth_pooled"].append(row)

allr = [r for r in recs if r["f1_at_detected"] is not None
        and r["detected"] is not None]
n_unscored = len(recs) - len(allr)
delta_all = [r["f1_at_detected"] - r["f1_fixed"]["2"] for r in allr]
summary["headline"] = dict(
    n_conditions=len(allr),
    n_records_total=len(recs),
    n_records_dropped_unscorable=n_unscored,
    overall_f1_detected=float(np.mean([r["f1_at_detected"] for r in allr])),
    overall_f1_c2=float(np.mean([r["f1_fixed"]["2"] for r in allr])),
    overall_f1_c1=float(np.mean([r["f1_fixed"]["1"] for r in allr])),
    overall_f1_c3=float(np.mean([r["f1_fixed"]["3"] for r in allr])),
    overall_f1_c5=float(np.mean([r["f1_fixed"]["5"] for r in allr])),
    overall_f1_optimal=float(np.mean([r["optimal_f1"] for r in allr])),
    frac_auto_beats_c2=float(np.mean([d > 0 for d in delta_all])),
    frac_auto_worse_c2=float(np.mean([d < 0 for d in delta_all])),
    mean_delta_f1=float(np.mean(delta_all)),
    frac_exact_recovery=float(np.mean(
        [r["detected"] == r["optimal_f1_c"] for r in allr])),
    hist_crosscheck_all_pass=bool(all(
        r["fgr2_hist_matches_ground_truth"] for r in allr)),
    catastrophic_rate=float(np.mean([r["f1_at_detected"] < 0.5 for r in allr])),
)

# depth floor: lowest depth D such that for ALL depths >= D the criterion holds.
# criterion: median |detected-optimal| <= 1 AND mean F1(detected) >= 0.95*mean F1(optimal)
def depth_floor(rows):
    ok = {}
    for r in rows:
        ok[r["depth"]] = (r["abs_err_median"] is not None
                          and r["abs_err_median"] <= 1
                          and r["f1_detected_mean"] >= 0.95 * r["f1_optimal_mean"])
    floor = None
    for d in sorted(ok, reverse=True):
        if ok[d]:
            floor = d
        else:
            break
    return floor, {str(d): bool(ok[d]) for d in sorted(ok)}


summary["depth_floor"] = {}
for p in PROFS:
    fl, det = depth_floor([r for r in summary["per_profile_depth"]
                           if r["profile"] == p])
    summary["depth_floor"][p] = dict(floor=fl, per_depth_pass=det)
fl, det = depth_floor(summary["per_depth_pooled"])
summary["depth_floor"]["pooled"] = dict(floor=fl, per_depth_pass=det)
summary["depth_floor"]["criterion"] = (
    "median |detected-optimal| <= 1 AND mean F1(detected) >= 0.95 * mean F1(optimal), "
    "required to hold at that depth and every higher depth tested")

json.dump(summary, open(os.path.join(RES, "e3_summary.json"), "w"), indent=1)

# ---------------------------------------------------------------- fig 1
fig, axes = plt.subplots(1, len(PROFS), figsize=(20, 4.2), sharey=True)
for ax, p in zip(axes, PROFS):
    rows = [r for r in summary["per_profile_depth"] if r["profile"] == p]
    x = [r["depth"] for r in rows]
    det = [r["detected_mean"] for r in rows]
    dets = [r["detected_sd"] for r in rows]
    opt = [r["optimal_mean"] for r in rows]
    ax.errorbar(x, det, yerr=dets, marker="o", color="crimson",
                label="fgr2 auto-detected", capsize=3)
    ax.plot(x, opt, marker="s", color="navy", label="optimal (max F1)")
    ax.axhline(2, ls="--", color="gray", lw=1, label="fixed c=2")
    for r in rows:
        ax.scatter([r["depth"]] * len(r["detected_values"]), r["detected_values"],
                   s=8, color="crimson", alpha=0.35, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(DEPTHS); ax.set_xticklabels(DEPTHS)
    ax.set_title(p, fontsize=10); ax.set_xlabel("read depth (x)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("coverage threshold")
axes[0].legend(fontsize=8)
fig.suptitle("E3: fgr2 auto-detected vs optimal coverage threshold "
             f"(k={meta['k']}, n={len(meta['genomes'])} genomes x "
             f"{len(meta['seeds'])} seeds per point)", fontsize=11)
fig.tight_layout()
f1p = os.path.join(FIG, "e3_detected_vs_optimal.png")
fig.savefig(f1p, dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 2
sel = [("illumina_1pct", d) for d in (2, 5, 20, 100)] + \
      [("longread_10pct", d) for d in (2, 5, 20, 100)]
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, (p, d) in zip(axes.ravel(), sel):
    rs = [r for r in by[(p, d)] if r["genome"] == "Ecoli_K12_MG1655"
          and r["seed"] == 0]
    if not rs:
        ax.set_visible(False); continue
    r = rs[0]
    ht = np.array(r["hist_true"], dtype=float)
    he = np.array(r["hist_error"], dtype=float)
    n = min(len(ht), max(6, int(d * 3)))
    xs = np.arange(1, n)
    ax.bar(xs, he[1:n], color="tomato", label="ERROR k-mers", width=1.0)
    ax.bar(xs, ht[1:n], bottom=he[1:n], color="steelblue",
           label="TRUE k-mers", width=1.0)
    ax.axvline(r["detected"], color="crimson", lw=2,
               label=f"fgr2 auto = {r['detected']}")
    ax.axvline(r["optimal_f1_c"], color="navy", ls="--", lw=2,
               label=f"optimal = {r['optimal_f1_c']}")
    if r["baseline_valley_c"]:
        ax.axvline(r["baseline_valley_c"], color="green", ls=":", lw=2,
                   label=f"valley baseline = {r['baseline_valley_c']}")
    ax.set_yscale("log")
    ax.set_title(f"{p}  {d}x  (E. coli K-12, seed 0)", fontsize=9)
    ax.set_xlabel("k-mer coverage"); ax.legend(fontsize=7)
axes[0, 0].set_ylabel("# distinct k-mers")
axes[1, 0].set_ylabel("# distinct k-mers")
fig.suptitle("E3: coverage spectra with fgr2's detected cutoff and the "
             "ground-truth optimum", fontsize=12)
fig.tight_layout()
f2p = os.path.join(FIG, "e3_representative_histograms.png")
fig.savefig(f2p, dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, axes = plt.subplots(1, len(PROFS) + 1, figsize=(22, 4.2), sharey=True)
cols = {"1": "#999999", "2": "#e08214", "3": "#8073ac", "5": "#4d9221"}
for ax, p in zip(axes, PROFS):
    rows = [r for r in summary["per_profile_depth"] if r["profile"] == p]
    x = [r["depth"] for r in rows]
    ax.plot(x, [r["f1_optimal_mean"] for r in rows], color="black", lw=2.5,
            marker="s", label="optimal")
    ax.errorbar(x, [r["f1_detected_mean"] for r in rows],
                yerr=[r["f1_detected_sd"] for r in rows], color="crimson",
                marker="o", lw=2, capsize=3, label="fgr2 auto")
    for c in FIXED:
        ax.plot(x, [r[f"f1_fixed_{c}_mean"] for r in rows], color=cols[c],
                ls="--", marker=".", label=f"fixed c={c}")
    ax.plot(x, [r["f1_baseline_mean"] for r in rows], color="green", ls=":",
            marker="^", label="valley baseline")
    ax.set_xscale("log"); ax.set_xticks(DEPTHS); ax.set_xticklabels(DEPTHS)
    ax.set_title(p, fontsize=10); ax.set_xlabel("read depth (x)"); ax.grid(alpha=0.3)
ax = axes[-1]
rows = summary["per_depth_pooled"]
x = [r["depth"] for r in rows]
ax.plot(x, [r["f1_optimal_mean"] for r in rows], color="black", lw=2.5, marker="s",
        label="optimal")
ax.errorbar(x, [r["f1_detected_mean"] for r in rows],
            yerr=[r["f1_detected_sd"] for r in rows], color="crimson", marker="o",
            lw=2, capsize=3, label="fgr2 auto")
for c in FIXED:
    ax.plot(x, [r[f"f1_fixed_{c}_mean"] for r in rows], color=cols[c], ls="--",
            marker=".", label=f"fixed c={c}")
ax.set_xscale("log"); ax.set_xticks(DEPTHS); ax.set_xticklabels(DEPTHS)
ax.set_title("all profiles pooled", fontsize=10); ax.set_xlabel("read depth (x)")
ax.grid(alpha=0.3); ax.legend(fontsize=7)
axes[0].set_ylabel("F1 (TRUE vs ERROR k-mer separation)")
fig.suptitle("E3: F1 achieved at fgr2's auto-detected threshold vs fixed "
             "thresholds vs the optimum", fontsize=12)
fig.tight_layout()
f3p = os.path.join(FIG, "e3_f1_auto_vs_fixed.png")
fig.savefig(f3p, dpi=200); plt.close(fig)

print(json.dumps(summary["headline"], indent=1))
print("\nper-depth pooled:")
for r in summary["per_depth_pooled"]:
    print(f"  {r['depth']:4d}x n={r['n']:3d} |det-opt| mean={r['abs_err_mean']:7.2f} "
          f"med={r['abs_err_median']:6.1f} exact={r['frac_exact']:.2f} "
          f"F1auto={r['f1_detected_mean']:.4f} F1c2={r['f1_c2_mean']:.4f} "
          f"F1c3={r['f1_fixed_3_mean']:.4f} F1opt={r['f1_optimal_mean']:.4f} "
          f"auto>c2 in {r['frac_auto_beats_c2']:.2f}")
print("figures:", f1p, f2p, f3p, sep="\n  ")
