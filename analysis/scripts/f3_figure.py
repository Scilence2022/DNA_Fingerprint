#!/usr/bin/env python3
"""F3 figure: depth-mismatch fragility of the abundance-weighted (cosine) result."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
P = os.path.join(REPO, "analysis", "results", "f3_depth_mismatch.json")
FIGDIR = os.path.join(REPO, "analysis", "figures")
os.makedirs(FIGDIR, exist_ok=True)

d = json.load(open(P))
A = d["stages"]["A_error1e-3"]
B = d["stages"]["B_error0"]
Cs, Cn = d["stages"]["C_sat"], d["stages"]["C_nosat"]
RA = d["parameters"]["stage_A"]["ratios"]
SA = d["parameters"]["stage_A"]["sigmas"]


def cell(st, n, r, s):
    return next((c for c in st["cells"]
                 if c["N"] == n and c["ratio"] == r and c["sigma"] == s), None)


fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.5))
cols = plt.cm.viridis(np.linspace(0, 0.85, len(SA)))

# --- a: raw cosine vs depth ratio ------------------------------------------
a = ax[0, 0]
for i, s in enumerate(SA):
    y = [cell(A, 1000, r, s)["cosine"]["mean"] for r in RA]
    e = [cell(A, 1000, r, s)["cosine"]["sd"] for r in RA]
    a.errorbar(RA, y, yerr=e, marker="o", ms=4, color=cols[i], label=f"$\\sigma$={s}")
a.set_xscale("log")
a.axvline(1.0, color="k", lw=0.8, ls=":")
a.set_xlabel("depth ratio (query reads / reference reads)")
a.set_ylabel("cosine vs reference")
a.set_title("(a) E5's raw cosine collapses under depth mismatch\n"
            "(E5 design, k=21, N=1000, err 1e-3, n=10 seeds/cell)", fontsize=10)
a.legend(fontsize=8)
a.grid(alpha=0.3)

# --- b: signal vs artifact -------------------------------------------------
a = ax[0, 1]
h = d["headline"]["signal_vs_artifact_N1000"]
sig = h["signal_sigma0_to_0.25_at_matched_depth"]
rs = sorted(float(k) for k in h["depth_artifact_at_sigma0"])
y = [abs(h["depth_artifact_at_sigma0"][str(r)]["delta_cosine_vs_ratio1"]) for r in rs]
a.plot(rs, y, "o-", color="crimson", label="depth artifact ($\\sigma$=0)")
a.axhline(sig, color="navy", ls="--",
          label=f"signal $\\sigma$ 0$\\to$0.25 = {sig:.5f}")
a.set_xscale("log")
a.set_yscale("log")
a.set_xlabel("depth ratio")
a.set_ylabel("|$\\Delta$ mean cosine| vs ratio 1.0")
a.set_title("(b) depth artifact exceeds the smallest claimed\nsignal beyond ~"
            "$\\pm$10% depth mismatch (up to 221$\\times$)", fontsize=10)
a.legend(fontsize=8)
a.grid(alpha=0.3, which="both")

# --- c: mixed-ratio AUC (the operating envelope) ---------------------------
a = ax[0, 2]
style = {"cosine": ("crimson", "-", "raw cosine (E5)"),
         "cosine_common": ("seagreen", "-", "cosine on common bottom-N index"),
         "cosine_min2": ("darkorange", "--", "cosine, count$\\geq$2 filter"),
         "cosine_log": ("purple", ":", "cosine of log1p(count)"),
         "jaccard_direct": ("grey", "-.", "Jaccard (direct)")}
for m, (c, ls, lab) in style.items():
    rows = [x for x in A["mixed_ratio_auc"]["1000"][m] if x["sigma"] == 0.25]
    a.plot([x["window"] for x in rows], [x["auc"] for x in rows],
           marker="o", ms=4, color=c, ls=ls, label=lab)
a.axhline(0.95, color="k", lw=0.8, ls=":")
a.set_xscale("log")
a.set_xlabel("depth-mismatch window $W$ (depth known only within $\\times W$)")
a.set_ylabel("AUC, $\\sigma$=0.25 vs $\\sigma$=0")
a.set_ylim(0.2, 1.03)
a.set_title("(c) operating envelope: raw cosine tolerates $W\\leq$1.1;\n"
            "common-index cosine holds AUC=1.000 to $W$=10", fontsize=10)
a.legend(fontsize=7.5, loc="lower left")
a.grid(alpha=0.3, which="both")

# --- d: mechanism 1 -- error k-mers, not coverage rescaling ----------------
a = ax[1, 0]
RB = d["parameters"]["stage_B"]["ratios"]
a.plot(RA, [cell(A, 1000, r, 0.0)["cosine"]["mean"] for r in RA], "o-",
       color="crimson", label="stage A: err = 1e-3")
a.plot(RB, [cell(B, 1000, r, 0.0)["cosine"]["mean"] for r in RB], "s-",
       color="navy", label="stage B: err = 0 (exactly 1.000000)")
a2 = a.twinx()
a2.plot(RA, [100 * cell(A, 1000, r, 0.0)["frac_singleton_mean"] for r in RA],
        "^--", color="grey", ms=4, label="% singleton k-mers in sketch")
a2.set_ylabel("% of sketch with count = 1", color="grey")
a.set_xscale("log")
a.set_xlabel("depth ratio")
a.set_ylabel("cosine, $\\sigma$=0")
a.set_title("(d) mechanism: error k-mers drive sketch-membership\n"
            "drift; coverage rescaling contributes exactly zero", fontsize=10)
a.legend(fontsize=8, loc="lower left")
a2.legend(fontsize=8, loc="upper right")
a.grid(alpha=0.3)

# --- e: mechanism 2 -- counter saturation ----------------------------------
a = ax[1, 1]
RC = d["parameters"]["stage_C"]["ratios"]
a.plot(RC, [cell(Cs, 1000, r, 0.0)["cosine"]["mean"]
            - cell(Cs, 1000, r, 0.25)["cosine"]["mean"] for r in RC],
       "o-", color="crimson", label="$\\sigma$=0.25 signal, saturating arm")
a.plot(RC, [cell(Cs, 1000, r, 0.0)["cosine"]["mean"]
            - cell(Cs, 1000, r, 1.0)["cosine"]["mean"] for r in RC],
       "s-", color="darkorange", label="$\\sigma$=1.0 signal, saturating arm")
a.axhline(cell(Cn, 1000, 1.0, 0.0)["cosine"]["mean"]
          - cell(Cn, 1000, 1.0, 0.25)["cosine"]["mean"], color="navy", ls="--",
          label="$\\sigma$=0.25 signal, non-saturating arm (flat at every ratio)")
a.set_yscale("log")
a5 = a.twinx()
a5.plot(RC, [100 * cell(Cs, 1000, r, 0.25)["frac_saturated_mean"] for r in RC],
        "^:", color="grey", ms=5, label="% counters saturated")
a5.set_ylabel("% of sketch counters saturated", color="grey")
a5.legend(fontsize=7.5, loc="center right")
a.set_xlabel("depth ratio (error-free pool, base copies 8000)")
a.set_ylabel("signal = cosine($\\sigma$=0) $-$ cosine($\\sigma$)")
a.set_title("(e) counter saturation ERASES the abundance signal\n"
            "(false negatives), error-free pool, n=5 seeds", fontsize=10)
a.legend(fontsize=8)
a.grid(alpha=0.3, which="both")

# --- f: the fix ------------------------------------------------------------
a = ax[1, 2]
for i, s in enumerate(SA):
    y = [cell(A, 1000, r, s)["cosine_common"]["mean"] for r in RA]
    e = [cell(A, 1000, r, s)["cosine_common"]["sd"] for r in RA]
    a.errorbar(RA, y, yerr=e, marker="o", ms=4, color=cols[i], label=f"$\\sigma$={s}")
a.set_xscale("log")
a.axvline(1.0, color="k", lw=0.8, ls=":")
a.set_xlabel("depth ratio")
a.set_ylabel("cosine on common bottom-N index")
a.set_title("(f) the fix: score both sketches on the N smallest\n"
            "hashes of their union (Mash-style common index)", fontsize=10)
a.legend(fontsize=8)
a.grid(alpha=0.3)

fig.suptitle("F3 -- sequencing-depth mismatch and the abundance-weighted (cosine) "
             "claim of E5", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = os.path.join(FIGDIR, "f3_depth_mismatch.png")
fig.savefig(out, dpi=200)
print("wrote", out)
