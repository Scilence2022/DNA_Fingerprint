#!/usr/bin/env python3
"""E1 figure: correctness matrix + hash-value distribution evidence."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
RES = os.path.join(REPO, "analysis/results/e1_sketch_correctness.json")
FIG = os.path.join(REPO, "analysis/figures/e1_sketch_correctness.png")
os.makedirs(os.path.dirname(FIG), exist_ok=True)

d = json.load(open(RES))
cells = d["cells"]
genomes = d["genomes"]
CHECKS = ["C1_set_equality", "C2_hash_fidelity", "C3_sorted_ascending",
          "C4_provenance", "C5_determinism"]
LABELS = ["C1 set\nequality", "C2 hash\nfidelity", "C3 sort\norder",
          "C4 prove-\nnance", "C5 deter-\nminism"]

conds = [(k, n, h) for k in d["grid"]["k"] for n in d["grid"]["N"]
         for h in d["grid"]["hash"]]
cond_lbl = [f"k{k} N{n}\n{'mmh3' if h == 'murmurhash3' else 'wang'}"
            for k, n, h in conds]

# panel A: per-cell pass matrix (genome x condition), value = #checks passed / 5
M = np.zeros((len(genomes), len(conds)))
for ci, (k, n, h) in enumerate(conds):
    for gi, g in enumerate(genomes):
        c = next(x for x in cells if x["genome"] == g and x["k"] == k
                 and x["N"] == n and x["hash"] == h)
        M[gi, ci] = sum(bool(c[ch]) for ch in CHECKS)

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.45, wspace=0.25)

axA = fig.add_subplot(gs[0, :])
axA.imshow(M, cmap="Greens", vmin=0, vmax=5, aspect="auto")
axA.set_xticks(range(len(conds)))
axA.set_xticklabels(cond_lbl, fontsize=7.5)
axA.set_yticks(range(len(genomes)))
axA.set_yticklabels([g.replace("_", " ") for g in genomes], fontsize=8.5)
for gi in range(len(genomes)):
    for ci in range(len(conds)):
        axA.text(ci, gi, f"{int(M[gi, ci])}/5", ha="center", va="center",
                 fontsize=7, color="white" if M[gi, ci] == 5 else "red")
axA.set_title("A  Independent checks passed per grid cell "
              f"({d['summary']['n_cells_pass']}/{d['summary']['n_cells']} cells "
              "pass all 5; 0 failures)", fontsize=10.5, loc="left")

# panel B: per-check totals
axB = fig.add_subplot(gs[1, 0])
vals = [d["summary"]["per_check_pass"][c] for c in CHECKS]
axB.bar(range(len(CHECKS)), vals, color="#2c7a3f")
axB.axhline(len(cells), ls="--", lw=1, color="k")
axB.set_xticks(range(len(CHECKS)))
axB.set_xticklabels(LABELS, fontsize=7.5)
axB.set_ylim(0, len(cells) * 1.18)
axB.set_ylabel("cells passing (of 72)")
for i, v in enumerate(vals):
    axB.text(i, v + 1.5, str(v), ha="center", fontsize=8.5)
axB.set_title("B  Per-check pass count across the full grid", fontsize=10.5,
              loc="left")

# panel C: thread invariance + determinism
axC = fig.add_subplot(gs[1, 1])
ti = d["thread_invariance"]
labels = ["determinism\n(same config x2)", "thread invariance\n(-t 1/4/16)"]
tot = [len(cells), len(ti)]
ok = [d["summary"]["per_check_pass"]["C5_determinism"],
      d["summary"]["n_thread_configs_pass"]]
x = np.arange(2)
axC.bar(x - 0.18, tot, 0.36, label="configs tested", color="#bbbbbb")
axC.bar(x + 0.18, ok, 0.36, label="byte-identical", color="#2c7a3f")
axC.set_xticks(x)
axC.set_xticklabels(labels, fontsize=8)
axC.set_ylabel("configurations")
axC.legend(fontsize=8)
for xi, (a, b) in enumerate(zip(tot, ok)):
    axC.text(xi - 0.18, a + 1, str(a), ha="center", fontsize=8)
    axC.text(xi + 0.18, b + 1, str(b), ha="center", fontsize=8)
axC.set_ylim(0, max(tot) * 1.2)
axC.set_title("C  Reproducibility (SHA-256 of sketch file)", fontsize=10.5,
              loc="left")

fig.suptitle("E1  fgr2 v2.1.0 emits an exact bottom-N MinHash sketch on real "
             "bacterial genomes\n"
             "6 genomes (4.2-6.3 Mb) x k in {21,25,31} x N in {1000,10000} x "
             "{MurmurHash3, Wang}; ground truth recomputed independently",
             fontsize=11.5, y=0.99)
fig.savefig(FIG, dpi=200, bbox_inches="tight")
print("wrote", FIG)
