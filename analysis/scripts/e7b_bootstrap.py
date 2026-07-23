"""E7(b) addendum: is the murmur3-vs-wang accuracy difference real?

Paired bootstrap over the 120 genome pairs (the same pairs are resampled for
both hashes, so the comparison is paired). Adds an "accuracy_bootstrap" block
to E7b_hash.json in place.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_common as C

B = 20000
SEED = 4242

p = os.path.join(C.RESULTS, "E7b_hash.json")
d = json.load(open(p))
pp = d["accuracy"]["per_pair"]
pairs = sorted(pp)
rng = np.random.default_rng(SEED)

out = {"n_bootstrap": B, "seed": SEED, "n_pairs": len(pairs), "comparisons": {}}
for est in ("union", "direct"):
    for N in (1000, 10000):
        em = np.array([pp[k][f"murmurhash3_{est}_N{N}"] - pp[k]["true_jaccard"]
                       for k in pairs])
        ew = np.array([pp[k][f"wang_{est}_N{N}"] - pp[k]["true_jaccard"]
                       for k in pairs])
        idx = rng.integers(0, len(pairs), size=(B, len(pairs)))
        rm = np.sqrt((em[idx] ** 2).mean(axis=1))
        rw = np.sqrt((ew[idx] ** 2).mean(axis=1))
        diff = rw - rm            # positive => wang worse
        bm = em[idx].mean(axis=1)
        bw = ew[idx].mean(axis=1)
        bdiff = bw - bm
        out["comparisons"][f"{est}_N{N}"] = {
            "rmse_murmur": float(np.sqrt((em ** 2).mean())),
            "rmse_wang": float(np.sqrt((ew ** 2).mean())),
            "rmse_diff_wang_minus_murmur": float(np.sqrt((ew ** 2).mean())
                                                 - np.sqrt((em ** 2).mean())),
            "rmse_diff_ci95": [float(np.percentile(diff, 2.5)),
                               float(np.percentile(diff, 97.5))],
            "rmse_diff_p_two_sided": float(2 * min((diff <= 0).mean(),
                                                   (diff >= 0).mean())),
            "bias_murmur": float(em.mean()),
            "bias_wang": float(ew.mean()),
            "bias_diff_ci95": [float(np.percentile(bdiff, 2.5)),
                               float(np.percentile(bdiff, 97.5))],
        }

d["accuracy_bootstrap"] = out
with open(p, "w") as fh:
    json.dump(d, fh)

for k, v in out["comparisons"].items():
    sig = "SIGNIFICANT" if v["rmse_diff_ci95"][0] * v["rmse_diff_ci95"][1] > 0 \
        else "not significant"
    print(f"{k:14s} rmse murmur={v['rmse_murmur']:.6f} wang={v['rmse_wang']:.6f} "
          f"diff={v['rmse_diff_wang_minus_murmur']:+.6f} "
          f"CI95=[{v['rmse_diff_ci95'][0]:+.6f},{v['rmse_diff_ci95'][1]:+.6f}] "
          f"p={v['rmse_diff_p_two_sided']:.4f}  {sig}")
print("updated", p)
