#!/usr/bin/env python3
"""
F3 analysis -- how fragile is E5's cosine result to sequencing-depth mismatch?

Reads the sketches left by f3_depth_mismatch.py and computes, for every
(depth ratio x amplification sigma) cell:

  metrics
    cosine            raw abundance cosine (E5's metric)
    cosine_l1         counts divided by their own sum first (must be identical:
                      cosine is invariant to any positive global scalar -- this
                      is included as a numerical proof that "normalise by total
                      counts" cannot possibly help)
    cosine_common     cosine on the COMMON bottom-N index (N globally smallest
                      hashes of the merged sketch, zero-filled) -- fixes sketch
                      MEMBERSHIP drift
    cosine_log        cosine of log1p(count)
    cosine_min2       drop count==1 k-mers from each sketch first
    cosine_common_min2  common index AND count>=2 filter
    jaccard_direct, jaccard_union, mash_distance

  discrimination
    fixed-ratio AUC   sigma=0.25 vs sigma=0 when BOTH pools share the same depth
    mixed-ratio AUC   sigma=0.25 vs sigma=0 when the depth ratio is unknown
                      within a window W (all ratios with max(r,1/r) <= W).
                      The largest W with AUC >= 0.95 is the depth-mismatch
                      TOLERANCE the manuscript must state.
    signal vs artifact  |cosine shift from sigma 0->0.25 at matched depth| vs
                      |cosine shift from depth mismatch alone at sigma=0|
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
sys.path.insert(0, os.path.join(REPO, "analysis", "scripts"))
import fgrlib  # noqa: E402
from f3_depth_mismatch import MASH, K, KC_MAX  # noqa: E402

RAW = os.path.join(REPO, "analysis", "results", "f3_depth_sweep_raw.json")
OUT = os.path.join(REPO, "analysis", "results", "f3_depth_mismatch.json")


# --------------------------------------------------------------- metric zoo
def _cos(x, y):
    nx, ny = math.sqrt(float(np.dot(x, x))), math.sqrt(float(np.dot(y, y)))
    return float(np.dot(x, y) / (nx * ny)) if nx and ny else 0.0


def cos_union(a, b):
    keys = set(a) | set(b)
    x = np.array([a.get(kk, 0) for kk in keys], float)
    y = np.array([b.get(kk, 0) for kk in keys], float)
    return _cos(x, y)


def cos_common_index(ah, ac, bh, bc, n):
    """Cosine on the N globally smallest hashes of the merged sketch."""
    merged = {}
    merged.update(ah)
    merged.update(bh)
    keys = [km for km, _ in sorted(merged.items(), key=lambda kv: kv[1])[:n]]
    x = np.array([ac.get(kk, 0) for kk in keys], float)
    y = np.array([bc.get(kk, 0) for kk in keys], float)
    return _cos(x, y), len(keys)


def filt(cov, minc):
    return {kk: v for kk, v in cov.items() if v >= minc}


def metrics(ref, qry, n):
    rh, rc = ref
    qh, qc = qry
    m = {}
    m["cosine"] = cos_union(rc, qc)
    tr, tq = sum(rc.values()), sum(qc.values())
    m["cosine_l1"] = cos_union({k: v / tr for k, v in rc.items()},
                               {k: v / tq for k, v in qc.items()})
    cc, nc = cos_common_index(rh, rc, qh, qc, n)
    m["cosine_common"] = cc
    m["cosine_common_index_size"] = nc
    m["cosine_log"] = cos_union({k: math.log1p(v) for k, v in rc.items()},
                                {k: math.log1p(v) for k, v in qc.items()})
    r2, q2 = filt(rc, 2), filt(qc, 2)
    m["cosine_min2"] = cos_union(r2, q2) if r2 and q2 else 0.0
    rh2 = {k: rh[k] for k in r2}
    qh2 = {k: qh[k] for k in q2}
    cc2, nc2 = cos_common_index(rh2, r2, qh2, q2, n)
    m["cosine_common_min2"] = cc2
    m["jaccard_direct"] = fgrlib.jaccard_direct(set(rh), set(qh))
    m["jaccard_union"] = fgrlib.jaccard_union(rh, qh, n)
    return m


def mash_dist(a, b):
    import subprocess
    r = subprocess.run([MASH, "dist", a, b], check=True, capture_output=True,
                       text=True)
    return float(r.stdout.split("\n")[0].split("\t")[2])


# ----------------------------------------------------------------- stats
def auc(neg, pos):
    """P(score_pos > score_neg), ties 0.5. Score = degradation (higher = worse)."""
    if not neg or not pos:
        return None
    t = 0.0
    for p in pos:
        for q in neg:
            t += 1.0 if p > q else (0.5 if p == q else 0.0)
    return t / (len(pos) * len(neg))


def mw_p(neg, pos):
    try:
        return float(stats.mannwhitneyu(pos, neg, alternative="two-sided")[1])
    except ValueError:
        return None


COSMETRICS = ["cosine", "cosine_l1", "cosine_common", "cosine_log",
              "cosine_min2", "cosine_common_min2"]
SIMMETRICS = COSMETRICS + ["jaccard_direct", "jaccard_union"]


def analyse_stage(recs, refrec, sizes, sigmas, ratios, do_mash, label):
    """recs: list of query records; refrec: reference record."""
    ref_sk = {}
    for n in sizes:
        h, c, _ = fgrlib.parse_sketch(refrec["fgr2"][str(n)]["path"])
        ref_sk[n] = (h, c)
    for r in recs:
        r["metrics"] = {}
        for n in sizes:
            h, c, _ = fgrlib.parse_sketch(r["fgr2"][str(n)]["path"])
            r["metrics"][str(n)] = metrics(ref_sk[n], (h, c), n)
        if do_mash and r["mash"]:
            s = list(r["mash"])[0]
            r["metrics"][str(sizes[0])]["mash_distance"] = mash_dist(
                refrec["mash"][s]["path"], r["mash"][s]["path"])

    idx = defaultdict(dict)          # (n, ratio, sigma) -> {metric: [values]}
    sat = defaultdict(list)
    sing = defaultdict(list)
    for r in recs:
        for n in sizes:
            key = (n, r["ratio"], r["sigma"])
            d = idx[key]
            for mname, v in r["metrics"][str(n)].items():
                d.setdefault(mname, []).append(v)
            d.setdefault("_seed", []).append(r["seed_index"])
        sat[(r["ratio"], r["sigma"])].append(
            r["fgr2"][str(sizes[0])]["frac_saturated"])
        sing[(r["ratio"], r["sigma"])].append(
            r["fgr2"][str(sizes[0])]["frac_count_eq_1"])

    cells = []
    for (n, ratio, sigma), d in sorted(idx.items()):
        cell = {"N": n, "ratio": ratio, "sigma": sigma,
                "n_replicates": len(d["_seed"]),
                "frac_saturated_mean": float(np.mean(sat[(ratio, sigma)])),
                "frac_singleton_mean": float(np.mean(sing[(ratio, sigma)]))}
        for mname, vals in d.items():
            if mname == "_seed":
                continue
            cell[mname] = {"mean": float(np.mean(vals)),
                           "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                           "values": [float(v) for v in vals]}
        cells.append(cell)

    def get(n, ratio, sigma, mname):
        for c in cells:
            if c["N"] == n and c["ratio"] == ratio and c["sigma"] == sigma:
                return c.get(mname, {}).get("values")
        return None

    res = {"label": label, "cells": cells}

    # ---- 1. fixed-ratio discrimination (both pools at the same depth) ------
    fixed = {}
    for n in sizes:
        fixed[str(n)] = {}
        for mname in SIMMETRICS:
            fixed[str(n)][mname] = []
            for ratio in ratios:
                row = {"ratio": ratio}
                neg = get(n, ratio, 0.0, mname)
                for sg in sigmas:
                    if sg == 0.0:
                        continue
                    pos = get(n, ratio, sg, mname)
                    if neg is None or pos is None:
                        continue
                    row[f"auc_sigma{sg}"] = auc([-v for v in neg],
                                                [-v for v in pos])
                    row[f"p_sigma{sg}"] = mw_p([-v for v in neg],
                                               [-v for v in pos])
                fixed[str(n)][mname].append(row)
    res["fixed_ratio_auc"] = fixed

    # ---- 2. mixed-ratio discrimination: depth unknown within window W -----
    windows = sorted({round(max(r, 1.0 / r), 4) for r in ratios})
    mixed = {}
    for n in sizes:
        mixed[str(n)] = {}
        for mname in SIMMETRICS:
            rows = []
            for W in windows:
                inw = [r for r in ratios if max(r, 1.0 / r) <= W + 1e-9]
                for sg in sigmas:
                    if sg == 0.0:
                        continue
                    neg, pos = [], []
                    for r in inw:
                        a = get(n, r, 0.0, mname)
                        b = get(n, r, sg, mname)
                        if a is None or b is None:
                            continue
                        neg += [-v for v in a]
                        pos += [-v for v in b]
                    if not neg or not pos:
                        continue
                    rows.append({"window": W, "sigma": sg,
                                 "ratios_included": inw,
                                 "n_neg": len(neg), "n_pos": len(pos),
                                 "auc": auc(neg, pos), "p": mw_p(neg, pos)})
            mixed[str(n)][mname] = rows
    res["mixed_ratio_auc"] = mixed

    # tolerance: largest window with AUC >= 0.95
    tol = {}
    for n in sizes:
        tol[str(n)] = {}
        for mname in SIMMETRICS:
            tol[str(n)][mname] = {}
            for sg in sigmas:
                if sg == 0.0:
                    continue
                ok = [r["window"] for r in mixed[str(n)][mname]
                      if r["sigma"] == sg and r["auc"] is not None
                      and r["auc"] >= 0.95]
                # must be a contiguous run starting at the smallest window
                best = None
                for W in windows:
                    if W in ok:
                        best = W
                    else:
                        break
                tol[str(n)][mname][str(sg)] = best
    res["depth_tolerance_auc0.95"] = tol

    # ---- 3. signal vs depth artifact --------------------------------------
    sig_art = {}
    for n in sizes:
        rows = []
        base = get(n, 1.0, 0.0, "cosine")
        if base is None:
            continue
        base_mu = float(np.mean(base))
        for sg in sigmas:
            if sg == 0.0:
                continue
            v = get(n, 1.0, sg, "cosine")
            if v is not None:
                rows.append({"kind": "signal", "sigma": sg,
                             "delta_cosine": base_mu - float(np.mean(v))})
        for ratio in ratios:
            if ratio == 1.0:
                continue
            v = get(n, ratio, 0.0, "cosine")
            if v is None:
                continue
            # paired: same seed index, same abundance profile (uniform), only depth
            d = base_mu - float(np.mean(v))
            try:
                w = stats.wilcoxon(np.array(base), np.array(v))
                pw = float(w.pvalue)
            except ValueError:
                pw = None
            rows.append({"kind": "depth_artifact", "ratio": ratio,
                         "delta_cosine": d, "wilcoxon_p_vs_ratio1": pw,
                         "n_pairs": len(base)})
        sig_art[str(n)] = rows
    res["signal_vs_artifact"] = sig_art
    return res


def main():
    raw = json.load(open(RAW))
    recs = raw["records"]
    par = raw["parameters"]
    out = {"experiment_id": "F3_depth_mismatch_analysis",
           "source_raw": RAW, "parameters": par, "stages": {}}

    byname = {r["tag"]: r for r in recs}

    A = [r for r in recs if r["stage"] == "A" and not r["tag"].startswith("ref")]
    out["stages"]["A_error1e-3"] = analyse_stage(
        A, byname["refA"], par["stage_A"]["N_sizes"], par["stage_A"]["sigmas"],
        par["stage_A"]["ratios"], True, "E5 control design + depth ratio, err=1e-3")

    B = [r for r in recs if r["stage"] == "B" and not r["tag"].startswith("ref")]
    out["stages"]["B_error0"] = analyse_stage(
        B, byname["refB"], [1000], par["stage_B"]["sigmas"],
        par["stage_B"]["ratios"], False, "same design, ZERO sequencing error")

    for arm in par["stage_C"]["arms"]:
        Cc = [r for r in recs if r["stage"] == "C" and r.get("arm") == arm
              and not r["tag"].startswith("ref")]
        out["stages"][f"C_{arm}"] = analyse_stage(
            Cc, byname[f"refC_{arm}"], [1000], par["stage_C"]["sigmas"],
            par["stage_C"]["ratios"], False,
            f"saturation arm '{arm}', base_copies={par['stage_C']['arms'][arm]}")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)

    # ------------------------------------------------------------- summary
    st = out["stages"]["A_error1e-3"]
    print("\n=== Stage A (E5 design, err 1e-3), N=1000: mean cosine ===")
    ratios = par["stage_A"]["ratios"]
    sigmas = par["stage_A"]["sigmas"]
    print("ratio   " + "".join(f"sig{s:<7}" for s in sigmas) + " sat%   single%")
    for r in ratios:
        row = f"{r:<7}"
        satv = singv = 0.0
        for s in sigmas:
            c = next((c for c in st["cells"] if c["N"] == 1000
                      and c["ratio"] == r and c["sigma"] == s), None)
            row += f"{c['cosine']['mean']:<10.5f}" if c else " " * 10
            if c and s == 0.0:
                satv, singv = c["frac_saturated_mean"], c["frac_singleton_mean"]
        print(row + f" {satv*100:.2f}   {singv*100:.1f}")

    print("\n=== signal vs depth artifact (N=1000, cosine) ===")
    for row in st["signal_vs_artifact"]["1000"]:
        print(" ", row)

    print("\n=== depth-mismatch tolerance (largest window W with AUC>=0.95) ===")
    for n in ("1000", "10000"):
        for mname in SIMMETRICS:
            print(f"  N={n:<6}{mname:<20}",
                  st["depth_tolerance_auc0.95"][n][mname])


if __name__ == "__main__":
    main()
