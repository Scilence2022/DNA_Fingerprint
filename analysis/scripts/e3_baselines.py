#!/usr/bin/env python3
"""Recompute threshold-selection baselines from the stored E3 histograms.

The first version of the valley baseline in e3_coverage_autodetect.py located
the signal peak as the RIGHTMOST local maximum of the smoothed spectrum, which
in practice latched onto noise in the high-coverage tail and returned an
absurdly high threshold (mean F1 ~0.004).  That implementation was wrong and is
replaced here.  Because the full TRUE/ERROR coverage histograms were persisted,
the baselines can be recomputed exactly without re-simulating any reads.

Baselines computed:
  valley_globalmax : 3-point moving average; signal peak = GLOBAL argmax over
                     c >= 3; threshold = argmin of the smoothed curve over
                     [1, peak], +1 (first bin kept).  None if no c >= 3 exists.
  fgr2_rule_smoothed : fgr2's own "first increase after a decreasing streak"
                     rule, applied to the 3-point-smoothed spectrum instead of
                     the raw one.  Isolates whether fgr2's failures are caused
                     by the rule itself or by its sensitivity to raw-count noise.
"""
import json
import os

import numpy as np

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
RES = os.path.join(REPO, "analysis", "results")
IN = os.path.join(RES, "e3_coverage_autodetect.json")
OUT = os.path.join(RES, "e3_baselines.json")


def smooth(hist, cmax):
    h = np.zeros(cmax + 3, dtype=float)
    n = min(cmax + 1, hist.size)
    h[1:n] = hist[1:n]
    sm = h.copy()
    for c in range(2, cmax + 1):
        sm[c] = (h[c - 1] + h[c] + h[c + 1]) / 3.0
    return sm


def valley_globalmax(hist, cmax):
    if cmax < 4:
        return None
    sm = smooth(hist, cmax)
    peak = int(np.argmax(sm[3:cmax + 1])) + 3
    if peak <= 1:
        return None
    valley = int(np.argmin(sm[1:peak + 1])) + 1
    return int(min(valley + 1, cmax))


def fgr2_rule(counts, hist_size):
    """Faithful re-implementation of fgr2.c find_first_increasing_coverage."""
    start = 1
    while start < hist_size and counts[start] == 0:
        start += 1
    if start >= hist_size:
        return 0
    prev = counts[start]
    dec = 0
    for i in range(start + 1, hist_size):
        if counts[i] == 0:
            continue
        if counts[i] < prev:
            dec += 1
            prev = counts[i]
            continue
        if dec > 0 and counts[i] > prev:
            return i
        prev = counts[i]
    return 0


def fgr2_rule_smoothed(hist, cmax):
    sm = smooth(hist, cmax)
    r = fgr2_rule(sm, cmax + 1)
    return int(r) if r > 0 else None


def metrics(hist_true, hist_err, cmax):
    n_true = int(hist_true.sum())
    n_err = int(hist_err.sum())
    tt = np.concatenate((np.cumsum(hist_true[::-1])[::-1], [0]))
    ee = np.concatenate((np.cumsum(hist_err[::-1])[::-1], [0]))
    f1, you = {}, {}
    for c in range(1, cmax + 1):
        tp = int(tt[c]) if c < tt.size else 0
        fp = int(ee[c]) if c < ee.size else 0
        fn = n_true - tp
        den = 2 * tp + fp + fn
        f1[c] = (2.0 * tp / den) if den else 0.0
        you[c] = (tp / n_true if n_true else 0.0) - (fp / n_err if n_err else 0.0)
    return f1, you


D = json.load(open(IN))
out = []
for r in D["records"]:
    ht = np.array(r["hist_true"], dtype=np.int64)
    he = np.array(r["hist_error"], dtype=np.int64)
    tot = ht + he
    cmax = r["cmax_searched"]
    f1, you = metrics(ht, he, cmax)
    v = valley_globalmax(tot, cmax)
    s = fgr2_rule_smoothed(tot, cmax)
    # sanity: the faithful re-implementation must reproduce fgr2's own choice
    # scan the FULL stored spectrum (601 bins), not just [1, cmax]: fgr2 itself
    # scans all 16384 bins, so truncating here would create spurious mismatches.
    raw = fgr2_rule(tot.astype(float), tot.size)
    out.append(dict(
        genome=r["genome"], profile=r["profile"], profile_class=r["profile_class"],
        seed=r["seed"], depth=r["depth"],
        detected=r["detected"], optimal_f1_c=r["optimal_f1_c"],
        optimal_f1=r["optimal_f1"], f1_at_detected=r["f1_at_detected"],
        valley_globalmax_c=v, f1_at_valley_globalmax=f1.get(v) if v else None,
        youden_at_valley_globalmax=you.get(v) if v else None,
        fgr2_rule_smoothed_c=s, f1_at_fgr2_rule_smoothed=f1.get(s) if s else None,
        reimpl_raw_rule_c=int(raw),
        # fgr2 falls back to c=1 when the rule returns 0 (no increase found)
        reimpl_matches_fgr2=bool((int(raw) if int(raw) > 0 else 1) == r["detected"]),
    ))

agree = float(np.mean([o["reimpl_matches_fgr2"] for o in out]))


def mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


depths = D["meta"]["depths"]
profs = [p["name"] for p in D["meta"]["profiles"]]
per_depth = []
for d in depths:
    rs = [o for o in out if o["depth"] == d]
    per_depth.append(dict(
        depth=d, n=len(rs),
        f1_auto=mean([o["f1_at_detected"] for o in rs]),
        f1_optimal=mean([o["optimal_f1"] for o in rs]),
        f1_valley_globalmax=mean([o["f1_at_valley_globalmax"] for o in rs]),
        n_valley_defined=sum(1 for o in rs if o["valley_globalmax_c"] is not None),
        f1_fgr2_rule_smoothed=mean([o["f1_at_fgr2_rule_smoothed"] for o in rs]),
        n_smoothed_defined=sum(1 for o in rs if o["fgr2_rule_smoothed_c"] is not None),
        valley_c_mean=mean([o["valley_globalmax_c"] for o in rs]),
        smoothed_c_mean=mean([o["fgr2_rule_smoothed_c"] for o in rs]),
    ))

per_prof_depth = []
for p in profs:
    for d in depths:
        rs = [o for o in out if o["depth"] == d and o["profile"] == p]
        if not rs:
            continue
        per_prof_depth.append(dict(
            profile=p, depth=d, n=len(rs),
            f1_auto=mean([o["f1_at_detected"] for o in rs]),
            f1_optimal=mean([o["optimal_f1"] for o in rs]),
            f1_valley_globalmax=mean([o["f1_at_valley_globalmax"] for o in rs]),
            f1_fgr2_rule_smoothed=mean([o["f1_at_fgr2_rule_smoothed"] for o in rs]),
        ))

summary = dict(
    note=__doc__,
    reimplementation_agreement_with_fgr2=agree,
    overall=dict(
        f1_auto=mean([o["f1_at_detected"] for o in out]),
        f1_optimal=mean([o["optimal_f1"] for o in out]),
        f1_valley_globalmax=mean([o["f1_at_valley_globalmax"] for o in out]),
        f1_fgr2_rule_smoothed=mean([o["f1_at_fgr2_rule_smoothed"] for o in out]),
        n_valley_defined=sum(1 for o in out if o["valley_globalmax_c"] is not None),
        n_smoothed_defined=sum(1 for o in out
                               if o["fgr2_rule_smoothed_c"] is not None),
        n=len(out),
    ),
    per_depth=per_depth,
    per_profile_depth=per_prof_depth,
)
json.dump(dict(summary=summary, records=out), open(OUT, "w"), indent=1)

print("re-implementation agreement with fgr2's own choice:", agree)
print(json.dumps(summary["overall"], indent=1))
print(f"\n{'d':>5s}{'F1auto':>9s}{'F1valley':>10s}{'F1smoothRule':>14s}{'F1opt':>9s}"
      f"{'valley_c':>10s}{'smooth_c':>10s}")
for r in per_depth:
    def f(x, w=9):
        return f"{x:{w}.4f}" if x is not None else f"{'--':>{w}s}"
    print(f"{r['depth']:5d}{f(r['f1_auto'])}{f(r['f1_valley_globalmax'],10)}"
          f"{f(r['f1_fgr2_rule_smoothed'],14)}{f(r['f1_optimal'])}"
          f"{f(r['valley_c_mean'],10)}{f(r['smoothed_c_mean'],10)}")
print("\nwrote", OUT)
