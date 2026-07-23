"""E7(c) addendum: ASYMMETRIC saturation.

In the symmetric sweep both samples are scaled together, so both saturate in
the same places and cosine (being scale invariant) stays close to the truth.
The realistic failure mode is comparing a deeply sequenced / high-copy sample
against a shallow one: only one side saturates, so the compression of its
abundance vector is not mirrored on the other side.

Here sample Y is pinned at scale 1000 (no k-mer anywhere near KC_MAX) and
only sample X is scaled. cosine_true is scale invariant, so any movement in
cosine_observed is attributable to saturation of X alone.

Appends "asymmetric_sweep" to E7c_saturation.json.
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_common as C
import e7c_saturation as E
import fgrlib

SEED = E.SEED
KC_MAX = E.KC_MAX
K = E.K
Y_SCALE = 1000


def main(n_oligos=60, oligo_len=200,
         scales=(1000, 3000, 8000, 16383, 25000, 40000)):
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)
    pool = fgrlib.make_oligo_pool(n_oligos, oligo_len, rng)
    wx = nprng.lognormal(0.0, 1.0, n_oligos); wx /= wx.mean()
    wy = wx * nprng.lognormal(0.0, 0.3, n_oligos); wy /= wy.mean()
    kmers = [sorted(fgrlib.canonical_kmers(o, K)) for o in pool]

    fa = os.path.join(E.WORK, "asym.fa")
    sk = os.path.join(E.WORK, "asym.fgr2")

    def build(w, s):
        copies = [max(1, int(round(s * wi))) for wi in w]
        E.write_records(fa, ((f"o{i}_{j}", pool[i])
                             for i in range(n_oligos)
                             for j in range(copies[i])))
        _, cov, _ = E.run_fgr2(fa, sk)
        t = {}
        for i in range(n_oligos):
            for km in kmers[i]:
                t[km] = t.get(km, 0) + copies[i]
        return cov, t, copies

    covy, truey, copies_y = build(wy, Y_SCALE)
    assert max(covy.values()) < KC_MAX, "reference sample must not saturate"

    out = {"y_scale": Y_SCALE, "n_oligos": n_oligos, "oligo_len": oligo_len,
           "seed": SEED, "y_max_count": max(covy.values()),
           "y_saturated": 0, "per_scale": {}}
    for s in scales:
        covx, truex, copies_x = build(wx, s)
        keys = sorted(set(truex) | set(truey))
        ct = E.cosine_np(truex, truey, keys)
        co = E.cosine_np(covx, covy, keys)
        nsat = sum(1 for v in covx.values() if v >= KC_MAX)
        rec = {
            "x_scale": s,
            "n_saturated_x": nsat,
            "frac_saturated_x": nsat / len(covx),
            "max_true_count_x": max(truex.values()),
            "max_reported_count_x": max(covx.values()),
            "cosine_true": ct,
            "cosine_observed": co,
            "cosine_signed_error": co - ct,
            "cosine_abs_error": abs(co - ct),
            "jaccard_observed": len(set(covx) & set(covy)) / len(set(covx) | set(covy)),
        }
        out["per_scale"][str(s)] = rec
        print(f"  x_scale {s:6d}: sat_x={100*rec['frac_saturated_x']:6.2f}% "
              f"cos_true={ct:.6f} cos_obs={co:.6f} err={co-ct:+.6f} "
              f"J={rec['jaccard_observed']:.6f}", flush=True)
    for p in (fa, sk):
        if os.path.exists(p):
            os.remove(p)

    errs = [abs(out["per_scale"][k]["cosine_signed_error"]) for k in out["per_scale"]]
    out["max_abs_cosine_error"] = float(max(errs))
    path = os.path.join(C.RESULTS, "E7c_saturation.json")
    d = json.load(open(path))
    d["asymmetric_sweep"] = out
    with open(path, "w") as fh:
        json.dump(d, fh)
    print("max |cosine error| (asymmetric):", out["max_abs_cosine_error"])
    print("updated", path)


if __name__ == "__main__":
    main()
