"""E7 figures. Reads only the persisted result JSONs -- no recomputation."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e7_common as C

os.makedirs(C.FIGURES, exist_ok=True)
DPI = 200


def fig_a():
    p = os.path.join(C.RESULTS, "E7a_variance.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    NS = d["N_values"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    bins = sorted(d["summary"]["by_N_and_Jbin"].keys(),
                  key=lambda s: float(s.split("-")[0]))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(bins)))
    for c, b in zip(cmap, bins):
        rec = d["summary"]["by_N_and_Jbin"][b]
        ns = [n for n in NS if str(n) in rec]
        emp = [rec[str(n)]["union_se_median"] for n in ns]
        theo = [rec[str(n)]["theoretical_se_median"] for n in ns]
        ax[0].loglog(ns, emp, "o-", color=c,
                     label=f"J∈[{b}]  n={rec[str(ns[0])]['n_pairs']}")
        ax[0].loglog(ns, theo, "--", color=c, alpha=0.55)
        ax[1].semilogx(ns, [rec[str(n)]["ratio_union_median"] for n in ns],
                       "o-", color=c, label=f"J∈[{b}]")
    ax[0].set_xlabel("sketch size N"); ax[0].set_ylabel("SE of Jaccard estimate")
    ax[0].set_title(f"(a) empirical SE (solid) vs $\\sqrt{{J(1-J)/N}}$ (dashed)\n"
                    f"n={d['n_replicates']} hash-seed replicates per pair")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which="both")
    ax[1].axhline(1.0, color="k", lw=1, ls=":")
    ax[1].set_xlabel("sketch size N"); ax[1].set_ylabel("empirical SE / theoretical SE")
    ax[1].set_title("(b) ratio to binomial prediction")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

    ub = [d["summary"]["by_N"][str(n)]["union_bias_median"] for n in NS]
    db = [d["summary"]["by_N"][str(n)]["direct_bias_median"] for n in NS]
    ax[2].semilogx(NS, ub, "o-", label="union (Mash) estimator")
    ax[2].semilogx(NS, db, "s-", label="direct estimator\n(calculate_similarity.py)")
    ax[2].axhline(0, color="k", lw=1, ls=":")
    ax[2].set_xlabel("sketch size N"); ax[2].set_ylabel("median bias (est − true J)")
    ax[2].set_title("(c) estimator bias"); ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(C.FIGURES, "E7a_variance_vs_N.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def fig_b():
    p = os.path.join(C.RESULTS, "E7b_hash.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    hs = ["murmurhash3", "wang"]
    cols = {"murmurhash3": "#2b6cb0", "wang": "#c05621"}

    g = "Ecoli_K12_MG1655"
    x = np.arange(64)
    for h in hs:
        f = np.array(d["uniformity"][g][h]["per_bit"]["fraction_set"])
        ax[0, 0].plot(x, f - 0.5, "o-", ms=3, color=cols[h], label=h)
    ax[0, 0].axhline(0, color="k", lw=1, ls=":")
    ax[0, 0].set_xlabel("output bit index (0 = LSB)")
    ax[0, 0].set_ylabel("P(bit set) − 0.5")
    ax[0, 0].set_title(f"(a) per-output-bit balance, real 31-mers of\n{g}")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    labels, mm, ww = [], [], []
    for g2 in d["uniformity"]:
        for part in ("top16", "low16"):
            labels.append(f"{g2.split('_')[0]}\n{part}")
            mm.append(d["uniformity"][g2]["murmurhash3"][part]["chi2_over_dof"])
            ww.append(d["uniformity"][g2]["wang"][part]["chi2_over_dof"])
    xi = np.arange(len(labels))
    ax[0, 1].bar(xi - 0.2, mm, 0.4, color=cols["murmurhash3"], label="murmurhash3")
    ax[0, 1].bar(xi + 0.2, ww, 0.4, color=cols["wang"], label="wang")
    ax[0, 1].axhline(1.0, color="k", lw=1, ls=":")
    ax[0, 1].set_xticks(xi); ax[0, 1].set_xticklabels(labels, fontsize=6)
    ax[0, 1].set_ylabel("chi² / dof   (1.0 = uniform)")
    ax[0, 1].set_title("(b) bucket uniformity, 65536 buckets")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3, axis="y")

    for h in hs:
        rows = np.array(d["avalanche"][h]["matrix_row_means"])
        ax[1, 0].plot(np.arange(rows.size), rows, "o-", ms=3, color=cols[h],
                      label=f"{h} (max|dev|={d['avalanche'][h]['max_abs_dev_from_half']:.3f})")
    ax[1, 0].axhline(0.5, color="k", lw=1, ls=":")
    ax[1, 0].set_xlabel("flipped input bit of the 2-bit k-mer code")
    ax[1, 0].set_ylabel("mean output-bit flip probability")
    ax[1, 0].set_title(f"(c) avalanche, n={d['avalanche']['wang']['n_sample']} real k-mers")
    ax[1, 0].legend(fontsize=7); ax[1, 0].grid(alpha=0.3)

    keys, bars = [], {h: [] for h in hs}
    for est in ("union", "direct"):
        for N in (1000, 10000):
            keys.append(f"{est}\nN={N}")
            for h in hs:
                bars[h].append(d["accuracy"]["summary"][f"{h}_{est}_N{N}"]["rmse"])
    xi = np.arange(len(keys))
    ax[1, 1].bar(xi - 0.2, bars["murmurhash3"], 0.4, color=cols["murmurhash3"],
                 label="murmurhash3")
    ax[1, 1].bar(xi + 0.2, bars["wang"], 0.4, color=cols["wang"], label="wang")
    ax[1, 1].set_xticks(xi); ax[1, 1].set_xticklabels(keys, fontsize=8)
    ax[1, 1].set_ylabel("RMSE vs exact Jaccard")
    ax[1, 1].set_title(f"(d) estimator accuracy, "
                       f"{d['accuracy']['summary']['wang_union_N1000']['n_pairs']} genome pairs")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(C.FIGURES, "E7b_hash_comparison.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def fig_c():
    p = os.path.join(C.RESULTS, "E7c_saturation.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    ds = sorted(int(x) for x in d["depth_sweep"]["per_depth"])
    fs = [100 * d["depth_sweep"]["per_depth"][str(x)]["frac_saturated"] for x in ds]
    mt = [d["depth_sweep"]["per_depth"][str(x)]["max_true_count"] for x in ds]
    ax[0].plot(ds, fs, "o-", color="#c53030")
    ax[0].set_xlabel("simulated sequencing depth (x)")
    ax[0].set_ylabel("% of sketch k-mers saturated", color="#c53030")
    ax[0].grid(alpha=0.3)
    a2 = ax[0].twinx()
    a2.plot(ds, mt, "s--", color="#2b6cb0", ms=4)
    a2.axhline(d["KC_MAX"], color="k", ls=":", lw=1)
    a2.set_ylabel("max true k-mer count", color="#2b6cb0")
    ax[0].set_title("(a) depth sweep, error-free reads\n"
                    "10 kb E. coli fragment, k=31, KC_MAX=16383")

    sc = sorted(int(x) for x in d["copy_sweep"]["per_scale"])
    sat = [100 * d["copy_sweep"]["per_scale"][str(x)]["frac_saturated_mean"] for x in sc]
    ax[1].semilogx(sc, sat, "o-", color="#c53030")
    ax[1].set_xlabel("copy-number scale s")
    ax[1].set_ylabel("% of sketch k-mers saturated")
    ax[1].set_title("(b) oligo-pool copy-number sweep")
    ax[1].grid(alpha=0.3, which="both")

    err = [d["copy_sweep"]["per_scale"][str(x)]["cosine_signed_error"] for x in sc]
    jerr = [d["copy_sweep"]["per_scale"][str(x)]["jaccard_observed"]
            - d["copy_sweep"]["per_scale"][str(x)]["jaccard_true"] for x in sc]
    ax[2].plot(sat, err, "o-", color="#805ad5",
               label="cosine error, symmetric\n(both samples saturate)")
    if "asymmetric_sweep" in d:
        a = d["asymmetric_sweep"]["per_scale"]
        asc = sorted(int(x) for x in a)
        ax[2].plot([100 * a[str(x)]["frac_saturated_x"] for x in asc],
                   [a[str(x)]["cosine_signed_error"] for x in asc],
                   "^-", color="#c53030",
                   label="cosine error, asymmetric\n(only one sample saturates)")
    ax[2].plot(sat, jerr, "s--", color="#38a169", label="Jaccard error (control)")
    ax[2].axhline(0, color="k", lw=1, ls=":")
    ax[2].set_xlabel("% of sketch k-mers saturated")
    ax[2].set_ylabel("observed − true similarity")
    ax[2].set_title("(c) saturation-induced similarity error")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(C.FIGURES, "E7c_saturation.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig_a, fig_b, fig_c):
        print(f.__name__, "->", f())
