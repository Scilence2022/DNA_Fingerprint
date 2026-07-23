#!/usr/bin/env python3
"""F4 figure: legacy vs corrected Jaccard estimator on the 16-genome E4 set."""
import json, os, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
R = json.load(open(os.path.join(REPO, 'analysis', 'results', 'f4_e4_recheck.json')))
FIGS = os.path.join(REPO, 'analysis', 'figures')
os.makedirs(FIGS, exist_ok=True)

names = R['genomes']
pairs = ['%s|%s' % p for p in itertools.combinations(names, 2)]
ex = np.array([R['raw_jaccard']['exact'][p] for p in pairs])
get = lambda tag: np.array([R['raw_jaccard'][tag][p] for p in pairs])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# (a) estimate vs exact, N=10000
ax = axes[0]
for tag, lab, c, mk in (('legacy_direct_N10000', 'legacy direct', '#c44e52', 'o'),
                        ('corrected_union_N10000', 'corrected union', '#4c72b0', '^')):
    ax.scatter(ex, get(tag), s=22, alpha=0.7, label=lab, color=c, marker=mk)
lim = [0, max(ex.max(), 1.0) * 1.02]
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='y = x')
ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 1.0)
ax.set_xlabel('exact Jaccard (full 31-mer sets)')
ax.set_ylabel('sketch estimate, N = 10000')
ax.set_title('(a) Estimate vs truth, 120 genome pairs')
ax.legend(frameon=False, fontsize=9)

# (b) signed error vs exact Jaccard, N=10000
ax = axes[1]
for tag, lab, c, mk in (('legacy_direct_N10000', 'legacy direct', '#c44e52', 'o'),
                        ('corrected_union_N10000', 'corrected union', '#4c72b0', '^')):
    ax.scatter(ex, get(tag) - ex, s=22, alpha=0.7, label=lab, color=c, marker=mk)
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xlabel('exact Jaccard')
ax.set_ylabel('estimate - exact')
ax.set_title('(b) Signed error, N = 10000')
ax.legend(frameon=False, fontsize=9)

# (c) RMSE vs N, all pairs and informative stratum
ax = axes[2]
Ns = [1000, 10000]
p = R['paired_abs_error_legacy_vs_corrected']
for stratum, ls, mk in (('all_120_pairs', '-', 'o'),
                        ('informative_exactJ_ge_0.01', '--', 's')):
    for est, lab, c in (('rmse_legacy', 'legacy', '#c44e52'),
                        ('rmse_corrected', 'corrected', '#4c72b0')):
        y = [p[stratum]['N%d' % N][est] for N in Ns]
        n = p[stratum]['N1000']['n']
        ax.plot(Ns, y, ls=ls, marker=mk, color=c,
                label='%s, %s (n=%d)' % (lab, stratum.split('_')[0], n))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('sketch size N'); ax.set_ylabel('RMSE vs exact Jaccard')
ax.set_title('(c) Error vs N: legacy floors, corrected shrinks')
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
out = os.path.join(FIGS, 'f4_estimator_correction.png')
fig.savefig(out, dpi=200)
print('wrote', out)
