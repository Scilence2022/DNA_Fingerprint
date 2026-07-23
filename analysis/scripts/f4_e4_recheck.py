#!/usr/bin/env python3
"""
F4 -- Re-run the E4 calibration/phylogeny pipeline through the CORRECTED
estimators now shipped in calculate_similarity.py, and report whether any E4
conclusion changes.

Everything here is computed in this session:
  * exact canonical-31-mer Jaccard is recomputed from the 16 RefSeq .fna files
    (and cross-checked against E4's cached values, reported as a diagnostic);
  * sketch Jaccards are computed by importing calculate_similarity.jaccard_union
    (new default) and calculate_similarity.jaccard_index (--legacy-jaccard), so
    the numbers below are the ones the shipped tool now produces;
  * NJ trees and Robinson-Foulds distances to the two reference topologies are
    rebuilt from those distance matrices.

The reference topologies are literature-derived INPUTS, copied verbatim from
E4's recorded strings; they are not measurements.
"""
import os, sys, json, math, itertools, importlib.util
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon
from skbio import DistanceMatrix
from skbio.tree import nj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import fgrlib

SCRATCH = ('/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-'
           '-claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad')
E4 = os.path.join(SCRATCH, 'e4')
GENOMES = os.path.join(SCRATCH, 'genomes')
RESULTS = os.path.join(REPO, 'analysis', 'results')
FIGS = os.path.join(REPO, 'analysis', 'figures')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

K = 31
SEED = 20260721
np.random.seed(SEED)

spec = importlib.util.spec_from_file_location("cs", os.path.join(REPO, "calculate_similarity.py"))
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)

E4_JSON = json.load(open(os.path.join(RESULTS, 'e4_calibration_phylogeny.json')))
REFS = {k: v for k, v in E4_JSON['reference_topologies_newick'].items() if k != 'note'}
NAMES = list(E4_JSON['genomes'])
NT = len(NAMES)
PAIRS = list(itertools.combinations(NAMES, 2))

# ------------------------------------------------------------------ split utils
def splits(tree, taxa):
    taxa = set(taxa); anchor = min(taxa)
    out = set()
    for node in tree.postorder(include_self=False):
        leaves = (frozenset(l.name for l in node.tips()) if not node.is_tip()
                  else frozenset([node.name]))
        if not leaves or leaves - taxa:
            continue
        comp = frozenset(taxa - leaves)
        if len(leaves) < 2 or len(comp) < 2:
            continue
        out.add(comp if anchor in leaves else leaves)
    return out

def rf(t1, t2, taxa):
    s1, s2 = splits(t1, taxa), splits(t2, taxa)
    return len(s1 ^ s2), len(s1), len(s2), len(s1 & s2)

# ------------------------------------------------------------------ exact truth
print('Recomputing exact canonical %d-mer Jaccard for %d genomes...' % (K, NT))
codes = {}
for n in NAMES:
    seq = fgrlib.load_genome(os.path.join(GENOMES, n + '.fna'))
    codes[n] = np.unique(fgrlib.fast_canonical_codes(seq, K))
    print('  %-32s %d distinct %d-mers' % (n, codes[n].size, K))

EX_J = {}
for a, b in PAIRS:
    inter = np.intersect1d(codes[a], codes[b], assume_unique=True).size
    union = codes[a].size + codes[b].size - inter
    EX_J[(a, b)] = inter / union

cached = E4_JSON['raw_jaccard_estimates']['exact']
max_dev = max(abs(EX_J[(a, b)] - cached['%s|%s' % (a, b)]) for a, b in PAIRS)
print('Max deviation of recomputed exact Jaccard from E4 cache: %.3e' % max_dev)

SUPPORT_RATIO = {(a, b): max(codes[a].size, codes[b].size) / min(codes[a].size, codes[b].size)
                 for a, b in PAIRS}

# ------------------------------------------------------------------ sketches
SK = {}
for N in (1000, 10000):
    SK[N] = {}
    for n in NAMES:
        recs, meta = cs.parse_sketch(os.path.join(E4, 'fgr2_%d' % N, n + '.fgr2'))
        assert recs is not None and len(recs) == N, (n, N, None if recs is None else len(recs))
        SK[N][n] = recs

methods = {'exact': {p: EX_J[p] for p in PAIRS}}
for N in (1000, 10000):
    methods['legacy_direct_N%d' % N] = {
        p: cs.jaccard_index(set(SK[N][p[0]]), set(SK[N][p[1]])) for p in PAIRS}
    methods['corrected_union_N%d' % N] = {
        p: cs.jaccard_union(SK[N][p[0]], SK[N][p[1]]) for p in PAIRS}

# cross-check: the shipped estimators must agree with E4's recorded values
xcheck = {}
for N in (1000, 10000):
    for tag, key in (('legacy_direct_N%d' % N, 'fgr2_direct_N%d' % N),
                     ('corrected_union_N%d' % N, 'fgr2_union_N%d' % N)):
        rec = E4_JSON['raw_jaccard_estimates'][key]
        xcheck[tag] = max(abs(methods[tag][(a, b)] - rec['%s|%s' % (a, b)]) for a, b in PAIRS)
print('Max deviation vs E4 recorded estimates:', json.dumps(xcheck))

def d_raw(j):  return 1.0 - j
def d_pois(j): return fgrlib.mash_distance(j, K)

# ------------------------------------------------------------------ calibration
calib = {}
ex = np.array([EX_J[p] for p in PAIRS])
for tag, m in methods.items():
    if tag == 'exact':
        continue
    v = np.array([m[p] for p in PAIRS])
    calib[tag] = {
        'n_pairs': len(PAIRS),
        'pearson_r_vs_exact': float(pearsonr(v, ex)[0]),
        'spearman_rho_vs_exact': float(spearmanr(v, ex)[0]),
        'rmse_vs_exact': float(np.sqrt(np.mean((v - ex) ** 2))),
        'mean_bias_vs_exact': float(np.mean(v - ex)),
        'max_abs_err_vs_exact': float(np.max(np.abs(v - ex))),
        'poisson_dist_rmse_vs_exact': float(np.sqrt(np.mean(
            (np.array([d_pois(x) for x in v]) - np.array([d_pois(x) for x in ex])) ** 2))),
    }

# Paired Wilcoxon on |error|, legacy vs corrected, at each N.
# Reported on ALL 120 pairs and on the "informative" stratum. Most of the 120
# cross-genus pairs have exact J near 0, where BOTH estimators are numerically
# near-exact; those near-ties dominate a rank test and mask the improvement,
# which is concentrated in the within-family pairs. Both are reported.
STRATA = {
    'all_120_pairs': [p for p in PAIRS],
    'informative_exactJ_ge_0.01': [p for p in PAIRS if EX_J[p] >= 0.01],
    'saturated_exactJ_lt_0.01': [p for p in PAIRS if EX_J[p] < 0.01],
}
paired = {}
for sname, subset in STRATA.items():
    if len(subset) < 3:
        continue
    exs = np.array([EX_J[p] for p in subset])
    paired[sname] = {}
    for N in (1000, 10000):
        e_leg = np.abs(np.array([methods['legacy_direct_N%d' % N][p] for p in subset]) - exs)
        e_cor = np.abs(np.array([methods['corrected_union_N%d' % N][p] for p in subset]) - exs)
        stat, p = wilcoxon(e_leg, e_cor)
        paired[sname]['N%d' % N] = {
            'n': len(subset),
            'mean_abs_err_legacy': float(np.mean(e_leg)),
            'mean_abs_err_corrected': float(np.mean(e_cor)),
            'median_abs_err_legacy': float(np.median(e_leg)),
            'median_abs_err_corrected': float(np.median(e_cor)),
            'rmse_legacy': float(np.sqrt(np.mean(e_leg ** 2))),
            'rmse_corrected': float(np.sqrt(np.mean(e_cor ** 2))),
            'n_corrected_better': int(np.sum(e_cor < e_leg)),
            'wilcoxon_stat': float(stat), 'wilcoxon_p': float(p),
            'significant_at_0.05': bool(p < 0.05),
            'favours': 'corrected' if np.mean(e_cor) < np.mean(e_leg) else 'legacy',
        }

# ------------------------------------------------------------------ trees + RF
def tree_from(m, transform):
    D = np.zeros((NT, NT))
    idx = {n: i for i, n in enumerate(NAMES)}
    for (a, b), j in m.items():
        d = transform(j)
        D[idx[a], idx[b]] = D[idx[b], idx[a]] = d
    np.fill_diagonal(D, 0.0)
    return nj(DistanceMatrix(D, ids=NAMES), disallow_negative_branch_length=True)

from skbio.tree import TreeNode
from io import StringIO

def read_ref(nwk):
    # skbio's Newick reader converts unquoted underscores to spaces; undo that.
    t = TreeNode.read(StringIO(nwk))
    for tip in t.tips():
        tip.name = tip.name.replace(' ', '_')
    return t

ref_trees = {k: read_ref(v) for k, v in REFS.items()}
for k, t in ref_trees.items():
    assert set(l.name for l in t.tips()) == set(NAMES), k
    assert len(splits(t, NAMES)) > 0, k

trees, rf_tbl, newicks = {}, {}, {}
for tag, m in methods.items():
    for dname, fn in (('raw1minusJ', d_raw), ('poisson', d_pois)):
        key = '%s__%s' % (tag, dname)
        t = tree_from(m, fn)
        trees[key] = t
        newicks[key] = str(t).strip()
        rf_tbl[key] = {}
        for rname, rt in ref_trees.items():
            d, s1, s2, shared = rf(t, rt, NAMES)
            rf_tbl[key][rname] = {'rf': d, 'splits_in_tree': s1, 'splits_in_ref': s2,
                                  'ref_splits_recovered': shared,
                                  'frac_ref_splits_recovered': shared / s2 if s2 else None}

# topology agreement between legacy- and corrected-derived trees
topo_agreement = {}
for N in (1000, 10000):
    for dname in ('raw1minusJ', 'poisson'):
        kl = 'legacy_direct_N%d__%s' % (N, dname)
        kc = 'corrected_union_N%d__%s' % (N, dname)
        ke = 'exact__%s' % dname
        d_lc, _, _, _ = rf(trees[kl], trees[kc], NAMES)
        d_le, _, _, _ = rf(trees[kl], trees[ke], NAMES)
        d_ce, _, _, _ = rf(trees[kc], trees[ke], NAMES)
        topo_agreement['N%d__%s' % (N, dname)] = {
            'rf_legacy_vs_corrected': d_lc,
            'rf_legacy_vs_exact': d_le,
            'rf_corrected_vs_exact': d_ce,
            'identical_topology': d_lc == 0,
        }

# clade recovery: do the named biological clades come out monophyletic?
CLADES = {
    'Escherichia_Shigella_8': ['Ecoli_K12_MG1655', 'Ecoli_O157H7_Sakai', 'Ecoli_CFT073',
                               'Ecoli_O111', 'Ecoli_UMN026', 'Ecoli_UTI89',
                               'Shigella_flexneri_2a', 'Shigella_sonnei'],
    'Salmonella_2': ['Salmonella_Typhimurium_LT2', 'Salmonella_Paratyphi'],
    'Klebsiella_Enterobacter_2': ['Klebsiella_pneumoniae', 'Enterobacter_cloacae'],
    'Enterobacteriaceae_14': [n for n in NAMES
                              if n not in ('Bacillus_subtilis_168', 'Pseudomonas_aeruginosa_PAO1')],
}
clade_rec = {}
for key, t in trees.items():
    ss = splits(t, NAMES)
    anchor = min(NAMES)
    clade_rec[key] = {}
    for cname, members in CLADES.items():
        s = frozenset(members)
        canon = frozenset(set(NAMES) - s) if anchor in s else s
        clade_rec[key][cname] = canon in ss

# ------------------------------------------------------------------ cosine check
cos = {}
for N in (1000, 10000):
    legacy = np.array([cs.cosine_similarity_manual(
        {k: v[1] for k, v in SK[N][a].items()},
        {k: v[1] for k, v in SK[N][b].items()}) for a, b in PAIRS])
    corrected = np.array([cs.cosine_union(SK[N][a], SK[N][b]) for a, b in PAIRS])
    stat, p = wilcoxon(np.abs(legacy - corrected), alternative='two-sided')
    cos['N%d' % N] = {
        'n_pairs': len(PAIRS),
        'mean_legacy': float(np.mean(legacy)), 'mean_corrected': float(np.mean(corrected)),
        'mean_signed_diff_corrected_minus_legacy': float(np.mean(corrected - legacy)),
        'max_abs_diff': float(np.max(np.abs(corrected - legacy))),
        'pearson_r_between_estimators': float(pearsonr(legacy, corrected)[0]),
        'spearman_rho_between_estimators': float(spearmanr(legacy, corrected)[0]),
        'note': 'No exact-cosine ground truth is computed here; F1 measured that. '
                'This records how much the shipped output moves on the E4 genome set.',
    }

out = {
    'experiment_id': 'F4_E4_recheck',
    'seed': SEED, 'k': K, 'n_genomes': NT, 'n_pairs': len(PAIRS),
    'genomes': NAMES,
    'genome_distinct_kmer_counts': {n: int(codes[n].size) for n in NAMES},
    'support_size_ratio': {
        'min': float(min(SUPPORT_RATIO.values())),
        'max': float(max(SUPPORT_RATIO.values())),
        'median': float(np.median(list(SUPPORT_RATIO.values()))),
    },
    'provenance': {
        'estimators_imported_from': os.path.join(REPO, 'calculate_similarity.py'),
        'sketches_reused_from': E4,
        'exact_jaccard_recomputed_this_session': True,
        'max_dev_recomputed_exact_vs_e4_cache': float(max_dev),
        'max_dev_shipped_estimator_vs_e4_recorded': xcheck,
        'reference_topologies_are_inputs_copied_from_E4': True,
    },
    'raw_jaccard': {tag: {'%s|%s' % p: v for p, v in m.items()} for tag, m in methods.items()},
    'calibration_vs_exact': calib,
    'paired_abs_error_legacy_vs_corrected': paired,
    'rf_to_reference_topologies': rf_tbl,
    'topology_agreement': topo_agreement,
    'clade_recovery': clade_rec,
    'cosine_shift_on_e4_genomes': cos,
    'inferred_trees_newick': newicks,
}
path = os.path.join(RESULTS, 'f4_e4_recheck.json')
json.dump(out, open(path, 'w'), indent=1)
print('\nWrote', path)

print('\n--- calibration vs exact ---')
for tag, c in calib.items():
    print('  %-24s RMSE %.6f  bias %+.6f  maxerr %.6f  r %.6f' % (
        tag, c['rmse_vs_exact'], c['mean_bias_vs_exact'],
        c['max_abs_err_vs_exact'], c['pearson_r_vs_exact']))
print('\n--- paired |error| legacy vs corrected ---')
for sname, byN in paired.items():
    for k, v in byN.items():
        print('  %-28s %-7s n=%3d  legacy %.6f  corrected %.6f  RMSE %.6f->%.6f  '
              'better %d/%d  p=%.3g %s' % (
                  sname, k, v['n'], v['mean_abs_err_legacy'], v['mean_abs_err_corrected'],
                  v['rmse_legacy'], v['rmse_corrected'], v['n_corrected_better'], v['n'],
                  v['wilcoxon_p'], '(SIG)' if v['significant_at_0.05'] else '(n.s.)'))
print('\n--- RF to reference topologies ---')
for k in sorted(rf_tbl):
    print('  %-34s NCBI rf=%2d  PHYLO rf=%2d' % (
        k, rf_tbl[k]['REF_NCBI_shigella_separate']['rf'],
        rf_tbl[k]['REF_PHYLO_shigella_nested']['rf']))
print('\n--- legacy vs corrected topology ---')
for k, v in topo_agreement.items():
    print('  %-26s rf(legacy,corrected)=%d  rf(leg,exact)=%d  rf(cor,exact)=%d' % (
        k, v['rf_legacy_vs_corrected'], v['rf_legacy_vs_exact'], v['rf_corrected_vs_exact']))
print('\n--- clade recovery ---')
for k in sorted(clade_rec):
    print('  %-34s %s' % (k, {c: ('yes' if b else 'no') for c, b in clade_rec[k].items()}))
print('\n--- cosine shift ---')
for k, v in cos.items():
    print('  %s  mean legacy %.4f -> corrected %.4f (mean diff %+.4f, max |diff| %.4f, rho %.4f)' % (
        k, v['mean_legacy'], v['mean_corrected'],
        v['mean_signed_diff_corrected_minus_legacy'], v['max_abs_diff'],
        v['spearman_rho_between_estimators']))
