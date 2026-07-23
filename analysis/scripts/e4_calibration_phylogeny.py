#!/usr/bin/env python3
"""
Experiment E4 - Distance calibration and phylogenetic accuracy.

(1) Do fgr2 distances recover known phylogeny?
(2) Does the Poisson (Mash) correction improve on the raw 1-J the pipeline uses?

All numbers written to analysis/results/e4_calibration_phylogeny.json
"""
import sys, os, json, math, random, itertools, subprocess
sys.path.insert(0, '/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/scripts')
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skbio import DistanceMatrix
from skbio.tree import nj, TreeNode
from io import StringIO

import fgrlib
sys.path.insert(0, '/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5')
from calculate_similarity import resample_kmer_universe

SCRATCH = '/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint--claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad'
E4 = os.path.join(SCRATCH, 'e4')
RESULTS = '/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/results'
FIGS = '/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/figures'
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
K = 31
SEED = 20260721
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- reference topologies
# Non-trivial splits only; E. coli intra-species relationships are left as an
# unresolved polytomy in BOTH references because there is no single accepted
# intra-species topology for these six strains.
#
# REF_NCBI: Shigella treated as a genus-level clade sister to Escherichia
#           (the NCBI *nomenclatural* arrangement).
# REF_PHYLO: Shigella nested inside Escherichia coli, i.e. Escherichia+Shigella
#            form one unresolved clade (the biologically accepted arrangement).
REF_NCBI = ("(Bacillus_subtilis_168,Pseudomonas_aeruginosa_PAO1,(Yersinia_pestis_CO92,"
            "((Klebsiella_pneumoniae,Enterobacter_cloacae),(Citrobacter_rodentium,"
            "((Salmonella_Typhimurium_LT2,Salmonella_Paratyphi),"
            "((Ecoli_K12_MG1655,Ecoli_O157H7_Sakai,Ecoli_CFT073,Ecoli_O111,"
            "Ecoli_UMN026,Ecoli_UTI89),(Shigella_flexneri_2a,Shigella_sonnei)))))));")
REF_PHYLO = ("(Bacillus_subtilis_168,Pseudomonas_aeruginosa_PAO1,(Yersinia_pestis_CO92,"
             "((Klebsiella_pneumoniae,Enterobacter_cloacae),(Citrobacter_rodentium,"
             "((Salmonella_Typhimurium_LT2,Salmonella_Paratyphi),"
             "(Ecoli_K12_MG1655,Ecoli_O157H7_Sakai,Ecoli_CFT073,Ecoli_O111,"
             "Ecoli_UMN026,Ecoli_UTI89,Shigella_flexneri_2a,Shigella_sonnei))))));")

# ---------------------------------------------------------------- split utilities
def splits(tree, taxa):
    """Non-trivial unrooted bipartitions, canonicalised as the frozenset of the
    side NOT containing the lexicographically first taxon."""
    taxa = set(taxa); n = len(taxa); anchor = min(taxa)
    out = set()
    for node in tree.postorder(include_self=False):
        leaves = frozenset(l.name for l in node.tips()) if not node.is_tip() else frozenset([node.name])
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

# ---------------------------------------------------------------- load data
exact = json.load(open(os.path.join(E4, 'exact_jaccard_k31.json')))
NAMES = exact['names']
NT = len(NAMES)
EX_J = {}
for key, v in exact['exact_jaccard'].items():
    a, b = key.split('|'); EX_J[(a, b)] = v; EX_J[(b, a)] = v
for n in NAMES:
    EX_J[(n, n)] = 1.0

def load_fgr2(N):
    d = {}
    for n in NAMES:
        h, cov, meta = fgrlib.parse_sketch(os.path.join(E4, f'fgr2_{N}', n + '.fgr2'))
        d[n] = h
    return d

FGR = {N: load_fgr2(N) for N in (1000, 10000)}
for N in (1000, 10000):
    assert all(len(v) == N for v in FGR[N].values()), "sketch size mismatch"

def load_mash(N):
    j, dist = {}, {}
    for line in open(os.path.join(E4, f'dist_s{N}.tsv') if os.path.exists(os.path.join(E4, f'dist_s{N}.tsv'))
                     else os.path.join(E4, 'mash', f'dist_s{N}.tsv')):
        f = line.split('\t')
        a = os.path.splitext(os.path.basename(f[0]))[0]
        b = os.path.splitext(os.path.basename(f[1]))[0]
        num, den = f[4].strip().split('/')
        j[(a, b)] = int(num) / int(den)
        dist[(a, b)] = float(f[2])
    return j, dist

MASH_J, MASH_D = {}, {}
for N in (1000, 10000):
    MASH_J[N], MASH_D[N] = load_mash(N)

PAIRS = list(itertools.combinations(NAMES, 2))

# ---------------------------------------------------------------- estimators
def fgr_j_direct(N, a, b):
    return fgrlib.jaccard_direct(set(FGR[N][a]), set(FGR[N][b]))

def fgr_j_union(N, a, b):
    return fgrlib.jaccard_union(FGR[N][a], FGR[N][b], N)

# ---------------------------------------------------------------- 1. calibration
methods = {}
methods['exact'] = {p: EX_J[p] for p in PAIRS}
for N in (1000, 10000):
    methods[f'fgr2_direct_N{N}'] = {p: fgr_j_direct(N, *p) for p in PAIRS}
    methods[f'fgr2_union_N{N}'] = {p: fgr_j_union(N, *p) for p in PAIRS}
    methods[f'mash_s{N}'] = {p: MASH_J[N][p] for p in PAIRS}

def d_raw(j):   return 1.0 - j
def d_pois(j):  return fgrlib.mash_distance(j, K)

calibration = {}
ex_raw = np.array([d_raw(EX_J[p]) for p in PAIRS])
ex_pois = np.array([d_pois(EX_J[p]) for p in PAIRS])
for m, jd in methods.items():
    if m == 'exact':
        continue
    est_j = np.array([jd[p] for p in PAIRS])
    exj = np.array([EX_J[p] for p in PAIRS])
    est_raw = 1.0 - est_j
    est_pois = np.array([d_pois(x) for x in est_j])
    entry = {}
    entry['jaccard_pearson_r_vs_exact'] = float(pearsonr(est_j, exj)[0])
    entry['jaccard_spearman_rho_vs_exact'] = float(spearmanr(est_j, exj)[0])
    entry['jaccard_rmse_vs_exact'] = float(np.sqrt(np.mean((est_j - exj) ** 2)))
    entry['jaccard_mean_bias_vs_exact'] = float(np.mean(est_j - exj))
    # restrict to informative pairs (exact J > 0) -- saturated pairs are all 0
    inf = exj > 1e-9
    entry['n_informative_pairs_exactJ_gt0'] = int(inf.sum())
    entry['jaccard_rmse_informative'] = float(np.sqrt(np.mean((est_j[inf] - exj[inf]) ** 2)))
    entry['jaccard_mean_bias_informative'] = float(np.mean(est_j[inf] - exj[inf]))
    entry['jaccard_mean_rel_bias_informative'] = float(np.mean((est_j[inf] - exj[inf]) / exj[inf]))
    entry['dist_raw_pearson_r_vs_exact_raw'] = float(pearsonr(est_raw, ex_raw)[0])
    entry['dist_pois_pearson_r_vs_exact_pois'] = float(pearsonr(est_pois, ex_pois)[0])
    entry['dist_raw_rmse_vs_exact_raw'] = float(np.sqrt(np.mean((est_raw - ex_raw) ** 2)))
    entry['dist_pois_rmse_vs_exact_pois'] = float(np.sqrt(np.mean((est_pois - ex_pois) ** 2)))
    calibration[m] = entry

# fgr2 vs mash head-to-head at matched sketch size
fgr_vs_mash = {}
for N in (1000, 10000):
    for est in ('direct', 'union'):
        a = np.array([methods[f'fgr2_{est}_N{N}'][p] for p in PAIRS])
        b = np.array([methods[f'mash_s{N}'][p] for p in PAIRS])
        inf = np.array([EX_J[p] for p in PAIRS]) > 1e-9
        fgr_vs_mash[f'fgr2_{est}_N{N}_vs_mash_s{N}'] = {
            'jaccard_pearson_r': float(pearsonr(a, b)[0]),
            'jaccard_rmse': float(np.sqrt(np.mean((a - b) ** 2))),
            'jaccard_mean_diff_fgr2_minus_mash': float(np.mean(a - b)),
            'jaccard_mean_diff_informative': float(np.mean(a[inf] - b[inf])),
            'jaccard_max_abs_diff': float(np.max(np.abs(a - b))),
        }
# fgr2 Poisson distance vs mash's own reported distance
for N in (1000, 10000):
    fp = np.array([d_pois(methods[f'fgr2_union_N{N}'][p]) for p in PAIRS])
    md = np.array([MASH_D[N][p] for p in PAIRS])
    fgr_vs_mash[f'fgr2_union_poissonD_N{N}_vs_mash_reportedD_s{N}'] = {
        'pearson_r': float(pearsonr(fp, md)[0]),
        'rmse': float(np.sqrt(np.mean((fp - md) ** 2))),
        'mean_diff': float(np.mean(fp - md)),
        'max_abs_diff': float(np.max(np.abs(fp - md))),
    }

# ---- saturation diagnostics: the Poisson transform is only defined away from J->0
sat = {}
exj_all = np.array([EX_J[p] for p in PAIRS])
sat['exact_jaccard_min'] = float(exj_all.min())
sat['exact_jaccard_median'] = float(np.median(exj_all))
sat['n_pairs_exactJ_lt_1e-4'] = int((exj_all < 1e-4).sum())
sat['n_pairs_exactJ_eq_0'] = int((exj_all == 0).sum())
for N in (1000, 10000):
    mj = np.array([MASH_J[N][p] for p in PAIRS])
    fj = np.array([methods[f'fgr2_union_N{N}'][p] for p in PAIRS])
    sat[f'n_pairs_mash_s{N}_zero_shared'] = int((mj == 0).sum())
    sat[f'n_pairs_fgr2_union_N{N}_zero_shared'] = int((fj == 0).sum())

# stratified fgr2-vs-mash: unsaturated pairs only (>=5 shared hashes in BOTH sketches)
fgr_vs_mash_unsat = {}
for N in (1000, 10000):
    fj = np.array([methods[f'fgr2_union_N{N}'][p] for p in PAIRS])
    mj = np.array([MASH_J[N][p] for p in PAIRS])
    keep = (fj >= 5.0 / N) & (mj >= 5.0 / N)
    fp = np.array([d_pois(x) for x in fj])[keep]
    md = np.array([MASH_D[N][p] for p in PAIRS])[keep]
    fgr_vs_mash_unsat[f'N{N}'] = {
        'criterion': 'both estimators report >=5/N shared hashes',
        'n_pairs_kept': int(keep.sum()), 'n_pairs_total': len(PAIRS),
        'jaccard_pearson_r': float(pearsonr(fj[keep], mj[keep])[0]),
        'jaccard_rmse': float(np.sqrt(np.mean((fj[keep] - mj[keep]) ** 2))),
        'poissonD_pearson_r': float(pearsonr(fp, md)[0]),
        'poissonD_rmse': float(np.sqrt(np.mean((fp - md) ** 2))),
        'poissonD_mean_diff': float(np.mean(fp - md)),
        'poissonD_max_abs_diff': float(np.max(np.abs(fp - md))),
    }

# ---------------------------------------------------------------- 2. trees
def matrix(jd, transform):
    M = np.zeros((NT, NT))
    for i, a in enumerate(NAMES):
        for j2, b in enumerate(NAMES):
            if i == j2:
                continue
            key = (a, b) if (a, b) in jd else (b, a)
            M[i, j2] = transform(jd[key])
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    return DistanceMatrix(M, NAMES)

TREE_SETS = {}
for m in ['exact', 'fgr2_direct_N1000', 'fgr2_direct_N10000',
          'fgr2_union_N1000', 'fgr2_union_N10000', 'mash_s1000', 'mash_s10000']:
    for tag, tr in (('raw1minusJ', d_raw), ('poisson', d_pois)):
        TREE_SETS[f'{m}__{tag}'] = nj(matrix(methods[m], tr))

def read_ref(nwk):
    # skbio's Newick reader converts unquoted underscores to spaces; undo that.
    t = TreeNode.read(StringIO(nwk))
    for tip in t.tips():
        tip.name = tip.name.replace(' ', '_')
    return t

ref_ncbi = read_ref(REF_NCBI)
ref_phylo = read_ref(REF_PHYLO)
taxa = set(NAMES)
assert set(l.name for l in ref_ncbi.tips()) == taxa
assert set(l.name for l in ref_phylo.tips()) == taxa

rf_table = {}
for name, t in TREE_SETS.items():
    row = {}
    for rname, rt in (('REF_NCBI_shigella_separate', ref_ncbi),
                      ('REF_PHYLO_shigella_nested', ref_phylo)):
        d, n1, n2, shared = rf(t, rt, taxa)
        row[rname] = {'rf': d, 'splits_in_tree': n1, 'splits_in_ref': n2,
                      'ref_splits_recovered': shared,
                      'frac_ref_splits_recovered': shared / n2}
    rf_table[name] = row

# topology identity: raw vs poisson for each method
topo_identity = {}
for m in ['exact', 'fgr2_direct_N1000', 'fgr2_direct_N10000',
          'fgr2_union_N1000', 'fgr2_union_N10000', 'mash_s1000', 'mash_s10000']:
    a = TREE_SETS[f'{m}__raw1minusJ']; b = TREE_SETS[f'{m}__poisson']
    d, n1, n2, shared = rf(a, b, taxa)
    la = np.array([nd.length for nd in a.postorder(include_self=False) if nd.length is not None])
    lb = np.array([nd.length for nd in b.postorder(include_self=False) if nd.length is not None])
    topo_identity[m] = {
        'rf_raw_vs_poisson': d, 'identical_topology': d == 0,
        'total_tree_length_raw': float(la.sum()), 'total_tree_length_poisson': float(lb.sum()),
        'branch_length_ratio_poisson_over_raw': float(lb.sum() / la.sum()),
        'n_negative_branches_raw': int((la < 0).sum()), 'n_negative_branches_poisson': int((lb < 0).sum()),
    }

# is Shigella nested inside E. coli in the inferred trees?
ECOLI = [n for n in NAMES if n.startswith('Ecoli_')]
SHIG = [n for n in NAMES if n.startswith('Shigella_')]
nesting = {}
for name, t in TREE_SETS.items():
    S = splits(t, taxa)
    ecoli_only = frozenset(ECOLI)
    shig_only = frozenset(SHIG)
    anchor = min(taxa)
    def canon(s):
        s = frozenset(s); c = frozenset(taxa - s)
        return c if anchor in s else s
    nesting[name] = {
        'ecoli6_monophyletic_excluding_shigella': canon(ecoli_only) in S,
        'shigella_monophyletic': canon(shig_only) in S,
        'ecoli_plus_shigella_clade': canon(ecoli_only | shig_only) in S,
    }

# explicit accounting: WHICH reference splits are recovered / missed
split_accounting = {}
for rname, rt in (('REF_NCBI_shigella_separate', ref_ncbi), ('REF_PHYLO_shigella_nested', ref_phylo)):
    S_ref = splits(rt, taxa)
    d = {}
    for name in ['fgr2_direct_N10000__raw1minusJ', 'fgr2_direct_N10000__poisson',
                 'exact__raw1minusJ', 'exact__poisson']:
        S = splits(TREE_SETS[name], taxa)
        d[name] = {'recovered': [sorted(s) for s in S_ref & S],
                   'missed': [sorted(s) for s in S_ref - S]}
    split_accounting[rname] = {'reference_splits': [sorted(s) for s in S_ref], 'per_tree': d}

# ---- FAIR TEST: Enterobacteriaceae-only 14 taxa (no saturated pairs).
# The Poisson correction is derived for the unsaturated regime, so this is the
# comparison on which it should be judged.
ENTERO14 = [n for n in NAMES if n not in ('Bacillus_subtilis_168', 'Pseudomonas_aeruginosa_PAO1')]
taxa14 = set(ENTERO14)
ref_ncbi14 = read_ref(REF_NCBI).shear(taxa14); ref_ncbi14.prune()
ref_phylo14 = read_ref(REF_PHYLO).shear(taxa14); ref_phylo14.prune()

def matrix_sub(jd, transform, subset):
    n = len(subset)
    M = np.zeros((n, n))
    for i, a in enumerate(subset):
        for j2, b in enumerate(subset):
            if i == j2: continue
            key = (a, b) if (a, b) in jd else (b, a)
            M[i, j2] = transform(jd[key])
    M = (M + M.T) / 2.0; np.fill_diagonal(M, 0.0)
    return DistanceMatrix(M, subset)

entero_trees, entero_rf = {}, {}
for m in ['exact', 'fgr2_direct_N1000', 'fgr2_direct_N10000',
          'fgr2_union_N1000', 'fgr2_union_N10000', 'mash_s1000', 'mash_s10000']:
    for tag, tr in (('raw1minusJ', d_raw), ('poisson', d_pois)):
        t = nj(matrix_sub(methods[m], tr, ENTERO14))
        entero_trees[f'{m}__{tag}'] = t
        row = {}
        for rname, rt in (('REF_NCBI_shigella_separate', ref_ncbi14),
                          ('REF_PHYLO_shigella_nested', ref_phylo14)):
            dd, n1, n2, shared = rf(t, rt, taxa14)
            row[rname] = {'rf': dd, 'splits_in_tree': n1, 'splits_in_ref': n2,
                          'ref_splits_recovered': shared, 'frac_ref_splits_recovered': shared / n2}
        entero_rf[f'{m}__{tag}'] = row

entero_topo_identity = {}
for m in ['exact', 'fgr2_direct_N1000', 'fgr2_direct_N10000',
          'fgr2_union_N1000', 'fgr2_union_N10000', 'mash_s1000', 'mash_s10000']:
    a = entero_trees[f'{m}__raw1minusJ']; b = entero_trees[f'{m}__poisson']
    dd, n1, n2, shared = rf(a, b, taxa14)
    la = np.array([nd.length for nd in a.postorder(include_self=False) if nd.length is not None])
    lb = np.array([nd.length for nd in b.postorder(include_self=False) if nd.length is not None])
    entero_topo_identity[m] = {
        'rf_raw_vs_poisson': dd, 'identical_topology': dd == 0,
        'total_tree_length_raw': float(la.sum()), 'total_tree_length_poisson': float(lb.sum()),
        'branch_length_ratio_poisson_over_raw': float(lb.sum() / la.sum()),
        'n_negative_branches_raw': int((la < 0).sum()),
        'n_negative_branches_poisson': int((lb < 0).sum())}

# ---------------------------------------------------------------- 3. additivity / four-point
def fourpoint(jd, transform, label):
    D = np.zeros((NT, NT))
    for i, a in enumerate(NAMES):
        for j2, b in enumerate(NAMES):
            if i == j2: continue
            key = (a, b) if (a, b) in jd else (b, a)
            D[i, j2] = transform(jd[key])
    D = (D + D.T) / 2.0
    rel, absv = [], []
    for q in itertools.combinations(range(NT), 4):
        i, j2, k2, l = q
        s = sorted([D[i, j2] + D[k2, l], D[i, k2] + D[j2, l], D[i, l] + D[j2, k2]])
        absv.append(s[2] - s[1])
        rel.append((s[2] - s[1]) / s[2] if s[2] > 0 else 0.0)
    rel = np.array(rel); absv = np.array(absv)
    # triangle inequality
    tri = 0; ntri = 0
    for i, j2, k2 in itertools.combinations(range(NT), 3):
        for (x, y, z) in ((i, j2, k2), (j2, k2, i), (k2, i, j2)):
            ntri += 1
            if D[x, y] > D[x, z] + D[z, y] + 1e-12: tri += 1
    return {
        'label': label, 'n_quartets': len(rel),
        'mean_relative_violation': float(rel.mean()),
        'median_relative_violation': float(np.median(rel)),
        'p90_relative_violation': float(np.percentile(rel, 90)),
        'max_relative_violation': float(rel.max()),
        'frac_quartets_rel_violation_gt_0.01': float((rel > 0.01).mean()),
        'frac_quartets_rel_violation_gt_0.05': float((rel > 0.05).mean()),
        'mean_absolute_violation': float(absv.mean()),
        'max_absolute_violation': float(absv.max()),
        'mean_pairwise_distance': float(D[np.triu_indices(NT, 1)].mean()),
        'mean_abs_violation_over_mean_distance': float(absv.mean() / D[np.triu_indices(NT, 1)].mean()),
        'n_triangle_inequality_violations': tri, 'n_triangle_tests': ntri,
        '_rel': rel.tolist(),
    }

additivity = {}
for m in ['exact', 'fgr2_direct_N10000', 'fgr2_union_N10000', 'mash_s10000',
          'fgr2_direct_N1000', 'fgr2_union_N1000', 'mash_s1000']:
    for tag, tr in (('raw1minusJ', d_raw), ('poisson', d_pois)):
        additivity[f'{m}__{tag}'] = fourpoint(methods[m], tr, f'{m} {tag}')

# additivity restricted to Enterobacteriaceae (no saturation)
ENTERO = [n for n in NAMES if n not in ('Bacillus_subtilis_168', 'Pseudomonas_aeruginosa_PAO1')]
def fourpoint_subset(jd, transform, subset):
    idx = subset
    D = {}
    for a in idx:
        for b in idx:
            if a == b: D[(a, b)] = 0.0; continue
            key = (a, b) if (a, b) in jd else (b, a)
            D[(a, b)] = transform(jd[key])
    rel, absv = [], []
    for q in itertools.combinations(idx, 4):
        i, j2, k2, l = q
        s = sorted([D[(i, j2)] + D[(k2, l)], D[(i, k2)] + D[(j2, l)], D[(i, l)] + D[(j2, k2)]])
        rel.append((s[2] - s[1]) / s[2] if s[2] > 0 else 0.0)
        absv.append(s[2] - s[1])
    rel = np.array(rel); absv = np.array(absv)
    pw = np.array([D[(a, b)] for a, b in itertools.combinations(idx, 2)])
    # SCALE-FREE comparator: four-point violation is homogeneous of degree 1 in D,
    # so dividing by the mean pairwise distance makes transforms with different
    # dynamic ranges directly comparable. This removes the compression confound
    # that the relative (s3-s2)/s3 statistic does not fully control for.
    return {'n_quartets': len(rel), 'mean_relative_violation': float(rel.mean()),
            'median_relative_violation': float(np.median(rel)),
            'max_relative_violation': float(rel.max()),
            'frac_gt_0.01': float((rel > 0.01).mean()),
            'mean_absolute_violation': float(absv.mean()),
            'mean_pairwise_distance': float(pw.mean()),
            'scalefree_mean_abs_violation_over_mean_distance': float(absv.mean() / pw.mean()),
            'scalefree_median_abs_violation_over_mean_distance': float(np.median(absv) / pw.mean())}

additivity_entero = {}
for m in ['exact', 'fgr2_direct_N10000', 'mash_s10000']:
    for tag, tr in (('raw1minusJ', d_raw), ('poisson', d_pois)):
        additivity_entero[f'{m}__{tag}'] = fourpoint_subset(methods[m], tr, ENTERO)

# CONTROL for the compression confound. 1-J is bounded above by 1 and for
# cross-genus pairs sits at ~0.99, so quartet sums are all ~2 and RELATIVE
# four-point violations are small almost by construction. The 8 Escherichia/
# Shigella genomes have J in a wide, unsaturated range, so 1-J is NOT compressed
# there; this is the honest test of which transform is more additive.
ECOLI8 = [n for n in NAMES if n.startswith('Ecoli_') or n.startswith('Shigella_')]
additivity_ecoli8 = {}
for m in ['exact', 'fgr2_direct_N10000', 'fgr2_union_N10000', 'mash_s10000']:
    for tag, tr in (('raw1minusJ', d_raw), ('poisson', d_pois)):
        r = fourpoint_subset(methods[m], tr, ECOLI8)
        vals = [tr(methods[m][(a, b)] if (a, b) in methods[m] else methods[m][(b, a)])
                for a, b in itertools.combinations(ECOLI8, 2)]
        r['distance_min'] = float(min(vals)); r['distance_max'] = float(max(vals))
        r['distance_dynamic_range_max_over_min'] = float(max(vals) / min(vals))
        additivity_ecoli8[f'{m}__{tag}'] = r
# and the same compression diagnostic for the full 16-taxon set
compression = {}
for m in ['exact', 'fgr2_direct_N10000']:
    for tag, tr in (('raw1minusJ', d_raw), ('poisson', d_pois)):
        vals = np.array([tr(methods[m][p]) for p in PAIRS])
        compression[f'{m}__{tag}'] = {
            'min': float(vals.min()), 'max': float(vals.max()),
            'median': float(np.median(vals)),
            'coefficient_of_variation': float(vals.std() / vals.mean()),
            'frac_pairs_within_1pct_of_max': float((vals > 0.99 * vals.max()).mean())}

# ---------------------------------------------------------------- 4. resampling support
NREP = 100
BOOT_N = 10000
support = {}
for tag, tr in (('raw1minusJ', d_raw), ('poisson', d_pois)):
    random.seed(SEED)
    sets = {n: set(FGR[BOOT_N][n]) for n in NAMES}
    universe = sorted(set().union(*sets.values()))
    counts = {}
    retained = []
    ok = 0
    for r in range(NREP):
        u = resample_kmer_universe(universe)
        retained.append(len(u) / len(universe))
        sub = {n: sets[n] & u for n in NAMES}
        jd = {}
        for p in PAIRS:
            jd[p] = fgrlib.jaccard_direct(sub[p[0]], sub[p[1]])
        try:
            t = nj(matrix(jd, tr))
        except Exception:
            continue
        ok += 1
        for s in splits(t, taxa):
            counts[s] = counts.get(s, 0) + 1
    obs = splits(TREE_SETS[f'fgr2_direct_N{BOOT_N}__{tag}'], taxa)
    support[tag] = {
        'n_replicates_requested': NREP, 'n_replicates_succeeded': ok,
        'mean_universe_retained_fraction': float(np.mean(retained)),
        'universe_size': len(universe),
        'observed_tree_clade_support': sorted(
            [{'clade': sorted(s), 'size': len(s), 'support_pct': 100.0 * counts.get(s, 0) / ok}
             for s in obs], key=lambda x: -x['support_pct']),
        'all_clades_seen_ge_50pct': sorted(
            [{'clade': sorted(s), 'support_pct': 100.0 * c / ok}
             for s, c in counts.items() if 100.0 * c / ok >= 50.0], key=lambda x: -x['support_pct']),
    }

# ---------------------------------------------------------------- figures
# Fig 1: calibration scatter
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
exj = np.array([EX_J[p] for p in PAIRS])
panels = [('fgr2_direct_N1000', 'fgr2 direct, N=1000'), ('fgr2_union_N1000', 'fgr2 union, N=1000'),
          ('mash_s1000', 'mash, s=1000'), ('fgr2_direct_N10000', 'fgr2 direct, N=10000'),
          ('fgr2_union_N10000', 'fgr2 union, N=10000'), ('mash_s10000', 'mash, s=10000')]
for ax, (m, title) in zip(axes.ravel(), panels):
    est = np.array([methods[m][p] for p in PAIRS])
    ax.plot([0, 1], [0, 1], 'k--', lw=1, zorder=1)
    ax.scatter(exj, est, s=18, alpha=0.75, c='#2b6cb0', zorder=2, edgecolors='none')
    r = pearsonr(est, exj)[0]
    rm = np.sqrt(np.mean((est - exj) ** 2))
    ax.set_title(f'{title}\nr={r:.5f}  RMSE={rm:.4f}', fontsize=10)
    ax.set_xlabel('exact Jaccard (full k-mer sets, k=31)')
    ax.set_ylabel('estimated Jaccard')
    ax.set_xscale('symlog', linthresh=1e-3); ax.set_yscale('symlog', linthresh=1e-3)
fig.suptitle('E4: sketch Jaccard calibration vs exact, 120 genome pairs (k=31)', fontsize=13)
fig.tight_layout()
f1 = os.path.join(FIGS, 'e4_calibration_scatter.png'); fig.savefig(f1, dpi=200); plt.close(fig)

# Fig 1b: fgr2 vs mash distances
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, N in zip(axes, (1000, 10000)):
    fp = np.array([d_pois(methods[f'fgr2_union_N{N}'][p]) for p in PAIRS])
    md = np.array([MASH_D[N][p] for p in PAIRS])
    ax.plot([0, max(md.max(), fp.max())], [0, max(md.max(), fp.max())], 'k--', lw=1)
    ax.scatter(md, fp, s=20, alpha=0.75, c='#c05621', edgecolors='none')
    ax.set_xlabel(f'mash distance (s={N})'); ax.set_ylabel(f'fgr2 union + Poisson (N={N})')
    ax.set_title(f'r={pearsonr(fp, md)[0]:.6f}  RMSE={np.sqrt(np.mean((fp-md)**2)):.5f}')
fig.suptitle('E4: fgr2 Poisson-corrected distance vs mash 2.3 reported distance')
fig.tight_layout()
f1b = os.path.join(FIGS, 'e4_fgr2_vs_mash.png'); fig.savefig(f1b, dpi=200); plt.close(fig)

# Fig 2: trees side by side
def draw(tree, ax, title):
    tips = list(tree.tips())
    ypos = {}
    for i, t in enumerate(tips):
        ypos[id(t)] = i
    for nd in tree.postorder(include_self=True):
        if not nd.is_tip():
            ypos[id(nd)] = np.mean([ypos[id(c)] for c in nd.children])
    xpos = {}
    def setx(nd, x):
        xpos[id(nd)] = x
        for c in nd.children:
            setx(c, x + max(c.length or 0.0, 0.0))
    setx(tree, 0.0)
    for nd in tree.postorder(include_self=True):
        if nd.children:
            ys = [ypos[id(c)] for c in nd.children]
            ax.plot([xpos[id(nd)]] * 2, [min(ys), max(ys)], c='#333', lw=1.1)
            for c in nd.children:
                ax.plot([xpos[id(nd)], xpos[id(c)]], [ypos[id(c)]] * 2, c='#333', lw=1.1)
    for t in tips:
        col = '#c53030' if t.name.startswith('Shigella') else ('#2b6cb0' if t.name.startswith('Ecoli') else '#333')
        ax.text(xpos[id(t)] + 0.005 * max(xpos.values()), ypos[id(t)], ' ' + t.name,
                va='center', fontsize=8, color=col)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.02 * max(xpos.values()), max(xpos.values()) * 1.75)
    ax.set_yticks([]); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False); ax.set_xlabel('distance')

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
draw(TREE_SETS['fgr2_direct_N10000__raw1minusJ'], axes[0],
     'NJ from raw 1 - J   (fgr2 direct, N=10000)  [what the pipeline uses]')
draw(TREE_SETS['fgr2_direct_N10000__poisson'], axes[1],
     'NJ from Poisson-corrected mash distance   (fgr2 direct, N=10000)')
fig.suptitle('E4: NJ topologies, raw 1-J vs Poisson-corrected distance', fontsize=13)
fig.tight_layout()
f2 = os.path.join(FIGS, 'e4_nj_trees.png'); fig.savefig(f2, dpi=200); plt.close(fig)

# Fig 3: additivity
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
keys = ['exact__raw1minusJ', 'exact__poisson', 'fgr2_direct_N10000__raw1minusJ',
        'fgr2_direct_N10000__poisson', 'mash_s10000__raw1minusJ', 'mash_s10000__poisson']
data = [additivity[k]['_rel'] for k in keys]
labels = [k.replace('__', '\n') for k in keys]
bp = axes[0].boxplot(data, labels=labels, showfliers=False, patch_artist=True)
for i, b in enumerate(bp['boxes']):
    b.set_facecolor('#90cdf4' if 'raw' in keys[i] else '#f6ad55')
axes[0].set_ylabel('relative four-point violation  (s3-s2)/s3')
axes[0].set_title(f'All 16 taxa, {additivity[keys[0]]["n_quartets"]} quartets')
axes[0].tick_params(axis='x', labelsize=7, rotation=30)
names_e = list(additivity_entero.keys())
vals = [additivity_entero[k]['mean_relative_violation'] for k in names_e]
axes[1].barh(range(len(names_e)), vals,
             color=['#90cdf4' if 'raw' in k else '#f6ad55' for k in names_e])
axes[1].set_yticks(range(len(names_e)))
axes[1].set_yticklabels([k.replace('__', '\n') for k in names_e], fontsize=7)
axes[1].set_xlabel('mean relative four-point violation')
axes[1].set_title(f'Enterobacteriaceae only (14 taxa, '
                  f'{additivity_entero[names_e[0]]["n_quartets"]} quartets)')
names_c = list(additivity_ecoli8.keys())
vals_c = [additivity_ecoli8[k]['scalefree_mean_abs_violation_over_mean_distance'] for k in names_c]
axes[2].barh(range(len(names_c)), vals_c,
             color=['#90cdf4' if 'raw' in k else '#f6ad55' for k in names_c])
axes[2].set_yticks(range(len(names_c)))
axes[2].set_yticklabels([k.replace('__', '\n') for k in names_c], fontsize=7)
axes[2].set_xlabel('scale-free violation\n(mean |s3-s2| / mean pairwise distance)')
axes[2].set_title(f'UNSATURATED CONTROL: Escherichia+Shigella\n(8 taxa, '
                  f'{additivity_ecoli8[names_c[0]]["n_quartets"]} quartets); '
                  f'scale-free removes\nthe 1-J compression confound')
fig.suptitle('E4: additivity / four-point condition by distance transform  '
             '(blue = raw 1-J, orange = Poisson)', fontsize=13)
fig.tight_layout()
f3 = os.path.join(FIGS, 'e4_additivity.png'); fig.savefig(f3, dpi=200); plt.close(fig)

# ---------------------------------------------------------------- write JSON
for k in additivity:
    additivity[k].pop('_rel')

out = {
    'experiment_id': 'E4',
    'seed': SEED, 'k': K, 'n_genomes': NT, 'n_pairs': len(PAIRS),
    'genomes': NAMES,
    'genome_distinct_kmer_counts': exact['genome_distinct_kmer_counts'],
    'tools': {'fgr2': 'v2.1.0 -k 31 -c 1', 'mash': subprocess.run(
        ['/opt/homebrew/bin/mash', '--version'], capture_output=True, text=True).stdout.strip()},
    'reference_topologies_newick': {
        'REF_NCBI_shigella_separate': REF_NCBI,
        'REF_PHYLO_shigella_nested': REF_PHYLO,
        'note': ('E. coli intra-species relationships left as an unresolved polytomy in both '
                 'references; no single accepted intra-species topology exists for these strains. '
                 'REF_PHYLO merges Escherichia+Shigella into one unresolved 8-taxon clade.')},
    'raw_jaccard_estimates': {m: {f'{a}|{b}': v for (a, b), v in jd.items()}
                              for m, jd in methods.items()},
    'mash_reported_distance': {N: {f'{a}|{b}': MASH_D[N][(a, b)] for (a, b) in PAIRS}
                               for N in (1000, 10000)},
    'calibration_vs_exact': calibration,
    'fgr2_vs_mash': fgr_vs_mash,
    'fgr2_vs_mash_unsaturated_only': fgr_vs_mash_unsat,
    'saturation_diagnostics': sat,
    'rf_to_reference_topologies': rf_table,
    'reference_split_accounting': split_accounting,
    'enterobacteriaceae14_rf': entero_rf,
    'enterobacteriaceae14_raw_vs_poisson': entero_topo_identity,
    'enterobacteriaceae14_trees_newick': {k: str(v).strip() for k, v in entero_trees.items()},
    'enterobacteriaceae14_taxa': ENTERO14,
    'raw_vs_poisson_topology_and_branchlengths': topo_identity,
    'clade_recovery': nesting,
    'additivity_fourpoint_all16': additivity,
    'additivity_fourpoint_enterobacteriaceae14': additivity_entero,
    'additivity_fourpoint_escherichia_shigella8_UNSATURATED_CONTROL': additivity_ecoli8,
    'escherichia_shigella8_taxa': ECOLI8,
    'distance_compression_diagnostics': compression,
    'resampling_support_fgr2_direct_N10000': support,
    'inferred_trees_newick': {k: str(v).strip() for k, v in TREE_SETS.items()},
    'figures': [f1, f1b, f2, f3],
}
p = os.path.join(RESULTS, 'e4_calibration_phylogeny.json')
json.dump(out, open(p, 'w'), indent=1)
print('wrote', p)

# console summary
print('\n--- calibration (Jaccard vs exact) ---')
for m, e in calibration.items():
    print(f"{m:24s} r={e['jaccard_pearson_r_vs_exact']:.5f} RMSE={e['jaccard_rmse_vs_exact']:.5f} "
          f"bias_inf={e['jaccard_mean_bias_informative']:+.5f} relbias_inf={e['jaccard_mean_rel_bias_informative']:+.4f}")
print('\n--- fgr2 vs mash ---')
for m, e in fgr_vs_mash.items():
    print(m, {k: round(v, 6) for k, v in e.items()})
print('\n--- RF ---')
for m, e in rf_table.items():
    print(f"{m:38s} NCBI rf={e['REF_NCBI_shigella_separate']['rf']:2d} "
          f"({e['REF_NCBI_shigella_separate']['ref_splits_recovered']}/{e['REF_NCBI_shigella_separate']['splits_in_ref']}) | "
          f"PHYLO rf={e['REF_PHYLO_shigella_nested']['rf']:2d} "
          f"({e['REF_PHYLO_shigella_nested']['ref_splits_recovered']}/{e['REF_PHYLO_shigella_nested']['splits_in_ref']})")
print('\n--- raw vs poisson ---')
for m, e in topo_identity.items():
    print(f"{m:22s} rf={e['rf_raw_vs_poisson']} identical={e['identical_topology']} "
          f"len_ratio={e['branch_length_ratio_poisson_over_raw']:.4f}")
print('\n--- additivity (all 16) ---')
for m, e in additivity.items():
    print(f"{m:38s} mean_rel={e['mean_relative_violation']:.5f} med={e['median_relative_violation']:.5f} "
          f"max={e['max_relative_violation']:.4f} tri_viol={e['n_triangle_inequality_violations']}")
print('\n--- additivity (entero 14) ---')
for m, e in additivity_entero.items():
    print(f"{m:38s} mean_rel={e['mean_relative_violation']:.5f}")
print('\n--- additivity ECOLI8 (UNSATURATED CONTROL) ---')
for m, e in additivity_ecoli8.items():
    print(f"{m:38s} mean_rel={e['mean_relative_violation']:.5f} "
          f"SCALEFREE={e['scalefree_mean_abs_violation_over_mean_distance']:.5f} "
          f"dyn_range={e['distance_dynamic_range_max_over_min']:.2f}x")
print('\n--- additivity ENTERO14 scale-free ---')
for m, e in additivity_entero.items():
    print(f"{m:38s} mean_rel={e['mean_relative_violation']:.5f} "
          f"SCALEFREE={e['scalefree_mean_abs_violation_over_mean_distance']:.5f}")
print('\n--- compression diagnostics (all 16) ---')
for m, e in compression.items():
    print(f"{m:38s} {({k: round(v,5) for k,v in e.items()})}")
print('\n--- support ---')
for tag, s in support.items():
    print(tag, 'retained', round(s['mean_universe_retained_fraction'], 4), 'nrep', s['n_replicates_succeeded'])
    for c in s['observed_tree_clade_support']:
        print('   ', round(c['support_pct'], 1), c['clade'])
print('\n--- saturation ---'); print(sat)
print('\n--- fgr2 vs mash, unsaturated pairs only ---')
for m, e in fgr_vs_mash_unsat.items():
    print(m, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()})
print('\n--- ENTERO14 RF (fair, unsaturated) ---')
for m, e in entero_rf.items():
    print(f"{m:38s} NCBI rf={e['REF_NCBI_shigella_separate']['rf']:2d} "
          f"({e['REF_NCBI_shigella_separate']['ref_splits_recovered']}/{e['REF_NCBI_shigella_separate']['splits_in_ref']}) | "
          f"PHYLO rf={e['REF_PHYLO_shigella_nested']['rf']:2d} "
          f"({e['REF_PHYLO_shigella_nested']['ref_splits_recovered']}/{e['REF_PHYLO_shigella_nested']['splits_in_ref']})")
print('\n--- ENTERO14 raw vs poisson ---')
for m, e in entero_topo_identity.items():
    print(f"{m:22s} rf={e['rf_raw_vs_poisson']} identical={e['identical_topology']} "
          f"len_ratio={e['branch_length_ratio_poisson_over_raw']:.4f} "
          f"negbr raw={e['n_negative_branches_raw']} pois={e['n_negative_branches_poisson']}")
print('\n--- missed reference splits (REF_PHYLO) ---')
for t, v in split_accounting['REF_PHYLO_shigella_nested']['per_tree'].items():
    print(t, 'MISSED:', v['missed'])
print('\n--- clade recovery ---')
for m, e in nesting.items():
    print(f"{m:38s} {e}")
