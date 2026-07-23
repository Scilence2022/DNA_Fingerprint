"""
Pairwise similarity between fgr/fgr2 bottom-N sketches, with dendrograms and
Neighbor-Joining trees.

ESTIMATORS AND THE LEGACY BIAS
------------------------------
An .fgr2 file is a bottom-N MinHash sketch: the N k-mers with the smallest hash
values in that sample. Two sketches built INDEPENDENTLY from two samples are not
samples from a common index, so set operations applied to them directly do not
estimate the corresponding quantity on the full k-mer sets.

* Jaccard. The historical estimator |S_A n S_B| / |S_A u S_B| ("direct") is NOT
  a MinHash estimator. Its bias is a function of the ratio of the two samples'
  distinct-k-mer counts and does NOT shrink as N grows. The correct construction
  (Broder / Mash) merges the two sketches, keeps the N globally smallest hashes
  of the merge -- which is exactly the bottom-N sketch of the TRUE union -- and
  reports the fraction of those k-mers present in both sketches. That is the
  default here (``jaccard_union``); ``--legacy-jaccard`` restores the old one.

* Cosine. The historical estimator took the dot product over S_A n S_B while
  taking each L2 norm over its own sketch, i.e. numerator and denominators lived
  on different supports. This carries a multiplicative downward bias
      E[cos_legacy] = cos_true / sqrt(r),   r = max(|A|,|B|) / min(|A|,|B|)
  which also does not vanish with N. The default here puts the numerator and
  BOTH norms on the bottom-N-of-union support, with absent k-mers contributing
  coverage 0 (``cosine_union``); ``--legacy-cosine`` restores the old one.

  WHERE THE COSINE FIX DOES AND DOES NOT HELP (measured, not assumed). The
  correction is decisive when the two inputs differ in size: on a controlled
  real-genome construction spanning r = 1.0-2.9 it cut mean bias from -0.131 to
  -0.004 and RMSE from 0.165 to 0.019 (n = 280, paired Wilcoxon p = 1.5e-44).
  But on the 120 natural pairs of 16 RefSeq genomes, where r only spans
  1.006-1.561, the legacy cosine is NOT significantly worse (p = 0.891 at
  N = 100, p = 0.225 at N = 10000); for whole bacterial genomes of similar size
  the fix changes little. And in a READ-based regime with coverage filtering
  disabled (-c 1), where the support is millions of distinct k-mers and N
  samples ~0.3% of it, BOTH estimators fall far below the true cosine and the
  corrected one was significantly LESS accurate than the legacy one
  (p = 8.6e-09 at N = 10000). Rank order is preserved (Spearman ~0.97), so
  relative comparisons survive, but neither value should be read as a point
  estimate of the cosine in that regime. The corrected estimator is the default
  because it is the one that is unbiased by construction and improves with N;
  it is not uniformly better on every input.

WARNING: similarity matrices produced with either legacy estimator are biased
whenever the two inputs differ in size (number of distinct k-mers). The bias is
small when the inputs are of comparable size -- e.g. whole bacterial genomes of
similar length -- and grows with the size ratio. Published .jac/.cos matrices in
this repository predate the fix; use the --legacy-* flags to reproduce them.
"""

import os
import glob
import math
import random
import argparse
from itertools import combinations
import numpy as np

# Try to import optional packages for dendrogram plotting
PLOTTING_AVAILABLE = False
try:
    import scipy.cluster.hierarchy as sch
    # Import squareform from its canonical location. Reaching it via
    # scipy.cluster.hierarchy.distance relied on an implicit re-export that newer
    # SciPy releases (>=1.15) no longer provide, which raised AttributeError and
    # aborted the whole run before any tree was written.
    from scipy.spatial.distance import squareform
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    pass  # We'll handle the absence of these packages later

# Try to import scikit-bio for Neighbor-Joining trees
SKBIO_AVAILABLE = False
SKBIO_MAJORITY_CONSENSUS_AVAILABLE = False
try:
    import skbio
    from skbio import DistanceMatrix
    from skbio.tree import nj, TreeNode # TreeNode for type hinting if needed
    SKBIO_AVAILABLE = True
    
    # Check if majority_consensus is available
    try:
        from skbio.tree import majority_consensus
        SKBIO_MAJORITY_CONSENSUS_AVAILABLE = True
    except (ImportError, AttributeError):
        # majority_consensus not available in this version of scikit-bio
        pass
except ImportError:
    pass

def parse_sketch(filepath):
    """
    Parses an .fgr/.fgr2 file into full sketch records, hash column included.

    The expected record format is three tab-separated columns:
        <kmer>\t<hash>\t<coverage>
    Lines beginning with '#' are comments. fgr2 emits a self-describing header
    (``#fgr2\tversion=..\tk=..\tN=..\thash=..\tcoverage_threshold=..``) which is
    parsed into the returned metadata dict so callers can verify that sketches
    were generated with compatible parameters.

    The hash column is the sketch's sort key. It is REQUIRED by the bottom-N-of-
    union estimators (``jaccard_union`` / ``cosine_union``), which is why it is
    retained here rather than discarded as it was historically.

    Returns
    -------
    (records, meta) on success, or (None, None) on failure, where ``records``
    maps kmer -> (hash, coverage) with both values ints.
    """
    records = {}
    meta = {}
    try:
        with open(filepath, 'r') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    # fgr2 metadata header, e.g. "#fgr2\tk=31\tN=10000\thash=murmurhash3"
                    if line.startswith('#fgr2'):
                        for token in line.split('\t')[1:]:
                            if '=' in token:
                                key, _, value = token.partition('=')
                                meta[key.strip()] = value.strip()
                    continue
                parts = line.split('\t')
                # Coverage lives in the 3rd column; a shorter line is malformed.
                if len(parts) < 3:
                    print(f"Warning: Skipping malformed line {lineno} in {filepath}: {line}")
                    continue
                kmer = parts[0]
                try:
                    coverage = int(parts[2])
                except ValueError:
                    print(f"Warning: Could not parse coverage for k-mer '{kmer}' at line "
                          f"{lineno} in {filepath}. Skipping line.")
                    continue
                try:
                    hash_value = int(parts[1])
                except ValueError:
                    # A k-mer with no usable hash cannot be ranked, so it cannot
                    # take part in the bottom-N-of-union construction. Dropping it
                    # is safer than silently excluding it from the union support
                    # while still counting it in the intersection.
                    print(f"Warning: Could not parse hash for k-mer '{kmer}' at line "
                          f"{lineno} in {filepath}. Skipping line.")
                    continue
                records[kmer] = (hash_value, coverage)
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None, None
    except OSError as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None
    return records, meta


def parse_fgr_file(filepath):
    """
    Backwards-compatible view of :func:`parse_sketch`.

    Returns ``(kmers, coverages, meta)`` exactly as it always has -- the hash
    column is dropped. New code should call :func:`parse_sketch` instead, since
    the corrected estimators need the hashes.

    Returns (None, None, None) on failure.
    """
    records, meta = parse_sketch(filepath)
    if records is None:
        return None, None, None
    kmers = set(records)
    coverages = {kmer: cov for kmer, (_h, cov) in records.items()}
    return kmers, coverages, meta


def sketch_kmers(records):
    """Set of k-mers in a sketch-record dict (kmer -> (hash, coverage))."""
    return set(records)


def sketch_coverages(records):
    """kmer -> coverage view of a sketch-record dict."""
    return {kmer: cov for kmer, (_h, cov) in records.items()}


def jaccard_index(set1, set2):
    """
    LEGACY (biased) Jaccard estimator: |A n B| / |A u B| applied directly to two
    independently built bottom-N sketches.

    This is NOT a MinHash estimator. Its bias is driven by the ratio of the two
    samples' distinct-k-mer counts and does not shrink as N grows. Retained only
    so ``--legacy-jaccard`` can reproduce previously published matrices; use
    :func:`jaccard_union` for new work.
    """
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


def union_support(records1, records2, n=None):
    """
    The bottom-N sketch of the TRUE union of the two samples' k-mer sets.

    Merging the two sketches and keeping the ``n`` globally smallest hashes is
    exact: a k-mer whose hash is among the n smallest of A u B must, if it is
    present in sample A at all, already be among the n smallest of A -- so it is
    guaranteed to be in sketch A and cannot be missed. Membership of the
    resulting support in each sample is therefore fully determined by the two
    sketches, with "absent from the sketch" meaning "absent from the sample".

    ``n`` defaults to min(|S_A|, |S_B|), which equals the sketch size N whenever
    both sketches are full and is the largest value for which the guarantee
    above holds for both samples.

    Ties in hash value are broken by k-mer string so the support -- and hence
    every estimate built on it -- is symmetric in its two arguments.
    """
    merged = {}
    for kmer, (h, _cov) in records1.items():
        merged[kmer] = h
    for kmer, (h, _cov) in records2.items():
        merged[kmer] = h
    if n is None:
        n = min(len(records1), len(records2))
    if n <= 0:
        return []
    return sorted(merged, key=lambda kmer: (merged[kmer], kmer))[:n]


def jaccard_union(records1, records2, n=None):
    """
    Corrected (bottom-N-of-union) Jaccard estimator -- the Broder/Mash
    construction, and the default in this tool.

    Takes the bottom-N sketch of the union (see :func:`union_support`) and
    reports the fraction of its k-mers present in BOTH samples. Because that
    support is a uniform random sample of the true union, this is a consistent
    estimator of |A n B| / |A u B| with error that shrinks as N grows.

    Parameters
    ----------
    records1, records2 : dict
        kmer -> (hash, coverage), as returned by :func:`parse_sketch`.
    n : int, optional
        Support size; defaults to min(|S_A|, |S_B|).
    """
    if not records1 and not records2:
        return 1.0
    support = union_support(records1, records2, n)
    if not support:
        return 0.0
    shared = sum(1 for kmer in support if kmer in records1 and kmer in records2)
    return shared / len(support)


def resample_kmer_universe(all_unique_kmers_list):
    """
    Draw a bootstrap k-mer universe by sampling with replacement.

    METHODOLOGICAL NOTE: k-mers are drawn with replacement and then collapsed into a
    set, which discards multiplicities. Sampling n items with replacement from n and
    keeping the distinct ones retains ~1 - 1/e ~= 63.2% of the universe, so each
    replicate is effectively a random ~63% *subsample* of the k-mers rather than a
    multinomial (Felsenstein) bootstrap. Support values should be read as
    subsampling/jackknife-style support, not textbook bootstrap proportions. For the
    presence/absence Jaccard distance multiplicities are irrelevant anyway (the
    statistic is a function of the sets), but for the coverage-weighted cosine
    distance a true bootstrap would weight each k-mer by its draw count. This
    function preserves the original behaviour; changing it would alter published
    support values and is left as a deliberate decision for the maintainer.
    """
    return set(random.choices(all_unique_kmers_list, k=len(all_unique_kmers_list)))

def vector_norm(coverages):
    """L2 norm of a k-mer coverage dictionary, viewed as a sparse vector."""
    if not coverages:
        return 0.0
    return math.sqrt(sum(float(v) * float(v) for v in coverages.values()))


def cosine_similarity_manual(dict1, dict2, standardize=False, norm1=None, norm2=None):
    """
    LEGACY (biased) cosine estimator between two k-mer coverage dictionaries.

    The dot product runs over A n B while each L2 norm runs over that sample's
    own dictionary. On two FULL coverage vectors this is exact -- absent entries
    contribute nothing to the dot product, so restricting it to the intersection
    is an optimisation, not an approximation. On two independently built bottom-N
    SKETCHES it is not: numerator and denominators are then estimated on
    different supports, giving a multiplicative downward bias
        E[cos_legacy] = cos_true / sqrt(r),  r = max(|A|,|B|) / min(|A|,|B|)
    that does not vanish as N grows.

    Retained for ``--legacy-cosine`` and for use on full (un-sketched) vectors,
    where it remains exact. New sketch comparisons should use
    :func:`cosine_union`.

    Parameters
    ----------
    dict1, dict2 : dict
        Dictionaries mapping k-mers to their coverage values.
    standardize : bool, optional
        Accepted for backwards compatibility. NOTE: this is a mathematical no-op.
        It rescales each vector by a positive constant, and cosine similarity is
        invariant under positive scaling: cos(a*u, b*v) == cos(u, v) for a, b > 0.
        Retained so existing command lines keep working.
    norm1, norm2 : float, optional
        Precomputed L2 norms of dict1/dict2. Supplied by callers that compare one
        sample against many, so each norm is computed once rather than per pair.

    Returns
    -------
    float
        Cosine similarity value between the two dictionaries.
    """
    if not dict1 and not dict2:
        return 1.0

    if norm1 is None:
        norm1 = vector_norm(dict1)
    if norm2 is None:
        norm2 = vector_norm(dict2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Iterate the smaller dictionary; only shared k-mers contribute to the dot product.
    if len(dict1) > len(dict2):
        dict1, dict2 = dict2, dict1
    get = dict2.get
    dot = 0.0
    for kmer, value in dict1.items():
        other = get(kmer)
        if other is not None:
            dot += float(value) * float(other)

    return dot / (norm1 * norm2)


def cosine_union(records1, records2, n=None):
    """
    Corrected (bottom-N-of-union) cosine estimator -- the default in this tool.

    The dot product AND both L2 norms are taken over one common, unbiasedly
    sampled support: the bottom-N sketch of the true union (see
    :func:`union_support`). A k-mer in the support that is absent from a sample
    contributes coverage 0 to that sample's vector, which is exact -- see the
    guarantee documented in :func:`union_support`.

    Note that no precomputed per-sample norm can be reused here: each norm is
    restricted to the pair's own union support, so it is pair-specific. This
    makes the corrected cosine strictly more expensive than the legacy one.

    Parameters
    ----------
    records1, records2 : dict
        kmer -> (hash, coverage), as returned by :func:`parse_sketch`.
    n : int, optional
        Support size; defaults to min(|S_A|, |S_B|).
    """
    if not records1 and not records2:
        return 1.0
    support = union_support(records1, records2, n)
    if not support:
        return 0.0
    dot = sum_aa = sum_bb = 0.0
    for kmer in support:
        rec_a = records1.get(kmer)
        rec_b = records2.get(kmer)
        a = float(rec_a[1]) if rec_a is not None else 0.0
        b = float(rec_b[1]) if rec_b is not None else 0.0
        dot += a * b
        sum_aa += a * a
        sum_bb += b * b
    if sum_aa == 0 or sum_bb == 0:
        return 0.0
    return dot / math.sqrt(sum_aa * sum_bb)


def write_matrix_to_file(matrix, filenames, output_filepath):
    """
    Writes a similarity matrix to a tab-separated file.
    """
    if matrix.shape[0] != len(filenames):
        print(f"Error: matrix is {matrix.shape} but {len(filenames)} labels were supplied; "
              f"refusing to write {output_filepath}.")
        return

    try:
        with open(output_filepath, 'w') as f:
            # Write header row
            f.write("\t" + "\t".join(filenames) + "\n")

            # Write each data row
            for filename, row in zip(filenames, matrix):
                row_values = "\t".join(f"{val:.4f}" for val in row)
                f.write(f"{filename}\t{row_values}\n")

        print(f"Successfully wrote matrix to {output_filepath}")
    except OSError as e:
        print(f"Error writing matrix to {output_filepath}: {e}")

def plot_and_save_dendrogram(distance_matrix, labels, tree_type, output_path_prefix):
    """
    Performs hierarchical clustering and plots/saves the dendrogram.
    Assumes distance_matrix is a condensed distance matrix or a square one.
    """
    if not PLOTTING_AVAILABLE:
        print("WARNING: Plotting libraries (scipy, matplotlib) are not available.")
        print("Please install them using: pip install scipy matplotlib")
        return
        
    try:
        if distance_matrix.ndim == 2 and distance_matrix.shape[0] == distance_matrix.shape[1]:
            # Convert square distance matrix to condensed form if necessary
            # Ensure it's not already a condensed matrix by checking shape
            if distance_matrix.shape[0] > 1: # Only condense if it's not a 1x1 matrix (single element)
                distance_matrix = squareform(distance_matrix, checks=False)
            elif distance_matrix.shape[0] == 1:
                print(f"Skipping dendrogram for {tree_type}: Only one item, cannot form a tree.")
                return

        if distance_matrix.size == 0 and len(labels) <= 1:
            print(f"Skipping dendrogram for {tree_type}: Not enough items to form a tree (labels: {len(labels)}).")
            return


        linked = sch.linkage(distance_matrix, method='average')
        plt.figure(figsize=(10, max(5, len(labels) * 0.5))) # Adjust figure size based on number of labels
        sch.dendrogram(linked, orientation='right', labels=labels, leaf_font_size=10)
        plt.title(f'Hierarchical Clustering Dendrogram ({tree_type} Distance)')
        plt.xlabel('Distance')
        plt.tight_layout() # Adjust layout to prevent labels from being cut off

        if output_path_prefix:
            output_filepath = f"{output_path_prefix}.{tree_type.lower()}.png"
            plt.savefig(output_filepath)
            print(f"Successfully saved {tree_type} dendrogram to {output_filepath}")
        else:
            plt.show()
        plt.close() # Close the figure to free memory

    except Exception as e:
        print(f"Error generating dendrogram for {tree_type}: {e}")
        print(f"Distance matrix shape: {distance_matrix.shape if hasattr(distance_matrix, 'shape') else 'N/A'}, size: {distance_matrix.size if hasattr(distance_matrix, 'size') else 'N/A'}")
        print(f"Labels: {labels}")

def strip_extensions(filename):
    """
    Removes .fgr2 and .fgr extensions from filenames for cleaner labels.
    """
    if filename.endswith('.fgr2'):
        return filename[:-5]
    elif filename.endswith('.fgr'):
        return filename[:-4]
    return filename


def disambiguate_labels(clean_names, fallback_names):
    """
    Ensure labels are unique.

    Stripping extensions can map distinct files onto the same label (e.g. both
    ``sample.fgr`` and ``sample.fgr2``). Duplicate ids raise an error in
    skbio.DistanceMatrix and make dendrogram labels ambiguous, so any name that is
    not unique reverts to the full basename (and, if still ambiguous, gets a suffix).
    """
    counts = {}
    for name in clean_names:
        counts[name] = counts.get(name, 0) + 1

    result = []
    used = set()
    for clean, fallback in zip(clean_names, fallback_names):
        label = clean if counts[clean] == 1 else fallback
        if label in used:
            suffix = 2
            while f"{label}_{suffix}" in used:
                suffix += 1
            label = f"{label}_{suffix}"
        used.add(label)
        result.append(label)

    changed = [(c, r) for c, r in zip(clean_names, result) if c != r]
    if changed:
        print("Note: some labels collided after extension stripping and were disambiguated:")
        for original, final in changed:
            print(f"  '{original}' -> '{final}'")
    return result


def check_sketch_compatibility(file_paths, file_meta):
    """
    Warn when sketches were not generated with comparable fgr2 parameters.

    Jaccard and Cosine similarities are only meaningful across sketches built with
    the same k, the same hash function, and a comparable sketch size / coverage
    threshold. Mixing them (e.g. k=21 against k=31, or MurmurHash3 against Wang)
    silently yields near-zero similarity that looks like a biological result.
    Sketches produced by older fgr2 builds carry no metadata header and are skipped.
    """
    annotated = {p: file_meta.get(p) or {} for p in file_paths}
    described = [p for p in file_paths if annotated[p]]
    if not described:
        return

    for field, label in (('k', 'k-mer size'), ('hash', 'hash function')):
        values = {}
        for p in described:
            value = annotated[p].get(field)
            if value is not None:
                values.setdefault(value, []).append(os.path.basename(p))
        if len(values) > 1:
            print(f"\nWARNING: sketches were built with different {label} ({field}):")
            for value, names in sorted(values.items()):
                shown = ', '.join(names[:4]) + (' ...' if len(names) > 4 else '')
                print(f"  {field}={value}: {shown}")
            print(f"  Similarities across different {label} values are not meaningful.")

    if len(described) < len(file_paths):
        missing = len(file_paths) - len(described)
        print(f"Note: {missing} sketch file(s) carry no fgr2 metadata header "
              f"(older format); their parameters could not be verified.")

def main():
    parser = argparse.ArgumentParser(description="Calculate Jaccard Index and Cosine Similarity between .fgr files, and optionally plot relationship trees and build Neighbor-Joining trees.")
    parser.add_argument("directory", nargs='?', default=".", help="Directory containing .fgr files (default: current directory)")
    parser.add_argument("-o", "--output_prefix", help="Output prefix for similarity matrix and dendrogram/tree files. If provided, matrices and plots/trees are saved to files.")
    parser.add_argument("--no-plot_trees", action='store_false', dest='plot_trees', help="Disable generation of hierarchical clustering relationship trees (dendrograms). Trees are generated by default.")
    parser.add_argument("--no-nj_tree", action='store_false', dest='nj_tree', help="Disable generation of Neighbor-Joining trees. Trees are generated by default.")
    parser.add_argument("--bootstrap_replicates", type=int, default=100, help="Number of bootstrap replicates for Neighbor-Joining trees (default: 100). Only used if NJ trees are enabled.")
    parser.add_argument("--standardize", action="store_true", help="Deprecated no-op: cosine similarity is invariant under the positive rescaling this applied, so results are identical with or without it. Accepted for backwards compatibility.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for bootstrap resampling, for reproducible support values.")
    parser.add_argument("--legacy-jaccard", action="store_true", dest="legacy_jaccard",
                        help="Use the old, biased direct Jaccard estimator "
                             "|A n B| / |A u B| on the two raw sketches. Only for "
                             "reproducing previously published .jac matrices.")
    parser.add_argument("--legacy-cosine", action="store_true", dest="legacy_cosine",
                        help="Use the old, biased cosine estimator (dot product over "
                             "the intersection, each norm over its own sketch). Only "
                             "for reproducing previously published .cos matrices.")
    args = parser.parse_args()

    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap_replicates must be >= 0")

    if args.legacy_jaccard or args.legacy_cosine:
        which = " and ".join(
            name for name, on in (("Jaccard", args.legacy_jaccard),
                                  ("cosine", args.legacy_cosine)) if on)
        print("*" * 78)
        print(f"WARNING: LEGACY {which.upper()} ESTIMATOR(S) ENABLED -- RESULTS ARE BIASED.")
        print("  The legacy estimators apply set/vector operations directly to two")
        print("  INDEPENDENTLY built bottom-N sketches. They are not MinHash estimators.")
        print("  Their bias is driven by the ratio of the two samples' distinct-k-mer")
        print("  counts and does NOT shrink as the sketch size N grows. Sketches that")
        print("  differ in size will be scored too low.")
        print("  Use these flags only to reproduce matrices published before the fix.")
        print("*" * 78)

    if args.standardize:
        print("NOTE: --standardize has no effect. Cosine similarity is invariant under the\n"
              "      positive per-vector rescaling it performs, so output is unchanged.")

    if args.seed is not None:
        random.seed(args.seed)

    # Check if plotting is requested but not available
    if args.plot_trees and not PLOTTING_AVAILABLE:
        print("WARNING: You requested dendrograms (--plot_trees) but required libraries are missing.")
        print("Please install scipy and matplotlib: pip install scipy matplotlib")
        print("Continuing with similarity calculation only...\n")

    fgr_dir = args.directory
    # Find both .fgr and .fgr2 files
    fgr_files_paths1 = glob.glob(os.path.join(fgr_dir, "*.fgr"))
    fgr_files_paths2 = glob.glob(os.path.join(fgr_dir, "*.fgr2"))
    # Combine and sort, removing duplicates
    fgr_files_paths = sorted(list(set(fgr_files_paths1 + fgr_files_paths2)))

    if len(fgr_files_paths) < 2:
        print("Need at least two .fgr or .fgr2 files to compare.")
        return

    print(f"Found {len(fgr_files_paths)} .fgr files in '{os.path.abspath(fgr_dir)}':")
    file_basenames = []
    for f_path in fgr_files_paths:
        basename = os.path.basename(f_path)
        # Store the original basename, but print the cleaned version
        clean_basename = strip_extensions(basename)
        print(f"  - {clean_basename}")
        file_basenames.append(basename)
    
    file_data = {}
    file_records = {}
    file_meta = {}
    valid_file_paths = []
    for f_path in fgr_files_paths:
        records, meta = parse_sketch(f_path)
        if records is None:
            continue
        kmers = sketch_kmers(records)
        coverages = sketch_coverages(records)
        if not kmers:
            # An empty sketch would compare as distance 0 to every other empty sketch
            # and as distance 1 to everything else, silently corrupting the tree.
            print(f"Warning: {os.path.basename(f_path)} contains no usable k-mers. Excluding it.")
            continue
        file_data[f_path] = (kmers, coverages)
        file_records[f_path] = records
        file_meta[f_path] = meta
        valid_file_paths.append(f_path)

    if len(valid_file_paths) < 2:
        print("Not enough valid .fgr files to compare after parsing (need at least 2).")
        return

    check_sketch_compatibility(valid_file_paths, file_meta)

    # Initialize matrices and name mapping
    file_basenames = [os.path.basename(p) for p in valid_file_paths]

    # Create clean basenames for display/output without .fgr2/.fgr extensions.
    # Stripping extensions can collide (e.g. sample.fgr and sample.fgr2), which would
    # produce duplicate tree/matrix labels, so fall back to the full basename.
    clean_basenames = disambiguate_labels([strip_extensions(b) for b in file_basenames],
                                          file_basenames)

    name_to_idx = {name: i for i, name in enumerate(file_basenames)}
    num_files = len(file_basenames)

    # Legacy-cosine only: norms depend solely on a sample's own coverages, so compute
    # each once here rather than re-deriving them inside every pairwise call. The
    # corrected cosine cannot reuse these -- its norms are restricted to the pair's
    # own union support and are therefore pair-specific.
    file_norms = ({p: vector_norm(file_data[p][1]) for p in valid_file_paths}
                  if args.legacy_cosine else None)

    print(f"\nJaccard estimator: {'LEGACY direct (biased)' if args.legacy_jaccard else 'bottom-N-of-union (default)'}")
    print(f"Cosine estimator:  {'LEGACY intersection/own-norms (biased)' if args.legacy_cosine else 'bottom-N-of-union (default)'}")

    jaccard_matrix = np.zeros((num_files, num_files))
    cosine_matrix = np.zeros((num_files, num_files))
    
    np.fill_diagonal(jaccard_matrix, 1.0)
    np.fill_diagonal(cosine_matrix, 1.0)

    if not args.output_prefix and not args.plot_trees:
        print("\n--- Similarity Scores (Printed to Console) ---")
    elif args.output_prefix:
         print(f"\n--- Processing for Output Prefix: {args.output_prefix} ---")

    for file1_path, file2_path in combinations(valid_file_paths, 2):
        kmers1, coverages1 = file_data[file1_path]
        kmers2, coverages2 = file_data[file2_path]

        idx1 = name_to_idx[os.path.basename(file1_path)]
        idx2 = name_to_idx[os.path.basename(file2_path)]

        if args.legacy_jaccard:
            ji = jaccard_index(kmers1, kmers2)
        else:
            ji = jaccard_union(file_records[file1_path], file_records[file2_path])

        if args.legacy_cosine:
            cs = cosine_similarity_manual(coverages1, coverages2, args.standardize,
                                          norm1=file_norms[file1_path],
                                          norm2=file_norms[file2_path])
        else:
            cs = cosine_union(file_records[file1_path], file_records[file2_path])

        jaccard_matrix[idx1, idx2] = ji
        jaccard_matrix[idx2, idx1] = ji
        cosine_matrix[idx1, idx2] = cs
        cosine_matrix[idx2, idx1] = cs

        if not args.output_prefix and not args.plot_trees:
            # Labels are index-aligned with valid_file_paths, so reuse them directly.
            print(f"\nComparing '{clean_basenames[idx1]}' and '{clean_basenames[idx2]}':")
            print(f"  Jaccard Index: {ji:.4f}")
            print(f"  Cosine Similarity (based on coverage): {cs:.4f}")

    if args.output_prefix:
        jac_output_path = args.output_prefix + ".jac"
        cos_output_path = args.output_prefix + ".cos"

        write_matrix_to_file(jaccard_matrix, clean_basenames, jac_output_path)
        write_matrix_to_file(cosine_matrix, clean_basenames, cos_output_path)

    if args.plot_trees and PLOTTING_AVAILABLE:
        print("\n--- Generating Relationship Trees (Dendrograms) ---")
        # Convert similarity to distance (1 - similarity)
        jaccard_distance_matrix_for_dendro = 1 - jaccard_matrix
        cosine_distance_matrix_for_dendro = 1 - cosine_matrix

        # Ensure diagonal is 0 for distance matrices
        np.fill_diagonal(jaccard_distance_matrix_for_dendro, 0)
        np.fill_diagonal(cosine_distance_matrix_for_dendro, 0)

        # Plot Jaccard Dendrogram
        if num_files >=2:
            plot_and_save_dendrogram(jaccard_distance_matrix_for_dendro, clean_basenames, "Jaccard", args.output_prefix)
            plot_and_save_dendrogram(cosine_distance_matrix_for_dendro, clean_basenames, "Cosine", args.output_prefix)
        else:
            print("Skipping dendrogram generation as there are less than 2 files to compare.")

    # --- Neighbor-Joining Tree Generation ---
    if args.nj_tree:
        if not SKBIO_AVAILABLE:
            print("\nWARNING: You requested Neighbor-Joining trees (--nj_tree) but scikit-bio is not installed.")
            print("Please install it using: pip install scikit-bio")
            print("Skipping NJ tree generation.")
        elif num_files < 2:
            print("\nSkipping Neighbor-Joining tree generation as there are less than 2 files to compare.")
        else:
            print("\n--- Generating Neighbor-Joining Trees ---")
            
            # Distance matrices (1 - similarity)
            # These are the full distance matrices based on all k-mers
            nj_jaccard_dist_matrix_np = 1 - jaccard_matrix
            nj_cosine_dist_matrix_np = 1 - cosine_matrix
            np.fill_diagonal(nj_jaccard_dist_matrix_np, 0)
            np.fill_diagonal(nj_cosine_dist_matrix_np, 0)

            # Prepare k-mer data for bootstrapping
            all_unique_kmers = set()
            if args.bootstrap_replicates > 0:
                for f_path in valid_file_paths: # Use valid_file_paths to get original kmer sets
                    kmers_set, _ = file_data[f_path]
                    all_unique_kmers.update(kmers_set)
                all_unique_kmers_list = list(all_unique_kmers)
                if not all_unique_kmers_list:
                    print("Warning: No k-mers found across all samples. Cannot perform bootstrap for NJ trees.")
                    # Proceed without bootstrapping if no k-mers
                    current_bootstrap_replicates = 0
                else:
                    current_bootstrap_replicates = args.bootstrap_replicates
            else:
                current_bootstrap_replicates = 0

            # --- Process Jaccard Distances for NJ Tree ---
            print("\nProcessing Jaccard distances for NJ tree...")
            try:
                skbio_jaccard_dm_orig = DistanceMatrix(nj_jaccard_dist_matrix_np, ids=clean_basenames)
                
                if current_bootstrap_replicates > 0:
                    print(f"Running {current_bootstrap_replicates} bootstrap replicates for Jaccard NJ tree...")
                    jaccard_bootstrap_trees = []
                    for i in range(current_bootstrap_replicates):
                        if (i + 1) % 10 == 0 or i == current_bootstrap_replicates - 1:
                             print(f"  Bootstrap replicate {i+1}/{current_bootstrap_replicates}...")
                        
                        # Create bootstrapped k-mer universe
                        current_bootstrap_kmer_universe = resample_kmer_universe(all_unique_kmers_list)

                        # Restrict each sample to the resampled universe ONCE per
                        # replicate. Doing this inside the pair loop below would repeat
                        # identical work num_files-1 times per sample, making the
                        # replicate O(n^2 * |kmers|) instead of O(n * |kmers|).
                        kmers_boot = [file_data[p][0] & current_bootstrap_kmer_universe
                                      for p in valid_file_paths]
                        # The corrected estimator needs the hashes as well, so the
                        # resampled universe is applied to the full records too.
                        records_boot = None
                        if not args.legacy_jaccard:
                            records_boot = [
                                {k: v for k, v in file_records[p].items()
                                 if k in current_bootstrap_kmer_universe}
                                for p in valid_file_paths]

                        boot_j_distances = np.zeros((num_files, num_files))

                        for r_idx in range(num_files): # row index
                            for c_idx in range(r_idx + 1, num_files): # column index
                                if args.legacy_jaccard:
                                    ji_boot = jaccard_index(kmers_boot[r_idx], kmers_boot[c_idx])
                                else:
                                    ji_boot = jaccard_union(records_boot[r_idx], records_boot[c_idx])
                                dist_boot = 1.0 - ji_boot

                                boot_j_distances[r_idx, c_idx] = dist_boot
                                boot_j_distances[c_idx, r_idx] = dist_boot
                        
                        skbio_boot_j_dm = DistanceMatrix(boot_j_distances, ids=clean_basenames)
                        
                        try:
                            if skbio_boot_j_dm.shape[0] >= 2:
                                if np.all(np.isclose(skbio_boot_j_dm.data, 0)) and skbio_boot_j_dm.shape[0] > 1 :
                                    # print(f"Warning: Jaccard Bootstrap replicate {i+1} resulted in all-zero distances. Skipping.")
                                    continue
                                if np.any(np.isnan(skbio_boot_j_dm.data)) or np.any(np.isinf(skbio_boot_j_dm.data)):
                                    # print(f"Warning: Jaccard Bootstrap replicate {i+1} resulted in NaN/Inf distances. Skipping.")
                                    continue
                                boot_tree = nj(skbio_boot_j_dm, disallow_negative_branch_length=True)
                                jaccard_bootstrap_trees.append(boot_tree)
                        except Exception as e_nj_boot:
                            # print(f"Warning: Could not build NJ tree for Jaccard bootstrap replicate {i+1}. Error: {e_nj_boot}. Skipping.")
                            pass # Fail silently for individual bootstrap replicates to avoid flooding console

                    if jaccard_bootstrap_trees:
                        # assign_supports=True will assign node.support = proportion of trees containing the bipartition
                        # A specific tree isn't used as the 'reference' for majority_consensus; it derives a new one.
                        if SKBIO_MAJORITY_CONSENSUS_AVAILABLE:
                            try:
                                final_tree_j = majority_consensus(jaccard_bootstrap_trees, cutoff=0.0, assign_supports=True)
                            except Exception as e:
                                print(f"Error using majority_consensus: {e}")
                                print("Falling back to simple consensus method...")
                                final_tree_j = simple_consensus_tree_with_supports(jaccard_bootstrap_trees, clean_basenames)
                        else:
                            final_tree_j = simple_consensus_tree_with_supports(jaccard_bootstrap_trees, clean_basenames)
                        print("Jaccard NJ consensus tree with bootstrap supports generated.")
                    else:
                        print("No Jaccard bootstrap trees were successfully generated. Building NJ tree on original data without supports.")
                        final_tree_j = nj(skbio_jaccard_dm_orig, disallow_negative_branch_length=True)
                else: # No bootstrapping
                    final_tree_j = nj(skbio_jaccard_dm_orig, disallow_negative_branch_length=True)
                    print("Jaccard NJ tree (no bootstrap) generated.")

                if args.output_prefix:
                    nj_j_output_path = args.output_prefix + ".jaccard.nj.newick"
                    final_tree_j.write(nj_j_output_path)
                    print(f"Jaccard NJ tree saved to {nj_j_output_path}")
                else:
                    print("\nJaccard Neighbor-Joining Tree (Newick format):")
                    print(final_tree_j.format_newick())
            except Exception as e_nj_main_j:
                print(f"Error during Jaccard NJ tree construction: {e_nj_main_j}")


            # --- Process Cosine Distances for NJ Tree ---
            print("\nProcessing Cosine distances for NJ tree...")
            try:
                skbio_cosine_dm_orig = DistanceMatrix(nj_cosine_dist_matrix_np, ids=clean_basenames)

                if current_bootstrap_replicates > 0:
                    print(f"Running {current_bootstrap_replicates} bootstrap replicates for Cosine NJ tree...")
                    cosine_bootstrap_trees = []
                    for i in range(current_bootstrap_replicates):
                        if (i + 1) % 10 == 0 or i == current_bootstrap_replicates -1:
                            print(f"  Bootstrap replicate {i+1}/{current_bootstrap_replicates}...")

                        current_bootstrap_kmer_universe = resample_kmer_universe(all_unique_kmers_list)

                        # Filter each sample's coverages and derive its norm ONCE per
                        # replicate, rather than rebuilding both dicts for every pair.
                        if args.legacy_cosine:
                            cov_boot = [{k: v for k, v in file_data[p][1].items()
                                         if k in current_bootstrap_kmer_universe}
                                        for p in valid_file_paths]
                            norms_boot = [vector_norm(c) for c in cov_boot]
                            records_boot_c = None
                        else:
                            records_boot_c = [
                                {k: v for k, v in file_records[p].items()
                                 if k in current_bootstrap_kmer_universe}
                                for p in valid_file_paths]

                        boot_c_distances = np.zeros((num_files, num_files))

                        for r_idx in range(num_files): # row index
                            for c_idx in range(r_idx + 1, num_files): # column index
                                if args.legacy_cosine:
                                    cs_boot = cosine_similarity_manual(
                                        cov_boot[r_idx], cov_boot[c_idx], args.standardize,
                                        norm1=norms_boot[r_idx], norm2=norms_boot[c_idx])
                                else:
                                    cs_boot = cosine_union(records_boot_c[r_idx],
                                                           records_boot_c[c_idx])
                                dist_boot = 1.0 - cs_boot

                                boot_c_distances[r_idx, c_idx] = dist_boot
                                boot_c_distances[c_idx, r_idx] = dist_boot
                        
                        skbio_boot_c_dm = DistanceMatrix(boot_c_distances, ids=clean_basenames)
                        try:
                            if skbio_boot_c_dm.shape[0] >= 2:
                                if np.all(np.isclose(skbio_boot_c_dm.data, 0)) and skbio_boot_c_dm.shape[0] > 1:
                                    # print(f"Warning: Cosine Bootstrap replicate {i+1} resulted in all-zero distances. Skipping.")
                                    continue
                                if np.any(np.isnan(skbio_boot_c_dm.data)) or np.any(np.isinf(skbio_boot_c_dm.data)):
                                    # print(f"Warning: Cosine Bootstrap replicate {i+1} resulted in NaN/Inf distances. Skipping.")
                                    continue
                                boot_tree = nj(skbio_boot_c_dm, disallow_negative_branch_length=True)
                                cosine_bootstrap_trees.append(boot_tree)
                        except Exception as e_nj_boot_c:
                            # print(f"Warning: Could not build NJ tree for Cosine bootstrap replicate {i+1}. Error: {e_nj_boot_c}. Skipping.")
                            pass

                    if cosine_bootstrap_trees:
                        if SKBIO_MAJORITY_CONSENSUS_AVAILABLE:
                            try:
                                final_tree_c = majority_consensus(cosine_bootstrap_trees, cutoff=0.0, assign_supports=True)
                            except Exception as e:
                                print(f"Error using majority_consensus: {e}")
                                print("Falling back to simple consensus method...")
                                final_tree_c = simple_consensus_tree_with_supports(cosine_bootstrap_trees, clean_basenames)
                        else:
                            final_tree_c = simple_consensus_tree_with_supports(cosine_bootstrap_trees, clean_basenames)
                        print("Cosine NJ consensus tree with bootstrap supports generated.")
                    else:
                        print("No Cosine bootstrap trees were successfully generated. Building NJ tree on original data without supports.")
                        final_tree_c = nj(skbio_cosine_dm_orig, disallow_negative_branch_length=True)
                else: # No bootstrapping
                    final_tree_c = nj(skbio_cosine_dm_orig, disallow_negative_branch_length=True)
                    print("Cosine NJ tree (no bootstrap) generated.")

                if args.output_prefix:
                    nj_c_output_path = args.output_prefix + ".cosine.nj.newick"
                    final_tree_c.write(nj_c_output_path)
                    print(f"Cosine NJ tree saved to {nj_c_output_path}")
                else:
                    print("\nCosine Neighbor-Joining Tree (Newick format):")
                    print(final_tree_c.format_newick())
            except Exception as e_nj_main_c:
                print(f"Error during Cosine NJ tree construction: {e_nj_main_c}")


# Custom function to build a simple consensus tree when skbio.tree.majority_consensus is not available
def simple_consensus_tree_with_supports(bootstrap_trees, ids):
    """
    A simplified approach to build a consensus tree with supports when skbio.tree.majority_consensus
    is not available. This uses the first tree as a reference and annotates it with support values.
    
    Parameters:
    -----------
    bootstrap_trees : list of TreeNode objects
        The bootstrap trees to consensus
    ids : list of str
        The IDs/labels for the tree leaves
        
    Returns:
    --------
    TreeNode
        A tree with branch support values
    """
    if not bootstrap_trees:
        return None
    
    # Use the first tree as the reference
    reference_tree = bootstrap_trees[0].copy()
    
    # Create a dict to track bipartitions across all trees
    bipartition_counts = {}
    total_trees = len(bootstrap_trees)
    
    # Function to get all bipartitions in a tree
    def get_bipartitions(tree):
        bipartitions = []
        for node in tree.non_tips():
            # Get all tip names descending from this node
            tips = {tip.name for tip in node.tips()}
            # Convert to a frozenset for hashing (order doesn't matter for bipartitions)
            bipartition = frozenset(tips)
            bipartitions.append((node, bipartition))
        return bipartitions
    
    # Count the frequency of each bipartition across bootstrap trees
    for tree in bootstrap_trees:
        for _, bipartition in get_bipartitions(tree):
            # Only count bipartitions that don't include all tips
            if len(bipartition) < len(ids) and len(bipartition) > 1:
                if bipartition in bipartition_counts:
                    bipartition_counts[bipartition] += 1
                else:
                    bipartition_counts[bipartition] = 1
    
    # Annotate the reference tree with support values
    for node, bipartition in get_bipartitions(reference_tree):
        if bipartition in bipartition_counts:
            support = bipartition_counts[bipartition] / total_trees
            node.support = support
    
    return reference_tree

if __name__ == "__main__":
    main() 