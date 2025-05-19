import os
import glob
import argparse
from itertools import combinations
import numpy as np

# Try to import optional packages for dendrogram plotting
PLOTTING_AVAILABLE = False
try:
    import scipy.cluster.hierarchy as sch
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
    import random # for bootstrapping
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

def parse_fgr_file(filepath):
    """
    Parses an .fgr file and returns a set of k-mers and a dictionary of k-mer:coverage.
    """
    kmers = set()
    coverages = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    kmer = parts[0]
                    try:
                        coverage = int(parts[2])
                        kmers.add(kmer)
                        coverages[kmer] = coverage
                    except ValueError:
                        print(f"Warning: Could not parse coverage for k-mer '{kmer}' in file {filepath}. Skipping line.")
                else:
                    print(f"Warning: Skipping malformed line in {filepath}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None, None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None
    return kmers, coverages

def jaccard_index(set1, set2):
    """
    Calculates the Jaccard Index between two sets.
    """
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def cosine_similarity_manual(dict1, dict2):
    """
    Calculates the Cosine Similarity between two k-mer coverage dictionaries.
    """
    all_kmers = set(dict1.keys()).union(set(dict2.keys()))
    if not all_kmers:
        return 1.0

    vec1 = []
    vec2 = []
    for kmer in all_kmers:
        vec1.append(dict1.get(kmer, 0))
        vec2.append(dict2.get(kmer, 0))

    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    dot_product = np.dot(vec1_np, vec2_np)
    norm_vec1 = np.linalg.norm(vec1_np)
    norm_vec2 = np.linalg.norm(vec2_np)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def write_matrix_to_file(matrix, filenames, output_filepath):
    """
    Writes a similarity matrix to a tab-separated file.
    """
    try:
        print(f"Writing matrix with shape {matrix.shape} to {output_filepath}")
        print(f"Number of filenames: {len(filenames)}")
        
        with open(output_filepath, 'w') as f:
            # Write header row
            header = "\t" + "\t".join(filenames)
            f.write(header + "\n")
            
            # Write each data row
            for i, filename in enumerate(filenames):
                if i >= matrix.shape[0]:
                    print(f"Warning: Filename index {i} exceeds matrix dimensions {matrix.shape}")
                    continue
                
                row_values = "\t".join([f"{val:.4f}" for val in matrix[i]])
                line = f"{filename}\t{row_values}"
                f.write(line + "\n")
                
        # Verify file was written correctly
        try:
            with open(output_filepath, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"Verification: {output_filepath} contains {line_count} lines (expected {len(filenames) + 1})")
            if line_count != len(filenames) + 1:
                print(f"WARNING: Expected {len(filenames) + 1} lines (header + {len(filenames)} data rows) but found {line_count} lines")
        except Exception as e:
            print(f"Error verifying file: {e}")
            
        print(f"Successfully wrote matrix to {output_filepath}")
    except Exception as e:
        print(f"Error writing matrix to {output_filepath}: {e}")
        print(f"Matrix shape: {matrix.shape}, Filenames length: {len(filenames)}")

def plot_and_save_dendrogram(distance_matrix, labels, tree_type, output_path_prefix):
    """
    Performs hierarchical clustering and plots/saves the dendrogram.
    Assumes distance_matrix is a condensed distance matrix or a square one.
    """
    if not PLOTTING_AVAILABLE:
        print("WARNING: Plotting libraries (scipy, matplotlib) are not available.")
        print("Please install them using: pip install scipy matplotlib")
        return
        
    if distance_matrix.ndim == 2 and distance_matrix.shape[0] == distance_matrix.shape[1]:
        # Convert square distance matrix to condensed form if necessary
        # Ensure it's not already a condensed matrix by checking shape
        if distance_matrix.shape[0] > 1: # Only condense if it's not a 1x1 matrix (single element)
             distance_matrix = sch.distance.squareform(distance_matrix, checks=False)
        elif distance_matrix.shape[0] == 1:
            print(f"Skipping dendrogram for {tree_type}: Only one item, cannot form a tree.")
            return 

    if distance_matrix.size == 0 and len(labels) <= 1:
        print(f"Skipping dendrogram for {tree_type}: Not enough items to form a tree (labels: {len(labels)}).")
        return
    
    try:
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

def main():
    parser = argparse.ArgumentParser(description="Calculate Jaccard Index and Cosine Similarity between .fgr files, and optionally plot relationship trees and build Neighbor-Joining trees.")
    parser.add_argument("directory", nargs='?', default=".", help="Directory containing .fgr files (default: current directory)")
    parser.add_argument("-o", "--output_prefix", help="Output prefix for similarity matrix and dendrogram/tree files. If provided, matrices and plots/trees are saved to files.")
    parser.add_argument("--plot_trees", action='store_true', help="Enable generation of hierarchical clustering relationship trees (dendrograms).")
    parser.add_argument("--nj_tree", action='store_true', help="Enable generation of Neighbor-Joining trees (requires scikit-bio).")
    parser.add_argument("--bootstrap_replicates", type=int, default=100, help="Number of bootstrap replicates for Neighbor-Joining trees (default: 100). Only used if --nj_tree is specified.")
    args = parser.parse_args()

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
        print(f"  - {basename}")
        file_basenames.append(basename)
    
    file_data = {}
    valid_file_paths = []
    for f_path in fgr_files_paths:
        kmers, coverages = parse_fgr_file(f_path)
        if kmers is not None and coverages is not None:
            file_data[f_path] = (kmers, coverages)
            valid_file_paths.append(f_path)

    if len(valid_file_paths) < 2:
        print("Not enough valid .fgr files to compare after parsing (need at least 2).")
        return

    file_basenames = [os.path.basename(p) for p in valid_file_paths]
    name_to_idx = {name: i for i, name in enumerate(file_basenames)}
    num_files = len(file_basenames)

    jaccard_matrix = np.zeros((num_files, num_files))
    cosine_matrix = np.zeros((num_files, num_files))
    
    np.fill_diagonal(jaccard_matrix, 1.0)
    np.fill_diagonal(cosine_matrix, 1.0)

    if not args.output_prefix and not args.plot_trees:
        print("\n--- Similarity Scores (Printed to Console) ---")
    elif args.output_prefix:
         print(f"\n--- Processing for Output Prefix: {args.output_prefix} ---")

    for (file1_path, data1), (file2_path, data2) in combinations(file_data.items(), 2):
        if file1_path not in valid_file_paths or file2_path not in valid_file_paths:
            continue

        kmers1, coverages1 = data1
        kmers2, coverages2 = data2
        
        basename1 = os.path.basename(file1_path)
        basename2 = os.path.basename(file2_path)

        idx1 = name_to_idx[basename1]
        idx2 = name_to_idx[basename2]

        ji = jaccard_index(kmers1, kmers2)
        cs = cosine_similarity_manual(coverages1, coverages2)

        jaccard_matrix[idx1, idx2] = ji
        jaccard_matrix[idx2, idx1] = ji
        cosine_matrix[idx1, idx2] = cs
        cosine_matrix[idx2, idx1] = cs

        if not args.output_prefix and not args.plot_trees:
            print(f"\nComparing '{basename1}' and '{basename2}':")
            print(f"  Jaccard Index: {ji:.4f}")
            print(f"  Cosine Similarity (based on coverage): {cs:.4f}")

    if args.output_prefix:
        jac_output_path = args.output_prefix + ".jac"
        cos_output_path = args.output_prefix + ".cos"
        
        # Debug the matrices before writing
        print(f"Jaccard matrix shape: {jaccard_matrix.shape}, values range: {np.min(jaccard_matrix):.4f} to {np.max(jaccard_matrix):.4f}")
        print(f"Cosine matrix shape: {cosine_matrix.shape}, values range: {np.min(cosine_matrix):.4f} to {np.max(cosine_matrix):.4f}")
        print(f"Number of file basenames: {len(file_basenames)}")
        
        write_matrix_to_file(jaccard_matrix, file_basenames, jac_output_path)
        write_matrix_to_file(cosine_matrix, file_basenames, cos_output_path)

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
            plot_and_save_dendrogram(jaccard_distance_matrix_for_dendro, file_basenames, "Jaccard", args.output_prefix)
            plot_and_save_dendrogram(cosine_distance_matrix_for_dendro, file_basenames, "Cosine", args.output_prefix)
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
                skbio_jaccard_dm_orig = DistanceMatrix(nj_jaccard_dist_matrix_np, ids=file_basenames)
                
                if current_bootstrap_replicates > 0:
                    print(f"Running {current_bootstrap_replicates} bootstrap replicates for Jaccard NJ tree...")
                    jaccard_bootstrap_trees = []
                    for i in range(current_bootstrap_replicates):
                        if (i + 1) % 10 == 0 or i == current_bootstrap_replicates - 1:
                             print(f"  Bootstrap replicate {i+1}/{current_bootstrap_replicates}...")
                        
                        # Create bootstrapped k-mer universe
                        current_bootstrap_kmer_universe = set(random.choices(all_unique_kmers_list, k=len(all_unique_kmers_list)))
                        
                        boot_j_distances = np.zeros((num_files, num_files))
                        
                        for r_idx in range(num_files): # row index
                            for c_idx in range(r_idx + 1, num_files): # column index
                                path1 = valid_file_paths[r_idx]
                                path2 = valid_file_paths[c_idx]
                                
                                kmers1_orig, _ = file_data[path1]
                                kmers2_orig, _ = file_data[path2]

                                kmers1_boot = kmers1_orig.intersection(current_bootstrap_kmer_universe)
                                kmers2_boot = kmers2_orig.intersection(current_bootstrap_kmer_universe)
                                
                                ji_boot = jaccard_index(kmers1_boot, kmers2_boot)
                                dist_boot = 1.0 - ji_boot
                                
                                boot_j_distances[r_idx, c_idx] = dist_boot
                                boot_j_distances[c_idx, r_idx] = dist_boot
                        
                        skbio_boot_j_dm = DistanceMatrix(boot_j_distances, ids=file_basenames)
                        
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
                                final_tree_j = simple_consensus_tree_with_supports(jaccard_bootstrap_trees, file_basenames)
                        else:
                            final_tree_j = simple_consensus_tree_with_supports(jaccard_bootstrap_trees, file_basenames)
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
                skbio_cosine_dm_orig = DistanceMatrix(nj_cosine_dist_matrix_np, ids=file_basenames)

                if current_bootstrap_replicates > 0:
                    print(f"Running {current_bootstrap_replicates} bootstrap replicates for Cosine NJ tree...")
                    cosine_bootstrap_trees = []
                    for i in range(current_bootstrap_replicates):
                        if (i + 1) % 10 == 0 or i == current_bootstrap_replicates -1:
                            print(f"  Bootstrap replicate {i+1}/{current_bootstrap_replicates}...")

                        current_bootstrap_kmer_universe = set(random.choices(all_unique_kmers_list, k=len(all_unique_kmers_list)))
                        boot_c_distances = np.zeros((num_files, num_files))

                        for r_idx in range(num_files): # row index
                            for c_idx in range(r_idx + 1, num_files): # column index
                                path1 = valid_file_paths[r_idx]
                                path2 = valid_file_paths[c_idx]

                                _, cov1_orig = file_data[path1]
                                _, cov2_orig = file_data[path2]

                                cov1_boot = {k: v for k, v in cov1_orig.items() if k in current_bootstrap_kmer_universe}
                                cov2_boot = {k: v for k, v in cov2_orig.items() if k in current_bootstrap_kmer_universe}

                                cs_boot = cosine_similarity_manual(cov1_boot, cov2_boot)
                                dist_boot = 1.0 - cs_boot
                                
                                boot_c_distances[r_idx, c_idx] = dist_boot
                                boot_c_distances[c_idx, r_idx] = dist_boot
                        
                        skbio_boot_c_dm = DistanceMatrix(boot_c_distances, ids=file_basenames)
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
                                final_tree_c = simple_consensus_tree_with_supports(cosine_bootstrap_trees, file_basenames)
                        else:
                            final_tree_c = simple_consensus_tree_with_supports(cosine_bootstrap_trees, file_basenames)
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