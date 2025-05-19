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
    parser = argparse.ArgumentParser(description="Calculate Jaccard Index and Cosine Similarity between .fgr files, and optionally plot relationship trees.")
    parser.add_argument("directory", nargs='?', default=".", help="Directory containing .fgr files (default: current directory)")
    parser.add_argument("-o", "--output_prefix", help="Output prefix for similarity matrix and dendrogram files. If provided, matrices and plots are saved to files.")
    parser.add_argument("--plot_trees", action='store_true', help="Enable generation of relationship trees (dendrograms).")
    args = parser.parse_args()

    # Check if plotting is requested but not available
    if args.plot_trees and not PLOTTING_AVAILABLE:
        print("WARNING: You requested dendrograms (--plot_trees) but required libraries are missing.")
        print("Please install scipy and matplotlib: pip install scipy matplotlib")
        print("Continuing with similarity calculation only...\n")

    fgr_dir = args.directory
    fgr_files_paths = sorted(glob.glob(os.path.join(fgr_dir, "*.fgr"))) 

    if len(fgr_files_paths) < 2:
        print("Need at least two .fgr files to compare.")
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
        jaccard_distance_matrix = 1 - jaccard_matrix
        cosine_distance_matrix = 1 - cosine_matrix

        # Ensure diagonal is 0 for distance matrices
        np.fill_diagonal(jaccard_distance_matrix, 0)
        np.fill_diagonal(cosine_distance_matrix, 0)

        # Plot Jaccard Dendrogram
        if num_files >=2:
            plot_and_save_dendrogram(jaccard_distance_matrix, file_basenames, "Jaccard", args.output_prefix)
            plot_and_save_dendrogram(cosine_distance_matrix, file_basenames, "Cosine", args.output_prefix)
        else:
            print("Skipping dendrogram generation as there are less than 2 files to compare.")

if __name__ == "__main__":
    main() 