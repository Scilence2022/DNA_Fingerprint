# CALCULATE_SIMILARITY.PY Program Flowchart

## Main Program Flow

```
START
  ↓
Parse Command Line Arguments
  ├─ directory: input directory (default: current)
  ├─ -o: output prefix for files
  ├─ --no-plot_trees: disable dendrograms (default: enabled)
  ├─ --no-nj_tree: disable NJ trees (default: enabled)
  ├─ --bootstrap_replicates: bootstrap count (default: 100)
  └─ --standardize: normalize coverage values
  ↓
Check Library Dependencies
  ├─ PLOTTING_AVAILABLE (scipy, matplotlib)
  └─ SKBIO_AVAILABLE (scikit-bio for NJ trees)
  ↓
Find Input Files
  ├─ Search for *.fgr files
  ├─ Search for *.fgr2 files
  └─ Combine and deduplicate
  ↓
Validate Input
  ├─ Need at least 2 files
  └─ Exit if insufficient files
  ↓
Parse All FGR Files
  ↓
Calculate Similarity Matrices
  ↓
Output Results
  ├─ Console output (if no output prefix)
  ├─ Matrix files (.jac, .cos)
  ├─ Dendrograms (.jaccard.png, .cosine.png)
  └─ NJ Trees (.jaccard.nj.newick, .cosine.nj.newick)
  ↓
END
```

## File Parsing Phase

```
Parse All FGR Files
  ↓
For each file in fgr_files_paths:
  ↓
  parse_fgr_file(filepath)
    ↓
    Open file and read line by line
      ↓
    For each line:
      ├─ Split by tab: [kmer, hash, coverage]
      ├─ Extract k-mer sequence
      ├─ Parse coverage value (integer)
      ├─ Add k-mer to set
      └─ Store coverage in dictionary
      ↓
    Return (kmers_set, coverages_dict)
  ↓
  Validate parsed data
    ├─ Check if kmers and coverages not None
    ├─ Add to file_data dictionary
    └─ Add to valid_file_paths list
  ↓
Check if at least 2 valid files
```

## Similarity Calculation Phase

```
Calculate Similarity Matrices
  ↓
Initialize matrices:
  ├─ jaccard_matrix: N×N zeros
  ├─ cosine_matrix: N×N zeros
  └─ Set diagonal to 1.0 (self-similarity)
  ↓
For each pair of files (combinations):
  ↓
  Calculate Jaccard Index:
    ├─ jaccard_index(set1, set2)
    ├─ intersection = set1 ∩ set2
    ├─ union = set1 ∪ set2
    └─ return |intersection| / |union|
  ↓
  Calculate Cosine Similarity:
    ├─ cosine_similarity_manual(dict1, dict2, standardize)
    ├─ Optional: standardize coverage values
    ├─ Create vectors for all unique k-mers
    ├─ Compute dot product and norms
    └─ return dot_product / (norm1 × norm2)
  ↓
  Store in matrices:
    ├─ jaccard_matrix[i,j] = jaccard_matrix[j,i] = ji
    └─ cosine_matrix[i,j] = cosine_matrix[j,i] = cs
```

## Output Generation Phase

```
Output Results
  ↓
┌─────────────────────────────────────────────────────┐
│ Matrix File Output (if output_prefix specified)     │
│   ↓                                                 │
│ write_matrix_to_file()                              │
│   ├─ Create .jac file (Jaccard matrix)              │
│   ├─ Create .cos file (Cosine matrix)               │
│   ├─ Format: tab-separated with headers             │
│   └─ Verify file integrity                          │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ Dendrogram Generation (if plot_trees enabled)       │
│   ↓                                                 │
│ Convert similarity to distance (1 - similarity)     │
│   ↓                                                 │
│ plot_and_save_dendrogram()                          │
│   ├─ Check if plotting libraries available          │
│   ├─ Convert to condensed distance matrix           │
│   ├─ Perform hierarchical clustering (linkage)      │
│   ├─ Generate dendrogram plot                       │
│   ├─ Save as .jaccard.png and .cosine.png           │
│   └─ Handle edge cases (single item, etc.)          │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ Neighbor-Joining Trees (if nj_tree enabled)         │
│   ↓                                                 │
│ Check scikit-bio availability                       │
│   ↓                                                 │
│ Prepare Bootstrap Data (if bootstrap > 0)           │
│   ├─ Collect all unique k-mers                      │
│   └─ Create k-mer universe list                     │
│   ↓                                                 │
│ Generate Bootstrap Trees                            │
│   ↓                                                 │
│ Build Consensus Tree                                │
│   ↓                                                 │
│ Save Newick Format Trees                            │
└─────────────────────────────────────────────────────┘
```

## Bootstrap and NJ Tree Generation

```
Bootstrap Process (for each distance type: Jaccard/Cosine)
  ↓
For i in range(bootstrap_replicates):
  ↓
  Create Bootstrap K-mer Universe:
    ├─ Random sample with replacement
    └─ Same size as original k-mer set
  ↓
  Calculate Bootstrap Distance Matrix:
    ├─ For each file pair:
    ├─ Intersect with bootstrap universe
    ├─ Calculate similarity on subset
    └─ Convert to distance (1 - similarity)
  ↓
  Build NJ Tree:
    ├─ Create DistanceMatrix object
    ├─ Validate (no NaN/Inf, not all zeros)
    ├─ Call nj() function
    └─ Add to bootstrap_trees list
  ↓
Build Consensus Tree:
  ├─ If majority_consensus available: use it
  ├─ Else: use simple_consensus_tree_with_supports()
  ├─ Assign support values to nodes
  └─ Return final tree with bootstrap supports
  ↓
Save Tree:
  ├─ Format as Newick string
  └─ Write to .newick file
```

## Consensus Tree Building (Custom Implementation)

```
simple_consensus_tree_with_supports(bootstrap_trees, ids)
  ↓
Use first tree as reference template
  ↓
Extract Bipartitions from All Trees:
  ├─ For each internal node in each tree:
  ├─ Get set of descendant tip names
  ├─ Create bipartition (frozenset)
  └─ Count frequency across all trees
  ↓
Annotate Reference Tree:
  ├─ For each internal node:
  ├─ Find corresponding bipartition
  ├─ Calculate support = count / total_trees
  └─ Assign node.support value
  ↓
Return annotated tree
```

## Key Functions and Data Flow

```
Core Similarity Functions:
├─ jaccard_index(set1, set2)
│   └─ |intersection| / |union|
├─ cosine_similarity_manual(dict1, dict2, standardize)
│   ├─ Optional standardization: scale to 100 × dict_size
│   ├─ Create vectors for all k-mers
│   └─ Compute cosine similarity
└─ strip_extensions(filename)
    └─ Remove .fgr/.fgr2 for clean display

Visualization Functions:
├─ plot_and_save_dendrogram()
│   ├─ scipy.cluster.hierarchy.linkage()
│   ├─ scipy.cluster.hierarchy.dendrogram()
│   └─ matplotlib.pyplot.savefig()
└─ write_matrix_to_file()
    ├─ Tab-separated format
    ├─ Header row with filenames
    └─ Data rows with similarity values

Tree Building Functions:
├─ skbio.tree.nj() - Neighbor-Joining algorithm
├─ skbio.tree.majority_consensus() - Consensus tree
└─ simple_consensus_tree_with_supports() - Fallback consensus
```

## Data Structures

```
Main Data Containers:
├─ file_data: {filepath: (kmers_set, coverages_dict)}
├─ jaccard_matrix: N×N numpy array
├─ cosine_matrix: N×N numpy array
├─ clean_basenames: list of display names
└─ valid_file_paths: list of successfully parsed files

K-mer Data:
├─ kmers_set: set of k-mer sequences
├─ coverages_dict: {kmer: coverage_count}
└─ all_unique_kmers: union of all k-mers (for bootstrap)

Tree Data:
├─ bootstrap_trees: list of TreeNode objects
├─ DistanceMatrix: scikit-bio distance matrix
└─ TreeNode: scikit-bio tree with support values
```

## Error Handling and Edge Cases

```
Input Validation:
├─ Check minimum 2 files
├─ Validate file parsing success
├─ Handle malformed lines gracefully
└─ Skip files with parsing errors

Library Dependencies:
├─ Graceful degradation if scipy/matplotlib missing
├─ Skip plotting if libraries unavailable
├─ Fallback consensus if majority_consensus missing
└─ Clear error messages for missing dependencies

Bootstrap Robustness:
├─ Skip bootstrap replicates with invalid distances
├─ Handle all-zero distance matrices
├─ Continue if some bootstrap trees fail
└─ Fallback to original tree if no bootstrap trees succeed
``` 