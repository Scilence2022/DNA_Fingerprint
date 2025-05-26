# DNA Fingerprint Programs Overview

## Program Relationship and Workflow

```
DNA Sequence Files (FASTA/FASTQ)
         ↓
    ┌─────────────┐
    │   fgr2.c    │  ← K-mer extraction and fingerprint generation
    │             │
    │ Input:      │
    │ - DNA files │
    │ - Parameters│
    │             │
    │ Output:     │
    │ - .fgr2     │  ← Fingerprint files (k-mer, hash, coverage)
    │ - .hist     │  ← Coverage histogram
    └─────────────┘
         ↓
    ┌─────────────┐
    │calculate_   │  ← Similarity analysis and phylogenetic trees
    │similarity.py│
    │             │
    │ Input:      │
    │ - .fgr/.fgr2│
    │ - Parameters│
    │             │
    │ Output:     │
    │ - .jac/.cos │  ← Similarity matrices
    │ - .png      │  ← Dendrograms
    │ - .newick   │  ← Phylogenetic trees
    └─────────────┘
```

## Program Purposes

### fgr2.c - DNA Fingerprint Generator
**Purpose**: Extract k-mer fingerprints from DNA sequences for efficient comparison

**Key Features**:
- Multi-threaded k-mer extraction from FASTA/FASTQ files
- Two hash function options (MurmurHash3, Wang hash)
- Automatic coverage threshold detection
- Selects top N k-mers with minimal hash values
- Generates coverage histograms for quality assessment

**Output Format**:
```
# .fgr2 file format:
ATCGATCGATCG    12345678901234567890    15
GCTAGCTAGCTA    23456789012345678901    22
...
# k-mer_sequence    hash_value    coverage_count
```

### calculate_similarity.py - Similarity Analysis Tool
**Purpose**: Compare DNA fingerprints and build phylogenetic relationships

**Key Features**:
- Jaccard Index calculation (k-mer presence/absence)
- Cosine Similarity calculation (coverage-weighted)
- Hierarchical clustering dendrograms
- Neighbor-Joining phylogenetic trees with bootstrap support
- Multiple output formats for downstream analysis

**Analysis Types**:
1. **Jaccard Index**: Binary similarity based on shared k-mers
2. **Cosine Similarity**: Weighted similarity considering k-mer coverage
3. **Dendrograms**: Hierarchical clustering visualization
4. **NJ Trees**: Phylogenetic trees with statistical support

## Technical Integration

### Data Flow
1. **fgr2.c** processes raw DNA sequences → generates fingerprint files
2. **calculate_similarity.py** reads fingerprint files → computes relationships

### File Format Compatibility
- Both programs handle `.fgr` and `.fgr2` formats
- Consistent tab-separated format: `k-mer \t hash \t coverage`
- Automatic file extension detection and processing

### Scalability Features
- **fgr2.c**: Multi-threaded processing for large genomic datasets
- **calculate_similarity.py**: Efficient matrix operations with NumPy
- Memory-efficient algorithms for handling thousands of samples

## Use Cases

### 1. Microbial Strain Comparison
```bash
# Generate fingerprints for bacterial genomes
./fgr2 -k 31 -N 10000 -o strain1 strain1.fasta
./fgr2 -k 31 -N 10000 -o strain2 strain2.fasta

# Compare strains and build phylogeny
python calculate_similarity.py -o comparison_results
```

### 2. DNA Storage Data Integrity
```bash
# Generate fingerprints for stored DNA data
./fgr2 -k 25 -c 5 -o original original_data.fq.gz
./fgr2 -k 25 -c 5 -o retrieved retrieved_data.fq.gz

# Assess data integrity
python calculate_similarity.py --standardize -o integrity_check
```

### 3. Large-scale Phylogenetic Analysis
```bash
# Process multiple samples
for file in *.fasta; do
    ./fgr2 -k 31 -N 5000 -o ${file%.fasta} $file
done

# Build comprehensive phylogeny with bootstrap
python calculate_similarity.py --bootstrap_replicates 1000 -o phylogeny
```

## Algorithm Highlights

### fgr2.c Algorithms
- **K-mer Extraction**: Sliding window with canonical k-mer selection
- **Hash-based Storage**: Distributed hash tables for memory efficiency
- **Coverage Analysis**: Automatic threshold detection using histogram analysis
- **Top-K Selection**: Heap-based algorithm for minimal hash value selection

### calculate_similarity.py Algorithms
- **Jaccard Index**: Set intersection/union operations
- **Cosine Similarity**: Vector dot product with optional normalization
- **Bootstrap Analysis**: Random sampling with replacement for statistical confidence
- **Consensus Trees**: Bipartition frequency analysis for tree support values

## Performance Characteristics

### fgr2.c Performance
- **Time Complexity**: O(n) for sequence length n, with multi-threading
- **Space Complexity**: O(k) for number of unique k-mers
- **Throughput**: Can process gigabytes of sequence data efficiently

### calculate_similarity.py Performance
- **Time Complexity**: O(n²) for n samples (pairwise comparisons)
- **Space Complexity**: O(n²) for similarity matrices
- **Bootstrap Scaling**: Linear with number of replicates

## Quality Control Features

### fgr2.c Quality Control
- Coverage histogram generation for threshold validation
- Automatic detection of optimal coverage cutoffs
- Hash function selection for different data characteristics

### calculate_similarity.py Quality Control
- Bootstrap confidence intervals for tree reliability
- Multiple similarity metrics for cross-validation
- Graceful handling of missing or corrupted data

This integrated workflow provides a complete solution for DNA sequence fingerprinting, comparison, and phylogenetic analysis, suitable for applications ranging from data integrity verification to evolutionary studies. 