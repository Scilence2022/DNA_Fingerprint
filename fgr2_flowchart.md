# FGR2.C Program Flowchart

## Main Program Flow

```
START
  ↓
Parse Command Line Arguments
  ├─ k: k-mer size (default: 31)
  ├─ p: prefix length (default: 14)
  ├─ b: block size (default: 10M)
  ├─ t: number of threads (default: 4)
  ├─ N: number of k-mers to output (default: 10000)
  ├─ c: coverage threshold (default: auto-detect)
  ├─ w: use Wang hash (default: MurmurHash3)
  └─ o: output filename
  ↓
Validate Arguments
  ├─ Check if input file provided
  ├─ Check if p >= KC_BITS (14)
  └─ Exit if invalid
  ↓
Select Hash Function
  ├─ If -w flag: use hash_wang()
  └─ Else: use hash_murmur3()
  ↓
COUNT_FILE() - Process Input File
  ↓
Generate Coverage Histogram
  ↓
Auto-detect Coverage Threshold (if not specified)
  ↓
Write Histogram to File
  ↓
Select Top N k-mers
  ↓
Output Results
  ↓
Cleanup Memory
  ↓
END
```

## COUNT_FILE() Function - Multi-threaded K-mer Counting

```
count_file(filename, k, p, block_size, n_thread)
  ↓
Open Input File (FASTA/FASTQ, gzipped or not)
  ↓
Initialize Hash Table Structure (kc_c4x_t)
  ├─ Create 2^p hash tables
  └─ Each table stores k-mers with coverage counts
  ↓
Start 3-Stage Pipeline (kt_pipeline)
  ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 1: Read Sequences                             │
│ worker_pipeline(step=0)                             │
│   ↓                                                 │
│ Read sequences in blocks                            │
│   ├─ Read until block_size reached                  │
│   ├─ Skip sequences shorter than k                  │
│   └─ Store sequences in stepdat_t                   │
│   ↓                                                 │
│ Return stepdat_t with sequences                     │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 2: Extract K-mers                             │
│ worker_pipeline(step=1)                             │
│   ↓                                                 │
│ For each sequence:                                  │
│   ↓                                                 │
│ count_seq_buf() - Extract k-mers                    │
│   ├─ Sliding window of size k                       │
│   ├─ Convert ACGT to 0123                          │
│   ├─ Generate forward and reverse complement        │
│   ├─ Take canonical k-mer (lexicographically smaller)│
│   ├─ Hash k-mer using hash64()                      │
│   └─ Store in buffer by prefix                      │
│   ↓                                                 │
│ Return stepdat_t with k-mer buffers                 │
└─────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────┐
│ STAGE 3: Insert to Hash Tables                      │
│ worker_pipeline(step=2)                             │
│   ↓                                                 │
│ Multi-threaded insertion (kt_for)                   │
│   ↓                                                 │
│ worker_for() - For each hash table:                 │
│   ├─ Insert k-mers from buffer                      │
│   ├─ Increment coverage count                       │
│   └─ Handle hash collisions                         │
│   ↓                                                 │
│ Free buffers                                        │
└─────────────────────────────────────────────────────┘
  ↓
Return populated hash table structure
```

## Coverage Histogram Generation

```
Generate Histogram
  ↓
Allocate histogram arrays for each thread
  ↓
Multi-threaded histogram calculation (kt_for)
  ↓
worker_hist() - For each hash table partition:
  ├─ Iterate through all k-mers
  ├─ Extract coverage count
  └─ Increment histogram bin
  ↓
Merge histograms from all threads
  ↓
find_first_increasing_coverage()
  ├─ Skip low coverage noise
  ├─ Find first increasing point after decrease
  └─ Return detected threshold
  ↓
print_hist() - Write histogram to .hist file
```

## Top K-mer Selection

```
select_top_kmers(h, N, coverage_threshold, k, output_fp)
  ↓
Initialize max-heap of size N
  ↓
For each hash table partition:
  ↓
  For each k-mer in partition:
    ↓
    Check if coverage >= threshold
      ↓ (Yes)
    Reconstruct original k-mer:
      ├─ Extract hashed k-mer from key
      ├─ Use hash64i() to invert hash
      └─ Compute final hash with selected function
      ↓
    Heap management:
      ├─ If heap not full: add k-mer
      ├─ If hash < max_heap_top: replace top
      └─ Maintain heap property
  ↓
Sort heap by hash values (qsort)
  ↓
Output k-mers:
  ├─ Convert k-mer integer to sequence
  ├─ Format: sequence \t hash_value \t coverage
  └─ Write to output file
```

## Key Data Structures

```
kc_c4x_t: Main hash table structure
  ├─ p: prefix length
  └─ h: array of 2^p hash tables

kc_c4_t: Individual hash table
  ├─ Stores k-mer keys with coverage
  └─ Key format: [hashed_k-mer][coverage_count]

buf_c4_t: Buffer for k-mer collection
  ├─ n, m: current and max size
  └─ a: array of hashed k-mers

stepdat_t: Pipeline step data
  ├─ sequences and lengths
  └─ k-mer buffers

kmer_t: K-mer with metadata
  ├─ y: original k-mer integer
  ├─ hash: hash value
  └─ count: coverage
```

## Hash Functions

```
Two hash function options:

1. MurmurHash3 (default):
   ├─ MurmurHash3_x64_64()
   ├─ High quality, good distribution
   └─ Seed = 42

2. Wang Hash (-w flag):
   ├─ hash_wang()
   ├─ Thomas Wang's 64-bit hash
   └─ Faster but simpler

Internal hash functions:
├─ hash64(): Invertible hash for storage
└─ hash64i(): Inverse of hash64()
``` 