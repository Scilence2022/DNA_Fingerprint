# FGR2 - Fast Genome Representation

FGR2 is a high-performance tool for extracting fingerprints of DNA data storage. It efficiently processes both FASTA and FASTQ files, including their gzipped variants.

## Features

- Fast k-mer counting and analysis
- Support for both FASTA and FASTQ formats (including gzipped files)
- Multi-threaded processing for improved performance
- Automatic coverage threshold detection
- Flexible output options including k-mer histograms
- Choice of hash functions (MurmurHash3 or Thomas Wang's hash)

## Installation

### Prerequisites

- C compiler (GCC recommended)
- zlib development library
- make

### Building from Source

```bash
git clone https://github.com/Scilence2022/DNA_Fingerprint.git
cd DNA_Fingerprint
make
```

## Usage

```bash
fgr2 [options] <in.fa|in.fq|in.fa.gz|in.fq.gz>
```

### Options

- `-k INT`     k-mer size [default: 31]
- `-p INT`     prefix length [default: 14]
- `-b INT`     block size [default: 10000000]
- `-t INT`     number of worker threads [default: 4]
- `-N INT`     number of k-mers to output [default: 10000]
- `-c INT`     minimum coverage threshold [default: auto-detect]
- `-w`         use Thomas Wang's hash function (default: MurmurHash3)
- `-o FILE`    Output file for top N k-mers and coverage histogram

### Input Formats

FGR2 supports the following input formats:
- FASTA format (*.fa, *.fa.gz)
- FASTQ format (*.fq, *.fq.gz)

### Output Files

When using the `-o` option, FGR2 generates two files:
1. The specified output file containing the top N k-mers with minimal hash values
2. A histogram file (*.hist) containing k-mer coverage distribution

## Examples

### Basic Usage

```bash
# Process a FASTA file with default settings
fgr2 input.fa

# Process a gzipped FASTQ file with custom k-mer size
fgr2 -k 25 input.fq.gz

# Process with custom coverage threshold
fgr2 -c 10 input.fa

# Save results to specific output file
fgr2 -o results.txt input.fa
```

### Advanced Usage

```bash
# Use multiple threads and custom block size
fgr2 -t 8 -b 20000000 input.fa

# Use Thomas Wang's hash function
fgr2 -w input.fa

# Output top 1000 k-mers with custom coverage
fgr2 -N 1000 -c 5 input.fa
```

## Output Format

### K-mer Output File

The main output file (when using -o) contains:
- One k-mer per line
- Format: `<sequence>\t<hash_value>\t<coverage>`

### Histogram File

The histogram file (*.hist) contains:
- Coverage distribution data
- Format: `<coverage>\t<number_of_kmers>`

## Performance Considerations

- Increase `-t` for better performance on multi-core systems
- Adjust `-b` based on available memory
- Larger `-k` values require more memory but may provide better specificity

## Author

Lifu Song <songlf@tib.cas.cn>

## License

[Specify your license here]

## Citation

If you use FGR2 in your research, please cite:
[Add citation information here]