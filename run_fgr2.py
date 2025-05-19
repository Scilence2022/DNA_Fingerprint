#!/usr/bin/env python3

import argparse
import os
import subprocess
import glob
import sys
from concurrent.futures import ProcessPoolExecutor

def process_file(input_file, options):
    """Process a single input file with fgr2"""
    try:
        # Create output filename according to the specified pattern
        output_filename = f".{input_file}c{options['coverage']}.fgr2.mm.fgr"
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build full path to fgr2 executable
        fgr2_path = os.path.join(script_dir, "fgr2")
        
        # Build the fgr2 command
        cmd = [
            fgr2_path,
            "-k", str(options['kmer_size']),
            "-t", str(options['threads']),
            "-c", str(options['coverage']),
            "-N", str(options['topn']),
            "-o", output_filename
        ]
        
        # Add wang hash option if specified
        if options['wang_hash']:
            cmd.append("-w")
            
        # Add the input file
        cmd.append(input_file)
        
        # Print the command being executed
        print(f"Running: {' '.join(cmd)}")
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for errors
        if result.returncode != 0:
            print(f"Error processing {input_file}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False
        
        print(f"Successfully processed {input_file} -> {output_filename}")
        return True
    
    except Exception as e:
        print(f"Exception while processing {input_file}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run fgr2 on multiple input files")
    parser.add_argument("input_files", nargs='+', help="Input files or glob patterns")
    parser.add_argument("-k", "--kmer-size", type=int, default=31, help="K-mer size (default: 31)")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads per file (default: 4)")
    parser.add_argument("-c", "--coverage", type=int, default=2, help="Coverage cutoff for k-mers (default: 2)")
    parser.add_argument("-N", "--topn", type=int, default=10000, help="Number of top k-mers to include (default: 10000)")
    parser.add_argument("-w", "--wang-hash", action="store_true", help="Use Wang's hash function (default: MurmurHash3)")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Number of files to process in parallel (default: 1)")
    
    args = parser.parse_args()
    
    # Expand any glob patterns in the input files
    expanded_files = []
    for pattern in args.input_files:
        matches = glob.glob(pattern)
        if matches:
            expanded_files.extend(matches)
        else:
            expanded_files.append(pattern)  # Keep the original if no matches
    
    # Remove duplicates while preserving order
    files_to_process = []
    for file in expanded_files:
        if file not in files_to_process:
            files_to_process.append(file)
    
    # Check that we have files to process
    if not files_to_process:
        print("No input files found.", file=sys.stderr)
        return 1
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Create options dictionary
    options = {
        'kmer_size': args.kmer_size,
        'threads': args.threads,
        'coverage': args.coverage,
        'topn': args.topn,
        'wang_hash': args.wang_hash
    }
    
    # Process files (in parallel if requested)
    success_count = 0
    if args.parallel > 1:
        print(f"Processing up to {args.parallel} files in parallel")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            results = list(executor.map(lambda f: process_file(f, options), files_to_process))
            success_count = sum(1 for r in results if r)
    else:
        for file in files_to_process:
            if process_file(file, options):
                success_count += 1
    
    print(f"Completed: {success_count}/{len(files_to_process)} files processed successfully")
    
    return 0 if success_count == len(files_to_process) else 1

if __name__ == "__main__":
    sys.exit(main()) 