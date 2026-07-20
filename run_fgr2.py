#!/usr/bin/env python3

import argparse
import os
import subprocess
import glob
import sys
from concurrent.futures import ProcessPoolExecutor

def build_output_filename(input_file_group, group_index, options):
    """Derive the .fgr2 output filename for a group of input files."""
    first_file_basename = os.path.basename(input_file_group[0])
    hash_suffix = "wang" if options['wang_hash'] else "mm3"
    if len(input_file_group) > 1:
        return f"{first_file_basename}_group{group_index + 1}.c{options['coverage']}.{hash_suffix}.fgr2"
    return f"{first_file_basename}.c{options['coverage']}.{hash_suffix}.fgr2"


def assign_output_filenames(grouped_files_list, options):
    """
    Build one output filename per group, guaranteeing uniqueness.

    Names are derived from the first file's *basename*, so two inputs with the same
    basename in different directories (e.g. runA/reads.fq and runB/reads.fq) would
    otherwise map to the same .fgr2 file and silently overwrite each other.
    """
    names = []
    used = set()
    for idx, group in enumerate(grouped_files_list):
        name = build_output_filename(group, idx, options)
        if name in used:
            stem, _, ext = name.rpartition('.fgr2')
            candidate = f"{stem}_group{idx + 1}.fgr2"
            suffix = 2
            while candidate in used:
                candidate = f"{stem}_group{idx + 1}_{suffix}.fgr2"
                suffix += 1
            print(f"Warning: output name '{name}' already used by another group; "
                  f"writing '{candidate}' instead.", file=sys.stderr)
            name = candidate
        used.add(name)
        names.append(name)
    return names


def process_file_group(input_file_group, group_index, options, output_filename=None):
    """Process a group of input files with a single fgr2 call"""
    if not input_file_group:
        print(f"Warning: Empty file group received for index {group_index}. Skipping.", file=sys.stderr)
        return False

    first_file_in_group = input_file_group[0]
    first_file_basename = os.path.basename(first_file_in_group)

    try:
        if output_filename is None:
            output_filename = build_output_filename(input_file_group, group_index, options)

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
        
        if options['wang_hash']:
            cmd.append("-w")
            
        # Add all input files in the group
        cmd.extend(input_file_group)
        
        group_name_for_log = f"group {group_index + 1} (first file: {first_file_basename})"
        print(f"Running for {group_name_for_log}: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"Error processing {group_name_for_log}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            if result.stdout: # Also print stdout if fgr2 writes errors there
                print(result.stdout, file=sys.stderr)
            return False
        
        print(f"Successfully processed {group_name_for_log} -> {output_filename}")
        return True
    
    except Exception as e:
        error_group_name = f"group {group_index + 1} (first file: {first_file_basename})" if input_file_group else f"group {group_index + 1}"
        print(f"Exception while processing {error_group_name}: {e}", file=sys.stderr)
        return False

def _process_file_group_wrapper(args_tuple):
    """Helper function to unpack arguments for ProcessPoolExecutor.map"""
    return process_file_group(*args_tuple)

def main():
    parser = argparse.ArgumentParser(description="Run fgr2 on multiple input files, potentially in groups.")
    parser.add_argument("input_files", nargs='+', help="Input files or glob patterns")
    parser.add_argument("-k", "--kmer-size", type=int, default=31, help="K-mer size (default: 31)")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads per fgr2 call (default: 4)")
    parser.add_argument("-c", "--coverage", type=int, default=2, help="Coverage cutoff for k-mers (default: 2)")
    parser.add_argument("-N", "--topn", type=int, default=10000, help="Number of top k-mers to include (default: 10000)")
    parser.add_argument("-w", "--wang-hash", action="store_true", help="Use Wang's hash function (default: MurmurHash3)")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Number of fgr2 processes to run in parallel (default: 1)")
    parser.add_argument("-g", "--group-size", type=int, default=1, help="Number of input files per fgr2 call. (default: 1)")

    args = parser.parse_args()

    if args.group_size <= 0:
        print("Warning: --group-size must be positive. Defaulting to 1.", file=sys.stderr)
        args.group_size = 1
    if args.parallel < 1:
        print("Warning: --parallel must be >= 1. Defaulting to 1.", file=sys.stderr)
        args.parallel = 1
    if args.topn < 1:
        parser.error("--topn must be >= 1")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if not 1 <= args.kmer_size <= 31:
        parser.error("--kmer-size must be between 1 and 31")
    if args.coverage < 0:
        parser.error("--coverage must be >= 0")


    expanded_files = []
    for pattern in args.input_files:
        matches = glob.glob(pattern)
        if matches:
            expanded_files.extend(matches)
        else:
            # If glob doesn't match, assume it's a literal filename
            if os.path.exists(pattern):
                expanded_files.append(pattern)
            else:
                print(f"Warning: Input pattern '{pattern}' did not match any files and is not an existing file. Skipping.", file=sys.stderr)
    
    files_to_process = []
    for file_path in expanded_files:
        if file_path not in files_to_process: # Avoid duplicates
            files_to_process.append(file_path)
    
    if not files_to_process:
        print("No input files found to process.", file=sys.stderr)
        return 1
    
    print(f"Found {len(files_to_process)} unique files to process.")

    # Group files
    grouped_files_list = [files_to_process[i:i + args.group_size] for i in range(0, len(files_to_process), args.group_size)]
    num_groups = len(grouped_files_list)
    print(f"Processing in {num_groups} group(s) of up to {args.group_size} file(s) each.")

    options = {
        'kmer_size': args.kmer_size,
        'threads': args.threads,
        'coverage': args.coverage,
        'topn': args.topn,
        'wang_hash': args.wang_hash
    }
    
    output_filenames = assign_output_filenames(grouped_files_list, options)

    success_count = 0
    if args.parallel > 1 and num_groups > 1:
        print(f"Processing up to {args.parallel} groups in parallel.")
        # Prepare arguments for each task: (input_file_group, group_index, options, out_name)
        tasks = [(group, idx, options, output_filenames[idx])
                 for idx, group in enumerate(grouped_files_list)]
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            results = list(executor.map(_process_file_group_wrapper, tasks))
            success_count = sum(1 for r in results if r)
    else:
        if args.parallel > 1 and num_groups <=1:
            print("Note: Parallel processing > 1 specified, but only one group to process. Running sequentially.")
        for idx, group in enumerate(grouped_files_list):
            if process_file_group(group, idx, options, output_filenames[idx]):
                success_count += 1
    
    print(f"Completed: {success_count}/{num_groups} groups processed successfully.")
    
    return 0 if success_count == num_groups else 1

if __name__ == "__main__":
    sys.exit(main()) 