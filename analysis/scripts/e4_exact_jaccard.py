#!/usr/bin/env python3
"""E4 step 1: exact pairwise Jaccard on full canonical k-mer sets (k=31)."""
import sys, os, json, itertools, time
sys.path.insert(0, '/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5/analysis/scripts')
import numpy as np
import fgrlib

GEN = '/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint--claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad/genomes'
OUT = '/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint--claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad/e4'
K = 31

names = sorted(os.path.splitext(f)[0] for f in os.listdir(GEN) if f.endswith('.fna'))
codes = {}
sizes = {}
for n in names:
    t0 = time.time()
    seq = fgrlib.load_genome(os.path.join(GEN, n + '.fna'))
    c = fgrlib.fast_canonical_codes(seq, K)
    codes[n] = c
    sizes[n] = int(c.size)
    print(f"{n}: len={len(seq)} distinct_{K}mers={c.size} ({time.time()-t0:.1f}s)", flush=True)
    del seq

pairs = {}
for a, b in itertools.combinations(names, 2):
    j = fgrlib.jaccard_from_codes(codes[a], codes[b])
    pairs[f"{a}|{b}"] = float(j)

json.dump({'k': K, 'names': names, 'genome_distinct_kmer_counts': sizes,
           'exact_jaccard': pairs},
          open(os.path.join(OUT, 'exact_jaccard_k31.json'), 'w'), indent=1)
print("wrote exact_jaccard_k31.json;", len(pairs), "pairs")
