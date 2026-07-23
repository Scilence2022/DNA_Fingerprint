"""
Shared ground-truth library for the fgr2 evaluation.

Everything here is *reference* implementation: exact, slow, and independent of
fgr2's C code, so that fgr2 can be checked against it. Where a definition must
agree with fgr2 (canonical k-mer, sketch hash), the agreement is asserted by a
test in `selftest()` rather than assumed.

Canonical k-mer convention (must match fgr2.c:243):
    fgr2 encodes A=0,C=1,G=2,T=3 with the leading base in the high bits and takes
    the numeric min of the forward and reverse-complement encodings. Because the
    2-bit codes are assigned in alphabetical order and the leading base dominates,
    that numeric min is exactly the lexicographic min of the two DNA strings.
"""
from __future__ import annotations

import gzip
import math
import random
import struct
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# Sequence I/O
# --------------------------------------------------------------------------

_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (name, sequence) pairs. Handles plain and gzipped FASTA."""
    op = gzip.open if path.endswith(".gz") else open
    name, chunks = None, []
    with op(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def load_genome(path: str) -> str:
    """Concatenate all records with 'N' separators, so no k-mer spans a contig join."""
    recs = [s for _, s in read_fasta(path)]
    return "N".join(recs)


def write_fasta(path: str, records: Sequence[Tuple[str, str]], width: int = 70) -> None:
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")


# --------------------------------------------------------------------------
# Exact canonical k-mer sets (ground truth)
# --------------------------------------------------------------------------

_VALID = set("ACGT")


def canonical_kmers(seq: str, k: int) -> Set[str]:
    """Exact set of canonical k-mers. k-mers containing non-ACGT are excluded,
    matching fgr2's window reset on ambiguous bases."""
    out: Set[str] = set()
    seq = seq.upper()
    n = len(seq)
    run = 0
    for i, ch in enumerate(seq):
        if ch in _VALID:
            run += 1
            if run >= k:
                km = seq[i - k + 1:i + 1]
                rc = revcomp(km)
                out.add(km if km <= rc else rc)
        else:
            run = 0
    return out


def canonical_kmer_counts(seq: str, k: int) -> Dict[str, int]:
    """Exact canonical k-mer -> occurrence count."""
    out: Dict[str, int] = {}
    seq = seq.upper()
    run = 0
    for i, ch in enumerate(seq):
        if ch in _VALID:
            run += 1
            if run >= k:
                km = seq[i - k + 1:i + 1]
                rc = revcomp(km)
                c = km if km <= rc else rc
                out[c] = out.get(c, 0) + 1
        else:
            run = 0
    return out


def true_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# --- Vectorised exact canonical k-mer sets (for genome-scale ground truth) ---
#
# Returns the same mathematical object as canonical_kmers(), but as a sorted
# numpy array of 2-bit encodings instead of a Python set of strings. Agreement
# with the string implementation is asserted in selftest().

def fast_canonical_codes(seq: str, k: int):
    """Exact canonical k-mer 2-bit encodings as a sorted, deduplicated
    numpy.uint64 array. Windows containing non-ACGT are excluded."""
    import numpy as np

    lut = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    codes = lut[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]

    n = codes.size
    m = n - k + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)

    valid_base = codes != 255
    # A window is valid iff all k of its bases are valid. Rolling sum via cumsum.
    cs = np.concatenate(([0], np.cumsum(valid_base, dtype=np.int64)))
    valid_win = (cs[k:] - cs[:-k]) == k

    safe = np.where(valid_base, codes, 0).astype(np.uint64)
    comp = (np.uint64(3) - safe)

    fwd = np.zeros(m, dtype=np.uint64)
    rev = np.zeros(m, dtype=np.uint64)
    for j in range(k):
        # forward: leading base occupies the high bits
        fwd = (fwd << np.uint64(2)) | safe[j:j + m]
        # reverse complement: base j lands at bit position 2*j
        rev |= comp[j:j + m] << np.uint64(2 * j)

    canon = np.minimum(fwd, rev)
    canon = canon[valid_win]
    return np.unique(canon)


def decode2bit(code: int, k: int) -> str:
    """Inverse of encode2bit."""
    out = []
    for j in range(k - 1, -1, -1):
        out.append("ACGT"[(code >> (2 * j)) & 3])
    return "".join(out)


def jaccard_from_codes(a, b) -> float:
    """Exact Jaccard between two sorted uint64 code arrays."""
    import numpy as np
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 0.0


# --------------------------------------------------------------------------
# Sketch hashes (must reproduce fgr2's selection hash bit-for-bit)
# --------------------------------------------------------------------------

_M64 = (1 << 64) - 1


def encode2bit(kmer: str) -> int:
    """2-bit encode with the leading base in the high bits (fgr2.c:240)."""
    v = 0
    for ch in kmer:
        v = ((v << 2) | "ACGT".index(ch)) & _M64
    return v


def murmur3_fmix64(k: int) -> int:
    """MurmurHash3 64-bit finalizer -- fgr2.c:100 fmix64."""
    k &= _M64
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & _M64
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & _M64
    k ^= k >> 33
    return k


def _rotl64(x: int, r: int) -> int:
    x &= _M64
    return ((x << r) | (x >> (64 - r))) & _M64


def murmur3_x64_64(key: int, seed: int = 42) -> int:
    """Faithful port of fgr2.c:110 MurmurHash3_x64_64 applied to an 8-byte
    uint64 key.

    NOTE for Methods: this is *not* stock MurmurHash3. It runs the x64_128
    mixing schedule and returns only h1, with a hard-coded seed of 42. Because
    len == 8 == one full block, the tail switch is never entered. The body
    therefore executes exactly one block iteration with k1 = key (a native
    little-endian load of the 8 bytes, which on x86-64 and arm64 equals the
    integer value).
    """
    c1 = 0x87C37B91114253D5
    c2 = 0x4CF5AD432745937F
    h1 = seed & _M64
    h2 = seed & _M64

    # Body: exactly one 8-byte block.
    k1 = key & _M64
    k1 = (k1 * c1) & _M64
    k1 = _rotl64(k1, 31)
    k1 = (k1 * c2) & _M64
    h1 ^= k1
    h1 = _rotl64(h1, 27)
    h1 = (h1 + h2) & _M64
    h1 = (h1 * 5 + 0x52DCE729) & _M64
    h2 = _rotl64(h2, 31)
    h2 = (h2 + h1) & _M64
    h2 = (h2 * 5 + 0x38495AB5) & _M64

    # Tail: len & 7 == 0, nothing to do.

    # Finalization.
    length = 8
    h1 ^= length
    h2 ^= length
    h1 = (h1 + h2) & _M64
    h2 = (h2 + h1) & _M64
    h1 = murmur3_fmix64(h1)
    h2 = murmur3_fmix64(h2)
    h1 = (h1 + h2) & _M64
    return h1


def wang_hash(key: int) -> int:
    """Thomas Wang 64-bit integer hash -- fgr2.c:85 hash_wang."""
    key &= _M64
    key = (~key + (key << 21)) & _M64
    key ^= key >> 24
    key = (key + (key << 3) + (key << 8)) & _M64
    key ^= key >> 14
    key = (key + (key << 2) + (key << 4)) & _M64
    key ^= key >> 28
    key = (key + (key << 31)) & _M64
    return key


def sketch_hash(kmer: str, hashname: str = "murmurhash3") -> int:
    v = encode2bit(kmer)
    return murmur3_x64_64(v) if hashname == "murmurhash3" else wang_hash(v)


def reference_bottom_n(kmers: Iterable[str], n: int,
                       hashname: str = "murmurhash3") -> List[Tuple[str, int]]:
    """Brute-force bottom-N sketch: the N smallest sketch-hash values."""
    hashed = sorted(((sketch_hash(km, hashname), km) for km in kmers))
    return [(km, h) for h, km in hashed[:n]]


# --------------------------------------------------------------------------
# fgr2 sketch file parsing
# --------------------------------------------------------------------------

def parse_sketch(path: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    """Return (kmer->hash, kmer->coverage, metadata)."""
    hashes: Dict[str, int] = {}
    cov: Dict[str, int] = {}
    meta: Dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("#fgr2"):
                    for field in line.split("\t")[1:]:
                        if "=" in field:
                            key, val = field.split("=", 1)
                            meta[key] = val
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                hashes[parts[0]] = int(parts[1])
                cov[parts[0]] = int(parts[2])
            except ValueError:
                continue
    return hashes, cov, meta


# --------------------------------------------------------------------------
# Jaccard estimators
# --------------------------------------------------------------------------

def jaccard_direct(a: Set[str], b: Set[str]) -> float:
    """The estimator as implemented in calculate_similarity.py:434 --
    direct intersection-over-union of two independently built bottom-N sketches."""
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def jaccard_union(a_h: Dict[str, int], b_h: Dict[str, int], n: int) -> float:
    """Unbiased bottom-N MinHash estimator (the Mash construction): take the N
    globally smallest hashes across the merged sketch, then ask what fraction of
    those k-mers occur in BOTH sketches."""
    merged = {}
    merged.update(a_h)
    merged.update(b_h)
    if not merged:
        return 1.0
    smallest = sorted(merged.items(), key=lambda kv: kv[1])[:n]
    if not smallest:
        return 1.0
    shared = sum(1 for km, _ in smallest if km in a_h and km in b_h)
    return shared / len(smallest)


def mash_distance(j: float, k: int) -> float:
    """Poisson-corrected mutation-rate estimate. Mash eq. (4)."""
    if j <= 0:
        return 1.0
    if j >= 1:
        return 0.0
    return -(1.0 / k) * math.log(2 * j / (1 + j))


def cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Abundance-weighted cosine over the union index (computed on the
    intersection, which is mathematically identical -- see Methods 3.6)."""
    if not a or not b:
        return 0.0
    na = math.sqrt(sum(float(v) * v for v in a.values()))
    nb = math.sqrt(sum(float(v) * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(float(v) * large[km] for km, v in small.items() if km in large)
    return dot / (na * nb)


# --------------------------------------------------------------------------
# Sequence evolution / read simulation
# --------------------------------------------------------------------------

def mutate(seq: str, rate: float, rng: random.Random) -> str:
    """Independent per-base substitution at `rate` (Jukes-Cantor-like: the new
    base is drawn uniformly from the other three)."""
    if rate <= 0:
        return seq
    out = list(seq)
    alts = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}
    for i, ch in enumerate(out):
        if ch in alts and rng.random() < rate:
            out[i] = rng.choice(alts[ch])
    return "".join(out)


def simulate_reads(genome: str, depth: float, read_len: int, err_rate: float,
                   rng: random.Random, indel_frac: float = 0.0) -> List[str]:
    """Uniform-coverage read sampling with per-base substitution error and an
    optional indel component (for long-read-like profiles)."""
    glen = len(genome)
    if glen <= read_len:
        return []
    n_reads = max(1, int(depth * glen / read_len))
    alts = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}
    reads = []
    for _ in range(n_reads):
        start = rng.randrange(0, glen - read_len)
        frag = genome[start:start + read_len]
        if rng.random() < 0.5:
            frag = revcomp(frag)
        if err_rate > 0:
            buf = []
            for ch in frag:
                r = rng.random()
                if r < err_rate:
                    if indel_frac > 0 and rng.random() < indel_frac:
                        if rng.random() < 0.5:
                            continue                      # deletion
                        buf.append(rng.choice("ACGT"))    # insertion
                        buf.append(ch)
                        continue
                    buf.append(rng.choice(alts.get(ch, "ACGT")))
                else:
                    buf.append(ch)
            frag = "".join(buf)
        reads.append(frag)
    return reads


def write_reads_fasta(path: str, reads: Sequence[str]) -> None:
    with open(path, "w") as fh:
        for i, r in enumerate(reads):
            fh.write(f">r{i}\n{r}\n")


# --------------------------------------------------------------------------
# DNA data storage pool simulation
# --------------------------------------------------------------------------

def make_oligo_pool(n_oligos: int, oligo_len: int, rng: random.Random) -> List[str]:
    """A synthetic encoded pool: random payloads flanked by shared primer sites,
    mirroring the architecture of real DNA storage libraries."""
    fwd = "".join(rng.choice("ACGT") for _ in range(20))
    rev = "".join(rng.choice("ACGT") for _ in range(20))
    payload_len = oligo_len - 40
    pool = []
    for _ in range(n_oligos):
        payload = "".join(rng.choice("ACGT") for _ in range(payload_len))
        pool.append(fwd + payload + rev)
    return pool


def pool_to_reads(pool: Sequence[str], copies: Sequence[int], err_rate: float,
                  rng: random.Random) -> List[str]:
    """Expand a pool into a read set at the given per-oligo copy numbers,
    applying per-base substitution error."""
    alts = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}
    reads = []
    for oligo, c in zip(pool, copies):
        for _ in range(c):
            if err_rate > 0:
                seq = "".join(
                    rng.choice(alts.get(ch, "ACGT")) if rng.random() < err_rate else ch
                    for ch in oligo
                )
            else:
                seq = oligo
            reads.append(seq)
    return reads


def pcr_bias_copies(n: int, base_copies: int, skew: float,
                    rng: random.Random) -> List[int]:
    """Copy numbers under PCR amplification bias, modelled as log-normal
    efficiency variation. skew=0 gives uniform copy number."""
    if skew <= 0:
        return [base_copies] * n
    out = []
    for _ in range(n):
        f = math.exp(rng.gauss(0.0, skew))
        out.append(max(0, int(round(base_copies * f))))
    return out


def dropout_copies(copies: Sequence[int], rate: float,
                   rng: random.Random) -> List[int]:
    """Strand dropout: each oligo species is lost entirely with probability `rate`.
    This is an abundance/presence failure mode distinct from substitution."""
    return [0 if rng.random() < rate else c for c in copies]


# --------------------------------------------------------------------------
# Self-test: assert agreement with the compiled fgr2
# --------------------------------------------------------------------------

def selftest(fgr2_bin: str, workdir: str, verbose: bool = True) -> bool:
    """Verify that this library's canonical-k-mer and sketch-hash definitions
    reproduce fgr2's actual output. Returns True on full agreement."""
    import os
    import subprocess

    os.makedirs(workdir, exist_ok=True)
    rng = random.Random(1234)
    seq = "".join(rng.choice("ACGT") for _ in range(50000))
    fa = os.path.join(workdir, "selftest.fa")
    write_fasta(fa, [("selftest", seq)])

    ok = True
    for k in (21, 25, 31):
        for hashname, flag in (("murmurhash3", []), ("wang", ["-w"])):
            out = os.path.join(workdir, f"selftest_k{k}_{hashname}.fgr2")
            subprocess.run([fgr2_bin, "-k", str(k), "-N", "2000", "-c", "1",
                            "-o", out, fa] + flag,
                           check=True, capture_output=True)
            got_h, _, meta = parse_sketch(out)

            truth = canonical_kmers(seq, k)
            ref = reference_bottom_n(truth, 2000, hashname)
            ref_kmers = {km for km, _ in ref}

            if set(got_h) != ref_kmers:
                ok = False
                if verbose:
                    print(f"  MISMATCH k={k} {hashname}: sketch k-mer sets differ "
                          f"({len(set(got_h) ^ ref_kmers)} symmetric difference)")
            bad = [km for km, h in got_h.items() if sketch_hash(km, hashname) != h]
            if bad:
                ok = False
                if verbose:
                    print(f"  MISMATCH k={k} {hashname}: {len(bad)} hash values differ")
            if verbose and not bad and set(got_h) == ref_kmers:
                print(f"  ok  k={k:2d} {hashname:12s} "
                      f"{len(got_h)} k-mers, hashes and selection both agree")

    # The vectorised enumerator must agree exactly with the string version,
    # including on sequences containing ambiguous bases.
    import numpy as np
    seq_n = seq[:20000] + "NNNN" + seq[20000:40000]
    for k in (21, 31):
        slow = canonical_kmers(seq_n, k)
        fast = fast_canonical_codes(seq_n, k)
        slow_codes = np.array(sorted(encode2bit(x) for x in slow), dtype=np.uint64)
        if slow_codes.size != fast.size or not np.array_equal(slow_codes, fast):
            ok = False
            if verbose:
                print(f"  MISMATCH k={k}: vectorised enumerator disagrees with "
                      f"string version ({slow_codes.size} vs {fast.size})")
        elif verbose:
            print(f"  ok  k={k:2d} vectorised    {fast.size} k-mers, "
                  f"matches string enumerator (incl. N handling)")
    return ok


if __name__ == "__main__":
    import sys
    binp = sys.argv[1] if len(sys.argv) > 1 else "./fgr2"
    wd = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fgrlib_selftest"
    print("fgrlib self-test against compiled fgr2:")
    good = selftest(binp, wd)
    print("RESULT:", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
