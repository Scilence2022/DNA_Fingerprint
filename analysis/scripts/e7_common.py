"""E7 common helpers: vectorised (numpy) ports of fgr2's two selection hashes.

Validated bit-for-bit against fgrlib.murmur3_x64_64 / fgrlib.wang_hash in
e7_validate().
"""
import os
import sys
import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import fgrlib  # noqa: E402

REPO = "/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5"
FGR2 = os.path.join(REPO, "fgr2")
GENOME_DIR = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
              "-claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468"
              "/scratchpad/genomes")
SCRATCH = ("/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint-"
           "-claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468"
           "/scratchpad")
RESULTS = os.path.join(REPO, "analysis", "results")
FIGURES = os.path.join(REPO, "analysis", "figures")

U64 = np.uint64


def _u(x):
    return np.uint64(x & 0xFFFFFFFFFFFFFFFF)


def rotl64(x, r):
    r = np.uint64(r)
    return (x << r) | (x >> np.uint64(64 - r))


def fmix64(k):
    k = k ^ (k >> U64(33))
    k = k * _u(0xFF51AFD7ED558CCD)
    k = k ^ (k >> U64(33))
    k = k * _u(0xC4CEB9FE1A85EC53)
    k = k ^ (k >> U64(33))
    return k


def murmur3_vec(key, seed=42):
    """Vectorised port of fgr2.c MurmurHash3_x64_64 on an 8-byte uint64 key."""
    key = np.asarray(key, dtype=U64)
    c1 = _u(0x87C37B91114253D5)
    c2 = _u(0x4CF5AD432745937F)
    h1 = np.full(key.shape, _u(seed), dtype=U64)
    h2 = np.full(key.shape, _u(seed), dtype=U64)
    k1 = key * c1
    k1 = rotl64(k1, 31)
    k1 = k1 * c2
    h1 = h1 ^ k1
    h1 = rotl64(h1, 27)
    h1 = h1 + h2
    h1 = h1 * U64(5) + _u(0x52DCE729)
    h2 = rotl64(h2, 31)
    h2 = h2 + h1
    h2 = h2 * U64(5) + _u(0x38495AB5)
    h1 = h1 ^ U64(8)
    h2 = h2 ^ U64(8)
    h1 = h1 + h2
    h2 = h2 + h1
    h1 = fmix64(h1)
    h2 = fmix64(h2)
    return h1 + h2


def wang_vec(key):
    """Vectorised port of fgr2.c hash_wang."""
    k = np.asarray(key, dtype=U64).copy()
    k = (~k) + (k << U64(21))
    k = k ^ (k >> U64(24))
    k = k + (k << U64(3)) + (k << U64(8))
    k = k ^ (k >> U64(14))
    k = k + (k << U64(2)) + (k << U64(4))
    k = k ^ (k >> U64(28))
    k = k + (k << U64(31))
    return k


HASHES = {"murmurhash3": murmur3_vec, "wang": wang_vec}


def genome_paths():
    return sorted(os.path.join(GENOME_DIR, f)
                  for f in os.listdir(GENOME_DIR) if f.endswith(".fna"))


def codes_cached(path, k=31):
    """Sorted unique canonical 2-bit codes for a genome, cached on disk."""
    name = os.path.basename(path).replace(".fna", "")
    cache = os.path.join(SCRATCH, "e7_cache", f"{name}.k{k}.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        return np.load(cache)
    seq = fgrlib.load_genome(path)
    codes = fgrlib.fast_canonical_codes(seq, k)
    np.save(cache, codes)
    return codes


def _canon_codes_with_dups(seq, k):
    """Canonical 2-bit codes for every valid k-mer window, WITHOUT dedup.
    Same construction as fgrlib.fast_canonical_codes minus the np.unique."""
    lut = np.full(256, 255, dtype=np.uint8)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    codes = lut[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]
    n = codes.size
    m = n - k + 1
    if m <= 0:
        return np.empty(0, dtype=U64)
    valid_base = codes != 255
    cs = np.concatenate(([0], np.cumsum(valid_base, dtype=np.int64)))
    valid_win = (cs[k:] - cs[:-k]) == k
    safe = np.where(valid_base, codes, 0).astype(U64)
    comp = U64(3) - safe
    fwd = np.zeros(m, dtype=U64)
    rev = np.zeros(m, dtype=U64)
    for j in range(k):
        fwd = (fwd << U64(2)) | safe[j:j + m]
        rev |= comp[j:j + m] << U64(2 * j)
    return np.minimum(fwd, rev)[valid_win]


def counts_from_reads(reads, k, chunk_bp=20_000_000):
    """Exact canonical k-mer occurrence counts over a read set, as
    dict[int code] -> int count. Reads are joined by 'N' so no k-mer spans a
    read junction; chunking keeps peak memory bounded."""
    from collections import defaultdict
    total = defaultdict(int)
    buf, blen = [], 0
    def flush(buf):
        if not buf:
            return
        c = _canon_codes_with_dups("N".join(buf), k)
        if c.size:
            u, n = np.unique(c, return_counts=True)
            for cc, nn in zip(u.tolist(), n.tolist()):
                total[cc] += nn
    for r in reads:
        buf.append(r)
        blen += len(r) + 1
        if blen >= chunk_bp:
            flush(buf); buf, blen = [], 0
    flush(buf)
    return dict(total)


def validate_counts(seed=5):
    """counts_from_reads must equal the reference string counter."""
    import random
    rng = random.Random(seed)
    reads = ["".join(rng.choice("ACGT") for _ in range(80)) for _ in range(60)]
    k = 31
    ref = {}
    for r in reads:
        for km, c in fgrlib.canonical_kmer_counts(r, k).items():
            ref[km] = ref.get(km, 0) + c
    got = counts_from_reads(reads, k, chunk_bp=500)
    got_s = {fgrlib.decode2bit(c, k): n for c, n in got.items()}
    assert got_s == ref, (len(got_s), len(ref))
    return True


def e7_validate(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 2**64, size=n, dtype=np.uint64)
    mv = murmur3_vec(keys, 42)
    wv = wang_vec(keys)
    for i in range(n):
        assert int(mv[i]) == fgrlib.murmur3_x64_64(int(keys[i]), 42), i
        assert int(wv[i]) == fgrlib.wang_hash(int(keys[i])), i
    # non-default seeds
    for s in (0, 7, 12345):
        mv = murmur3_vec(keys[:200], s)
        for i in range(200):
            assert int(mv[i]) == fgrlib.murmur3_x64_64(int(keys[i]), s)
    return True


if __name__ == "__main__":
    old = np.seterr(over="ignore")
    print("vectorised hash validation:", e7_validate())
    print("counts_from_reads validation:", validate_counts())
