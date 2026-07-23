#!/usr/bin/env python3
"""Vectorised read simulator + streaming k-mer counter for F2 (full-genome E3 re-run).

The read-generation MODEL is identical to fgrlib.simulate_reads:
  n_reads = max(1, int(depth*glen/read_len)); start ~ U[0, glen-read_len);
  revcomp with p=0.5; per base P(error)=err_rate; given an error,
  P(indel)=indel_frac (50% deletion / 50% insertion of a uniform base BEFORE the
  base), else substitution uniform over the 3 other bases.
Only the RNG stream differs (numpy Generator instead of random.Random), which is
required for tractability at ~1 Gbp scale.  f2_validate.py checks distributional
equivalence against fgrlib.

Everything is chunked so peak memory stays bounded regardless of depth.
"""
import numpy as np

CODE2CH = np.frombuffer(b"ACGT", dtype=np.uint8)
SEP = np.uint8(255)

# Multi-contig genomes are joined with 'N' by fgrlib.load_genome, so reads can
# contain non-ACGT bases.  fgrlib.simulate_reads emits them as 'N'; codes 4..255
# all map to 'N' here so the FASTA is byte-identical in those positions.
CODE2CH_FULL = np.full(256, ord("N"), dtype=np.uint8)
CODE2CH_FULL[:4] = CODE2CH


def genome_codes(seq: str) -> np.ndarray:
    lut = np.full(256, SEP, dtype=np.uint8)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    return lut[np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)]


def canonical_codes_from_bytes(codes: np.ndarray, k: int) -> np.ndarray:
    """Canonical k-mer codes from a byte array of 0..3 with SEP(255) separators.
    Same convention as fgrlib.fast_canonical_codes; multiplicities preserved."""
    n = codes.size
    m = n - k + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    valid_base = codes != SEP
    cs = np.concatenate(([0], np.cumsum(valid_base, dtype=np.int64)))
    valid_win = (cs[k:] - cs[:-k]) == k
    safe = np.where(valid_base, codes, 0).astype(np.uint64)
    comp = np.uint64(3) - safe
    fwd = np.zeros(m, dtype=np.uint64)
    rev = np.zeros(m, dtype=np.uint64)
    for j in range(k):
        fwd <<= np.uint64(2)
        fwd |= safe[j:j + m]
        rev |= comp[j:j + m] << np.uint64(2 * j)
    np.minimum(fwd, rev, out=fwd)
    return fwd[valid_win]


def _simulate_chunk(gcodes, n_reads, read_len, err_rate, indel_frac, rng):
    """Return (seq_codes_concatenated, lens) for n_reads reads."""
    glen = gcodes.size
    starts = rng.integers(0, glen - read_len, size=n_reads)
    mat = gcodes[starts[:, None] + np.arange(read_len)[None, :]]
    rc = rng.random(n_reads) < 0.5
    if rc.any():
        sub = mat[rc]
        valid = sub != SEP
        comp = np.where(valid, 3 - np.minimum(sub, 3), SEP).astype(np.uint8)
        mat[rc] = comp[:, ::-1]
    flat = mat.reshape(-1).copy()
    del mat

    lens = np.full(n_reads, read_len, dtype=np.int64)
    if err_rate > 0:
        n = flat.size
        # fgrlib applies errors to N too (alts.get(ch, "ACGT")), so no mask here
        is_err = rng.random(n) < err_rate
        if indel_frac > 0:
            is_indel = is_err & (rng.random(n) < indel_frac)
            is_del = is_indel & (rng.random(n) < 0.5)
            is_ins = is_indel & ~is_del
        else:
            is_indel = np.zeros(n, dtype=bool)
            is_del = is_indel
            is_ins = is_indel
        is_sub = is_err & ~is_indel
        if is_sub.any():
            b = flat[is_sub].astype(np.int16)
            newb = ((b + 1 + rng.integers(0, 3, size=b.size)) % 4).astype(np.uint8)
            # a substituted N becomes a uniform draw over ACGT, as in fgrlib
            isn = b > 3
            if isn.any():
                newb[isn] = rng.integers(0, 4, size=int(isn.sum())).astype(np.uint8)
            flat[is_sub] = newb
        if indel_frac > 0 and (is_ins.any() or is_del.any()):
            lens = (read_len
                    - is_del.reshape(n_reads, read_len).sum(1)
                    + is_ins.reshape(n_reads, read_len).sum(1)).astype(np.int64)
            rep = np.ones(n, dtype=np.int8)
            rep[is_del] = 0
            rep[is_ins] = 2
            out = np.repeat(flat, rep)
            ends = np.cumsum(rep, dtype=np.int64)
            ins_pos = ends[is_ins] - 2
            out[ins_pos] = rng.integers(0, 4, size=ins_pos.size).astype(np.uint8)
            flat = out
    return flat, lens


def _sep_join(seq, lens):
    """Insert SEP between reads so no k-mer window spans two reads."""
    n = lens.size
    total = int(lens.sum()) + n - 1
    out = np.full(total, SEP, dtype=np.uint8)
    seq_start = np.concatenate(([0], np.cumsum(lens[:-1])))
    out_start = seq_start + np.arange(n)
    within = np.arange(seq.size, dtype=np.int64) - np.repeat(seq_start, lens)
    out[np.repeat(out_start, lens) + within] = seq
    return out


def _fasta_bytes(seq, lens, first_id):
    n = lens.size
    hdr_w = 14                      # b">r" + 11 digits + b"\n"
    seg = hdr_w + lens + 1
    starts = np.concatenate(([0], np.cumsum(seg[:-1])))
    out = np.empty(int(seg.sum()), dtype=np.uint8)
    hdr = np.empty((n, hdr_w), dtype=np.uint8)
    hdr[:, 0] = ord(">")
    hdr[:, 1] = ord("r")
    hdr[:, -1] = ord("\n")
    ids = np.arange(first_id, first_id + n, dtype=np.int64)
    for d in range(11):
        hdr[:, 12 - d] = (ids // (10 ** d)) % 10 + ord("0")
    out[(starts[:, None] + np.arange(hdr_w)[None, :]).reshape(-1)] = hdr.reshape(-1)
    seq_start = np.concatenate(([0], np.cumsum(lens[:-1])))
    within = np.arange(seq.size, dtype=np.int64) - np.repeat(seq_start, lens)
    out[np.repeat(starts + hdr_w, lens) + within] = CODE2CH_FULL[seq]
    out[starts + hdr_w + lens] = ord("\n")
    return out


def simulate_stream(gcodes, depth, read_len, err_rate, indel_frac, rng,
                    chunk_reads=200_000):
    """Yield (seq_codes, lens) chunks covering the whole read set."""
    glen = gcodes.size
    n_reads = max(1, int(depth * glen / read_len))
    done = 0
    while done < n_reads:
        m = min(chunk_reads, n_reads - done)
        yield _simulate_chunk(gcodes, m, read_len, err_rate, indel_frac, rng), done
        done += m
