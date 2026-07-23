#!/usr/bin/env python3
"""Vectorised uint64 port of fgrlib.murmur3_x64_64 (= fgr2's selection hash).
Validated element-for-element against the scalar version in f1_synthetic.py."""
import numpy as np

U = np.uint64
_C1 = U(0x87C37B91114253D5)
_C2 = U(0x4CF5AD432745937F)


def _rotl(x, r):
    r = U(r)
    return (x << r) | (x >> U(64 - r))


def murmur3_vec(keys, seed=42):
    with np.errstate(over="ignore"):
        k1 = np.asarray(keys, dtype=np.uint64).copy()
        h1 = np.full(k1.shape, seed, dtype=np.uint64)
        h2 = np.full(k1.shape, seed, dtype=np.uint64)
        k1 = k1 * _C1
        k1 = _rotl(k1, 31)
        k1 = k1 * _C2
        h1 = h1 ^ k1
        h1 = _rotl(h1, 27)
        h1 = h1 + h2
        h1 = h1 * U(5) + U(0x52DCE729)
        h2 = _rotl(h2, 31)
        h2 = h2 + h1
        h2 = h2 * U(5) + U(0x38495AB5)
        h1 = h1 ^ U(8)
        h2 = h2 ^ U(8)
        h1 = h1 + h2
        h2 = h2 + h1
        h1 = _fmix(h1)
        h2 = _fmix(h2)
        return h1 + h2


def _fmix(k):
    k = k ^ (k >> U(33))
    k = k * U(0xFF51AFD7ED558CCD)
    k = k ^ (k >> U(33))
    k = k * U(0xC4CEB9FE1A85EC53)
    k = k ^ (k >> U(33))
    return k
