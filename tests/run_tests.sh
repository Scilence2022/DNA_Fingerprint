#!/usr/bin/env bash
#
# Regression tests for the DNA_Fingerprint toolchain.
#
# Each test pins down a defect that was previously shipped, so a regression fails
# loudly rather than silently corrupting fingerprints. Run with `make check`.
#
# Requires: a built ./fgr2 and python3 with numpy.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FGR2="$REPO_DIR/fgr2"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL + 1)); }
info() { printf '\n== %s ==\n' "$1"; }

if [ ! -x "$FGR2" ]; then
    echo "error: $FGR2 not found or not executable. Run 'make' first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
python3 - "$WORK" <<'PY'
import random, sys, os
random.seed(20240501)
work = sys.argv[1]
base = ''.join(random.choice('ACGT') for _ in range(20000))

def mutate(seq, rate):
    out = list(seq)
    for i in range(len(out)):
        if random.random() < rate:
            out[i] = random.choice('ACGT')
    return ''.join(out)

# near: 0.5% divergence, far: 20% divergence
for name, seq in (('base', base), ('near', mutate(base, 0.005)), ('far', mutate(base, 0.20))):
    with open(os.path.join(work, name + '.fa'), 'w') as fh:
        fh.write('>%s\n%s\n' % (name, seq))

with open(os.path.join(work, 'expected_base.txt'), 'w') as fh:
    fh.write(base)
PY

info "fgr2: fingerprint integrity"

# The decisive test for the k-mer encoding round-trip. Before the c4x_insert_buf
# fix, the emitted k-mers were reconstructed from a truncated key and NONE of them
# occurred in the input at all.
"$FGR2" -k 31 -N 500 -c 1 -o "$WORK/base.fgr2" "$WORK/base.fa" >/dev/null 2>&1
if python3 - "$WORK" <<'PY'
import sys
work = sys.argv[1]
genome = open(work + '/expected_base.txt').read().strip()
rc = lambda s: s.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
kmers = [l.split('\t')[0] for l in open(work + '/base.fgr2') if not l.startswith('#')]
missing = [k for k in kmers if k not in genome and rc(k) not in genome]
sys.exit(0 if kmers and not missing else 1)
PY
then ok "every emitted k-mer occurs in the input sequence"
else bad "emitted k-mers are not present in the input (k-mer reconstruction broken)"
fi

# The max-heap must yield exactly the N smallest hashes. A broken heapify_down
# silently returns a non-minimal set, which is not a valid MinHash sketch.
"$FGR2" -k 31 -N 1000000 -c 1 -o "$WORK/all.fgr2" "$WORK/base.fa" >/dev/null 2>&1
"$FGR2" -k 31 -N 100     -c 1 -o "$WORK/top100.fgr2" "$WORK/base.fa" >/dev/null 2>&1
grep -v '^#' "$WORK/all.fgr2" | cut -f2 | sort -n | head -100 | sort > "$WORK/truth.txt"
grep -v '^#' "$WORK/top100.fgr2" | cut -f2 | sort > "$WORK/got.txt"
if [ "$(comm -12 "$WORK/truth.txt" "$WORK/got.txt" | wc -l | tr -d ' ')" = "100" ]; then
    ok "bottom-N selection returns exactly the N minimal hashes"
else
    bad "bottom-N selection is not the true minimum set (heap property violated)"
fi

# Output must be sorted ascending by hash.
if grep -v '^#' "$WORK/top100.fgr2" | cut -f2 | sort -c -n 2>/dev/null; then
    ok "output is sorted ascending by hash value"
else
    bad "output is not sorted by hash value"
fi

# Determinism: identical input must give byte-identical sketches.
"$FGR2" -k 31 -N 200 -c 1 -o "$WORK/d1.fgr2" "$WORK/base.fa" >/dev/null 2>&1
"$FGR2" -k 31 -N 200 -c 1 -o "$WORK/d2.fgr2" "$WORK/base.fa" >/dev/null 2>&1
if cmp -s "$WORK/d1.fgr2" "$WORK/d2.fgr2"; then
    ok "repeated runs produce identical sketches"
else
    bad "sketch generation is not deterministic"
fi

# Self-describing header, used by calculate_similarity.py for compatibility checks.
if head -1 "$WORK/base.fgr2" | grep -q 'k=31' && head -1 "$WORK/base.fgr2" | grep -q 'hash=murmurhash3'; then
    ok "output carries a self-describing metadata header"
else
    bad "metadata header missing or malformed"
fi

info "fgr2: stdout mode"

# With no -o, stdout must carry ONLY the fingerprint: status text and the histogram
# previously interleaved into it, corrupting `fgr2 in.fa > out.fgr2`.
"$FGR2" -k 31 -N 5 -c 1 "$WORK/base.fa" 2>/dev/null > "$WORK/stdout.fgr2"
# Every non-comment line must be a well-formed record: <ACGT k-mer>\t<hash>\t<cov>.
# Status text and histogram rows ("4\t30") both fail this shape. Comment lines are
# excluded because the metadata header legitimately contains words like "threshold".
if [ "$(grep -vc '^#' "$WORK/stdout.fgr2")" = "5" ] \
   && [ "$(grep -v '^#' "$WORK/stdout.fgr2" | grep -cE '^[ACGT]+	[0-9]+	[0-9]+$')" = "5" ]; then
    ok "stdout contains only the fingerprint (no status/histogram noise)"
else
    bad "stdout is polluted with status or histogram lines"
fi

info "fgr2: argument validation and error handling"

# Previously a NULL dereference -> segfault.
"$FGR2" -o "$WORK/x" "$WORK/definitely_missing.fa" >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 1 ]; then ok "missing input file exits cleanly (rc=1, no segfault)"
else bad "missing input file returned rc=$rc (expected 1; 139 = segfault)"; fi

# k >= 32 would shift a uint64 by >= 64 bits (undefined behaviour).
"$FGR2" -k 32 -o "$WORK/x" "$WORK/base.fa" >/dev/null 2>&1
[ $? -eq 1 ] && ok "rejects -k 32 (out of range)" || bad "accepted -k 32"

"$FGR2" -k 0 -o "$WORK/x" "$WORK/base.fa" >/dev/null 2>&1
[ $? -eq 1 ] && ok "rejects -k 0" || bad "accepted -k 0"

"$FGR2" -t 0 -o "$WORK/x" "$WORK/base.fa" >/dev/null 2>&1
[ $? -eq 1 ] && ok "rejects -t 0" || bad "accepted -t 0"

# All sequences shorter than k: empty histogram must not read out of bounds.
printf '>tiny\nACGT\n' > "$WORK/tiny.fa"
"$FGR2" -k 31 -N 5 -o "$WORK/tiny.fgr2" "$WORK/tiny.fa" >/dev/null 2>&1
[ $? -eq 0 ] && ok "handles input with no k-mers >= k" || bad "crashed on input with no k-mers"

info "calculate_similarity.py: parsing and similarity"

if python3 -c 'import numpy' 2>/dev/null; then
    python3 - "$REPO_DIR" "$WORK" <<'PY'
import importlib.util, sys, os
repo, work = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("cs", os.path.join(repo, "calculate_similarity.py"))
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)

failures = []

# A 2-field line previously raised an uncaught IndexError (the guard checked
# len(parts) >= 2 but the code read parts[2]), killing the whole run.
p = os.path.join(work, "malformed.fgr2")
with open(p, "w") as fh:
    fh.write("#fgr2\tk=31\tN=10\thash=murmurhash3\n")
    fh.write("ACGT\t123\n")             # only 2 fields -> must be skipped, not crash
    fh.write("ACGTA\t456\t7\n")         # valid
    fh.write("\n")                      # blank
    fh.write("BADCOV\t1\tnotanint\n")   # unparseable coverage
try:
    kmers, cov, meta = cs.parse_fgr_file(p)
    if kmers != {"ACGTA"}:      failures.append(f"expected {{ACGTA}}, got {kmers}")
    if cov.get("ACGTA") != 7:   failures.append("coverage not parsed")
    if meta.get("k") != "31":   failures.append(f"metadata not parsed: {meta}")
except Exception as e:
    failures.append(f"parser raised {type(e).__name__}: {e}")

# Jaccard
if cs.jaccard_index({"a","b"}, {"b","c"}) != 1/3: failures.append("jaccard wrong")
if cs.jaccard_index(set(), set()) != 1.0:         failures.append("jaccard empty/empty")

# Cosine: identical vectors -> 1, disjoint -> 0, and scale invariance
if abs(cs.cosine_similarity_manual({"a":1,"b":2}, {"a":1,"b":2}) - 1.0) > 1e-12:
    failures.append("cosine self != 1")
if cs.cosine_similarity_manual({"a":1}, {"b":1}) != 0.0:
    failures.append("cosine disjoint != 0")
a = cs.cosine_similarity_manual({"a":1,"b":2}, {"a":3,"b":1})
b = cs.cosine_similarity_manual({"a":10,"b":20}, {"a":3,"b":1})
if abs(a-b) > 1e-12: failures.append("cosine not scale-invariant")

# --standardize is a documented no-op
s = cs.cosine_similarity_manual({"a":1,"b":2}, {"a":3,"b":1}, standardize=True)
if abs(a-s) > 1e-12: failures.append("--standardize changed the result")

# Label disambiguation for colliding stems
labels = cs.disambiguate_labels(["s","s"], ["s.fgr","s.fgr2"])
if len(set(labels)) != 2: failures.append(f"labels not disambiguated: {labels}")

for f in failures: print("FAILURE:", f)
sys.exit(1 if failures else 0)
PY
    [ $? -eq 0 ] && ok "python unit checks (parser, jaccard, cosine, labels)" \
                 || bad "python unit checks"

    # End-to-end: similarity must track true divergence (near > far).
    "$FGR2" -k 31 -N 2000 -c 1 -o "$WORK/near.fgr2" "$WORK/near.fa" >/dev/null 2>&1
    "$FGR2" -k 31 -N 2000 -c 1 -o "$WORK/far.fgr2"  "$WORK/far.fa"  >/dev/null 2>&1
    cp "$WORK/base.fgr2" "$WORK/e2e_base.fgr2" 2>/dev/null
    mkdir -p "$WORK/e2e" && cp "$WORK/base.fgr2" "$WORK/near.fgr2" "$WORK/far.fgr2" "$WORK/e2e/" 2>/dev/null
    (cd "$WORK/e2e" && python3 "$REPO_DIR/calculate_similarity.py" . -o r \
        --no-nj_tree --no-plot_trees >/dev/null 2>&1)
    if python3 - "$WORK" <<'PY'
import sys
work = sys.argv[1]
rows = {}
with open(work + '/e2e/r.jac') as fh:
    header = fh.readline().strip('\n').split('\t')[1:]
    for line in fh:
        parts = line.strip('\n').split('\t')
        rows[parts[0]] = dict(zip(header, (float(x) for x in parts[1:])))
# base-near (0.5% divergence) must be far more similar than base-far (20%)
sys.exit(0 if rows['base']['near'] > rows['base']['far'] else 1)
PY
    then ok "end-to-end: similarity tracks sequence divergence (near > far)"
    else bad "end-to-end: similarity does not track divergence"
    fi
else
    echo "  SKIP python tests (numpy not available)"
fi

printf '\n=====================================\n'
printf 'Passed: %d   Failed: %d\n' "$PASS" "$FAIL"
printf '=====================================\n'
[ "$FAIL" -eq 0 ] || exit 1
