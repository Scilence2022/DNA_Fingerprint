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

# The hash column (parts[1]) was historically parsed and then DISCARDED, which
# made the bottom-N-of-union estimators impossible to compute. It must now be
# threaded through parse_sketch.
p2 = os.path.join(work, "hashes.fgr2")
with open(p2, "w") as fh:
    fh.write("#fgr2\tk=31\tN=4\thash=murmurhash3\n")
    fh.write("AAAA\t900\t3\n")
    fh.write("CCCC\t10\t5\n")
    fh.write("GGGG\t400\t2\n")
    fh.write("TTTT\tnotanint\t9\n")   # unparseable hash -> skipped, not crash
try:
    recs, meta2 = cs.parse_sketch(p2)
    if set(recs) != {"AAAA", "CCCC", "GGGG"}:
        failures.append(f"parse_sketch kmers wrong: {sorted(recs)}")
    if recs.get("AAAA") != (900, 3) or recs.get("CCCC") != (10, 5) or recs.get("GGGG") != (400, 2):
        failures.append(f"hash/coverage not threaded through: {recs}")
    if meta2.get("N") != "4":
        failures.append(f"parse_sketch metadata not parsed: {meta2}")
    # parse_fgr_file must remain a compatible 3-tuple view of the same data.
    k3, c3, m3 = cs.parse_fgr_file(p2)
    if k3 != set(recs) or c3 != {k: v[1] for k, v in recs.items()} or m3 != meta2:
        failures.append("parse_fgr_file is no longer a consistent view of parse_sketch")
    # union_support must order by HASH (not by k-mer, not by insertion order).
    if cs.union_support(recs, recs, 2) != ["CCCC", "GGGG"]:
        failures.append(f"union_support not ordered by hash: {cs.union_support(recs, recs, 2)}")
    # ...and be symmetric in its arguments.
    other = {"AAAA": (900, 1), "TTTA": (5, 4)}
    if cs.union_support(recs, other, 3) != cs.union_support(other, recs, 3):
        failures.append("union_support is not symmetric")
except Exception as e:
    failures.append(f"parse_sketch raised {type(e).__name__}: {e}")

# Jaccard: legacy direct estimator (unchanged semantics)
if cs.jaccard_index({"a","b"}, {"b","c"}) != 1/3: failures.append("jaccard wrong")
if cs.jaccard_index(set(), set()) != 1.0:         failures.append("jaccard empty/empty")

# Corrected estimators on a hand-built example. Support = 3 smallest hashes of the
# merge = a(1), b(2), c(3); a and b are shared, c is not -> 2/3.
ra = {"a": (1, 2), "b": (2, 4), "d": (9, 1)}
rb = {"a": (1, 1), "b": (2, 2), "c": (3, 7)}
if abs(cs.jaccard_union(ra, rb, 3) - 2/3) > 1e-12:
    failures.append(f"jaccard_union wrong: {cs.jaccard_union(ra, rb, 3)}")
if cs.jaccard_union({}, {}) != 1.0:
    failures.append("jaccard_union empty/empty")
# Cosine on that support: a=(2,4,0), b=(1,2,7) -> dot 10, norms sqrt(20), sqrt(54)
expected = 10.0 / ((20 ** 0.5) * (54 ** 0.5))
if abs(cs.cosine_union(ra, rb, 3) - expected) > 1e-12:
    failures.append(f"cosine_union wrong: {cs.cosine_union(ra, rb, 3)} != {expected}")
# Identical sketches must give exactly 1 under both corrected estimators.
if abs(cs.jaccard_union(ra, ra) - 1.0) > 1e-12: failures.append("jaccard_union self != 1")
if abs(cs.cosine_union(ra, ra) - 1.0) > 1e-12:  failures.append("cosine_union self != 1")
# Disjoint sketches -> 0.
if cs.jaccard_union({"a": (1, 1)}, {"b": (2, 1)}) != 0.0:
    failures.append("jaccard_union disjoint != 0")
if cs.cosine_union({"a": (1, 1)}, {"b": (2, 1)}) != 0.0:
    failures.append("cosine_union disjoint != 0")

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

    info "calculate_similarity.py: corrected vs legacy estimators"

    # The decisive test for the estimator fix. Two samples with a ~2x ratio in
    # distinct-k-mer count: A is a 20 kb random sequence, B shares A's first 10 kb
    # and adds 30 kb of independent sequence. The true Jaccard is computed exactly
    # by enumerating canonical 31-mers.
    #
    # The legacy direct estimator |A n B| / |A u B| on two independently built
    # bottom-N sketches carries a structural bias set by that ratio: its error does
    # NOT shrink as N grows. The bottom-N-of-union estimator's error does.
    # Everything here is seeded, so the numbers are deterministic.
    if python3 - "$REPO_DIR" "$WORK" <<'PY'
import importlib.util, os, random, subprocess, sys, statistics
repo, work = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("cs", os.path.join(repo, "calculate_similarity.py"))
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
fgr2 = os.path.join(repo, "fgr2")
d = os.path.join(work, "ratio"); os.makedirs(d, exist_ok=True)

TR = str.maketrans("ACGT", "TGCA")
def canonical(seq, k=31):
    out = set()
    for i in range(len(seq) - k + 1):
        m = seq[i:i+k]; r = m.translate(TR)[::-1]
        out.add(m if m < r else r)
    return out

REPS, NS = 5, (100, 10000)
err = {(e, n): [] for e in ("union", "direct") for n in NS}
for rep in range(REPS):
    random.seed(20240502 + rep)
    A = "".join(random.choice("ACGT") for _ in range(20000))
    B = A[:10000] + "".join(random.choice("ACGT") for _ in range(30000))
    for name, seq in (("A", A), ("B", B)):
        with open(os.path.join(d, name + ".fa"), "w") as fh:
            fh.write(">%s\n%s\n" % (name, seq))
    ca, cb = canonical(A), canonical(B)
    true_j = len(ca & cb) / len(ca | cb)
    for N in NS:
        for name in ("A", "B"):
            subprocess.run([fgr2, "-k", "31", "-N", str(N), "-c", "1",
                            "-o", os.path.join(d, f"{name}.{N}.fgr2"),
                            os.path.join(d, f"{name}.fa")], capture_output=True)
        ra, _ = cs.parse_sketch(os.path.join(d, f"A.{N}.fgr2"))
        rb, _ = cs.parse_sketch(os.path.join(d, f"B.{N}.fgr2"))
        err[("union", N)].append(abs(cs.jaccard_union(ra, rb) - true_j))
        err[("direct", N)].append(abs(cs.jaccard_index(set(ra), set(rb)) - true_j))

m = {k: statistics.mean(v) for k, v in err.items()}
for k in sorted(m, key=str):
    print("  mean|error| %-7s N=%-6d %.6f" % (k[0], k[1], m[k]))

failures = []
# (1) At N=10000 the corrected estimator is far more accurate.
if not (m[("union", 10000)] < 0.25 * m[("direct", 10000)]):
    failures.append("union not much more accurate than direct at N=10000: "
                    f"{m[('union',10000)]:.6f} vs {m[('direct',10000)]:.6f}")
if not m[("union", 10000)] < 0.01:
    failures.append(f"union error too large at N=10000: {m[('union',10000)]:.6f}")
# (2) The corrected error shrinks with N; the legacy error floors at its bias.
if not (m[("union", 100)] > 5 * m[("union", 10000)]):
    failures.append("union error does not shrink with N: "
                    f"{m[('union',100)]:.6f} -> {m[('union',10000)]:.6f}")
if not (m[("direct", 100)] < 3 * m[("direct", 10000)]):
    failures.append("direct error shrank with N more than expected (bias should persist)")
if not m[("direct", 10000)] > 0.03:
    failures.append("direct estimator no longer shows its size-ratio bias floor: "
                    f"{m[('direct',10000)]:.6f}")
for f in failures: print("FAILURE:", f)
sys.exit(1 if failures else 0)
PY
    then ok "union estimator recovers true Jaccard accurately; error shrinks with N, legacy error does not"
    else bad "corrected/legacy Jaccard accuracy contract violated"
    fi

    # Legacy mode must reproduce the pre-fix numbers bit-for-bit, since published
    # .jac/.cos matrices were produced with it -- and the default must differ.
    (cd "$WORK/e2e" && python3 "$REPO_DIR/calculate_similarity.py" . -o legacy \
        --legacy-jaccard --legacy-cosine --no-nj_tree --no-plot_trees \
        > "$WORK/legacy.log" 2>&1)
    if grep -qi 'WARNING: LEGACY' "$WORK/legacy.log"; then
        ok "legacy mode emits a loud bias warning"
    else
        bad "legacy mode did not warn about the bias"
    fi
    if python3 - "$REPO_DIR" "$WORK" <<'PY'
import importlib.util, os, sys, glob
repo, work = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("cs", os.path.join(repo, "calculate_similarity.py"))
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
d = os.path.join(work, "e2e")

def read_matrix(path):
    rows = {}
    with open(path) as fh:
        header = fh.readline().strip("\n").split("\t")[1:]
        for line in fh:
            p = line.strip("\n").split("\t")
            rows[p[0]] = dict(zip(header, (float(x) for x in p[1:])))
    return rows

recs = {}
for p in sorted(glob.glob(os.path.join(d, "*.fgr2"))):
    r, _ = cs.parse_sketch(p)
    recs[cs.strip_extensions(os.path.basename(p))] = r

failures = []
leg_j, leg_c = read_matrix(os.path.join(d, "legacy.jac")), read_matrix(os.path.join(d, "legacy.cos"))
new_j, new_c = read_matrix(os.path.join(d, "r.jac")),      read_matrix(os.path.join(d, "r.cos"))
names = sorted(recs)
differed_j = differed_c = False
for i, a in enumerate(names):
    for b in names[i+1:]:
        # Legacy CLI output must equal the legacy formulas applied to the sketches.
        want_j = cs.jaccard_index(set(recs[a]), set(recs[b]))
        want_c = cs.cosine_similarity_manual({k: v[1] for k, v in recs[a].items()},
                                             {k: v[1] for k, v in recs[b].items()})
        if abs(leg_j[a][b] - want_j) > 5e-5:
            failures.append(f"legacy jaccard {a}/{b}: {leg_j[a][b]} != {want_j}")
        if abs(leg_c[a][b] - want_c) > 5e-5:
            failures.append(f"legacy cosine {a}/{b}: {leg_c[a][b]} != {want_c}")
        # Default CLI output must equal the corrected formulas.
        if abs(new_j[a][b] - cs.jaccard_union(recs[a], recs[b])) > 5e-5:
            failures.append(f"default jaccard {a}/{b} is not the union estimator")
        if abs(new_c[a][b] - cs.cosine_union(recs[a], recs[b])) > 5e-5:
            failures.append(f"default cosine {a}/{b} is not the union estimator")
        if abs(new_j[a][b] - leg_j[a][b]) > 5e-5: differed_j = True
        if abs(new_c[a][b] - leg_c[a][b]) > 5e-5: differed_c = True
if not differed_j: failures.append("default Jaccard is identical to legacy everywhere (fix not wired up?)")
if not differed_c: failures.append("default cosine is identical to legacy everywhere (fix not wired up?)")
for f in failures: print("FAILURE:", f)
sys.exit(1 if failures else 0)
PY
    then ok "--legacy-jaccard/--legacy-cosine reproduce the old estimators; default uses the corrected ones"
    else bad "legacy/default estimator wiring is wrong"
    fi
else
    echo "  SKIP python tests (numpy not available)"
fi

printf '\n=====================================\n'
printf 'Passed: %d   Failed: %d\n' "$PASS" "$FAIL"
printf '=====================================\n'
[ "$FAIL" -eq 0 ] || exit 1
