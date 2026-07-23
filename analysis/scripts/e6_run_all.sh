#!/bin/bash
# E6 driver. Runs the three benchmark phases strictly sequentially, each gated
# on the shared machine being close to idle. Never run two phases at once.
set -u
REPO=/Users/song/Github-Repos/DNA_Fingerprint/.claude/worktrees/session-64bad5
S=/private/tmp/claude-501/-Users-song-Github-Repos-DNA-Fingerprint--claude-worktrees-session-64bad5/2d7ad55d-8bc0-467b-bea7-515e6cd39468/scratchpad
LT=${LOAD_THRESHOLD:-6.0}
NREPS=${NREPS:-5}
GMW=${GATE_MAX_WAIT:-600}
R=$REPO/analysis/results
mkdir -p "$R" "$S/e6/work" "$S/e6/tools" "$S/e6/down"

echo "=================== PHASE 1: scaling ==================="
python3 -u "$REPO/analysis/scripts/e6_bench_scaling.py" \
  --fgr2 "$REPO/fgr2" --manifest "$S/e6/manifest.json" \
  --workdir "$S/e6/work" --out "$R/e6_scaling.json" \
  --nreps "$NREPS" --load-threshold "$LT" --gate-max-wait "$GMW"
echo "phase1 rc=$?"

echo "=================== PHASE 2: tools ==================="
python3 -u "$REPO/analysis/scripts/e6_bench_tools.py" \
  --fgr2 "$REPO/fgr2" --repo "$REPO" --genomes "$S/genomes" \
  --manifest "$S/e6/manifest.json" --workdir "$S/e6/tools" \
  --out "$R/e6_tools.json" --nreps "$NREPS" --load-threshold "$LT" --gate-max-wait "$GMW"
echo "phase2 rc=$?"

echo "=================== PHASE 3: downstream ==================="
python3 -u "$REPO/analysis/scripts/e6_bench_downstream.py" \
  --repo "$REPO" --sketchdir "$S/e6/tools/avaa_fgr2" \
  --workdir "$S/e6/down" --out "$R/e6_downstream.json" \
  --nreps "$NREPS" --load-threshold "$LT" --gate-max-wait "$GMW" --per-run-timeout 900
echo "phase3 rc=$?"

echo "=================== ALL PHASES DONE ==================="
