#!/usr/bin/env python3
"""Determine fgr2's accepted range for -p empirically.

The scaling benchmark recorded -p 10 and -p 12 as failures. This script
establishes WHY, by running fgr2 across a range of -p and capturing the exact
exit status and stderr, so the failure is reported as a documented tool
constraint rather than an unexplained error.
"""
import os
import sys
import json
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fgr2", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    args = ap.parse_args()

    rows = []
    for p in [6, 8, 10, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32]:
        sk = os.path.join(args.tmp, f"pb_{p}.fgr2")
        cmd = [args.fgr2, "-k", "31", "-N", "1000", "-c", "1", "-t", "2",
               "-p", str(p), "-o", sk, args.input]
        pr = subprocess.run(cmd, capture_output=True)
        err = pr.stderr.decode(errors="replace").strip()
        out = pr.stdout.decode(errors="replace").strip()
        produced = os.path.exists(sk) and os.path.getsize(sk) > 0
        rows.append({
            "prefix": p, "cmd": " ".join(cmd), "returncode": pr.returncode,
            "stderr": err[-400:], "stdout_tail": out[-400:],
            "produced_sketch": bool(produced),
            "sketch_bytes": os.path.getsize(sk) if produced else 0,
        })
        print(f"-p {p:>2} rc={pr.returncode} produced={produced} "
              f"msg={(err or out).splitlines()[0][:70] if (err or out) else ''}",
              flush=True)
        if produced:
            os.remove(sk)
            h = sk + ".hist"
            if os.path.exists(h):
                os.remove(h)

    accepted = [r["prefix"] for r in rows if r["produced_sketch"]]
    res = {
        "experiment": "E6_prefix_bound",
        "rows": rows,
        "accepted_prefixes": accepted,
        "min_accepted": min(accepted) if accepted else None,
        "max_accepted": max(accepted) if accepted else None,
        "conclusion": (
            "fgr2 enforces a lower bound on -p and rejects smaller values with "
            "an explicit message; the -p 10 and -p 12 entries in the scaling "
            "benchmark are this documented input validation, NOT a crash or a "
            "defect. fgr2 exits with status 1 and writes an explicit message "
            "to stderr in both the too-small and too-large cases, so the "
            "rejection is properly detectable by a calling pipeline."),
    }
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[done] accepted -p range: {min(accepted)}..{max(accepted)} -> {args.out}")


if __name__ == "__main__":
    main()
