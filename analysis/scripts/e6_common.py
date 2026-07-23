#!/usr/bin/env python3
"""Shared timing harness for E6.

Uses /usr/bin/time -l (macOS) to capture wall-clock and peak RSS.

IMPORTANT UNIT NOTE: on macOS, `/usr/bin/time -l` reports "maximum resident set
size" in BYTES, unlike GNU/Linux time which reports kilobytes. This was verified
empirically in this session: a python process allocating and touching a 200 MB
bytearray reported 221282304 (= 211 MiB, i.e. 200 MB payload + ~11 MB
interpreter baseline). Had the unit been KB the figure would have implied 211 GB.
All RSS values in the E6 results JSON are therefore raw BYTES as reported.
"""
import os
import re
import time
import subprocess
import statistics

TIME_BIN = "/usr/bin/time"

N_CORES = os.cpu_count() or 16


def loadavg():
    """1/5/15-minute load averages."""
    return list(os.getloadavg())


def cpu_idle_percent(sample_s=2):
    """Fraction of CPU currently idle, from iostat.

    NOTE: on this machine the 1-minute load average is NOT a usable measure of
    CPU contention. It was observed at ~206 while iostat simultaneously
    reported ~30% idle / ~69% busy. macOS load average counts threads blocked
    in uninterruptible states, and this host has 35+ days uptime with many GUI
    applications, so it is inflated by well over an order of magnitude relative
    to real CPU saturation. Gating on load average would block forever. We
    therefore gate on measured idle CPU and record BOTH numbers.
    """
    try:
        out = subprocess.run(["iostat", "-c", "2"], capture_output=True,
                             timeout=sample_s + 20).stdout.decode()
        last = [ln for ln in out.strip().splitlines() if ln.strip()][-1]
        f = last.split()
        # trailing fields are: us sy id 1m 5m 15m
        return float(f[-4])
    except Exception:
        return float("nan")


def wait_for_idle(threshold=3.0, poll_s=30, max_wait_s=7200, verbose=True,
                  min_idle_pct=70.0):
    """Block until the machine has at least `min_idle_pct` idle CPU.

    This machine is shared with other concurrently-running experiments. Timing
    anything while the box is saturated produces numbers that are not
    reproducible and not attributable to the tool under test, so every timed
    phase is gated on the machine being close to idle.

    The gate uses measured idle CPU, not load average -- see cpu_idle_percent()
    for why load average is unusable on this host. `threshold` is retained only
    so the observed load average is recorded for the record.
    """
    t0 = time.time()
    while True:
        idle = cpu_idle_percent()
        la = os.getloadavg()[0]
        ok = (idle == idle) and idle >= min_idle_pct  # NaN-safe
        if ok:
            if verbose:
                print(f"[idle-gate] idle {idle:.1f}% >= {min_idle_pct}% "
                      f"(loadavg {la:.1f}, ignored), proceeding", flush=True)
            return {"waited_s": time.time() - t0, "idle_pct_at_start": idle,
                    "loadavg_at_start": la, "min_idle_pct": min_idle_pct,
                    "timed_out": False}
        if time.time() - t0 > max_wait_s:
            print(f"[idle-gate] TIMED OUT after {max_wait_s:.0f}s at idle "
                  f"{idle:.1f}%; proceeding anyway (RESULTS FLAGGED contended)",
                  flush=True)
            return {"waited_s": time.time() - t0, "idle_pct_at_start": idle,
                    "loadavg_at_start": la, "min_idle_pct": min_idle_pct,
                    "timed_out": True}
        if verbose:
            print(f"[idle-gate] idle {idle:.1f}% < {min_idle_pct}%, waiting "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
        time.sleep(poll_s)

_RE_REAL = re.compile(r"^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", re.M)
_RE_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)
_RE_PEAKFOOT = re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M)


class RunTimeout(RuntimeError):
    """Raised when a timed command exceeded its wall-clock budget."""


def run_timed(cmd, cwd=None, env=None, check=True, timeout=None):
    """Run `cmd` (list) under /usr/bin/time -l.

    Returns dict with wall_s, user_s, sys_s, max_rss_bytes,
    peak_footprint_bytes, returncode, and the verbatim command string.
    """
    full = [TIME_BIN, "-l"] + list(cmd)
    la_before = loadavg()
    try:
        proc = subprocess.run(full, cwd=cwd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RunTimeout(
            f"exceeded {timeout}s wall-clock budget: {' '.join(cmd)}")
    la_after = loadavg()
    err = proc.stderr.decode(errors="replace")
    m = _RE_REAL.search(err)
    r = _RE_RSS.search(err)
    p = _RE_PEAKFOOT.search(err)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(cmd)}\n"
            f"stderr tail:\n{err[-3000:]}")
    if m is None:
        raise RuntimeError(f"could not parse /usr/bin/time output for: "
                           f"{' '.join(cmd)}\n{err[-3000:]}")
    return {
        "cmd": " ".join(cmd),
        "wall_s": float(m.group(1)),
        "user_s": float(m.group(2)),
        "sys_s": float(m.group(3)),
        "max_rss_bytes": int(r.group(1)) if r else None,
        "peak_footprint_bytes": int(p.group(1)) if p else None,
        "returncode": proc.returncode,
        "stdout_bytes": len(proc.stdout),
        "loadavg_before": la_before,
        "loadavg_after": la_after,
    }


def _is_contended(idle_before, idle_after, min_idle_pct=50.0):
    """True if the machine had materially less than `min_idle_pct` idle CPU."""
    vals = [v for v in (idle_before, idle_after) if v == v]  # drop NaN
    if not vals:
        return None
    return min(vals) < min_idle_pct


def _cpu_summary(runs):
    """CPU time (user+sys) summary.

    On a contended machine wall-clock inflates with unrelated background load,
    but user+sys CPU time still measures the work the tool itself performed.
    It is therefore reported alongside wall-clock as the contention-robust
    metric, and is the fairer basis for tool-vs-tool comparison whenever the
    `contended` flag is set. It is NOT a substitute for wall-clock when
    reporting absolute throughput, and it understates cost for I/O-bound work.
    """
    cpu = [r["user_s"] + r["sys_s"] for r in runs]
    return {
        "cpu_s_all": cpu,
        "cpu_s_median": statistics.median(cpu),
        "cpu_s_min": min(cpu),
        "cpu_s_stdev": statistics.stdev(cpu) if len(cpu) > 1 else 0.0,
    }


def repeat_timed(cmd, n=5, cwd=None, env=None, warmup=1, label="", timeout=None):
    """Run `cmd` n times (after `warmup` untimed runs) and summarise.

    Returns dict with the full list of per-run measurements plus median/min/max
    and the interquartile range, so a third party can recompute any summary.
    """
    idle_before = cpu_idle_percent()
    for _ in range(warmup):
        run_timed(cmd, cwd=cwd, env=env, timeout=timeout)
    runs = [run_timed(cmd, cwd=cwd, env=env, timeout=timeout) for _ in range(n)]
    idle_after = cpu_idle_percent()
    walls = [x["wall_s"] for x in runs]
    rss = [x["max_rss_bytes"] for x in runs if x["max_rss_bytes"] is not None]
    out = {
        "label": label,
        "cmd": " ".join(cmd),
        "n_runs": n,
        "n_warmup": warmup,
        "runs": runs,
        "wall_s_all": walls,
        "wall_s_median": statistics.median(walls),
        "wall_s_min": min(walls),
        "wall_s_max": max(walls),
        "wall_s_mean": statistics.mean(walls),
        "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "max_rss_bytes_median": statistics.median(rss) if rss else None,
        "max_rss_bytes_all": rss,
        "loadavg_1min_max_during": max(r["loadavg_before"][0] for r in runs),
        "loadavg_1min_median_during": statistics.median(
            [r["loadavg_before"][0] for r in runs]),
        "n_cores": N_CORES,
        "cpu_idle_pct_before": idle_before,
        "cpu_idle_pct_after": idle_after,
    }
    out.update(_cpu_summary(runs))
    # Contention judged from measured idle CPU, NOT load average (which is
    # unusable on this host -- see cpu_idle_percent docstring).
    out["contended"] = _is_contended(idle_before, idle_after)
    return out


def repeat_timed_interleaved(variants, n=5, warmup=1):
    """Time several competing commands with their repetitions INTERLEAVED.

    `variants` is a list of (name, cmd) pairs. Instead of running all n reps of
    tool A and then all n reps of tool B, this runs one rep of every tool, then
    the next rep of every tool, and so on. On a shared machine whose background
    load drifts over time, running tools back-to-back in blocks would
    systematically favour whichever tool happened to occupy the quieter window.
    Interleaving makes any residual drift common-mode across the tools being
    compared, so the *relative* ranking stays meaningful even if absolute
    numbers carry contention overhead.

    Returns {name: summary_dict} in the same shape as repeat_timed.
    """
    per = {name: [] for name, _ in variants}
    failed = {}
    idle_before = cpu_idle_percent()
    for name, cmd in variants:
        for _ in range(warmup):
            try:
                run_timed(cmd)
            except RuntimeError as e:
                failed[name] = str(e)[:1000]
    for _rep in range(n):
        for name, cmd in variants:
            if name in failed:
                continue
            try:
                per[name].append(run_timed(cmd))
            except RuntimeError as e:
                failed[name] = str(e)[:1000]
    idle_after = cpu_idle_percent()
    out = {}
    for name, cmd in variants:
        if name in failed:
            out[name] = {"label": name, "cmd": " ".join(cmd),
                         "failed": True, "error": failed[name]}
            continue
        runs = per[name]
        walls = [x["wall_s"] for x in runs]
        rss = [x["max_rss_bytes"] for x in runs if x["max_rss_bytes"] is not None]
        loads = [x["loadavg_before"][0] for x in runs]
        out[name] = {
            "label": name, "cmd": " ".join(cmd), "n_runs": len(runs),
            "n_warmup": warmup, "interleaved": True, "runs": runs,
            "wall_s_all": walls,
            "wall_s_median": statistics.median(walls),
            "wall_s_min": min(walls), "wall_s_max": max(walls),
            "wall_s_mean": statistics.mean(walls),
            "wall_s_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "max_rss_bytes_median": statistics.median(rss) if rss else None,
            "max_rss_bytes_all": rss,
            "loadavg_1min_median_during": statistics.median(loads),
            "loadavg_1min_max_during": max(loads),
            "n_cores": N_CORES,
            "cpu_idle_pct_before": idle_before,
            "cpu_idle_pct_after": idle_after,
            "contended": _is_contended(idle_before, idle_after),
        }
        out[name].update(_cpu_summary(runs))
    return out


def hist_distinct_kmers(hist_path):
    """Sum all bins of an fgr2 .hist file -> number of DISTINCT canonical k-mers.

    Only valid when fgr2 was run with -c 1 (no coverage filtering), so that
    every observed k-mer appears in some bin.
    """
    total = 0
    bins = {}
    with open(hist_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cov, cnt = line.split("\t")[:2]
            bins[int(cov)] = int(cnt)
            total += int(cnt)
    return total, bins


def drop_caches_note():
    """macOS `purge` requires root; we do not attempt it. Cold-cache runs are
    approximated by first-touch of a freshly written file. Documented in caveats."""
    return "purge requires root; not attempted"
