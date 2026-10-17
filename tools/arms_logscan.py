#!/usr/bin/env python3
"""Scan the arm ladder engine logs (strata-arm-<arm>.log) for the pollution signature.

A polluted reply runs at junk speed and the suffix drafts accept perfectly: the
model degenerates into one repeated token, so every draft window is confirmed.
The client gate (strata_bench.py) already stops a session on the reply text; this
scanner is the second opinion that reads the engine's own numbers, and it finds
storms in logs whose results.json predates the gate.

usage:
  python tools/arms_logscan.py                     # every strata-arm-*.log
  python tools/arms_logscan.py strata-arm-k22ref.log --arm k22ref
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

PROMPT_RE = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms "
    r"\(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), "
    r"drafts accepted (\d+) of (\d+), (\d+) checkpoints"
)
START_RE = re.compile(r"engine started:")

# a junk window accepts every draft and runs faster than the session's normal pace
MIN_DRAFTS = 8
ACCEPT_RATIO = 1.0
SPEED_RATIO = 1.25


def parse_log(path):
    """Return one dict per engine session: {"start": line, "rows": [...]}"""
    sessions = []
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if START_RE.search(line):
                cur = {"start": line.strip(), "rows": []}
                sessions.append(cur)
                continue
            m = PROMPT_RE.search(line)
            if not m or cur is None:
                continue
            g = [int(m.group(i)) for i in (1, 2, 3, 4, 6, 7, 9, 10, 11)]
            row = {
                "prompt_tokens": g[0],
                "reused_tokens": g[1],
                "read_tokens": g[2],
                "prompt_ms": g[3],
                "generated_tokens": g[4],
                "decode_ms": g[5],
                "decode_tps": float(m.group(8)),
                "accepted": g[6],
                "offered": g[7],
                "checkpoints": g[8],
            }
            cur["rows"].append(row)
    return sessions


def session_median(rows):
    """Median decode over the rows that look normal (acceptance under 90%)."""
    normal = [r["decode_tps"] for r in rows if r["offered"] == 0 or r["accepted"] / r["offered"] < 0.9]
    return statistics.median(normal) if normal else None


def scan_session(rows):
    """Flag rows that carry the junk signature."""
    med = session_median(rows)
    flagged = []
    for i, r in enumerate(rows):
        if r["offered"] < MIN_DRAFTS or r["accepted"] < r["offered"]:
            continue
        fast = r["decode_tps"] >= SPEED_RATIO * med if med else False
        if fast:
            flagged.append(i)
    return med, flagged


def find_results(arm, session_idx, results_root):
    """The Nth session of arm/ with a results.json, in mtime order."""
    folder = os.path.join(results_root, arm)
    if not os.path.isdir(folder):
        return None
    runs = sorted(
        (d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)),
        key=os.path.getmtime,
    )
    if session_idx >= len(runs):
        return None
    return os.path.join(runs[session_idx], "results.json")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="log files (default: strata-arm-*.log at the repo root)")
    ap.add_argument("--arm", help="arm name override (default: parsed from the file name)")
    ap.add_argument("--results", default=os.path.join("bench", "results", "arms"))
    ap.add_argument("--quiet-clean", action="store_true", help="print only sessions with a flag")
    args = ap.parse_args(argv)

    logs = args.logs or sorted(glob.glob("strata-arm-*.log"))
    if not logs:
        print("no strata-arm-*.log found")
        return 2

    bad_sessions = 0
    for path in logs:
        arm = args.arm or re.sub(r"^strata-arm-|\.log$", "", os.path.basename(path))
        sessions = parse_log(path)
        if not sessions:
            continue
        print(f"== {os.path.basename(path)}: {len(sessions)} session(s)")
        for si, s in enumerate(sessions):
            rows = s["rows"]
            if not rows:
                print(f"  session {si + 1}: no request lines")
                continue
            med, flagged = scan_session(rows)
            gate = ""
            rp = find_results(arm, si, args.results)
            if rp and os.path.exists(rp):
                try:
                    with open(rp, "r", encoding="utf-8") as f:
                        pol = json.load(f).get("results", {}).get("pollution") or []
                    if pol:
                        gate = " [gate: %d flagged, first %s]" % (len(pol), pol[0].get("where"))
                except (OSError, ValueError):
                    gate = " [gate: unreadable]"
            mark = "LOGSIGNAL" if flagged else "clean"
            if flagged:
                bad_sessions += 1
            med_s = "%.1f" % med if med else "-"
            if flagged or not args.quiet_clean:
                print(f"  session {si + 1}: {len(rows)} requests, normal pace {med_s} tok/s -> {mark}{gate}")
            for i in flagged:
                r = rows[i]
                print(
                    "    row %d: prompt %d tokens, %d generated, %.1f tok/s, drafts %d of %d"
                    % (i + 1, r["prompt_tokens"], r["generated_tokens"], r["decode_tps"],
                       r["accepted"], r["offered"])
                )
    print()
    print("sessions with the junk signature: %d" % bad_sessions)
    return 1 if bad_sessions else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
