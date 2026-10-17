"""Summarize every optimization arm under bench/results/arms: one row per arm aggregated over its server SESSIONS.

Each strata_bench run is one fresh server session (BENCH-ARMS.bat runs each arm 3 sessions per round). A polluted
session is void from the suite where the pollution appeared onwards: the suites measured before it keep their numbers
(the fault is sticky, it does not reach backwards), so a session that stormed in the agent phase still counts for
chat and the prompt tests. The spread of the clean sessions' medians decides consistency. Read-only; safe any time.

Run: python tools/arms_summary.py [--arms bench/results/arms] [--baseline k22ref] [--tol 12] [--sessions]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

KEYS = (("decode", ("chat", "decode_tps")), ("4K", ("prompts", "4K", "prompt_tps")),
        ("16K", ("prompts", "16K", "prompt_tps")))
KEY_SUITE = {"decode": "chat", "4K": "4K", "16K": "16K"}
# the order the suite runs them; pollution voids its suite and every later one
SUITE_ORDER = ["chat", "4K", "16K", "agent", "clients", "32K"]


def void_suites(where: str) -> set[str]:
    """The suites a pollution report at `where` makes void (warm-ups and calibration void everything)."""
    w = (where or "").strip()
    for suite in reversed(SUITE_ORDER):
        if w.startswith(suite):
            return set(SUITE_ORDER[SUITE_ORDER.index(suite):])
    return set(SUITE_ORDER)             # warm-up, calibration or unknown: play safe and void the session


def arm_sessions(root: Path) -> dict[str, list[Path]]:
    out = {}
    for arm in sorted(root.iterdir()):
        if not arm.is_dir():
            continue
        runs = [d for d in arm.iterdir() if (d / "results.json").is_file()]
        if runs:
            out[arm.name] = sorted(runs, key=lambda d: d.stat().st_mtime)
    return out


def med(d: dict, *path):
    node = d
    for part in path:
        node = (node or {}).get(part)
    return (node or {}).get("median") if isinstance(node, dict) else None


def session_row(d: Path) -> dict:
    r = json.loads((d / "results.json").read_text(encoding="utf-8"))
    res = r.get("results") or {}
    pol = res.get("pollution") or []
    void = void_suites(pol[0].get("where")) if pol else set()
    row = {"stamp": time.strftime("%H:%M", time.localtime(d.stat().st_mtime)),
           "polluted": len(pol),
           "first_pollution": pol[0].get("where") if pol else None,
           "void": void,
           "agent_hit": med(res, "agent", "hit_pct") if "agent" not in void else None}
    for key, path in KEYS:
        node = res
        row[key] = med(node, *path) if KEY_SUITE[key] not in void else None
    return row


def agg(rows: list[dict], key: str) -> tuple[float | None, float]:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None, 0.0
    m = statistics.median(vals)
    spread = (max(vals) - min(vals)) / m * 100 if m else 0.0
    return m, spread


def fmt(v, digits=1, spread=None):
    if v is None:
        return "-"
    s = f"{v:.{digits}f}"
    if spread is not None:
        s += f" ({spread:.0f}%)"
    return s


def epoch_drift(rows: list[dict]) -> float | None:
    """Percent change in decode from the first clean session to the last, in time order.

    A long evening of benchmarking drifts down (about 5% over an hour here: heat, background load), and an arm measured
    late against a baseline measured early looks better than it is. Reading the drift next to the deltas keeps the two
    effects apart.
    """
    vals = [r["decode"] for r in rows if r["decode"] is not None]
    if len(vals) < 2 or not vals[0]:
        return None
    return (vals[-1] - vals[0]) / vals[0] * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="bench/results/arms")
    ap.add_argument("--baseline", default="k22ref")
    ap.add_argument("--tol", type=float, default=12.0, help="max spread between clean session medians, percent")
    ap.add_argument("--newest", type=int, default=0, help="use only the newest N sessions per arm (0: all; use it when a new round piled onto the old ones)")
    ap.add_argument("--sessions", action="store_true", help="also list every session row")
    a = ap.parse_args()
    root = Path(a.arms)
    if not root.is_dir():
        print(f"no arms folder at {root}")
        return 1
    data = {name: [session_row(d) for d in (dirs[-a.newest:] if a.newest else dirs)]
            for name, dirs in arm_sessions(root).items()}
    if not data:
        print(f"no results under {root} yet (run BENCH-ARMS.bat)")
        return 1

    def arm_agg(name):
        rows = data[name]
        cells = {k: agg(rows, k) for k, _ in KEYS}
        used = [r for r in rows if any(r[k] is not None for k in KEY_SUITE)]
        worst = max((s for _, s in cells.values()), default=0.0)
        if not used:
            verdict = "all polluted"
        elif len(used) == 1:
            verdict = "1 session"
        elif worst > a.tol:
            verdict = f"varies (worst {worst:.0f}%)"
        else:
            verdict = "consistent"
        return cells, verdict, used

    base_cells = arm_agg(a.baseline)[0] if a.baseline in data else None

    def delta(name, key):
        if not base_cells or name == a.baseline:
            return ""
        v, _ = arm_agg(name)[0][key]
        b, _ = base_cells[key]
        return "" if not v or not b else f"{(v - b) / b * 100:+.0f}%"

    print(f"{'arm':22s} {'sessions':8s} {'decode tok/s':15s} {'4K tok/s':15s} {'16K tok/s':15s} "
          f"{'agent hit':9s} consistency        vs {a.baseline}: decode / 4K / 16K")
    for name, rows in data.items():
        cells, verdict, clean = arm_agg(name)
        n = f"{len(clean)}/{len(rows)}"
        print(f"{name:22s} {n:>8s} " +
              " ".join(f"{fmt(cells[k][0], 1, cells[k][1]):>14s} " for k, _ in KEYS) +
              f"{fmt(next((r['agent_hit'] for r in clean if r['agent_hit'] is not None), None), 0):>9s}  "
              f"{verdict:15s}  {delta(name, 'decode')} / {delta(name, '4K')} / {delta(name, '16K')}")
    if a.sessions:
        print("\nsessions (per arm, oldest first):")
        for name, rows in data.items():
            for r in rows:
                if not r["polluted"]:
                    note = ""
                elif r["void"] and len(r["void"]) < len(SUITE_ORDER):
                    note = f" POLLUTED at {r['first_pollution']} (later suites void)"
                else:
                    note = f" POLLUTED at {r['first_pollution']} (whole session void)"
                print(f"  {name:22s} {r['stamp']}  decode {fmt(r['decode']):>6s}  4K {fmt(r['4K'], 0):>5s}  "
                      f"16K {fmt(r['16K'], 0):>5s}  polluted {r['polluted']}{note}")
            drift = epoch_drift(rows)
            if drift is not None:
                print(f"  {name:22s}        drift over the round (last clean vs first clean): {drift:+.1f}%")
    print(f"\nAggregates use every session that reached a suite clean ({len(clean)} of {len(rows)} where shown); a "
          f"session polluted later keeps its earlier suites. Spread is max-min over those session medians. tol "
          f"{a.tol:.0f}%. A 'consistent' arm still needs its numbers read against the baseline row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
