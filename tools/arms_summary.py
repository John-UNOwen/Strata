"""Summarize every optimization arm under bench/results/arms: one row per arm (its newest run), and percentage
changes against a baseline arm (default k22ref). Read-only; safe any time.

Run: python tools/arms_summary.py [--arms bench/results/arms] [--baseline k22ref]
"""
import argparse
import json
from pathlib import Path


def arm_dirs(root: Path) -> dict[str, Path]:
    out = {}
    for arm in sorted(root.iterdir()):
        if not arm.is_dir():
            continue
        runs = [d for d in arm.iterdir() if (d / "results.json").is_file()]
        if runs:
            out[arm.name] = sorted(runs)[-1]      # newest dated folder
    return out


def med(*path):
    node = path[0]
    for part in path[1:]:
        node = (node or {}).get(part)
    return (node or {}).get("median") if isinstance(node, dict) else None


def row(arm: str, d: Path) -> dict:
    r = json.loads((d / "results.json").read_text(encoding="utf-8"))
    res = r.get("results") or {}
    prompts = res.get("prompts") or {}
    return {
        "arm": arm,
        "date": r.get("date") or "?",
        "decode": med(res, "chat", "decode_tps"),
        "prompt4k": prompts.get("4K") and med(prompts, "4K", "prompt_tps"),
        "prompt16k": prompts.get("16K") and med(prompts, "16K", "prompt_tps"),
        "agent_hit": med(res, "agent", "hit_pct"),
        "polluted": len(res.get("pollution") or []),
        "failures": len(res.get("failures") or []),
    }


def fmt(v, digits=1):
    return "-" if v is None else f"{v:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="bench/results/arms")
    ap.add_argument("--baseline", default="k22ref")
    a = ap.parse_args()
    root = Path(a.arms)
    if not root.is_dir():
        print(f"no arms folder at {root}")
        return 1
    rows = [row(name, d) for name, d in arm_dirs(root).items()]
    if not rows:
        print(f"no results under {root} yet (run bench/arms/run-arm.bat or BENCH-ARMS.bat first)")
        return 1
    base = next((r for r in rows if r["arm"] == a.baseline), None)

    def delta(r, key):
        if not base or r is base or r[key] is None or base[key] in (None, 0):
            return ""
        return f"{(r[key] - base[key]) / base[key] * 100:+.0f}%"

    print(f"{'arm':22s} {'date':11s} {'decode tok/s':13s} {'4K tok/s':10s} {'16K tok/s':10s} "
          f"{'agent hit %':11s} {'polluted':8s} vs baseline ({a.baseline}): decode / 4K / 16K")
    for r in rows:
        flag = " POLLUTED" if r["polluted"] else ""
        print(f"{r['arm']:22s} {r['date']:11s} {fmt(r['decode']):>8s}  {fmt(r['prompt4k'], 0):>8s}  "
              f"{fmt(r['prompt16k'], 0):>8s}  {fmt(r['agent_hit']):>9s}  {r['polluted']:>6d}{flag}   "
              f"{delta(r, 'decode')} / {delta(r, 'prompt4k')} / {delta(r, 'prompt16k')}")
    print("\nRemember: every table comes from one server session; a POLLUTED row is void, not slow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
