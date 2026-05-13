"""
eval/run_p3_charger_sweep.py
────────────────────────────
Sweep P3 (opportunistic) charger count at 20k ticks.

For each num_chargers in NUM_CHARGERS_SWEEP, build a symmetric layout that
fills charger-priority tiers one at a time (5 cells per tier, 1 per picker
station), then run the simulation at 20k ticks and capture snapshot +
per-robot state.

Priority tiers (each adds 5 cells, 1 per picker station, in station order
1..5 at rows 1/7/13/19/25):
  T1 (processing):     (col=2, picker_row)       — path[-1]  cells
  T2 (queue-back):     (col=4, picker_row+2)     — path[0]   cells
  T3 (mid-queue):      (col=3, picker_row+2)     — path[1]   cells
  T4 (middle-gate):    (col=2, picker_row+1)     — path[3]   cells
  T5 (corner):         (col=2, picker_row+2)     — path[2]   cells
  T6 (entry-deep):     (col=4, picker_row)       — transit at picker row
  T7 (entry-mid):      (col=3, picker_row)       — transit at picker row
  T8 (gate-mid):       (col=3, picker_row+1)     — between gate and queue
  T9 (gate-deep):      (col=4, picker_row+1)     — between gate and queue
  T10 (corner+1):      (col=2, picker_row+3)     — extend col-2 cluster down

Results go to eval/runs/p3_sweep/n{NUM}/ (per-num snapshots) and
eval/runs/p3_sweep/sweep_summary.json (aggregated across all runs).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "eval" / "runs" / "p3_sweep"

PICKER_ROWS = [1, 7, 13, 19, 25]

TIERS = [
    [[r, 2]     for r in PICKER_ROWS],  # T1 processing
    [[r + 2, 4] for r in PICKER_ROWS],  # T2 queue-back
    [[r + 2, 3] for r in PICKER_ROWS],  # T3 mid-queue
    [[r + 1, 2] for r in PICKER_ROWS],  # T4 middle-gate
    [[r + 2, 2] for r in PICKER_ROWS],  # T5 corner
    [[r,     4] for r in PICKER_ROWS],  # T6 entry-deep
    [[r,     3] for r in PICKER_ROWS],  # T7 entry-mid
    [[r + 1, 3] for r in PICKER_ROWS],  # T8 gate-mid
    [[r + 1, 4] for r in PICKER_ROWS],  # T9 gate-deep
    [[r + 3, 2] for r in PICKER_ROWS],  # T10 corner+1
]

NUM_CHARGERS_SWEEP = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def build_layout(num: int) -> list[list[int]]:
    cells: list[list[int]] = []
    for tier in TIERS:
        cells.extend(tier)
    return cells[:num]


def run_one_num(num: int, horizon: int, heartbeat: int) -> dict:
    """Run P3 at a given num_chargers. Returns run metadata."""
    print(f"\n[sweep] === num_chargers={num}  horizon={horizon} ===")
    layout = build_layout(num)
    print(f"[sweep] layout ({len(layout)} cells): {layout}")

    # Skip if already complete (re-runnable).
    dest = SWEEP_DIR / f"n{num}"
    if (dest / "run_summary.json").exists():
        print(f"[sweep] n{num} already complete — skipping (delete to rerun)")
        return {"num": num, "skipped": True}

    # Write a per-num override file that run_one.py reads via --config-override.
    override = {
        "pipeline": 3,
        "num_chargers": num,
        "disable_active_charging": True,
        "selective_picker_chargers": True,
        "charger_positions": layout,
    }
    override_path = SWEEP_DIR / f"override_n{num}.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    with open(override_path, "w") as f:
        json.dump(override, f)

    # run_one.py writes snapshots to --out-root/p{P}; we re-point --out-root
    # to our sweep dir so each run lands in its own n{num}_tmp/p3 folder.
    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, str(ROOT / "eval" / "run_one.py"),
         "--pipeline", "3",
         "--horizon", str(horizon),
         "--heartbeat", str(heartbeat),
         "--out-root", f"eval/runs/p3_sweep/n{num}_tmp",
         "--config-override", str(override_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    print(f"[sweep] n{num} rc={rc}  elapsed={elapsed / 60:.1f} min")

    # Relocate the tmp snapshot folder to the final name (n{num}).
    tmp = SWEEP_DIR / f"n{num}_tmp" / "p3"
    if tmp.exists():
        dest.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            f.replace(dest / f.name)
        try:
            tmp.rmdir()
            tmp.parent.rmdir()
        except OSError:
            pass  # leftover files OK; snapshot already relocated

    return {"num": num, "rc": rc, "elapsed_min": round(elapsed / 60, 1),
            "layout": layout}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=20000)
    p.add_argument("--heartbeat", type=int, default=4000)
    p.add_argument("--nums", type=int, nargs="+", default=NUM_CHARGERS_SWEEP)
    args = p.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] nums={args.nums}  horizon={args.horizon}")
    all_results = []
    t0 = time.time()
    for num in args.nums:
        meta = run_one_num(num, args.horizon, args.heartbeat)
        all_results.append(meta)
    total = (time.time() - t0) / 60
    print(f"[sweep] DONE — {len(all_results)} runs in {total:.1f} min")

    with open(SWEEP_DIR / "sweep_summary.json", "w") as f:
        json.dump({"horizon": args.horizon, "runs": all_results,
                   "total_elapsed_min": round(total, 1)},
                  f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
