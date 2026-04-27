"""
eval/run_p12_charger_sweep.py
─────────────────────────────
Sweep charger budget for P1 (set cover) and P2 (affinity propagation)
at the same horizon as the existing P3 sweep, so the three curves can
be plotted on a single axis for direct comparison.

For each (pipeline, num_chargers) we:
  1. Write an override JSON {pipeline, num_chargers}
  2. Call run_one.py with --config-override
     -- the algorithm (set cover for P1, affinity propagation for P2)
        produces num_chargers cells based on the traffic heatmap.
  3. Relocate snapshots into eval/runs/p{P}_sweep/n{num}/

Usage:
    python eval/run_p12_charger_sweep.py
    python eval/run_p12_charger_sweep.py --pipelines 1 --nums 8 12 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM_CHARGERS_SWEEP = [5, 10, 15, 20, 25]


def run_one_num(pipeline: int, num: int, horizon: int, heartbeat: int) -> dict:
    sweep_dir = ROOT / "eval" / "runs" / f"p{pipeline}_sweep"
    dest = sweep_dir / f"n{num}"
    if (dest / "run_summary.json").exists():
        print(f"[sweep] p{pipeline} n{num} already complete -skipping")
        return {"pipeline": pipeline, "num": num, "skipped": True}

    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[sweep] === pipeline={pipeline} num_chargers={num} "
          f"horizon={horizon} ===")

    override = {"pipeline": pipeline, "num_chargers": num}
    override_path = sweep_dir / f"override_n{num}.json"
    override_path.write_text(json.dumps(override))

    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, str(ROOT / "eval" / "run_one.py"),
         "--pipeline", str(pipeline),
         "--horizon", str(horizon),
         "--heartbeat", str(heartbeat),
         "--out-root", f"eval/runs/p{pipeline}_sweep/n{num}_tmp",
         "--config-override", str(override_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    print(f"[sweep] p{pipeline} n{num} rc={rc} elapsed={elapsed / 60:.1f} min")

    tmp = sweep_dir / f"n{num}_tmp" / f"p{pipeline}"
    if tmp.exists():
        dest.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            f.replace(dest / f.name)
        try:
            tmp.rmdir()
            tmp.parent.rmdir()
        except OSError:
            pass

    return {"pipeline": pipeline, "num": num, "rc": rc,
            "elapsed_min": round(elapsed / 60, 1)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pipelines", type=int, nargs="+", default=[1, 2])
    p.add_argument("--nums", type=int, nargs="+", default=NUM_CHARGERS_SWEEP)
    p.add_argument("--horizon", type=int, default=20000)
    p.add_argument("--heartbeat", type=int, default=4000)
    args = p.parse_args()

    print(f"[sweep] pipelines={args.pipelines} nums={args.nums} "
          f"horizon={args.horizon}")
    all_results = []
    t0 = time.time()
    for pipeline in args.pipelines:
        for num in args.nums:
            meta = run_one_num(pipeline, num, args.horizon, args.heartbeat)
            all_results.append(meta)
    total = (time.time() - t0) / 60
    print(f"\n[sweep] DONE -{len(all_results)} runs in {total:.1f} min")

    summary_path = ROOT / "eval" / "runs" / "p12_sweep_summary.json"
    summary_path.write_text(json.dumps(
        {"pipelines": args.pipelines, "nums": args.nums,
         "horizon": args.horizon, "runs": all_results,
         "total_elapsed_min": round(total, 1)},
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
