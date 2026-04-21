"""
eval/run_all.py
───────────────
Run all 4 charging-placement pipelines sequentially at the same horizon,
then extract KPIs and print the comparison table.

Resumable:
    If ``eval/runs/p{P}/run_summary.json`` already exists, the pipeline is
    skipped.  Delete that file (or the whole directory) to force a re-run.

Usage
─────
    python eval/run_all.py                    # default horizon 28800
    python eval/run_all.py --horizon 100000   # ~28 hr sim, ~2.5 hr wall-clock
    python eval/run_all.py --pipelines 1 2    # subset
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"


def run_pipeline(P: int, horizon: int, heartbeat: int) -> int:
    dest = RUNS / f"p{P}"
    summary = dest / "run_summary.json"
    if summary.exists():
        print(f"[run_all] p{P} already has run_summary.json — skipping (delete it to rerun)")
        return 0
    print(f"[run_all] launching p{P} (horizon={horizon})")
    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, str(ROOT / "eval" / "run_one.py"),
         "--pipeline", str(P),
         "--horizon", str(horizon),
         "--heartbeat", str(heartbeat)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    print(f"[run_all] p{P} finished in {elapsed / 60:.1f} min (rc={rc})")
    return rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=28800)
    p.add_argument("--heartbeat", type=int, default=2000)
    p.add_argument("--pipelines", type=int, nargs="+", default=[1, 2, 3, 4])
    args = p.parse_args()

    print(f"[run_all] pipelines={args.pipelines} horizon={args.horizon}")
    t0 = time.time()
    for P in args.pipelines:
        if run_pipeline(P, args.horizon, args.heartbeat) != 0:
            print(f"[run_all] p{P} failed; aborting batch")
            return 1
    print(f"[run_all] batch done in {(time.time() - t0) / 60:.1f} min")

    # Extract KPIs
    subprocess.call(
        [sys.executable, str(ROOT / "eval" / "extract_kpis.py"), "--all"],
        cwd=str(ROOT),
    )
    subprocess.call(
        [sys.executable, str(ROOT / "eval" / "aggregate.py")],
        cwd=str(ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
