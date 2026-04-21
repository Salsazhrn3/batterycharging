"""
eval/run_one.py
───────────────
Headless runner for a single (pipeline, horizon) evaluation trial.

Keeps the warehouse grid (``generated_pod.csv``) and the workload
(``generated_order.csv``, ``pods.csv``, etc.) frozen so every run compares
the same physics — the only thing that changes between pipelines is the
charging-station overlay written into ``charging_config.json``.

Procedure
─────────
1. Clear transient state that would contaminate the run:
       ``netlogo.state``, ``assign_order.csv``, ``pod_info.csv``,
       ``output/order-finished.csv`` (appended to by the sim — must be
       cleared, setup() does not clear it).
2. Build the pipeline-P overlay via ``ChargingLayoutGenerator`` on the
   *existing* grid, then write ``charging_config.json``.
3. ``netlogo.setup()`` → ``netlogo.console_tick(max_ticks)``.
4. Copy ``output/order-finished.csv``, ``charging_config.json``, and
   ``netlogo.state`` into ``eval/runs/p{P}/``.

Usage
─────
    python eval/run_one.py --pipeline 1 --horizon 28800
    python eval/run_one.py --pipeline 2 --horizon 200000
    python eval/run_one.py --pipeline 3
    python eval/run_one.py --pipeline 4

Requires the project's generated CSVs to already exist (as they do in the
current working dir).  ``run_one.py`` does not regenerate them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Imports after sys.path tweak
from model.charging_layout_generator import ChargingLayoutGenerator  # noqa: E402
import netlogo  # noqa: E402


PIPELINE_DEFAULTS = {
    1: {"pipeline": 1, "d": 15},
    2: {"pipeline": 2, "c": 1.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
        "num_chargers": 12},
    3: {"pipeline": 3, "num_chargers": 12},
    4: {"pipeline": 4, "num_chargers": 12},
}


def load_grid(path: Path) -> np.ndarray:
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append([int(x) for x in r])
    return np.array(rows, dtype=int)


def clear_transient_state(workdir: Path) -> None:
    """Delete files that would contaminate a fresh run."""
    for rel in (
        "netlogo.state",
        "assign_order.csv",
        "pod_info.csv",
        "output/order-finished.csv",
    ):
        p = workdir / rel
        if p.exists():
            p.unlink()


def write_charger_overlay(workdir: Path, pipeline: int) -> dict:
    """Rerun the charging overlay for pipeline P on the EXISTING grid."""
    grid = load_grid(workdir / "generated_pod.csv")
    config = dict(PIPELINE_DEFAULTS[pipeline])
    gen = ChargingLayoutGenerator(grid, config)
    gen.generate()  # sets config["charger_positions"] and ["num_chargers"]
    with open(workdir / "charging_config.json", "w") as f:
        json.dump(config, f)
    return config


def snapshot_outputs(workdir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for rel in (
        "charging_config.json",
        "output/order-finished.csv",
        "netlogo.state",
    ):
        src = workdir / rel
        if src.exists():
            target = dest / Path(rel).name
            shutil.copy2(src, target)


def extract_per_robot_snapshot(state_path: Path) -> list[dict]:
    """Load the final universe and extract per-robot end-of-run stats."""
    with open(state_path, "rb") as f:
        universe = pickle.load(f)
    rows = []
    for obj in universe._objects:
        if getattr(obj, "object_type", None) != "robot":
            continue
        rows.append({
            "robot_id": getattr(obj, "id", None),
            "battery_level_j": float(getattr(obj, "battery_level_j", 0.0)),
            "battery_pct": float(getattr(obj, "battery_pct", 0.0)),
            "energy_consumption_j": float(getattr(obj, "energy_consumption", 0.0)),
            "current_state": getattr(obj, "current_state", None),
            "is_charging": bool(getattr(obj, "is_charging", False)),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--horizon", type=int, default=28800,
                   help="Max simulation ticks (default 28800 ≈ 72 sim-min).")
    p.add_argument("--heartbeat", type=int, default=2000,
                   help="Print heartbeat every N ticks (0 = silent).")
    p.add_argument("--out-root", type=str, default="eval/runs",
                   help="Where to snapshot per-run artifacts.")
    args = p.parse_args()

    workdir = ROOT
    dest = ROOT / args.out_root / f"p{args.pipeline}"

    print(f"[run_one] pipeline={args.pipeline} horizon={args.horizon}")
    clear_transient_state(workdir)
    cfg = write_charger_overlay(workdir, args.pipeline)
    print(f"[run_one] overlay: {cfg.get('num_chargers')} chargers selected")

    t0 = time.time()
    setup_result = netlogo.setup()
    if isinstance(setup_result, str) and setup_result.startswith("An error"):
        print("[run_one] SETUP FAILED:", setup_result)
        return 1
    t_setup = time.time() - t0
    print(f"[run_one] setup() done in {t_setup:.1f}s")

    t1 = time.time()
    tick_result = netlogo.console_tick(
        max_ticks=args.horizon,
        heartbeat_every=args.heartbeat,
    )
    t_tick = time.time() - t1
    print(f"[run_one] console_tick() done in {t_tick:.1f}s  result={tick_result!r}")

    # Snapshot artefacts
    snapshot_outputs(workdir, dest)

    # End-of-run per-robot snapshot
    per_robot = extract_per_robot_snapshot(dest / "netlogo.state")
    with open(dest / "per_robot.json", "w") as f:
        json.dump(per_robot, f, indent=2, default=str)

    # Wall-clock + tick summary for quick inspection
    summary = {
        "pipeline": args.pipeline,
        "horizon": args.horizon,
        "setup_seconds": round(t_setup, 2),
        "tick_seconds": round(t_tick, 2),
        "ticks_per_second": round(args.horizon / t_tick, 1) if t_tick else None,
        "tick_result": tick_result,
        "num_robots": len(per_robot),
        "num_chargers": cfg.get("num_chargers"),
        "charger_positions": cfg.get("charger_positions"),
    }
    with open(dest / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[run_one] summary written to {dest / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
