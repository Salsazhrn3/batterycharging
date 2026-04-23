"""
eval/extract_kpis.py
────────────────────
Parse a per-pipeline run directory (``eval/runs/p{P}/``) and emit a single
CSV row of evaluation KPIs into ``eval/results/run_p{P}.csv``.

Inputs (per run directory)
──────────────────────────
* ``run_summary.json``     — horizon, wall-clock, sim totals (energy,
                             stop_and_go, turning, pending_jobs)
* ``order-finished.csv``   — one row per completed order with
                             (order_id, order_arrival, process_start_time,
                              order_complete_time, station_id)
* ``per_robot.json``       — end-of-run per-robot snapshot
                             (battery_pct, energy_consumption_j,
                              current_state, is_charging)
* ``charging_config.json`` — overlay used (num_chargers, positions)

Emitted KPIs
────────────
Throughput
    orders_completed           count
    throughput_per_sim_hour     orders per sim-hour

Latency
    cycle_time_mean_s           mean(order_complete − order_arrival)
    cycle_time_median_s         median of the same
    cycle_time_p95_s            95th percentile of the same

Energy
    total_energy_j              universe.total_energy at end of run
    energy_per_order_j          total_energy / orders_completed

Congestion
    stop_and_go                 sim-wide counter
    total_turning               sim-wide counter

Battery state (end-of-run snapshot, per-robot aggregated)
    battery_pct_mean            mean across robots
    battery_pct_min             worst robot
    robots_charging_now         count where is_charging == True
    robots_in_going_to_charge   count where state == "going_to_charge"

Usage
─────
    python eval/extract_kpis.py --pipeline 1
    python eval/extract_kpis.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "eval" / "runs"
RESULTS_DIR = ROOT / "eval" / "results"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def parse_one(run_dir: Path) -> dict:
    with open(run_dir / "run_summary.json") as f:
        summary = json.load(f)
    with open(run_dir / "per_robot.json") as f:
        robots = json.load(f)
    with open(run_dir / "charging_config.json") as f:
        charging = json.load(f)

    horizon_s = summary["horizon"]
    tick_result = summary.get("tick_result") or {}
    # Defensive: a failed run stores the error as a string; treat as missing.
    if not isinstance(tick_result, dict):
        tick_result = {}

    # ── Orders ──────────────────────────────────────────────────────────
    cycle_times: list[float] = []
    order_count = 0
    of_path = run_dir / "order-finished.csv"
    if of_path.exists():
        with open(of_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    arrival = float(row["order_arrival"])
                    complete = float(row["order_complete_time"])
                    cycle_times.append(complete - arrival)
                    order_count += 1
                except (KeyError, ValueError):
                    continue

    throughput_per_hour = (
        order_count / (horizon_s / 3600.0) if horizon_s else float("nan")
    )

    # ── Battery / robot snapshot ────────────────────────────────────────
    batt = [r.get("battery_pct", 0.0) for r in robots]
    batt_mean = statistics.mean(batt) if batt else float("nan")
    batt_min = min(batt) if batt else float("nan")
    charging_now = sum(1 for r in robots if r.get("is_charging"))
    going_to_charge = sum(
        1 for r in robots if r.get("current_state") == "going_to_charge"
    )

    # ── Energy ──────────────────────────────────────────────────────────
    total_energy = tick_result.get("total_energy")
    if total_energy is None:
        total_energy = sum(r.get("energy_consumption_j", 0.0) for r in robots)
    energy_per_order = (
        total_energy / order_count if order_count else float("nan")
    )

    # ── Pack ────────────────────────────────────────────────────────────
    return {
        "pipeline": summary["pipeline"],
        "horizon_s": horizon_s,
        "wallclock_s": summary.get("tick_seconds"),
        "num_robots": summary.get("num_robots"),
        "num_chargers": charging.get("num_chargers"),
        "orders_completed": order_count,
        "throughput_per_sim_hour": round(throughput_per_hour, 2),
        "cycle_time_mean_s": round(statistics.mean(cycle_times), 2) if cycle_times else None,
        "cycle_time_median_s": round(statistics.median(cycle_times), 2) if cycle_times else None,
        "cycle_time_p95_s": round(percentile(cycle_times, 95), 2) if cycle_times else None,
        "total_energy_j": round(float(total_energy), 1) if total_energy is not None else None,
        "energy_per_order_j": round(energy_per_order, 1) if order_count and total_energy else None,
        "stop_and_go": tick_result.get("stop_and_go"),
        "total_turning": tick_result.get("total_turning"),
        "pending_jobs_at_end": tick_result.get("pending_jobs"),
        "battery_pct_mean": round(batt_mean, 2),
        "battery_pct_min": round(batt_min, 2),
        "robots_charging_now": charging_now,
        "robots_going_to_charge": going_to_charge,
    }


def write_row(row: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    targets: list[int]
    if args.all:
        targets = [1, 2, 3, 4]
    elif args.pipeline is not None:
        targets = [args.pipeline]
    else:
        p.error("Pass --pipeline N or --all")
        return 2

    for P in targets:
        run_dir = RUNS_DIR / f"p{P}"
        if not (run_dir / "run_summary.json").exists():
            print(f"[skip] p{P}: no run_summary.json at {run_dir}")
            continue
        row = parse_one(run_dir)
        out = RESULTS_DIR / f"run_p{P}.csv"
        write_row(row, out)
        print(f"[ok] p{P} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
