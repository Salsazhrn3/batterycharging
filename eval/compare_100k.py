"""
eval/compare_100k.py
────────────────────
Cross-pipeline comparison for 100k-tick runs.

Ingests per_robot.json + run_summary.json + order-finished.csv from:
  eval/runs/p1_100k/p1/
  eval/runs/p2_100k/p2/
  eval/runs/p4_100k/p4/
  (optionally) eval/runs/p3_100k/p3/

Produces:
  - console summary table
  - eval/results/p100k_summary.csv (machine-readable)
  - eval/results/p100k_summary.md (human-readable markdown table)

Metrics per pipeline:
  - robots_dead (headline survivorship)
  - battery_pct (min / mean / max)
  - charging_now, going_to_charge (fleet-state snapshot)
  - orders_completed, throughput_per_hr
  - cycle_time_mean, p95
  - energy (consumed, depleted, gained, replenish ratio, sustain ratio)
  - wallclock_sec

Usage:
    python eval/compare_100k.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "eval" / "runs"
RESULTS_DIR = ROOT / "eval" / "results"

BATTERY_CAPACITY_J = 6_480_000.0
BASE_DRAIN_J_PER_S = 90.0


def load_one(pipeline: int) -> dict | None:
    run_dir = RUNS_DIR / f"p{pipeline}_100k" / f"p{pipeline}"
    per_robot_p = run_dir / "per_robot.json"
    summary_p = run_dir / "run_summary.json"
    of_p = run_dir / "order-finished.csv"

    if not per_robot_p.exists() or not summary_p.exists():
        return None

    per_robot = json.loads(per_robot_p.read_text())
    summary = json.loads(summary_p.read_text())

    horizon = summary.get("horizon", 0)
    num_robots = len(per_robot)
    wallclock = summary.get("tick_seconds", None)

    # robot states
    states: dict[str, int] = {}
    for r in per_robot:
        s = r.get("current_state", "?")
        states[s] = states.get(s, 0) + 1
    charging_now = sum(1 for r in per_robot if r.get("is_charging"))
    going_to_charge = states.get("going_to_charge", 0)
    dead = states.get("dead", 0) + sum(
        1 for r in per_robot if r.get("battery_pct", 100) <= 0.5
        and r.get("current_state") != "dead"
    )

    # battery
    batt = [r.get("battery_pct", 0.0) for r in per_robot]
    batt_min = min(batt)
    batt_mean = statistics.mean(batt)
    batt_max = max(batt)

    # energy balance
    tick_result = summary.get("tick_result") or {}
    motion_j = tick_result.get("total_energy") if isinstance(tick_result, dict) else None
    if motion_j is None:
        motion_j = sum(r.get("energy_consumption_j", 0.0) for r in per_robot)
    base_j = BASE_DRAIN_J_PER_S * horizon * num_robots
    consumed_j = motion_j + base_j
    depleted_j = sum(
        BATTERY_CAPACITY_J - r.get("battery_level_j", 0) for r in per_robot
    )
    gained_j = consumed_j - depleted_j
    replenish_ratio = gained_j / consumed_j if consumed_j else 0
    sustain_ratio = gained_j / depleted_j if depleted_j else float("inf")

    # orders / cycle
    order_count = 0
    cycle_times: list[float] = []
    if of_p.exists():
        with open(of_p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    a = float(row["order_arrival"])
                    c = float(row["order_complete_time"])
                    cycle_times.append(c - a)
                    order_count += 1
                except (KeyError, ValueError):
                    continue

    throughput_per_hr = (
        order_count / (horizon / 3600.0) if horizon else float("nan")
    )

    return {
        "pipeline": pipeline,
        "horizon": horizon,
        "wallclock_s": wallclock,
        "num_robots": num_robots,
        "num_chargers": summary.get("num_chargers"),
        "robots_dead": dead,
        "batt_min": round(batt_min, 1),
        "batt_mean": round(batt_mean, 1),
        "batt_max": round(batt_max, 1),
        "charging_now": charging_now,
        "going_to_charge": going_to_charge,
        "states": states,
        "orders_completed": order_count,
        "throughput_per_hr": round(throughput_per_hr, 2),
        "cycle_mean_s": (
            round(statistics.mean(cycle_times), 1) if cycle_times else None
        ),
        "cycle_p95_s": (
            round(sorted(cycle_times)[int(0.95 * (len(cycle_times) - 1))], 1)
            if len(cycle_times) > 1 else None
        ),
        "motion_mj": round(motion_j / 1e6, 2),
        "base_mj": round(base_j / 1e6, 2),
        "consumed_mj": round(consumed_j / 1e6, 2),
        "depleted_mj": round(depleted_j / 1e6, 2),
        "gained_mj": round(gained_j / 1e6, 2),
        "replenish_pct": round(replenish_ratio * 100, 1),
        "sustain_pct": round(sustain_ratio * 100, 1),
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("[compare] no runs found")
        return
    print(f"\n{'=' * 92}")
    print(f"  100k-tick pipeline comparison")
    print(f"{'=' * 92}")
    header = (
        f"{'P':>2}  {'dead':>4}  {'batt_min':>8}  {'batt_mean':>9}  "
        f"{'orders':>6}  {'rep%':>5}  {'sust%':>6}  "
        f"{'cons_MJ':>7}  {'dep_MJ':>6}  {'gain_MJ':>7}"
    )
    print(header)
    print("-" * 92)
    for r in rows:
        print(
            f"{r['pipeline']:>2}  {r['robots_dead']:>4}  "
            f"{r['batt_min']:>8.1f}  {r['batt_mean']:>9.1f}  "
            f"{r['orders_completed']:>6}  {r['replenish_pct']:>5.1f}  "
            f"{r['sustain_pct']:>6.1f}  "
            f"{r['consumed_mj']:>7.2f}  {r['depleted_mj']:>6.2f}  "
            f"{r['gained_mj']:>7.2f}"
        )
    print(f"{'=' * 92}\n")

    # state breakdown
    print("State breakdown (end-of-run):")
    all_states = sorted({s for r in rows for s in r["states"]})
    hdr = f"  {'state':<22}  " + "  ".join(f"P{r['pipeline']:<3}" for r in rows)
    print(hdr)
    for s in all_states:
        cells = "  ".join(
            f"{r['states'].get(s, 0):<4}" for r in rows
        )
        print(f"  {s:<22}  {cells}")
    print()


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    keys = [
        "pipeline", "horizon", "wallclock_s", "num_robots", "num_chargers",
        "robots_dead", "batt_min", "batt_mean", "batt_max",
        "charging_now", "going_to_charge",
        "orders_completed", "throughput_per_hr",
        "cycle_mean_s", "cycle_p95_s",
        "motion_mj", "base_mj", "consumed_mj", "depleted_mj", "gained_mj",
        "replenish_pct", "sustain_pct",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"[csv] {out_path}")


def write_md(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    lines = [
        "# 100k-tick pipeline comparison",
        "",
        "| P | Dead | Batt min | Batt mean | Orders | Replenish% | Sustain% | Cons (MJ) | Dep (MJ) | Gain (MJ) |",
        "|---|------|---------|-----------|--------|-----------|---------|-----------|----------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| P{r['pipeline']} | {r['robots_dead']} | "
            f"{r['batt_min']:.1f}% | {r['batt_mean']:.1f}% | "
            f"{r['orders_completed']} | {r['replenish_pct']:.1f}% | "
            f"{r['sustain_pct']:.1f}% | {r['consumed_mj']:.2f} | "
            f"{r['depleted_mj']:.2f} | {r['gained_mj']:.2f} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[md] {out_path}")


def main() -> int:
    rows = []
    for P in (1, 2, 3, 4):
        r = load_one(P)
        if r:
            rows.append(r)
            print(f"[load] p{P}_100k OK ({r['horizon']}s, "
                  f"dead={r['robots_dead']}/{r['num_robots']})")
        else:
            print(f"[skip] p{P}_100k not found")

    print_table(rows)
    write_csv(rows, RESULTS_DIR / "p100k_summary.csv")
    write_md(rows, RESULTS_DIR / "p100k_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
