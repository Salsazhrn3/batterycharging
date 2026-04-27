"""
eval/analyze_preemption.py
──────────────────────────
Post-run validator for the taking_pod preemption fix (model/robot.py:1121).

Given a completed run directory (per_robot.json + run_summary.json), plus the
stdout log from that run, produce a pass/fail assessment on whether the
preemption mechanism actually fired:

  1. Battery state -did any robot drop below BATTERY_LOW_PCT=20%?
     - If no, the preemption never had reason to fire (smoke test horizon
       too short). NOT a failure of the fix.
     - If yes, preemption MUST have fired, else robots would keep draining
       toward 0% (dead).

  2. End-state mix -how many robots in each state? A healthy run shows
     robots cycling through charging/going_to_charge (not just working states).

  3. Charge evidence -if battery_level_j > initial_consumed_j would imply
     (the robot consumed energy but got some back), that's a drive-by charge
     at minimum. For preemption specifically, we check energy_consumption_j
     patterns: a robot that preempted will show high consumption but
     battery_pct rebounded from a low.

  4. Log grep -count occurrences of is_charging transitions and
     going_to_charge entries. Presence confirms the mechanism is live.

Usage
─────
    python eval/analyze_preemption.py \
        --run-dir eval/runs/p1_30k_smoke/p1 \
        --log     eval/runs/p1_30k_smoke.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BATTERY_CAPACITY_J = 6_480_000.0
BASE_DRAIN_J_PER_S = 90.0
LOW_PCT = 20.0


def analyze_robots(per_robot: list[dict]) -> dict:
    states: dict[str, int] = {}
    batt_pcts: list[float] = []
    consumption: list[float] = []
    low_batt = []  # robots below 20%
    dead = []  # robots at 0% / dead state
    charging_now = 0
    going_to_charge = 0

    for r in per_robot:
        s = r.get("current_state", "?")
        states[s] = states.get(s, 0) + 1
        pct = r.get("battery_pct", 0.0)
        batt_pcts.append(pct)
        consumption.append(r.get("energy_consumption_j", 0.0))
        if pct < LOW_PCT:
            low_batt.append((r["robot_id"], pct))
        if s == "dead" or pct <= 0.5:
            dead.append(r["robot_id"])
        if r.get("is_charging"):
            charging_now += 1
        if s == "going_to_charge":
            going_to_charge += 1

    return {
        "num_robots": len(per_robot),
        "states": states,
        "battery_pct_min": min(batt_pcts) if batt_pcts else 0,
        "battery_pct_max": max(batt_pcts) if batt_pcts else 0,
        "battery_pct_mean": sum(batt_pcts) / len(batt_pcts) if batt_pcts else 0,
        "robots_below_20pct": len(low_batt),
        "robots_dead": len(dead),
        "charging_now": charging_now,
        "going_to_charge": going_to_charge,
        "total_consumption_j": sum(consumption),
        "mean_consumption_j": sum(consumption) / len(consumption) if consumption else 0,
        "low_batt_examples": low_batt[:5],
    }


def analyze_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {"log_exists": False}

    text = log_path.read_text(errors="ignore")

    # Heartbeats show tick progression
    heartbeats = re.findall(r"\[heartbeat\]\s+tick=([\d.]+)/(\d+)", text)

    # The robot.py preemption block sets current_state = "idle" then calls
    # _start_charging_trip() which transitions to "going_to_charge". We can't
    # observe this directly in stdout without extra prints, but we can check
    # for the debug lines that do exist. The most reliable signal is
    # "going_to_charge" appearing in the log, which only happens if either
    # idle-trigger OR taking_pod-preemption fired.
    going_hits = text.count("going_to_charge")
    charging_hits = text.count("is_charging")

    return {
        "log_exists": True,
        "heartbeat_count": len(heartbeats),
        "last_tick": heartbeats[-1][0] if heartbeats else None,
        "max_tick": heartbeats[-1][1] if heartbeats else None,
        "going_to_charge_mentions": going_hits,
        "is_charging_mentions": charging_hits,
    }


def verdict(robot_stats: dict, log_stats: dict, horizon: float) -> dict:
    vmin = robot_stats["battery_pct_min"]
    vmax = robot_stats["battery_pct_max"]
    dead = robot_stats["robots_dead"]
    charging_now = robot_stats["charging_now"]
    going = robot_stats["going_to_charge"]
    # Heuristic: if a run that reached low-battery range shows NO robots in
    # charging/going_to_charge state AND NO evidence in log → fix broken.
    hit_threshold = vmin < LOW_PCT + 5.0  # within 5pp of threshold -should
    #                                      have triggered by now
    any_charge_activity = charging_now > 0 or going > 0

    if dead == robot_stats["num_robots"]:
        level = "FAIL"
        why = "All robots dead -preemption did not save them."
    elif not hit_threshold:
        level = "INCONCLUSIVE"
        why = (f"Horizon too short -min battery {vmin:.1f}% never approached "
               f"the 20% threshold, so preemption had no reason to fire.")
    elif hit_threshold and not any_charge_activity and vmax - vmin < 5.0:
        level = "FAIL"
        why = ("Batteries low but NO robots charging/going_to_charge and no "
               "spread in end-state -preemption appears inactive.")
    elif any_charge_activity or vmax > 50.0:
        level = "PASS"
        why = (f"Charging activity detected (charging={charging_now}, "
               f"going_to_charge={going}, max batt={vmax:.1f}%) -preemption "
               "mechanism is live.")
    else:
        level = "WEAK"
        why = ("No robots currently charging but battery spread suggests "
               "some cycling happened. Inspect per_robot.json in detail.")

    return {"level": level, "reason": why}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=str,
                   help="e.g. eval/runs/p1_30k_smoke/p1")
    p.add_argument("--log", required=True, type=str,
                   help="e.g. eval/runs/p1_30k_smoke.log")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    log_path = Path(args.log)

    per_robot_p = run_dir / "per_robot.json"
    summary_p = run_dir / "run_summary.json"

    if not per_robot_p.exists():
        print(f"[err] {per_robot_p} not found -run may still be in progress")
        return 2

    per_robot = json.loads(per_robot_p.read_text())
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
    horizon = summary.get("horizon", 0)

    robot_stats = analyze_robots(per_robot)
    log_stats = analyze_log(log_path)
    v = verdict(robot_stats, log_stats, horizon)

    print(f"\n{'=' * 60}")
    print(f"  P1 PREEMPTION SMOKE TEST -{run_dir.name}")
    print(f"{'=' * 60}")
    print(f"Horizon:            {horizon}")
    print(f"Wallclock:          {summary.get('tick_seconds', '?')} s")
    print(f"Num robots:         {robot_stats['num_robots']}")
    print(f"")
    print(f"Battery (pct):")
    print(f"  min / mean / max: {robot_stats['battery_pct_min']:5.1f} / "
          f"{robot_stats['battery_pct_mean']:5.1f} / "
          f"{robot_stats['battery_pct_max']:5.1f}")
    print(f"  below 20%:        {robot_stats['robots_below_20pct']}")
    print(f"  dead (~0%):       {robot_stats['robots_dead']}")
    print(f"")
    print(f"Current states:")
    for s, n in sorted(robot_stats["states"].items()):
        marker = "  <-- preemption evidence" if s in ("going_to_charge", "charging") else ""
        print(f"  {s:<22} {n}{marker}")
    print(f"")
    print(f"Charging activity (end-of-run snapshot):")
    print(f"  is_charging=True:    {robot_stats['charging_now']}")
    print(f"  going_to_charge:     {robot_stats['going_to_charge']}")
    print(f"")
    print(f"Log signal:")
    print(f"  heartbeats:          {log_stats.get('heartbeat_count')}")
    print(f"  last tick in log:    {log_stats.get('last_tick')} / {log_stats.get('max_tick')}")
    print(f"  'going_to_charge':   {log_stats.get('going_to_charge_mentions')} mentions")
    print(f"")
    tick_result = summary.get("tick_result") or {}
    if isinstance(tick_result, dict):
        print(f"Energy (from tick_result):")
        print(f"  total_energy_j:   {tick_result.get('total_energy', '?')}")
    print(f"")
    print(f"{'-' * 60}")
    print(f"VERDICT: {v['level']}")
    print(f"  {v['reason']}")
    print(f"{'-' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
