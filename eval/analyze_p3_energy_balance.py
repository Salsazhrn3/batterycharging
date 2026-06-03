"""
eval/analyze_p3_energy_balance.py
─────────────────────────────────
P3 picker-corridor charger-count analysis (Phase 4 redirect):

The question is no longer "what is the best threshold policy on P3?"
(P3 is opportunistic; threshold matters little). The question is:

    What is the minimum |C| at the picker stations such that the energy
    gained per dwell cycle covers the energy spent in one trip cycle,
    without overcharging (E_in ≫ E_out wastes infrastructure)?

For each existing P3 charger-sweep run at horizon 20k we compute:

  orders               # rows in order-finished.csv minus header
  energy_consumed_MJ   tick_result.total_energy / 1e6
  kJ_per_order         total_energy / orders / 1e3
  mean_final_battery   mean per-robot battery_pct at end of horizon
  frac_dead            fraction of fleet with battery_pct <= 0.5
  E_gain_total_MJ      total energy delivered by chargers (per-fleet)
                       = sum_r (E_consumed_r - (E_initial_r - E_final_r))
  cycles_per_robot     orders / fleet_size   (mean order-completions per robot)
  E_spend_per_cycle_kJ E_consumed_per_robot / cycles_per_robot   (fleet mean)
  E_gain_per_cycle_kJ  E_gain_per_robot / cycles_per_robot
  balance_ratio        E_gain_per_cycle / E_spend_per_cycle      (target ≥ 1)
  overcharge_index     mean_final_battery > 90 indicator of slack

Outputs:
  eval/results/p3_energy_balance.csv   summary table
  eval/results/figures/p3_energy_balance.png   E_gain/E_spend vs |C|

Usage
─────
    python eval/analyze_p3_energy_balance.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "eval" / "runs" / "p3_sweep"
OUT_CSV = ROOT / "eval" / "results" / "p3_energy_balance.csv"
OUT_FIG = ROOT / "eval" / "results" / "figures" / "p3_energy_balance.png"

BATTERY_CAPACITY_J = 6_480_000.0   # Robot.BATTERY_CAPACITY_J (1.8 kWh)
BASE_DRAIN_J_PER_S = 90.0          # Robot.BASE_DRAIN_RATE_PER_S
CHARGE_POWER_W = 397.6             # Robot.CHARGE_POWER_W
FLEET_SIZE_DEFAULT = 20


def analyse_cell(cell_dir: Path) -> dict | None:
    summary_p = cell_dir / "run_summary.json"
    per_robot_p = cell_dir / "per_robot.json"
    orders_p = cell_dir / "order-finished.csv"
    if not (summary_p.exists() and per_robot_p.exists() and orders_p.exists()):
        return None

    summary = json.loads(summary_p.read_text())
    robots = json.loads(per_robot_p.read_text())
    with open(orders_p) as f:
        orders = max(0, sum(1 for _ in f) - 1)

    num_chargers = summary.get("num_chargers")
    fleet_size = summary.get("num_robots", FLEET_SIZE_DEFAULT) or FLEET_SIZE_DEFAULT
    horizon_s = float(summary.get("horizon", 20000))
    total_motion_energy = float(summary["tick_result"]["total_energy"])

    # Per-robot energy accounting.
    #   Motion energy (lifting, rotation, drive-by drain accounted elsewhere):
    #     stored in per_robot.energy_consumption_j.
    #   Base drain (electronics, constant 90 J/s while not parked):
    #     NOT included in energy_consumption_j; add separately.
    #   Battery change ΔB = E_initial - E_final  (positive = depletion).
    #   Energy balance: ΔB = motion + base_drain - charger_gain.
    #   ∴ charger_gain = motion + base_drain - ΔB.
    base_drain_per_robot = BASE_DRAIN_J_PER_S * horizon_s
    e_motion = [float(r["energy_consumption_j"]) for r in robots]
    e_final = [float(r["battery_pct"]) / 100.0 * BATTERY_CAPACITY_J
               for r in robots]
    delta_batt = [BATTERY_CAPACITY_J - f for f in e_final]    # depletion
    e_spend = [m + base_drain_per_robot for m in e_motion]    # what cycle used
    e_gain = [s - d for s, d in zip(e_spend, delta_batt)]     # from charger

    cycles_per_robot = orders / fleet_size if fleet_size else 0.0
    mean_spend = sum(e_spend) / len(e_spend) if e_spend else 0.0
    mean_gain = sum(e_gain) / len(e_gain) if e_gain else 0.0
    e_spend_cycle = (mean_spend / cycles_per_robot
                     if cycles_per_robot > 0 else 0.0)
    e_gain_cycle = (mean_gain / cycles_per_robot
                    if cycles_per_robot > 0 else 0.0)
    balance_ratio = (e_gain_cycle / e_spend_cycle
                     if e_spend_cycle > 0 else 0.0)
    mean_final_pct = (sum(r["battery_pct"] for r in robots) / len(robots)
                      if robots else 0.0)
    frac_dead = (sum(1 for r in robots
                     if float(r["battery_pct"]) <= 0.5) / len(robots)
                 if robots else 0.0)

    return {
        "num_chargers": num_chargers,
        "orders": orders,
        "motion_energy_MJ": round(total_motion_energy / 1e6, 2),
        "base_drain_fleet_MJ": round(base_drain_per_robot * fleet_size / 1e6, 2),
        "total_spend_fleet_MJ": round(sum(e_spend) / 1e6, 2),
        "charger_gain_fleet_MJ": round(sum(e_gain) / 1e6, 2),
        "kJ_per_order": round(sum(e_spend) / orders / 1e3, 2)
                        if orders else 0.0,
        "cycles_per_robot": round(cycles_per_robot, 2),
        "E_spend_per_cycle_kJ": round(e_spend_cycle / 1e3, 2),
        "E_gain_per_cycle_kJ": round(e_gain_cycle / 1e3, 2),
        "balance_ratio": round(balance_ratio, 3),
        "mean_final_battery_pct": round(mean_final_pct, 2),
        "frac_dead": round(frac_dead, 3),
        "overcharge_flag": 1 if mean_final_pct > 90.0 else 0,
    }


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell in sorted(SWEEP_DIR.glob("n*"),
                       key=lambda p: int(p.name[1:])):
        result = analyse_cell(cell)
        if result is None:
            print(f"[skip] {cell.name} missing artefacts")
            continue
        rows.append(result)

    if not rows:
        print("[err] no cells found")
        return 1

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {OUT_CSV} ({len(rows)} rows)")

    # Console table.
    col_w = max(len(c) for c in fieldnames) + 1
    print()
    print(" | ".join(c.rjust(col_w) for c in fieldnames))
    for r in rows:
        print(" | ".join(str(r[c]).rjust(col_w) for c in fieldnames))

    # Knee-point heuristic: smallest |C| with balance_ratio >= 1 and
    # frac_dead == 0.
    viable = [r for r in rows
              if r["balance_ratio"] >= 1.0 and r["frac_dead"] == 0]
    if viable:
        knee = min(viable, key=lambda r: r["num_chargers"])
        print()
        print(f"[knee] minimum viable |C| = {knee['num_chargers']} "
              f"(balance={knee['balance_ratio']}, "
              f"frac_dead={knee['frac_dead']}, "
              f"kJ/order={knee['kJ_per_order']})")
    else:
        print()
        print("[knee] no |C| achieved balance_ratio>=1 with zero deaths "
              "— constraint is more demanding than current sweep covers.")

    # Plot.
    try:
        import matplotlib.pyplot as plt
        xs = [r["num_chargers"] for r in rows]
        gain = [r["E_gain_per_cycle_kJ"] for r in rows]
        spend = [r["E_spend_per_cycle_kJ"] for r in rows]
        ratio = [r["balance_ratio"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(xs, gain, "o-", label="E_gain/cycle (kJ)")
        ax1.plot(xs, spend, "s-", label="E_spend/cycle (kJ)")
        ax1.set_xlabel("|C| (picker-station chargers)")
        ax1.set_ylabel("Energy per cycle (kJ)")
        ax1.set_title("Per-cycle energy balance")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.plot(xs, ratio, "^-", color="C2")
        ax2.axhline(1.0, ls="--", color="red", label="balance threshold")
        ax2.set_xlabel("|C| (picker-station chargers)")
        ax2.set_ylabel("E_gain / E_spend ratio")
        ax2.set_title("Energy balance ratio vs |C|")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=150)
        print(f"[ok] wrote {OUT_FIG}")
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
