"""
eval/plot_charger_sweep.py
──────────────────────────
Read each n{NUM}/per_robot.json + run_summary.json under
eval/runs/p3_sweep/ and plot energy-ratio metrics vs num_chargers.

Per run (n robots = 20, BATTERY_CAPACITY = 6.48 MJ each):
  motion_j    = sum(robot.energy_consumption_j)         [motion only]
  base_j      = BASE_DRAIN_RATE * horizon * num_robots  [idle drain approx]
  consumed_j  = motion_j + base_j                       [total drain from battery]
  depleted_j  = sum(BATT_CAP - robot.battery_level_j)   [net loss observed]
  gained_j    = consumed_j - depleted_j                 [charge replenished]

Note: base drain is skipped while is_charging=True, so `base_j` slightly
overestimates actual base drain (~14% charging time → ~14% overstated).
Conservative upper bound on "consumed".

Ratios:
  replenish_ratio = gained_j  / consumed_j   — fraction of consumption recovered
  sustain_ratio   = gained_j  / depleted_j   — >1 impossible, =1 sustainable
                                                 (here always = consumed/depleted - 1
                                                  so we plot replenish only +
                                                  raw MJ bars for context)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "eval" / "runs" / "p3_sweep"
BATTERY_CAPACITY_J = 6_480_000.0
BASE_DRAIN_RATE_J_PER_S = 90.0


def collect() -> list[dict]:
    rows = []
    for sub in sorted(SWEEP_DIR.glob("n*"), key=lambda p: int(p.name[1:].split("_")[0])):
        if sub.name.endswith("_tmp"):
            continue
        per_robot_p = sub / "per_robot.json"
        summary_p = sub / "run_summary.json"
        if not per_robot_p.exists() or not summary_p.exists():
            continue
        per_robot = json.loads(per_robot_p.read_text())
        summary = json.loads(summary_p.read_text())
        n = int(sub.name[1:])
        horizon = summary.get("horizon", 0) or 0
        num_robots = len(per_robot)
        motion = sum(r["energy_consumption_j"] for r in per_robot)
        base = BASE_DRAIN_RATE_J_PER_S * horizon * num_robots
        consumed = motion + base
        depleted = sum(BATTERY_CAPACITY_J - r["battery_level_j"] for r in per_robot)
        gained = consumed - depleted
        n_dead = sum(1 for r in per_robot if r.get("current_state") == "dead")
        rows.append({
            "num_chargers": n,
            "num_robots": num_robots,
            "motion_mj": motion / 1e6,
            "base_mj": base / 1e6,
            "consumed_mj": consumed / 1e6,
            "depleted_mj": depleted / 1e6,
            "gained_mj": gained / 1e6,
            "replenish_ratio": gained / consumed if consumed else 0.0,
            "sustain_ratio": gained / depleted if depleted else float("inf"),
            "num_dead": n_dead,
            "horizon": horizon,
        })
    return rows


def plot(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("[plot] no completed runs found")
        return
    nums = [r["num_chargers"] for r in rows]
    consumed = [r["consumed_mj"] for r in rows]
    depleted = [r["depleted_mj"] for r in rows]
    gained = [r["gained_mj"] for r in rows]
    replenish = [r["replenish_ratio"] for r in rows]
    sustain = [r["sustain_ratio"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(nums, replenish, "o-", color="tab:blue", label="replenish ratio = gained / consumed")
    ax.plot(nums, sustain, "s--", color="tab:green", label="sustain ratio = gained / depleted")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("num_chargers")
    ax.set_ylabel("ratio")
    ax.set_title("P3 energy ratios vs num_chargers (20k ticks)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    width = 1.2
    x = nums
    ax.bar([n - width for n in x], consumed, width=width, label="consumed (motion+drain)",
           color="tab:red", alpha=0.75)
    ax.bar(x, depleted, width=width, label="net depleted (battery)",
           color="tab:orange", alpha=0.85)
    ax.bar([n + width for n in x], gained, width=width, label="charge gained",
           color="tab:blue", alpha=0.85)
    ax.set_xlabel("num_chargers")
    ax.set_ylabel("energy (MJ, fleet total)")
    ax.set_title("P3 energy budget vs num_chargers")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[plot] saved {out_path}")


def main() -> int:
    rows = collect()
    print(f"[plot] {len(rows)} runs")
    for r in rows:
        print(
            f"  n={r['num_chargers']:>3}  "
            f"consumed={r['consumed_mj']:6.2f}MJ  "
            f"depleted={r['depleted_mj']:6.2f}MJ  "
            f"gained={r['gained_mj']:6.2f}MJ  "
            f"replenish={r['replenish_ratio']*100:5.1f}%  "
            f"sustain={r['sustain_ratio']*100:5.1f}%  "
            f"dead={r['num_dead']}/{r['num_robots']}"
        )
    out_csv = SWEEP_DIR / "sweep_metrics.json"
    out_csv.write_text(json.dumps(rows, indent=2))
    plot(rows, SWEEP_DIR / "sweep_energy_ratio.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
