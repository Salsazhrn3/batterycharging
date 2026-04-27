"""
eval/plot_combined_sweep.py
───────────────────────────
Combined charger-budget sweep figure for P1, P2, P3 at the 20k horizon.

Reads each n{N}/per_robot.json + run_summary.json under
eval/runs/p{1,2,3}_sweep/ and overlays three curves on the same axes:

  - replenish_ratio = gained / consumed (fraction of consumption recovered)
  - orders_completed (raw productivity)

Note: at 20k, batteries do not drop below ~30% (preemption rarely
fires); these curves measure pure-placement effectiveness, not the
preemption-policy effect tested at 100k.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BATTERY_CAPACITY_J = 6_480_000.0
BASE_DRAIN_J_PER_S = 90.0

PIPELINES = {
    1: ("P1 Set Cover",          "#1f77b4", "o"),
    2: ("P2 Affinity Prop.",     "#2ca02c", "s"),
    3: ("P3 Picker-Tier (opp.)", "#ff7f0e", "^"),
}


def collect(pipeline: int) -> list[dict]:
    sweep = ROOT / "eval" / "runs" / f"p{pipeline}_sweep"
    rows = []
    if not sweep.exists():
        return rows
    for sub in sorted(sweep.glob("n*"), key=lambda p: int(p.name[1:].split("_")[0])):
        if sub.name.endswith("_tmp"):
            continue
        # P1/P2 sweep stores files directly in n{N}/, but P3 sweep nests
        # them in n{N}/p3/ historically -try both.
        candidates = [sub, sub / f"p{pipeline}"]
        run_dir = next((c for c in candidates
                        if (c / "per_robot.json").exists()), None)
        if run_dir is None:
            continue
        per_robot = json.loads((run_dir / "per_robot.json").read_text())
        summary = json.loads((run_dir / "run_summary.json").read_text())
        n = int(sub.name[1:])
        horizon = summary.get("horizon", 0) or 0
        num_robots = len(per_robot)
        motion = sum(r["energy_consumption_j"] for r in per_robot)
        base = BASE_DRAIN_J_PER_S * horizon * num_robots
        consumed = motion + base
        depleted = sum(BATTERY_CAPACITY_J - r["battery_level_j"]
                       for r in per_robot)
        gained = consumed - depleted
        n_dead = sum(1 for r in per_robot
                     if r.get("current_state") == "dead"
                     or r.get("battery_pct", 100) <= 0.5)

        # orders from order-finished.csv if present
        orders = 0
        of = run_dir / "order-finished.csv"
        if of.exists():
            with open(of) as f:
                orders = sum(1 for _ in f) - 1

        rows.append({
            "num_chargers": n,
            "consumed_mj": consumed / 1e6,
            "depleted_mj": depleted / 1e6,
            "gained_mj": gained / 1e6,
            "replenish_pct": gained / consumed * 100 if consumed else 0,
            "orders": orders,
            "num_dead": n_dead,
        })
    return rows


def main() -> int:
    data: dict[int, list[dict]] = {}
    for P in PIPELINES:
        data[P] = collect(P)
        print(f"[load] p{P}_sweep: {len(data[P])} runs")
        for r in data[P]:
            print(f"  n={r['num_chargers']:>2}  "
                  f"replenish={r['replenish_pct']:5.1f}%  "
                  f"orders={r['orders']}  dead={r['num_dead']}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: replenish ratio
    ax = axes[0]
    for P, (label, color, marker) in PIPELINES.items():
        rows = data.get(P, [])
        if not rows:
            continue
        ax.plot([r["num_chargers"] for r in rows],
                [r["replenish_pct"] for r in rows],
                f"{marker}-", color=color, label=label,
                linewidth=2, markersize=9)
    ax.axhline(100, color="grey", linewidth=0.8, linestyle=":")
    ax.text(5.2, 102, "100% (sustainable)", color="grey", fontsize=9)
    ax.set_xlabel("Number of chargers")
    ax.set_ylabel("Replenish ratio (%)")
    ax.set_title("Energy replenishment vs charger budget (T = 20,000 s)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # right: orders completed
    ax = axes[1]
    for P, (label, color, marker) in PIPELINES.items():
        rows = data.get(P, [])
        if not rows:
            continue
        ax.plot([r["num_chargers"] for r in rows],
                [r["orders"] for r in rows],
                f"{marker}-", color=color, label=label,
                linewidth=2, markersize=9)
    ax.set_xlabel("Number of chargers")
    ax.set_ylabel("Orders completed")
    ax.set_title("Throughput vs charger budget (T = 20,000 s)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = ROOT / "eval" / "results" / "figures" / "combined_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"[fig] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
