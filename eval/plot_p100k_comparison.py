"""
eval/plot_p100k_comparison.py
─────────────────────────────
Publication-quality figures from eval/results/p100k_summary.csv.

Generates four figures:
  1. survivorship.png       - bar chart of robots_dead per pipeline
  2. throughput.png         - bar chart of orders_completed
  3. tradeoff.png           - scatter of throughput vs (20 - dead)
  4. energy_balance.png     - stacked bars: consumed / depleted / gained
  5. state_breakdown.png    - stacked bars of end-of-run robot states

All figures use a consistent pipeline color scheme.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
RUNS_DIR = ROOT / "eval" / "runs"
SUMMARY_CSV = RESULTS_DIR / "p100k_summary.csv"
OUT_DIR = RESULTS_DIR / "figures"

PIPELINE_LABELS = {
    1: "P1\nSet Cover",
    2: "P2\nAffinity\nProp.",
    3: "P3\nPicker-\nTier",
    4: "P4\nPerimeter",
}
PIPELINE_COLORS = {
    1: "#1f77b4",
    2: "#2ca02c",
    3: "#ff7f0e",
    4: "#d62728",
}


def load_summary() -> list[dict]:
    rows = []
    with open(SUMMARY_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "pipeline": int(r["pipeline"]),
                "robots_dead": int(r["robots_dead"]),
                "orders": int(r["orders_completed"]),
                "consumed": float(r["consumed_mj"]),
                "depleted": float(r["depleted_mj"]),
                "gained": float(r["gained_mj"]),
                "replenish_pct": float(r["replenish_pct"]),
                "batt_mean": float(r["batt_mean"]),
                "batt_min": float(r["batt_min"]),
                "wallclock_s": float(r["wallclock_s"]),
            })
    return sorted(rows, key=lambda r: r["pipeline"])


def load_states() -> dict[int, dict[str, int]]:
    """Per-pipeline end-of-run state breakdown from per_robot.json."""
    out = {}
    for P in (1, 2, 3, 4):
        per_robot = RUNS_DIR / f"p{P}_100k" / f"p{P}" / "per_robot.json"
        if not per_robot.exists():
            continue
        data = json.loads(per_robot.read_text())
        states: dict[str, int] = {}
        for r in data:
            s = r.get("current_state", "?")
            states[s] = states.get(s, 0) + 1
        out[P] = states
    return out


def plot_survivorship(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [PIPELINE_LABELS[r["pipeline"]] for r in rows]
    survivors = [20 - r["robots_dead"] for r in rows]
    dead = [r["robots_dead"] for r in rows]
    colors = [PIPELINE_COLORS[r["pipeline"]] for r in rows]

    bars1 = ax.bar(x, survivors, color=colors, label="Surviving")
    bars2 = ax.bar(x, dead, bottom=survivors, color="lightgrey",
                   edgecolor="black", label="Dead", hatch="///")
    ax.set_ylabel("Robots (out of 20)")
    ax.set_title("Fleet survivorship at T = 100,000 s (20 robots, 12 chargers)")
    ax.set_ylim(0, 22)
    ax.axhline(20, color="black", linewidth=0.5, linestyle=":")
    ax.legend(loc="upper right")
    for b, v in zip(bars1, survivors):
        ax.text(b.get_x() + b.get_width() / 2, v / 2, str(v),
                ha="center", va="center", color="white", fontweight="bold")
    fig.tight_layout()
    p = OUT_DIR / "survivorship.png"
    fig.savefig(p, dpi=160)
    print(f"[fig] {p}")
    plt.close(fig)


def plot_throughput(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [PIPELINE_LABELS[r["pipeline"]] for r in rows]
    orders = [r["orders"] for r in rows]
    colors = [PIPELINE_COLORS[r["pipeline"]] for r in rows]
    bars = ax.bar(x, orders, color=colors)
    ax.set_ylabel("Orders completed")
    ax.set_title("Throughput at T = 100,000 s")
    for b, v in zip(bars, orders):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, str(v),
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(orders) * 1.12)
    fig.tight_layout()
    p = OUT_DIR / "throughput.png"
    fig.savefig(p, dpi=160)
    print(f"[fig] {p}")
    plt.close(fig)


def plot_tradeoff(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in rows:
        ax.scatter(20 - r["robots_dead"], r["orders"],
                   s=350, color=PIPELINE_COLORS[r["pipeline"]],
                   edgecolor="black", linewidth=1.2, zorder=3)
        ax.annotate(f"P{r['pipeline']}",
                    (20 - r["robots_dead"], r["orders"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold")
    ax.set_xlabel("Robots surviving (out of 20)  -- higher is better")
    ax.set_ylabel("Orders completed  -- higher is better")
    ax.set_title("Productivity-survival tradeoff (T = 100,000 s)")
    ax.grid(alpha=0.3, zorder=0)
    ax.set_xlim(0, 21)
    ax.set_ylim(2200, max(r["orders"] for r in rows) + 200)
    fig.tight_layout()
    p = OUT_DIR / "tradeoff.png"
    fig.savefig(p, dpi=160)
    print(f"[fig] {p}")
    plt.close(fig)


def plot_energy(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [PIPELINE_LABELS[r["pipeline"]] for r in rows]
    consumed = [r["consumed"] for r in rows]
    depleted = [r["depleted"] for r in rows]
    gained = [r["gained"] for r in rows]
    width = 0.27
    xpos = list(range(len(x)))
    ax.bar([p - width for p in xpos], consumed, width=width,
           color="tab:red", alpha=0.8, label="Consumed (motion + drain)")
    ax.bar(xpos, depleted, width=width,
           color="tab:orange", alpha=0.85, label="Net depleted (battery)")
    ax.bar([p + width for p in xpos], gained, width=width,
           color="tab:blue", alpha=0.85, label="Gained (charged)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(x)
    ax.set_ylabel("Energy (MJ, fleet total)")
    ax.set_title("Fleet energy balance at T = 100,000 s (12 chargers)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for i, r in enumerate(rows):
        ax.text(i + width, gained[i] + 2,
                f"{r['replenish_pct']:.1f}%",
                ha="center", va="bottom", fontsize=9,
                color="tab:blue", fontweight="bold")
    fig.tight_layout()
    p = OUT_DIR / "energy_balance.png"
    fig.savefig(p, dpi=160)
    print(f"[fig] {p}")
    plt.close(fig)


def plot_states(states: dict[int, dict[str, int]]) -> None:
    if not states:
        return
    all_states = sorted({s for d in states.values() for s in d})
    state_colors = {
        "dead": "#444444",
        "going_to_charge": "#ff7f0e",
        "idle": "#cccccc",
        "taking_pod": "#9467bd",
        "delivering_pod": "#1f77b4",
        "returning_pod": "#2ca02c",
        "station_processing": "#17becf",
    }

    pipelines = sorted(states.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = [0] * len(pipelines)
    for s in all_states:
        vals = [states[P].get(s, 0) for P in pipelines]
        ax.bar([PIPELINE_LABELS[P] for P in pipelines], vals,
               bottom=bottom, label=s,
               color=state_colors.get(s, "#888888"),
               edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("Robots (out of 20)")
    ax.set_title("End-of-run robot state distribution (T = 100,000 s)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.set_ylim(0, 22)
    fig.tight_layout()
    p = OUT_DIR / "state_breakdown.png"
    fig.savefig(p, dpi=160)
    print(f"[fig] {p}")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_summary()
    states = load_states()
    if not rows:
        print(f"[err] {SUMMARY_CSV} empty -run compare_100k.py first")
        return 1
    plot_survivorship(rows)
    plot_throughput(rows)
    plot_tradeoff(rows)
    plot_energy(rows)
    plot_states(states)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
