"""
eval/plot_deadlock_and_depletion.py
───────────────────────────────────
Two diagnostic figures for §8.8 (mechanism analysis):

  1. deadlock_schematic.png : three-panel schematic of the deadlock mechanisms
     observed in P1, P2, and P3 deadlock cells. Conceptual (not a reconstruction
     of exact robot positions) because we have only end-of-run state from the
     simulator. Designed to clarify each mechanism's spatial structure.

  2. depletion_distribution.png : per-robot end-of-run battery distribution at
     T = 100,000 s, one panel per pipeline (P1-P4). Shows the bimodal pattern
     of P3 and P4 (heavy mass at 0 with a small surviving subset) vs the more
     spread-out distributions of P1 and P2.

Outputs land in eval/results/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "eval" / "results" / "figures"
P100K = ROOT / "eval" / "runs"


# =============================================================================
# Figure 1: Deadlock schematic
# =============================================================================

def plot_deadlock_schematic() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    pipeline_titles = [
        ("P1 deadlock pattern",
         r"$\tau_\mathrm{low}=20,\ \tau_\mathrm{full}=75,\ \mathrm{int}=\mathrm{on}@50$",
         "Frequent low-trigger cycling on scattered storage chargers\n"
         "creates traffic on the same corridors as productive pod motion;\n"
         "interrupt cannot fully break the resulting concurrent block."),
        ("P2 deadlock pattern",
         r"$\tau_\mathrm{low}=25,\ \tau_\mathrm{full}=60,\ \mathrm{int}=\mathrm{off}$",
         "Tight charge cycle (35-pp buffer) + interrupt off:\n"
         "robots occupy scattered cluster cells for full charge duration;\n"
         "mutual blocking on access paths between clusters."),
        ("P3 deadlock pattern",
         r"$\tau_\mathrm{low}\approx 20,\ \tau_\mathrm{full}\geq 89,\ \mathrm{int}=\mathrm{on}@50$",
         "Synchronisation resonance: all 20 robots hit threshold at the same\n"
         "tick; convergence wave on picker-corridor chargers is catastrophic\n"
         "because those cells are also the productive-workflow cells."),
    ]

    grid_color = "#dddddd"
    aisle_color = "#f6f6f6"
    charger_color = "#ff7f0e"
    picker_color = "#1f77b4"
    free_robot_color = "#2ca02c"
    blocked_robot_color = "#d62728"
    arrow_color = "#555555"

    def draw_grid(ax, label_xy):
        # Schematic 12x10 grid
        for r in range(10):
            for c in range(12):
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=aisle_color,
                                       edgecolor=grid_color, linewidth=0.5))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(*label_xy, "Storage area", fontsize=8, color="#666",
                ha="center", va="center", style="italic")

    def draw_chargers(ax, positions, label="Charger"):
        for x, y in positions:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=charger_color,
                                   edgecolor="#a04000", linewidth=1))
        if positions:
            ax.scatter([], [], marker="s", s=120, c=charger_color,
                       edgecolor="#a04000", label=label)

    def draw_pickers(ax, rows):
        for r in rows:
            ax.add_patch(Rectangle((0, r), 0.4, 1, facecolor=picker_color,
                                   edgecolor="#003366", linewidth=1))
        if rows:
            ax.scatter([], [], marker="s", s=120, c=picker_color,
                       edgecolor="#003366", label="Picker station")

    def draw_robot(ax, x, y, color, label_robot=False):
        ax.scatter(x + 0.5, y + 0.5, s=200, c=color, edgecolor="black",
                   linewidth=1, zorder=5)
        if label_robot:
            ax.scatter([], [], marker="o", s=120, c=blocked_robot_color,
                       edgecolor="black", label="Robot (blocked)")

    def arrow(ax, p1, p2, dashed=False):
        a = FancyArrowPatch((p1[0] + 0.5, p1[1] + 0.5),
                            (p2[0] + 0.5, p2[1] + 0.5),
                            arrowstyle="->", mutation_scale=14,
                            color=arrow_color, linewidth=1.4,
                            linestyle="--" if dashed else "-",
                            zorder=4, shrinkA=8, shrinkB=8)
        ax.add_patch(a)

    # ── Panel 1: P1 deadlock — scattered interior chargers, traffic conflict
    ax = axes[0]
    draw_grid(ax, (6, 0.5))
    # P1 chargers: scattered in storage
    p1_chargers = [(3, 6), (5, 4), (7, 7), (9, 5), (8, 2)]
    draw_chargers(ax, p1_chargers)
    draw_pickers(ax, [2, 5, 8])
    # Robots in conflict
    draw_robot(ax, 3, 6, blocked_robot_color, label_robot=True)
    draw_robot(ax, 5, 4, blocked_robot_color)
    draw_robot(ax, 7, 7, blocked_robot_color)
    draw_robot(ax, 4, 6, blocked_robot_color)
    draw_robot(ax, 6, 4, blocked_robot_color)
    # Arrows showing mutual blocking
    arrow(ax, (4, 6), (3, 6))
    arrow(ax, (6, 4), (5, 4))
    arrow(ax, (5, 4), (4, 6), dashed=True)
    arrow(ax, (7, 7), (3, 6), dashed=True)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(pipeline_titles[0][0], fontsize=11, fontweight="bold")
    ax.text(6, -0.7, pipeline_titles[0][1], fontsize=8, ha="center",
            color="#444")
    ax.text(6, -2.1, pipeline_titles[0][2], fontsize=7.5, ha="center",
            color="#444")

    # ── Panel 2: P2 deadlock — tight cycle, mutual blocking on clusters
    ax = axes[1]
    draw_grid(ax, (6, 0.5))
    p2_chargers = [(4, 6), (5, 6), (4, 5), (8, 3), (9, 3)]
    draw_chargers(ax, p2_chargers)
    draw_pickers(ax, [2, 5, 8])
    draw_robot(ax, 4, 6, blocked_robot_color)
    draw_robot(ax, 5, 6, blocked_robot_color)
    draw_robot(ax, 4, 5, blocked_robot_color)
    draw_robot(ax, 8, 3, blocked_robot_color)
    draw_robot(ax, 9, 3, blocked_robot_color)
    draw_robot(ax, 6, 6, blocked_robot_color)
    draw_robot(ax, 7, 3, blocked_robot_color)
    arrow(ax, (6, 6), (5, 6))
    arrow(ax, (4, 5), (4, 6), dashed=True)
    arrow(ax, (7, 3), (8, 3))
    arrow(ax, (8, 3), (9, 3), dashed=True)
    ax.set_title(pipeline_titles[1][0], fontsize=11, fontweight="bold")
    ax.text(6, -0.7, pipeline_titles[1][1], fontsize=8, ha="center", color="#444")
    ax.text(6, -2.1, pipeline_titles[1][2], fontsize=7.5, ha="center",
            color="#444")

    # ── Panel 3: P3 deadlock — convergence wave on picker corridor
    ax = axes[2]
    draw_grid(ax, (6, 0.5))
    p3_chargers = [(1, 1), (1, 4), (1, 7),
                   (2, 1), (2, 4), (2, 7),
                   (3, 1), (3, 4), (3, 7),
                   (1, 2), (2, 5), (3, 8)]
    draw_chargers(ax, p3_chargers)
    draw_pickers(ax, [1, 4, 7])
    # Convergence wave of robots
    for x, y in [(5, 1), (5, 4), (5, 7), (6, 1), (6, 4), (7, 4), (7, 7), (6, 7)]:
        draw_robot(ax, x, y, blocked_robot_color)
    # Arrows showing convergence
    for (x, y) in [(5, 1), (5, 4), (5, 7), (6, 4), (7, 4)]:
        arrow(ax, (x, y), (3, y), dashed=False)
    ax.set_title(pipeline_titles[2][0], fontsize=11, fontweight="bold")
    ax.text(6, -0.7, pipeline_titles[2][1], fontsize=8, ha="center", color="#444")
    ax.text(6, -2.1, pipeline_titles[2][2], fontsize=7.5, ha="center",
            color="#444")

    plt.subplots_adjust(top=0.88, bottom=0.27, wspace=0.1, left=0.02, right=0.98)
    fig.suptitle("Three observed deadlock mechanisms (schematic; not exact robot positions)",
                 fontsize=12, fontweight="bold")
    out = FIG_DIR / "deadlock_schematic.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


# =============================================================================
# Figure 2: End-of-run battery distribution at T = 100,000 s
# =============================================================================

def plot_depletion_distribution() -> None:
    pipelines = [
        ("P1 Set Cover", "p1_100k/p1", "#1f77b4"),
        ("P2 Affinity Prop.", "p2_100k/p2", "#2ca02c"),
        ("P3 Picker-Tier", "p3_100k/p3", "#ff7f0e"),
        ("P4 Perimeter", "p4_100k/p4", "#d62728"),
    ]
    data = []
    for label, subdir, color in pipelines:
        p = P100K / subdir / "per_robot.json"
        if not p.exists():
            print(f"[warn] {p} missing — skipping")
            continue
        per = json.loads(p.read_text())
        batt = [r["battery_pct"] for r in per]
        # Categorise dead-or-alive
        dead = [b for b in batt if b <= 0.5]
        alive = [b for b in batt if b > 0.5]
        data.append({"label": label, "color": color, "dead": dead,
                     "alive": alive, "all": batt})

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    for ax, d in zip(axes, data):
        # Strip plot of all batteries with dead/alive colours
        x_dead = np.random.uniform(-0.15, 0.15, len(d["dead"]))
        x_alive = np.random.uniform(-0.15, 0.15, len(d["alive"]))
        ax.scatter(x_dead, d["dead"], s=150, c="black",
                   edgecolor="#660000", linewidth=1, alpha=0.85,
                   label=f"Dead ({len(d['dead'])})")
        ax.scatter(x_alive, d["alive"], s=150, c=d["color"],
                   edgecolor="black", linewidth=0.6, alpha=0.85,
                   label=f"Alive ({len(d['alive'])})")
        # Threshold lines
        ax.axhline(20, color="#888", linestyle=":", linewidth=1)
        ax.axhline(50, color="#bbb", linestyle=":", linewidth=1)
        ax.text(0.18, 20.5, r"$\tau_\mathrm{low}=20\%$", fontsize=7, color="#666")
        ax.text(0.18, 50.5, r"$\tau_\mathrm{int}=50\%$", fontsize=7, color="#aaa")
        # Mean line
        mean = sum(d["all"]) / len(d["all"])
        ax.axhline(mean, color=d["color"], linestyle="-", linewidth=2,
                   alpha=0.5)
        ax.text(-0.4, mean + 1.5, f"mean = {mean:.1f}%", fontsize=8,
                color=d["color"], fontweight="bold")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-2, 102)
        ax.set_xticks([])
        ax.set_title(d["label"], fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("End-of-run battery (\%)", fontsize=11)
    fig.suptitle("End-of-run battery distribution across the 20-robot fleet at $T = 100{,}000$\\,s",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.98, wspace=0.1)
    out = FIG_DIR / "depletion_distribution.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[fig] {out}")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_deadlock_schematic()
    plot_depletion_distribution()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
