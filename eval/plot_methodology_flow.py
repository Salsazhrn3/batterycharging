"""
eval/plot_methodology_flow.py
─────────────────────────────
Render a methodology flow diagram for the joint placement-policy
optimisation paper: Phase 1 (screening) -> Phase 2 (CRSM) -> Phase 3
(Taguchi L9). Produces eval/results/figures/methodology_flow.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "eval" / "results" / "figures"


def main() -> int:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    phase_colors = {
        "phase1": "#cfe7ff",
        "phase2": "#d4f0d4",
        "phase3": "#ffe5cc",
        "out":    "#fff7c4",
        "input":  "#ffffff",
    }
    edge_colors = {
        "phase1": "#3a6ea5",
        "phase2": "#2d7a2d",
        "phase3": "#c87420",
        "out":    "#a08000",
    }

    def box(x, y, w, h, title, body, fill, edge, fontsize=8.5, title_size=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                               linewidth=1.6, edgecolor=edge, facecolor=fill)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=edge)
        ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top",
                fontsize=fontsize, color="black", wrap=True)

    def arrow(p1, p2, color="#444", text="", text_offset=(0, 0.18)):
        a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,
                             color=color, linewidth=1.6, shrinkA=4, shrinkB=4)
        ax.add_patch(a)
        if text:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(mx + text_offset[0], my + text_offset[1], text,
                    ha="center", va="bottom", fontsize=8,
                    fontstyle="italic", color="#333")

    # Title
    ax.text(7, 9.55, "Phased Joint Placement-Policy Optimisation Methodology",
            ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(7, 9.15,
            "Phase 1: factorial screening | Phase 2: constrained RSM | Phase 3: Taguchi robust design",
            ha="center", va="center", fontsize=9, color="#555", style="italic")

    # Inputs row
    box(0.4, 6.95, 3.3, 1.5,
        "INPUTS",
        "4 placement pipelines (P1-P4)\n"
        "Policy parameters: tau_low, tau_full, interrupt\n"
        "Simulator: 20-robot RMFS, 30x40 grid, |C|=12",
        phase_colors["input"], "#555", fontsize=8, title_size=9.5)

    # Phase 1 box
    box(0.2, 4.0, 4.2, 2.6,
        "PHASE 1\nFactorial Screening",
        "Design: 4 placements x 2^3 policy = 32 cells\n"
        "Levels: tau_low {15,25}, tau_full {60,90},\n"
        "        interrupt {off, on@50%}\n"
        "Horizon: 20,000 s, |C|=12, single seed\n"
        "Statistic: per-placement main effects;\n"
        "          Pareto-domination drop criterion\n"
        "Reference: Montgomery [DoE 9e, ch.6,8]",
        phase_colors["phase1"], edge_colors["phase1"],
        fontsize=8, title_size=10)

    # Phase 1 finding
    box(0.4, 2.1, 4.0, 1.55,
        "Phase 1 finding",
        "P3 dominates by 30x in replenish ratio.\n"
        "P1, P2, P4 policy-immune at |C|=12.\n"
        "Discovered: deadlock cell at (P2, lo=25,\n"
        "fu=60, int=off) -> infeasibility flag",
        phase_colors["out"], edge_colors["out"], fontsize=8, title_size=9)

    # Phase 2 box
    box(4.9, 4.0, 4.2, 2.6,
        "PHASE 2\nConstrained RSM (CRSM)",
        "Design: face-centred CCD on P3\n"
        "Factors: tau_low, tau_full (continuous)\n"
        "Levels: 5 each, axials at face centres\n"
        "Probes: 4-cell deadlock-boundary scan\n"
        "Fit: quadratic OLS on 8 clean points\n"
        "Constraint: D_P3 = {19.5<=lo<=20.5\n"
        "            and fu>=89} excluded\n"
        "Reference: Myers/Montgomery/Anderson-\n"
        "Cook [RSM 4e, sec.7.5]",
        phase_colors["phase2"], edge_colors["phase2"],
        fontsize=8, title_size=10)

    # Phase 2 finding
    box(5.1, 2.1, 4.0, 1.55,
        "Phase 2 finding",
        "Constrained optimum at (tau_low=18.4%,\n"
        "tau_full=60%, interrupt=on@50%).\n"
        "Predicted replenish 27.4%; constraint\n"
        "non-binding for the recommended policy.",
        phase_colors["out"], edge_colors["out"], fontsize=8, title_size=9)

    # Phase 3 box
    box(9.6, 4.0, 4.2, 2.6,
        "PHASE 3\nTaguchi L9 Robustness",
        "Design: L9(3^4) orthogonal array\n"
        "Noise factors:\n"
        "  A: demand load {0.7, 1.0, 1.3}\n"
        "  B: fleet size {18, 20, 22}\n"
        "  C: charger faults {0, 1, 2}\n"
        "  D: initial battery {0.5, 0.7, 1.0}\n"
        "Candidates: Robust vs Peak\n"
        "Statistic: S/N ratios per response\n"
        "Reference: Taguchi 1986 / Phadke 1989",
        phase_colors["phase3"], edge_colors["phase3"],
        fontsize=8, title_size=10)

    # Phase 3 finding
    box(9.8, 2.1, 4.0, 1.55,
        "Phase 3 finding",
        "Robust dominates Peak: 2/9 vs 4/9\n"
        "deadlocks; orders S/N +5.79 dB.\n"
        "init_batt is dominant noise factor;\n"
        "demand/robots/faults nearly flat.",
        phase_colors["out"], edge_colors["out"], fontsize=8, title_size=9)

    # Final recommendation
    box(2.4, 0.05, 9.2, 1.7,
        "FINAL JOINT OPTIMUM (CapEx-OpEx-Procedural Package)",
        "CapEx: P3 placement, |C|=12 picker-corridor cells (canonical layout)\n"
        "OpEx:  tau_low=18%, tau_full=60%, interrupt=on@50%, active charging enabled\n"
        "Procedural: ensure fleet starts each shift at >=70% battery (dominant noise lever)",
        phase_colors["out"], "#a06000", fontsize=9.5, title_size=10.5)

    # Arrows
    arrow((2.0, 6.95), (2.3, 6.6), text="")
    arrow((4.4, 5.3), (4.9, 5.3),
          text="surviving\nplacement", text_offset=(0, 0.05))
    arrow((9.1, 5.3), (9.6, 5.3),
          text="optimum\npolicy", text_offset=(0, 0.05))

    arrow((2.4, 4.0), (2.4, 3.65))
    arrow((7.0, 4.0), (7.1, 3.65))
    arrow((11.7, 4.0), (11.8, 3.65))

    arrow((2.4, 2.1), (4.0, 1.75))
    arrow((7.1, 2.1), (7.0, 1.75))
    arrow((11.8, 2.1), (10.0, 1.75))

    fig.tight_layout()
    out = FIG_DIR / "methodology_flow.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
