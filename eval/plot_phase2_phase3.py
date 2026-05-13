"""
eval/plot_phase2_phase3.py
──────────────────────────
Generate paper figures for Phase 2 (CRSM) and Phase 3 (Taguchi).

Outputs (160 dpi PNG, written to eval/results/figures/):
  - phase2_rsm_surface.png   : 2D contour of fitted P3 quadratic surface for
                                replenish ratio, with deadlock zone shaded
                                infeasible and the constrained optimum marked.
  - phase3_taguchi_response.png : 4-panel response graph (one panel per noise
                                   factor) comparing Robust vs Peak mean orders
                                   at each level.
  - phase3_robust_vs_peak.png : bar chart of headline metrics (mean orders,
                                 worst-case orders, deadlock count, S/N).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "eval" / "results" / "figures"
RSM_FIT = ROOT / "eval" / "results" / "p3_rsm_fit.json"
PHASE3_CSV = ROOT / "eval" / "results" / "doe_phase3_responses.csv"


# -----------------------------------------------------------------------------
# Phase 2 — RSM surface plot
# -----------------------------------------------------------------------------

def plot_phase2_surface() -> None:
    fit = json.loads(RSM_FIT.read_text())
    coefs = fit["fits"]["replenish_pct"]["coefficients"]
    design_points = fit["fits"]["replenish_pct"]["design_points"]
    constrained = fit["fits"]["replenish_pct"]["constrained_optimum"]

    # Predict on a grid.
    grid_lo = np.linspace(15, 25, 121)
    grid_fu = np.linspace(60, 90, 121)
    LL, FF = np.meshgrid(grid_lo, grid_fu, indexing="ij")
    xL = (LL - 20) / 5
    xF = (FF - 75) / 15
    Z = (coefs["intercept"]
         + coefs["tau_low"] * xL
         + coefs["tau_full"] * xF
         + coefs["tau_low_sq"] * xL ** 2
         + coefs["tau_full_sq"] * xF ** 2
         + coefs["tau_low_x_tau_full"] * xL * xF)

    # Mask deadlock zone D_{P3} = {(lo, fu) : 19.5 <= lo <= 20.5 AND fu >= 89}.
    deadlock_mask = (LL >= 19.5) & (LL <= 20.5) & (FF >= 89)

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(LL, FF, Z, levels=20, cmap="viridis", alpha=0.85)
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label("Predicted replenish ratio (%)")
    cs_lines = ax.contour(LL, FF, Z, levels=8, colors="white",
                           linewidths=0.6, alpha=0.6)
    ax.clabel(cs_lines, inline=True, fontsize=7, fmt="%.2f")

    # Deadlock zone: cross-hatched red overlay.
    ax.contourf(LL, FF, deadlock_mask.astype(float), levels=[0.5, 1.5],
                colors=["none"], hatches=["///"], alpha=0)
    ax.contour(LL, FF, deadlock_mask.astype(float), levels=[0.5],
               colors="red", linewidths=2, linestyles="--")
    # Manual hatched fill (matplotlib's hatch fill is finicky)
    ax.fill_between([19.5, 20.5], 89, 90, color="red", alpha=0.25,
                    hatch="///", edgecolor="darkred", linewidth=0)
    # Annotate deadlock zone
    ax.text(20, 89.5, r"$\mathcal{D}_{P3}$" + "\n(infeasible)",
            ha="center", va="bottom", fontsize=9, color="darkred",
            fontweight="bold")

    # Design points used in the fit.
    for dp in design_points:
        ax.plot(dp["tau_low"], dp["tau_full"], "wo",
                markeredgecolor="black", markersize=8)
        ax.text(dp["tau_low"] + 0.15, dp["tau_full"] + 0.4,
                f"{dp['y']:.1f}", fontsize=7, color="black")

    # Excluded deadlock cell (20, 90) — show as red X.
    ax.plot(20, 90, "rx", markersize=14, markeredgewidth=3,
            label=r"Excluded (deadlock)")
    ax.text(20.3, 90, "(20, 90)\n5.2%", fontsize=7, color="darkred",
            va="center")

    # Constrained optimum
    ax.plot(constrained["tau_low"], constrained["tau_full"], "*",
            markersize=22, color="gold", markeredgecolor="black",
            markeredgewidth=1.5,
            label=f"Constrained optimum: ({constrained['tau_low']:.1f}, "
                  f"{constrained['tau_full']:.0f}), "
                  f"{constrained['predicted']:.2f}%")

    ax.set_xlabel(r"$\tau_{\mathrm{low}}$  (\%)", fontsize=11)
    ax.set_ylabel(r"$\tau_{\mathrm{full}}$  (\%)", fontsize=11)
    ax.set_title("P3 quadratic response surface --- replenish ratio\n"
                 "(interrupt = on@50%, |C| = 12, T = 20{,}000 s)",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(15, 25)
    ax.set_ylim(60, 90)
    fig.tight_layout()
    out = FIG_DIR / "phase2_rsm_surface.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[fig] {out}")


# -----------------------------------------------------------------------------
# Phase 3 — Taguchi response graph
# -----------------------------------------------------------------------------

def load_phase3() -> list[dict]:
    rows = []
    with open(PHASE3_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "candidate": r["candidate"],
                "demand": float(r["demand"]),
                "robots_setting": int(r["robots_setting"]),
                "faults": int(r["faults"]),
                "init_batt": float(r["init_batt"]),
                "orders": int(r["orders"]),
                "replenish_pct": float(r["replenish_pct"]),
                "deadlock": r["deadlock"].lower() == "true",
            })
    return rows


def plot_phase3_response() -> None:
    rows = load_phase3()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    factor_specs = [
        ("demand", "Demand load", [0.7, 1.0, 1.3]),
        ("robots_setting", "Fleet size", [18, 20, 22]),
        ("faults", "Charger faults", [0, 1, 2]),
        ("init_batt", "Initial battery", [0.5, 0.7, 1.0]),
    ]
    colors = {"robust": "#1f77b4", "peak": "#d62728"}
    markers = {"robust": "o", "peak": "s"}
    for ax, (attr, label, levels) in zip(axes, factor_specs):
        for cand in ("robust", "peak"):
            sub = [r for r in rows if r["candidate"] == cand]
            means = []
            for lv in levels:
                cell_orders = [r["orders"] for r in sub if r[attr] == lv]
                means.append(sum(cell_orders) / len(cell_orders) if cell_orders else float("nan"))
            ax.plot(range(len(levels)), means, marker=markers[cand],
                    color=colors[cand], linewidth=2, markersize=10,
                    label=cand.capitalize())
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(lv) for lv in levels])
        ax.set_xlabel(label)
        ax.set_title(f"{label}", fontsize=10)
        ax.grid(alpha=0.3)
        if attr == "demand":
            ax.set_ylabel("Mean orders completed")
            ax.legend(loc="lower right")
    fig.suptitle("Phase 3 Taguchi $L_9$ response graph: orders vs noise factor level",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "phase3_taguchi_response.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[fig] {out}")


# -----------------------------------------------------------------------------
# Phase 3 — Robust vs Peak headline bar chart
# -----------------------------------------------------------------------------

def sn_larger_better(values: list[float]) -> float:
    safe = [v for v in values if v > 0]
    if not safe:
        return float("nan")
    return -10 * math.log10(sum(1 / v ** 2 for v in safe) / len(safe))


def plot_phase3_robust_vs_peak() -> None:
    rows = load_phase3()
    by_cand: dict[str, list[dict]] = {}
    for r in rows:
        by_cand.setdefault(r["candidate"], []).append(r)

    metrics = []
    for cand in ("robust", "peak"):
        cells = by_cand[cand]
        orders = [c["orders"] for c in cells]
        replen = [c["replenish_pct"] for c in cells]
        n_dl = sum(1 for c in cells if c["deadlock"])
        metrics.append({
            "candidate": cand,
            "mean_orders": sum(orders) / len(orders),
            "worst_orders": min(orders),
            "deadlock_count": n_dl,
            "sn_orders": sn_larger_better(orders),
            "sn_replen": sn_larger_better([v + 100 for v in replen]),
        })

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    labels = ["Robust\n(CRSM)", "Peak\n(Phase 1 best)"]
    colors = ["#1f77b4", "#d62728"]

    panels = [
        ("Mean orders", "mean_orders", "{:.0f}", False),
        ("Worst-case orders", "worst_orders", "{:.0f}", False),
        ("Deadlock count\n(of 9 noise combos)", "deadlock_count", "{:.0f}", True),
        ("S/N ratio (orders, dB)", "sn_orders", "{:+.2f}", False),
    ]
    for ax, (title, key, fmt, lower_is_better) in zip(axes, panels):
        vals = [m[key] for m in metrics]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * (0.95 if v >= 0 else 1.05),
                    fmt.format(v),
                    ha="center", va="top" if v >= 0 else "bottom",
                    fontsize=11, fontweight="bold", color="white")
        if lower_is_better:
            best = min(range(len(vals)), key=lambda i: vals[i])
        else:
            best = max(range(len(vals)), key=lambda i: vals[i])
        bars[best].set_edgecolor("gold")
        bars[best].set_linewidth(3)

    fig.suptitle("Phase 3 verdict --- Robust dominates Peak on every aggregate metric",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "phase3_robust_vs_peak.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[fig] {out}")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_phase2_surface()
    plot_phase3_response()
    plot_phase3_robust_vs_peak()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
