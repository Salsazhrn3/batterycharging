"""
eval/plot_timeseries.py
───────────────────────
Build time-series plots for the 4-pipeline comparison from each pipeline's
``eval/runs/p{P}/order-finished.csv``.

Three panels are produced (one figure per metric), each with 4 lines:
  * Cumulative orders completed vs sim-time
  * Rolling throughput (orders / sim-hr, 600 s window) vs sim-time
  * Per-order cycle time scattered against completion time,
    plus rolling-mean cycle-time curve

Only end-of-run battery / pending-jobs / stop-and-go are available in the
current logs — those are NOT time-series and are not plotted here.

Usage
─────
    python eval/plot_timeseries.py
    python eval/plot_timeseries.py --window 1200

Produces PNGs under ``eval/results/plots/``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "eval" / "runs"
PLOTS_DIR = ROOT / "eval" / "results" / "plots"

PIPELINE_NAMES = {
    1: "P1 SetCover",
    2: "P2 AffinityProp",
    3: "P3 PickStation (Opp)",
    4: "P4 Perimeter",
}
PIPELINE_COLORS = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd"}


def load_order_finished(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (complete_times, cycle_times) sorted by complete_time."""
    path = run_dir / "order-finished.csv"
    if not path.exists():
        return np.array([]), np.array([])
    arrivals, completes = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                a = float(row["order_arrival"])
                c = float(row["order_complete_time"])
            except (KeyError, ValueError):
                continue
            arrivals.append(a)
            completes.append(c)
    if not completes:
        return np.array([]), np.array([])
    completes = np.array(completes)
    cycles = np.array(completes) - np.array(arrivals)
    order = np.argsort(completes)
    return completes[order], cycles[order]


def rolling_throughput(completes: np.ndarray, window_s: float,
                       step_s: float, horizon_s: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_grid, orders_per_sim_hour) over a sliding window.

    Grid spans [0, horizon_s] so every pipeline's line covers the same
    x-axis range — points past the last completion naturally drop to 0.
    """
    t_grid = np.arange(0.0, horizon_s + step_s, step_s)
    if completes.size == 0:
        return t_grid, np.zeros_like(t_grid)
    thr = np.zeros_like(t_grid)
    for i, t in enumerate(t_grid):
        lo, hi = t - window_s, t
        count = np.count_nonzero((completes > lo) & (completes <= hi))
        thr[i] = count * (3600.0 / window_s)
    return t_grid, thr


def rolling_mean(x: np.ndarray, y: np.ndarray, window_s: float,
                 step_s: float, horizon_s: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_grid, mean_y_in_window) — grid spans [0, horizon_s]."""
    t_grid = np.arange(0.0, horizon_s + step_s, step_s)
    if x.size == 0:
        return t_grid, np.full_like(t_grid, np.nan, dtype=float)
    out = np.full_like(t_grid, np.nan, dtype=float)
    for i, t in enumerate(t_grid):
        mask = (x > t - window_s) & (x <= t)
        if mask.any():
            out[i] = float(np.mean(y[mask]))
    return t_grid, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=600,
                    help="Rolling window (sim-seconds) for throughput + cycle mean.")
    ap.add_argument("--step", type=int, default=100,
                    help="Step between rolling-window samples (sim-seconds).")
    ap.add_argument("--pipelines", type=str, default="1,2,3,4",
                    help="Comma-separated pipeline IDs to include.")
    ap.add_argument("--horizon", type=int, default=100000,
                    help="Sim horizon (sec) — sets x-axis range for all plots.")
    args = ap.parse_args()
    horizon_s = float(args.horizon)

    pipelines = [int(x) for x in args.pipelines.split(",") if x.strip()]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for every pipeline that has an order-finished.csv
    data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for P in pipelines:
        run_dir = RUNS_DIR / f"p{P}"
        completes, cycles = load_order_finished(run_dir)
        if completes.size:
            data[P] = (completes, cycles)
            print(f"[ok] P{P}: {completes.size} orders loaded")
        else:
            print(f"[skip] P{P}: no order-finished.csv or empty")

    if not data:
        print("No data to plot.")
        return 1

    # ── Figure 1: Cumulative orders completed vs sim-time ───────────────
    # Extend each line flat past its last completion to horizon_s so the
    # reader can see "no further completions occurred" rather than thinking
    # the data is missing.
    plt.figure(figsize=(10, 5))
    for P, (completes, _) in data.items():
        cum = np.arange(1, completes.size + 1)
        x = np.concatenate([completes, [horizon_s]])
        y = np.concatenate([cum, [cum[-1]]])
        plt.plot(x, y, label=PIPELINE_NAMES[P],
                 color=PIPELINE_COLORS[P], lw=1.8)
    plt.xlim(0, horizon_s)
    plt.xlabel("Sim-time (seconds)")
    plt.ylabel("Cumulative orders completed")
    plt.title(f"Cumulative orders completed vs sim-time ({int(horizon_s):,}-sec horizon)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cumulative_orders.png", dpi=120)
    plt.close()
    print(f"[ok] wrote {PLOTS_DIR / 'cumulative_orders.png'}")

    # ── Figure 2: Rolling throughput (orders/sim-hr) ───────────────────
    plt.figure(figsize=(10, 5))
    for P, (completes, _) in data.items():
        t, thr = rolling_throughput(completes, args.window, args.step, horizon_s)
        plt.plot(t, thr, label=PIPELINE_NAMES[P],
                 color=PIPELINE_COLORS[P], lw=1.6)
    plt.xlim(0, horizon_s)
    plt.xlabel("Sim-time (seconds)")
    plt.ylabel(f"Orders / sim-hour (rolling {args.window}s)")
    plt.title(f"Rolling throughput ({args.window}s window)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rolling_throughput.png", dpi=120)
    plt.close()
    print(f"[ok] wrote {PLOTS_DIR / 'rolling_throughput.png'}")

    # ── Figure 3: Cycle time (rolling mean) ────────────────────────────
    plt.figure(figsize=(10, 5))
    for P, (completes, cycles) in data.items():
        t, mean_c = rolling_mean(completes, cycles, args.window, args.step, horizon_s)
        plt.plot(t, mean_c, label=PIPELINE_NAMES[P],
                 color=PIPELINE_COLORS[P], lw=1.6)
    plt.xlim(0, horizon_s)
    plt.xlabel("Sim-time (seconds)")
    plt.ylabel(f"Cycle time mean (s, rolling {args.window}s)")
    plt.title(f"Order cycle time vs sim-time ({args.window}s window)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cycle_time_rolling.png", dpi=120)
    plt.close()
    print(f"[ok] wrote {PLOTS_DIR / 'cycle_time_rolling.png'}")

    # ── Figure 4: Cycle time scatter + rolling overlay ─────────────────
    plt.figure(figsize=(10, 5))
    for P, (completes, cycles) in data.items():
        plt.scatter(completes, cycles, s=4, alpha=0.15,
                    color=PIPELINE_COLORS[P])
        t, mean_c = rolling_mean(completes, cycles, args.window, args.step, horizon_s)
        plt.plot(t, mean_c, label=PIPELINE_NAMES[P],
                 color=PIPELINE_COLORS[P], lw=1.8)
    plt.xlim(0, horizon_s)
    plt.xlabel("Sim-time order completed (seconds)")
    plt.ylabel("Cycle time (s)")
    plt.title("Per-order cycle time (dots) and rolling mean (lines)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cycle_time_scatter.png", dpi=120)
    plt.close()
    print(f"[ok] wrote {PLOTS_DIR / 'cycle_time_scatter.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
