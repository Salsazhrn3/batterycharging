"""
eval/analyze_doe_phase2b_stage2.py
──────────────────────────────────
Phase 2b Stage 2: per-placement |C|/N saturation curves.

Stage 2 design: 5 |C| levels {3, 8, 13, 17, 20} × 4 placements at the
per-placement τ optimum carried over from Stage 1, interrupt fixed at 50%.
Fleet size N = 20, so |C|/N spans {0.15, 0.40, 0.65, 0.85, 1.00}.

Per placement we report:
  - The raw response curve (replenish_pct, orders) versus |C|/N
  - A saturating Michaelis–Menten-style fit  y = y∞ · (|C|/N) / (K + |C|/N)
    fitted in log-residual sense by closed-form linearisation, then
    refined by 1-D grid search on K.
  - The empirical knee: smallest |C|/N at which marginal gain
    Δy / Δ(|C|/N) falls below 5 % of the curve's total range.
  - The operating recommendation: knee · N rounded up to the next
    integer charger count.

Outputs:
  eval/results/phase2b_stage2_curves.json
  eval/results/figures/phase2b_stage2_saturation.png
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "eval" / "runs" / "doe_phase2b"
RESULTS = ROOT / "eval" / "results"
FIGURES = RESULTS / "figures"
BATTERY_J = 6_480_000.0
BASE_DRAIN = 90.0
FLEET_N = 20
STAGE2_LEVELS = {3, 8, 13, 17, 20}
PLACEMENT_TAU = {
    1: (21, 82),
    2: (15, 90),
    3: (17, 60),
    4: (15, 76),
}
LABELS = {
    1: "P1 Set Cover",
    2: "P2 Affinity Prop.",
    3: "P3 Picker Heuristic",
    4: "P4 Perimeter",
}

CELL_RE = re.compile(r"^p(?P<pl>[1-4])_n(?P<nc>\d+)_lo(?P<lo>\d+)_fu(?P<fu>\d+)_int(?P<intr>\d+)$")


def parse_cell(name: str) -> dict | None:
    m = CELL_RE.match(name)
    if not m:
        return None
    return {
        "placement": int(m["pl"]),
        "num_chargers": int(m["nc"]),
        "tau_low": int(m["lo"]),
        "tau_full": int(m["fu"]),
        "interrupt": int(m["intr"]),
    }


def load_cell(d: Path) -> dict | None:
    rs = d / "run_summary.json"
    pr = d / "per_robot.json"
    if not rs.exists() or not pr.exists():
        return None
    s = json.loads(rs.read_text())
    per = json.loads(pr.read_text())
    nr = len(per)
    h = s.get("horizon", 0)
    motion = sum(r["energy_consumption_j"] for r in per)
    consumed = motion + BASE_DRAIN * h * nr
    depleted = sum(BATTERY_J - r["battery_level_j"] for r in per)
    gained = consumed - depleted
    of = d / "order-finished.csv"
    orders = 0
    last_tick = 0.0
    if of.exists():
        with open(of) as f:
            for row in csv.DictReader(f):
                orders += 1
                try:
                    last_tick = max(last_tick, float(row.get("order_complete_time") or 0))
                except (TypeError, ValueError):
                    pass
    tr = s.get("tick_result")
    final_tick = tr.get("final_tick", 0.0) if isinstance(tr, dict) else 0.0
    status = tr.get("status", "") if isinstance(tr, dict) else str(tr or "")
    deadlock = (h > 0) and (status != "done" or final_tick < 0.95 * h)
    return {
        "id": d.name,
        "orders": orders,
        "replenish_pct": (gained / consumed * 100) if consumed else 0.0,
        "consumed_mj": consumed / 1e6,
        "deadlock": deadlock,
        "final_tick": final_tick,
        "status": status,
    }


def collect_stage2() -> dict[int, list[dict]]:
    """Group Stage 2 cells by placement, filtered to {3,8,13,17,20} at τ*."""
    by_pl: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: []}
    for d in sorted(RUN_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_cell(d.name)
        if meta is None:
            continue
        pl = meta["placement"]
        if meta["num_chargers"] not in STAGE2_LEVELS:
            continue
        tau_target = PLACEMENT_TAU[pl]
        if (meta["tau_low"], meta["tau_full"]) != tau_target:
            continue
        if meta["interrupt"] != 50:
            continue
        cell = load_cell(d)
        if cell is None:
            continue
        cell.update(meta)
        cell["ratio"] = meta["num_chargers"] / FLEET_N
        by_pl[pl].append(cell)
    for pl in by_pl:
        by_pl[pl].sort(key=lambda r: r["num_chargers"])
    return by_pl


def fit_mm(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = y_inf * x / (K + x) by closed-form linearisation + grid refine.

    Linearised:  1/y = (K/y_inf) * (1/x) + 1/y_inf
    Falls back to OLS through the origin when y is monotone increasing
    and non-zero. Returns (y_inf, K, R²).
    """
    pos = (y > 0) & (x > 0)
    if pos.sum() < 2:
        return float(y.max()), float("nan"), float("nan")
    xp, yp = x[pos], y[pos]
    inv_x = 1.0 / xp
    inv_y = 1.0 / yp
    A = np.column_stack([np.ones_like(inv_x), inv_x])
    coef, *_ = np.linalg.lstsq(A, inv_y, rcond=None)
    inv_yinf, K_over_yinf = coef
    if inv_yinf <= 0:
        # Degenerate; fall back to last observed value
        return float(y.max()), float("nan"), float("nan")
    y_inf0 = 1.0 / inv_yinf
    K0 = K_over_yinf * y_inf0

    # Refine on K with a small log grid around K0; pick min SSE on original scale
    K_grid = np.logspace(math.log10(max(K0 / 8, 1e-3)),
                         math.log10(max(K0 * 8, 1e-2)), 81)
    best = (float("inf"), y_inf0, K0)
    for K in K_grid:
        denom = K + x
        if np.any(denom <= 0):
            continue
        # Closed-form y_inf at this K: y_inf = Σ(yi · ri) / Σ(ri²) where ri = xi/(K+xi)
        r = x / denom
        denom2 = float(np.sum(r * r))
        if denom2 <= 0:
            continue
        y_inf = float(np.sum(y * r) / denom2)
        if y_inf <= 0:
            continue
        sse = float(np.sum((y - y_inf * r) ** 2))
        if sse < best[0]:
            best = (sse, y_inf, K)
    sse, y_inf, K = best
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    return float(y_inf), float(K), float(r2)


def empirical_knee(x: np.ndarray, y: np.ndarray, gain_floor_pct: float = 5.0) -> dict:
    """Smallest ratio at which marginal gain Δy/Δx drops below
    gain_floor_pct % of the curve's total range (y.max − y.min)."""
    if len(x) < 2:
        return {"ratio": float("nan"), "num_chargers": None, "criterion_met": False}
    rng = float(y.max() - y.min())
    if rng <= 0:
        return {"ratio": float(x[0]), "num_chargers": int(round(x[0] * FLEET_N)),
                "criterion_met": True}
    threshold = (gain_floor_pct / 100.0) * rng
    # Marginal gain between consecutive points (per unit of x)
    knee_ratio = float(x[-1])
    knee_idx = len(x) - 1
    for i in range(1, len(x)):
        dy = float(y[i] - y[i - 1])
        dx = float(x[i] - x[i - 1])
        marginal = dy / dx if dx > 0 else 0.0
        # First i where marginal gain to next step is small
        if marginal < threshold / max(x[i] - x[i - 1], 1e-6):
            knee_ratio = float(x[i - 1])
            knee_idx = i - 1
            break
    nc_recommend = math.ceil(knee_ratio * FLEET_N)
    return {
        "ratio": knee_ratio,
        "num_chargers": nc_recommend,
        "criterion_met": True,
        "gain_floor_pct": gain_floor_pct,
        "y_at_knee": float(y[knee_idx]),
        "y_max_observed": float(y.max()),
        "pct_of_max": float(y[knee_idx] / y.max() * 100) if y.max() else float("nan"),
    }


def analyse_placement(pl: int, cells: list[dict], response: str) -> dict:
    nc = np.array([c["num_chargers"] for c in cells], dtype=float)
    ratio = np.array([c["ratio"] for c in cells], dtype=float)
    y = np.array([c[response] for c in cells], dtype=float)
    y_inf, K, r2 = fit_mm(ratio, y)
    knee = empirical_knee(ratio, y)
    deadlocked = [c["id"] for c in cells if c["deadlock"]]
    return {
        "placement": pl,
        "label": LABELS[pl],
        "response": response,
        "tau_low": PLACEMENT_TAU[pl][0],
        "tau_full": PLACEMENT_TAU[pl][1],
        "fleet_N": FLEET_N,
        "design_points": [
            {"id": c["id"], "num_chargers": int(c["num_chargers"]),
             "ratio": float(c["ratio"]), "y": float(c[response]),
             "deadlock": bool(c["deadlock"])}
            for c in cells
        ],
        "mm_fit": {
            "model": "y = y_inf * (|C|/N) / (K + |C|/N)",
            "y_inf": y_inf,
            "K": K,
            "r_squared": r2,
            "ratio_at_half_max": K,
            "y_at_max_ratio": (y_inf * 1.0 / (K + 1.0)) if K == K else float("nan"),
        },
        "knee": knee,
        "deadlocked": deadlocked,
    }


def plot_saturation(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    responses = ("replenish_pct", "orders")
    titles = {
        "replenish_pct": "Energy replenishment (%)",
        "orders": "Orders completed",
    }
    colours = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}

    for ax, response in zip(axes, responses):
        for pl in (1, 2, 3, 4):
            entry = results["by_response"][response][f"p{pl}"]
            dps = entry["design_points"]
            xs = np.array([d["ratio"] for d in dps])
            ys = np.array([d["y"] for d in dps])
            ax.plot(xs, ys, marker=markers[pl], color=colours[pl],
                    label=entry["label"], linewidth=1.4, markersize=7)
            # Overlay MM fit
            mm = entry["mm_fit"]
            y_inf, K = mm["y_inf"], mm["K"]
            if K == K:
                xg = np.linspace(xs.min(), xs.max(), 100)
                yg = y_inf * xg / (K + xg)
                ax.plot(xg, yg, color=colours[pl], linewidth=0.8,
                        linestyle="--", alpha=0.6)
            # Mark knee
            kn = entry["knee"]
            if kn["criterion_met"] and kn["ratio"] == kn["ratio"]:
                ax.axvline(kn["ratio"], color=colours[pl], linewidth=0.6,
                           linestyle=":", alpha=0.4)
        ax.set_xlabel("Charger-to-robot ratio  |C| / N")
        ax.set_ylabel(titles[response])
        ax.set_title(titles[response])
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.suptitle("Phase 2b Stage 2 — Charger-count saturation curves "
                 "(τ fixed at per-placement RSM optimum, interrupt = 50%)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def print_table(results: dict) -> None:
    for response in ("replenish_pct", "orders"):
        print()
        print("=" * 88)
        print(f"  RESPONSE: {response}")
        print("=" * 88)
        print(f"{'Pl':<3} {'τ*':>10} {'|C|':>4} {'|C|/N':>6} "
              f"{'y':>10} {'knee':>6} {'reco |C|':>9} {'%max':>6} "
              f"{'y∞':>10} {'K':>6} {'R²':>6}")
        for pl in (1, 2, 3, 4):
            entry = results["by_response"][response][f"p{pl}"]
            mm = entry["mm_fit"]
            kn = entry["knee"]
            tau = f"({entry['tau_low']},{entry['tau_full']})"
            for d in entry["design_points"]:
                marker = " <-- knee" if kn["criterion_met"] and abs(d["ratio"] - kn["ratio"]) < 1e-6 else ""
                print(f"P{pl:<2} {tau:>10} {d['num_chargers']:>4} {d['ratio']:>6.2f} "
                      f"{d['y']:>10.2f}{marker}")
            print(f"   → knee at |C|/N={kn['ratio']:.2f}  "
                  f"(recommend |C| ≥ {kn['num_chargers']}, "
                  f"{kn['pct_of_max']:.1f}% of observed max)")
            print(f"   → MM fit: y∞={mm['y_inf']:.2f}, K={mm['K']:.3f}, R²={mm['r_squared']:.3f}")
            print()


def main() -> int:
    by_pl = collect_stage2()
    print(f"Loaded Stage 2 cells from {RUN_DIR}")
    for pl in (1, 2, 3, 4):
        print(f"  P{pl}: {len(by_pl[pl])} cells (|C| = "
              f"{[c['num_chargers'] for c in by_pl[pl]]})")
    expected = sorted(STAGE2_LEVELS)
    for pl in (1, 2, 3, 4):
        got = sorted([c["num_chargers"] for c in by_pl[pl]])
        if got != expected:
            print(f"  WARN P{pl}: missing levels — expected {expected}, got {got}")

    results = {
        "stage": "phase2b_stage2",
        "design": {
            "fleet_N": FLEET_N,
            "C_levels": sorted(STAGE2_LEVELS),
            "ratios": [c / FLEET_N for c in sorted(STAGE2_LEVELS)],
            "placement_tau": {f"p{pl}": {"tau_low": t[0], "tau_full": t[1]}
                              for pl, t in PLACEMENT_TAU.items()},
            "interrupt_pct": 50,
            "horizon": 100000,
        },
        "by_response": {},
    }
    for response in ("replenish_pct", "orders"):
        results["by_response"][response] = {}
        for pl in (1, 2, 3, 4):
            results["by_response"][response][f"p{pl}"] = analyse_placement(
                pl, by_pl[pl], response)

    print_table(results)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "phase2b_stage2_curves.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"[json] {out_json}")

    out_fig = FIGURES / "phase2b_stage2_saturation.png"
    plot_saturation(results, out_fig)
    print(f"[fig]  {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
