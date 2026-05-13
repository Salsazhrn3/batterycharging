"""
eval/fit_p3_rsm.py
──────────────────
Phase 2 P3 quadratic response surface fit, with empirical deadlock
manifold treated as an infeasibility constraint (CRSM, per Myers,
Montgomery & Anderson-Cook §7.5).

Inputs:
    Phase 1 P3 cells with interrupt=on@50% (4 corners)
    Phase 2 P3 cells with interrupt=on@50% (axials + center + boundary probes)

Excluded:
    Any P3 cell where deadlock fired (last_order_tick < 0.9 * horizon)

Fit: y = β0 + β1·xL + β2·xF + β11·xL² + β22·xF² + β12·xL·xF
      where xL = (tau_low - 20) / 5  and  xF = (tau_full - 75) / 15
      (centered + scaled to roughly [-1, 1] over the design ranges).

Outputs:
    eval/results/p3_rsm_fit.json  — coefficients, R², optima
    Console summary with both unconstrained and constrained optima.

Feasible region F:
    F = {(tau_low, tau_full) : NOT (19.5 ≤ tau_low ≤ 20.5 AND tau_full ≥ 89)}

Empirical basis: deadlock observed at (20, 90, on@50) and (20, 92, on@50).
                Clean at (19, 90), (21, 90), (20, 88), all (15-25, 60-75).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT / "eval" / "runs" / "doe_phase1"
PHASE2 = ROOT / "eval" / "runs" / "doe_phase2"
RESULTS = ROOT / "eval" / "results"
BATTERY_J = 6_480_000.0
BASE_DRAIN = 90.0
HORIZON = 20000


def load_cell(cell_dir: Path) -> dict | None:
    rs = cell_dir / "run_summary.json"
    if not rs.exists():
        return None
    s = json.loads(rs.read_text())
    per = json.loads((cell_dir / "per_robot.json").read_text())
    nr = len(per)
    h = s.get("horizon", 0)
    motion = sum(r["energy_consumption_j"] for r in per)
    consumed = motion + BASE_DRAIN * h * nr
    depleted = sum(BATTERY_J - r["battery_level_j"] for r in per)
    gained = consumed - depleted
    of = cell_dir / "order-finished.csv"
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
    deadlock = h > 0 and last_tick < 0.9 * h
    return {
        "id": cell_dir.name,
        "orders": orders,
        "replenish_pct": gained / consumed * 100 if consumed else 0,
        "consumed_mj": consumed / 1e6,
        "deadlock": deadlock,
        "last_tick": last_tick,
    }


def parse_p3_cell(cid: str) -> tuple | None:
    """Return (tau_low, tau_full, interrupt) for a 'p3_loX_fuY_intZ' cell."""
    if not cid.startswith("p3_lo"):
        return None
    try:
        parts = cid.split("_")
        lo = int(parts[1][2:])
        fu = int(parts[2][2:])
        intr = int(parts[3][3:])
    except (IndexError, ValueError):
        return None
    return lo, fu, intr


def collect_p3_int50() -> list[dict]:
    """Collect all P3 cells with interrupt=on@50%, with deadlock flagging."""
    rows = []
    for src in (PHASE1, PHASE2):
        if not src.exists():
            continue
        for d in sorted(src.iterdir()):
            if not d.is_dir():
                continue
            parsed = parse_p3_cell(d.name)
            if parsed is None:
                continue
            lo, fu, intr = parsed
            if intr != 50:
                continue
            cell = load_cell(d)
            if cell is None:
                continue
            cell.update({"tau_low": lo, "tau_full": fu, "interrupt": intr})
            rows.append(cell)
    return rows


def design_matrix_quadratic(xL: np.ndarray, xF: np.ndarray) -> np.ndarray:
    """Build the design matrix [1, xL, xF, xL², xF², xL*xF]."""
    n = len(xL)
    X = np.column_stack([
        np.ones(n),
        xL,
        xF,
        xL ** 2,
        xF ** 2,
        xL * xF,
    ])
    return X


def is_in_deadlock_zone(tau_low: float, tau_full: float) -> bool:
    """Empirical deadlock manifold for P3 at interrupt=on@50%."""
    return (19.5 <= tau_low <= 20.5) and (tau_full >= 89)


def fit_response(rows: list[dict], response: str) -> dict:
    """Fit a quadratic on the given response across the clean cells."""
    clean = [r for r in rows if not r["deadlock"]]
    tau_low = np.array([r["tau_low"] for r in clean], dtype=float)
    tau_full = np.array([r["tau_full"] for r in clean], dtype=float)
    y = np.array([r[response] for r in clean], dtype=float)

    # Center + scale (improves conditioning of the regression)
    xL = (tau_low - 20.0) / 5.0
    xF = (tau_full - 75.0) / 15.0

    X = design_matrix_quadratic(xL, xF)
    n, p = X.shape
    if n < p:
        raise ValueError(f"Need >= {p} clean points, have {n}")

    # OLS regression (closed-form; defensible, reproducible)
    beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    coefs = dict(zip(
        ["intercept", "tau_low", "tau_full", "tau_low_sq",
         "tau_full_sq", "tau_low_x_tau_full"],
        beta.tolist(),
    ))

    # ── Grid search for unconstrained and constrained optima ───────────────
    grid_lo = np.linspace(15, 25, 101)
    grid_fu = np.linspace(60, 90, 121)
    LL, FF = np.meshgrid(grid_lo, grid_fu, indexing="ij")
    xLg = (LL - 20.0) / 5.0
    xFg = (FF - 75.0) / 15.0
    Xg = np.stack([
        np.ones_like(xLg), xLg, xFg, xLg ** 2, xFg ** 2, xLg * xFg
    ], axis=-1)
    Yg = (Xg @ beta)

    # Unconstrained
    idx_u = np.unravel_index(np.argmax(Yg), Yg.shape)
    unc = {
        "tau_low": float(LL[idx_u]),
        "tau_full": float(FF[idx_u]),
        "predicted": float(Yg[idx_u]),
        "in_feasible_region": not is_in_deadlock_zone(LL[idx_u], FF[idx_u]),
    }

    # Constrained (mask the deadlock zone)
    feasible_mask = ~np.vectorize(is_in_deadlock_zone)(LL, FF)
    Yg_c = np.where(feasible_mask, Yg, -np.inf)
    idx_c = np.unravel_index(np.argmax(Yg_c), Yg_c.shape)
    con = {
        "tau_low": float(LL[idx_c]),
        "tau_full": float(FF[idx_c]),
        "predicted": float(Yg_c[idx_c]),
    }

    return {
        "response": response,
        "n_observations": int(n),
        "deadlock_excluded": [r["id"] for r in rows if r["deadlock"]],
        "coefficients": coefs,
        "r_squared": float(r2),
        "ss_residual": ss_res,
        "ss_total": ss_tot,
        "unconstrained_optimum": unc,
        "constrained_optimum": con,
        "design_points": [
            {"tau_low": int(r["tau_low"]), "tau_full": int(r["tau_full"]),
             "y": float(r[response])} for r in clean
        ],
    }


def main() -> int:
    rows = collect_p3_int50()
    print(f"Loaded {len(rows)} P3 cells at interrupt=on@50%")
    print(f"  clean: {sum(1 for r in rows if not r['deadlock'])}")
    print(f"  deadlocked: {sum(1 for r in rows if r['deadlock'])}")
    for r in rows:
        flag = "DEADLOCK" if r["deadlock"] else ""
        print(f"  ({r['tau_low']:>2}, {r['tau_full']:>2}): replen={r['replenish_pct']:>5.2f}%  "
              f"orders={r['orders']:>4}  {flag}")

    fits = {}
    for resp in ["replenish_pct", "orders", "consumed_mj"]:
        fit = fit_response(rows, resp)
        fits[resp] = fit
        print()
        print("=" * 80)
        print(f"P3 quadratic fit on  {resp}")
        print("=" * 80)
        print(f"  observations used: {fit['n_observations']} (deadlock excluded)")
        print(f"  R² = {fit['r_squared']:.4f}")
        print(f"  coefficients:")
        for k, v in fit["coefficients"].items():
            print(f"    {k:<24} {v:>+.4f}")
        print(f"\n  Unconstrained optimum:")
        u = fit["unconstrained_optimum"]
        flag = "(in feasible region)" if u["in_feasible_region"] else "(IN DEADLOCK ZONE!)"
        print(f"    tau_low = {u['tau_low']:.1f},  tau_full = {u['tau_full']:.1f}")
        print(f"    predicted {resp} = {u['predicted']:.3f}  {flag}")
        print(f"\n  Constrained optimum (feasible region only):")
        c = fit["constrained_optimum"]
        print(f"    tau_low = {c['tau_low']:.1f},  tau_full = {c['tau_full']:.1f}")
        print(f"    predicted {resp} = {c['predicted']:.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "p3_rsm_fit.json"
    out.write_text(json.dumps({
        "feasible_region_F": "NOT (19.5 <= tau_low <= 20.5 AND tau_full >= 89)",
        "interrupt_fixed": "on@50% (BATTERY_INTERRUPT_PCT=50)",
        "fits": fits,
    }, indent=2, default=str))
    print(f"\n[json] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
