"""
eval/analyze_doe_phase2b_stage1.py
──────────────────────────────────
Phase 2b Stage 1 Response-Surface fit on the face-centred CCD restricted
to |C| ∈ {6, 18}, per placement pipeline (P1, P2, P3, P4).

Rationale (single-seed, no formal pre-tests):
  - The simulator is fully deterministic given a seed (Kleijnen, 2015,
    §3.1; Law, 2015, §11): a single run per design point is an unbiased
    realisation of the response, so the noise structure that justifies
    replication in stochastic-experiment DoE (Montgomery, 2017, §6.2) is
    absent. We therefore commit one seed per cell and budget the freed
    compute to wider factor coverage (Sanchez & Wan, 2015).
  - With n = 1 per cell the within-cell variance is identically zero, so
    no pure-error term exists and the formal lack-of-fit F-test
    (Myers et al., 2016, §2.4) is not applicable. We instead report
    R² and adj-R² and inspect residuals visually (Kutner et al., 2005,
    §3.4–3.7) — the recommended practice for moderate n where formal
    normality / homoscedasticity tests have low power (Montgomery, 2017,
    §4.4).
  - The OLS regression is implemented in NumPy (closed-form normal
    equations) with statsmodels-equivalent algebra (Seabold & Perktold,
    2010); coefficients, R², and F-statistics are mathematically
    identical to those produced by Minitab, JMP, or Design-Expert.

|C| = 30 cells are excluded (Phase 2b design memo, 2026-05-19): |C| > N
fleet size is physically meaningless for active dispatch since at most
N = 20 robots can simultaneously occupy chargers, so the |C| = 30 axial
contributes no decision-relevant information.

Fit (per placement):
  y = β₀ + β_C·x_C + β_L·x_L + β_F·x_F
      + β_LL·x_L² + β_FF·x_F²            (no x_C² — only 2 levels)
      + β_LF·x_L·x_F                     (within-policy curvature)

Coded variables (face-centred):
  x_C = (|C| − 12) / 6           ∈ {−1, +1}
  x_L = (τ_low − 20) / 5         ∈ {−1, 0, +1}
  x_F = (τ_full − 75) / 15       ∈ {−1, 0, +1}

Inputs:
  /Users/admin/netlogo_sa_8new/eval/runs/doe_phase2b/p{1..4}_n{6,18,30}_*

Outputs:
  eval/results/phase2b_stage1_rsm.json          coefficients, R², optima
  eval/results/figures/phase2b_stage1_diag.png  residual + Q-Q diagnostics

References (consulted; full bibitems appended to paper_full.tex):
  Kleijnen, J.P.C. (2015). Design and Analysis of Simulation Experiments
    (2nd ed.). Springer.
  Law, A.M. (2015). Simulation Modeling and Analysis (5th ed.). McGraw-Hill.
  Sanchez, S.M. & Wan, H. (2015). Work smarter, not harder. WSC 2015.
  Montgomery, D.C. (2017). Design and Analysis of Experiments (9th ed.). Wiley.
  Myers, R.H., Montgomery, D.C. & Anderson-Cook, C.M. (2016). Response
    Surface Methodology (4th ed.). Wiley.
  Kutner, M.H., Nachtsheim, C.J., Neter, J. & Li, W. (2005). Applied
    Linear Statistical Models (5th ed.). McGraw-Hill/Irwin.
  Seabold, S. & Perktold, J. (2010). statsmodels. SciPy 2010.
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
    tr_pre = s.get("tick_result")
    final_tick_pre = tr_pre.get("final_tick", 0.0) if isinstance(tr_pre, dict) else 0.0
    status_pre = tr_pre.get("status", "") if isinstance(tr_pre, dict) else str(tr_pre or "")
    deadlock = (h > 0) and (status_pre != "done" or final_tick_pre < 0.95 * h)
    final_tick = final_tick_pre
    status = status_pre
    return {
        "id": d.name,
        "orders": orders,
        "replenish_pct": (gained / consumed * 100) if consumed else 0.0,
        "consumed_mj": consumed / 1e6,
        "deadlock": deadlock,
        "last_tick": last_tick,
        "final_tick": final_tick,
        "status": status,
    }


def collect_cells() -> list[dict]:
    rows = []
    for d in sorted(RUN_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_cell(d.name)
        if meta is None:
            continue
        cell = load_cell(d)
        if cell is None:
            continue
        cell.update(meta)
        rows.append(cell)
    return rows


def design_matrix(x_C: np.ndarray, x_L: np.ndarray, x_F: np.ndarray) -> np.ndarray:
    n = len(x_C)
    return np.column_stack([
        np.ones(n),       # β₀
        x_C,              # β_C
        x_L,              # β_L
        x_F,              # β_F
        x_L ** 2,         # β_LL
        x_F ** 2,         # β_FF
        x_L * x_F,        # β_LF
    ])


COEF_NAMES = ["intercept", "x_C", "x_L", "x_F", "x_L^2", "x_F^2", "x_L*x_F"]


def fit_placement(rows: list[dict], response: str) -> dict:
    """Fit reduced quadratic RSM for one placement (|C| linear, τ quadratic)."""
    clean = [r for r in rows if not r["deadlock"] and r["status"] == "done"]
    if len(clean) < 7:
        raise ValueError(f"Need ≥7 clean cells, have {len(clean)}")

    nc = np.array([r["num_chargers"] for r in clean], dtype=float)
    lo = np.array([r["tau_low"] for r in clean], dtype=float)
    fu = np.array([r["tau_full"] for r in clean], dtype=float)
    y = np.array([r[response] for r in clean], dtype=float)

    x_C = (nc - 12.0) / 6.0
    x_L = (lo - 20.0) / 5.0
    x_F = (fu - 75.0) / 15.0

    X = design_matrix(x_C, x_L, x_F)
    n, p = X.shape

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if (n - p) > 0 else float("nan")

    # Per-coefficient SE / t / p via (XᵀX)⁻¹ σ̂² (statsmodels-equivalent)
    df_res = max(n - p, 1)
    sigma2 = ss_res / df_res
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        t_stat = beta / se
        # Two-sided p from t-distribution; closed-form Welch approx not needed
        # for n=10 we use the exact Student-t via numpy's special; fall back to
        # the survival function from scipy if available, else asymptotic normal
        try:
            from scipy.stats import t as student_t
            p_val = 2.0 * (1.0 - student_t.cdf(np.abs(t_stat), df_res))
        except Exception:
            from math import erf, sqrt
            p_val = np.array([2.0 * (1.0 - 0.5 * (1 + erf(abs(ts) / sqrt(2)))) for ts in t_stat])
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
        t_stat = np.full(p, np.nan)
        p_val = np.full(p, np.nan)

    # Stationary point of τ surface: solve 2β_LL x_L + β_LF x_F = -β_L,
    #                                       β_LF x_L + 2β_FF x_F = -β_F
    bL, bF, bLL, bFF, bLF = beta[2], beta[3], beta[4], beta[5], beta[6]
    H = np.array([[2 * bLL, bLF], [bLF, 2 * bFF]])
    g = np.array([-bL, -bF])
    try:
        x_star = np.linalg.solve(H, g)
        eig = np.linalg.eigvalsh(H)
        is_max = bool(np.all(eig < 0))
    except np.linalg.LinAlgError:
        x_star = np.array([np.nan, np.nan])
        is_max = False

    tau_low_star = float(20.0 + 5.0 * x_star[0])
    tau_full_star = float(75.0 + 15.0 * x_star[1])

    # Clip to design face for the reported optimum
    tau_low_star_clip = float(np.clip(tau_low_star, 15, 25))
    tau_full_star_clip = float(np.clip(tau_full_star, 60, 90))

    # Grid search confirms (and handles saddle / boundary optima)
    grid_lo = np.linspace(15, 25, 101)
    grid_fu = np.linspace(60, 90, 121)
    LL, FF = np.meshgrid(grid_lo, grid_fu, indexing="ij")
    xLg = (LL - 20.0) / 5.0
    xFg = (FF - 75.0) / 15.0
    # Evaluate at both |C| levels and take the higher
    grid_per_C = {}
    for c_label, x_c_val in [("C6", -1.0), ("C18", +1.0)]:
        Yg = (beta[0]
              + beta[1] * x_c_val
              + beta[2] * xLg
              + beta[3] * xFg
              + beta[4] * xLg ** 2
              + beta[5] * xFg ** 2
              + beta[6] * xLg * xFg)
        idx = np.unravel_index(np.argmax(Yg), Yg.shape)
        grid_per_C[c_label] = {
            "tau_low": float(LL[idx]),
            "tau_full": float(FF[idx]),
            "predicted": float(Yg[idx]),
        }

    coefs = {name: float(b) for name, b in zip(COEF_NAMES, beta)}
    coef_stats = {
        name: {
            "estimate": float(b),
            "std_err": float(s),
            "t": float(t),
            "p": float(pv),
        }
        for name, b, s, t, pv in zip(COEF_NAMES, beta, se, t_stat, p_val)
    }

    return {
        "response": response,
        "n_observations": int(n),
        "n_parameters": int(p),
        "df_residual": int(df_res),
        "coefficients": coefs,
        "coef_stats": coef_stats,
        "r_squared": float(r2),
        "adj_r_squared": float(adj_r2),
        "ss_residual": ss_res,
        "ss_total": ss_tot,
        "sigma_hat": float(math.sqrt(sigma2)),
        "stationary_point": {
            "tau_low": tau_low_star,
            "tau_full": tau_full_star,
            "is_maximum": is_max,
            "interior_to_design": (15 <= tau_low_star <= 25) and (60 <= tau_full_star <= 90),
        },
        "optimum_clipped": {
            "tau_low": tau_low_star_clip,
            "tau_full": tau_full_star_clip,
        },
        "grid_optimum_per_C": grid_per_C,
        "design_points": [
            {"id": r["id"],
             "num_chargers": r["num_chargers"],
             "tau_low": r["tau_low"],
             "tau_full": r["tau_full"],
             "y": float(r[response]),
             "y_hat": float(yh),
             "residual": float(rs)}
            for r, yh, rs in zip(clean, y_hat, resid)
        ],
        "deadlock_excluded": [r["id"] for r in rows if r["deadlock"]],
        "incomplete_excluded": [r["id"] for r in rows if r["status"] != "done" and not r["deadlock"]],
    }


def plot_diagnostics(fits_by_pl: dict[int, dict], response: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    placements = [1, 2, 3, 4]
    labels = {1: "P1 Set Cover", 2: "P2 Affinity Prop", 3: "P3 Picker Heur", 4: "P4 Perimeter"}
    for col, pl in enumerate(placements):
        fit = fits_by_pl[pl]
        y_hat = np.array([d["y_hat"] for d in fit["design_points"]])
        resid = np.array([d["residual"] for d in fit["design_points"]])

        # Top row: residuals vs fitted
        ax = axes[0, col]
        ax.scatter(y_hat, resid, s=30, color="steelblue", edgecolor="black")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel(f"Fitted {response}")
        if col == 0:
            ax.set_ylabel("Residual")
        ax.set_title(f"{labels[pl]}\nR²={fit['r_squared']:.3f}, adj={fit['adj_r_squared']:.3f}")

        # Bottom row: Q-Q
        ax = axes[1, col]
        sresid = np.sort(resid)
        n_pts = len(sresid)
        q_theor = np.array([(i + 0.5) / n_pts for i in range(n_pts)])
        try:
            from scipy.stats import norm
            z = norm.ppf(q_theor)
        except Exception:
            from math import sqrt
            z = np.array([math.sqrt(2) * math.erfinv(2 * q - 1) for q in q_theor])
        # Standardise residuals for the Q-Q (use sigma_hat)
        sigma_hat = fit["sigma_hat"] or 1.0
        ax.scatter(z, sresid / sigma_hat, s=30, color="darkorange", edgecolor="black")
        zmin, zmax = float(z.min()), float(z.max())
        ax.plot([zmin, zmax], [zmin, zmax], color="gray", lw=0.8, linestyle="--")
        ax.set_xlabel("Theoretical N(0,1) quantile")
        if col == 0:
            ax.set_ylabel("Standardised residual")
        ax.set_title("Q-Q plot")
    fig.suptitle(f"Phase 2b Stage 1 RSM diagnostics — response: {response}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    rows = collect_cells()
    print(f"Loaded {len(rows)} Stage A cells from {RUN_DIR}")
    by_pl_full = {pl: [r for r in rows if r["placement"] == pl] for pl in (1, 2, 3, 4)}
    # Apply |C| ≤ N = 20 cap per Phase 2b design decision (drop |C|=30)
    by_pl = {pl: [r for r in cells if r["num_chargers"] <= 20]
             for pl, cells in by_pl_full.items()}
    for pl, cells in by_pl.items():
        kept = len(cells)
        dropped = len(by_pl_full[pl]) - kept
        deadlock = sum(1 for r in cells if r["deadlock"])
        print(f"  P{pl}: kept {kept} cells (dropped {dropped} |C|=30); deadlock-excluded {deadlock}")

    output = {
        "stage": "phase2b_stage1",
        "factor_cap": "|C| <= N = 20 (|C|=30 cells dropped per design decision 2026-05-19)",
        "coded_scale": {
            "x_C": "(|C| - 12) / 6  in {-1, +1}",
            "x_L": "(tau_low - 20) / 5  in {-1, 0, +1}",
            "x_F": "(tau_full - 75) / 15  in {-1, 0, +1}",
        },
        "model_form": "y = b0 + bC*xC + bL*xL + bF*xF + bLL*xL^2 + bFF*xF^2 + bLF*xL*xF",
        "by_response": {},
    }

    for response in ("orders", "replenish_pct"):
        print()
        print("=" * 86)
        print(f"  RESPONSE: {response}")
        print("=" * 86)
        fits_by_pl = {}
        for pl in (1, 2, 3, 4):
            cells = by_pl[pl]
            try:
                fit = fit_placement(cells, response)
            except ValueError as e:
                print(f"  P{pl}: SKIPPED — {e}")
                continue
            fits_by_pl[pl] = fit
            cs = fit["coef_stats"]
            print(f"  P{pl} (n={fit['n_observations']}, p={fit['n_parameters']}, "
                  f"df_res={fit['df_residual']}):")
            print(f"     R² = {fit['r_squared']:.4f}   adj R² = {fit['adj_r_squared']:.4f}   "
                  f"σ̂ = {fit['sigma_hat']:.3f}")
            for nm in COEF_NAMES:
                s = cs[nm]
                tag = "***" if s["p"] < 0.01 else ("**" if s["p"] < 0.05 else ("*" if s["p"] < 0.1 else ""))
                print(f"        {nm:<10} = {s['estimate']:>+10.4f}  "
                      f"(SE {s['std_err']:>7.4f}, t {s['t']:>+6.2f}, p {s['p']:>.4f}) {tag}")
            sp = fit["stationary_point"]
            oc = fit["optimum_clipped"]
            sp_tag = "INTERIOR MAX" if sp["is_maximum"] and sp["interior_to_design"] else (
                "SADDLE" if not sp["is_maximum"] else "EXTERIOR")
            print(f"     Stationary point: τ_low*={sp['tau_low']:.2f}, "
                  f"τ_full*={sp['tau_full']:.2f}  [{sp_tag}]")
            print(f"     Reported optimum (clipped to design face): "
                  f"τ_low={oc['tau_low']:.2f}, τ_full={oc['tau_full']:.2f}")
            for c_label, g in fit["grid_optimum_per_C"].items():
                print(f"        grid optimum @ {c_label}: τ_low={g['tau_low']:.1f}, "
                      f"τ_full={g['tau_full']:.1f}, ŷ={g['predicted']:.3f}")

        output["by_response"][response] = {f"p{pl}": fit for pl, fit in fits_by_pl.items()}
        # Diagnostics figure
        if len(fits_by_pl) == 4:
            fig_path = FIGURES / f"phase2b_stage1_diag_{response}.png"
            plot_diagnostics(fits_by_pl, response, fig_path)
            print(f"  [figure] {fig_path}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "phase2b_stage1_rsm.json"
    out.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n[json]   {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
