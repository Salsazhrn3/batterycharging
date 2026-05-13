"""
eval/analyze_doe_phase3.py
──────────────────────────
Phase 3 Taguchi-style robustness analysis.

Reads the 18 cells under eval/runs/doe_phase3/ (2 candidates x L9), computes
signal-to-noise (S/N) ratios per candidate and per response, and reports
which P3 candidate survives the noise envelope better.

S/N formulas:
  Larger-the-better (orders, replenish):
    eta = -10 log10( (1/n) * sum(1/y_i**2) )
  Smaller-the-better (energy):
    eta = -10 log10( (1/n) * sum(y_i**2) )
  Nominal-the-best (mean_battery): not used here.

Plus:
  - Deadlock count per candidate (constraint metric).
  - Per-noise-factor level means (which factor matters most).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOE_DIR = ROOT / "eval" / "runs" / "doe_phase3"
RESULTS = ROOT / "eval" / "results"
BATTERY_J = 6_480_000.0
BASE_DRAIN = 90.0
HORIZON = 20000

L9 = [
    (1, 1, 1, 1), (1, 2, 2, 2), (1, 3, 3, 3),
    (2, 1, 2, 3), (2, 2, 3, 1), (2, 3, 1, 2),
    (3, 1, 3, 2), (3, 2, 1, 3), (3, 3, 2, 1),
]
DEMAND = [0.7, 1.0, 1.3]
ROBOTS = [18, 20, 22]
FAULTS = [0, 1, 2]
INIT_BATT = [0.5, 0.7, 1.0]


def load_cell(d: Path) -> dict | None:
    rs = d / "run_summary.json"
    if not rs.exists():
        return None
    s = json.loads(rs.read_text())
    per = json.loads((d / "per_robot.json").read_text())
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
    return {
        "id": d.name,
        "orders": orders,
        "replenish_pct": gained / consumed * 100 if consumed else 0,
        "consumed_mj": consumed / 1e6,
        "deadlock": h > 0 and last_tick < 0.9 * h,
        "last_tick": last_tick,
        "num_robots": nr,
    }


def parse_cell_id(cid: str) -> tuple[str, int] | None:
    """Return (candidate, run_idx) for 'robust_run07' or 'peak_run01'."""
    if cid.startswith("robust_run"):
        return "robust", int(cid.split("run")[1])
    if cid.startswith("peak_run"):
        return "peak", int(cid.split("run")[1])
    return None


def sn_larger_better(values: list[float]) -> float:
    """Larger-the-better S/N: eta = -10 log10(mean(1/y^2))."""
    safe = [v for v in values if v > 0]
    if not safe:
        return float("nan")
    return -10 * math.log10(sum(1 / v ** 2 for v in safe) / len(safe))


def sn_smaller_better(values: list[float]) -> float:
    """Smaller-the-better S/N: eta = -10 log10(mean(y^2))."""
    if not values:
        return float("nan")
    return -10 * math.log10(sum(v ** 2 for v in values) / len(values))


def main() -> int:
    rows = []
    for d in sorted(DOE_DIR.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_cell_id(d.name)
        if parsed is None:
            continue
        cand, run_idx = parsed
        cell = load_cell(d)
        if cell is None:
            continue
        a, b, c, dd = L9[run_idx - 1]
        cell.update({"candidate": cand, "run_idx": run_idx,
                     "demand": DEMAND[a-1], "robots_setting": ROBOTS[b-1],
                     "faults": FAULTS[c-1], "init_batt": INIT_BATT[dd-1]})
        rows.append(cell)

    print(f"Loaded {len(rows)} cells")
    print()

    # ── Per-cell summary table ──────────────────────────────────────────
    print("=" * 100)
    print(f'{"id":<18} {"D":>5} {"R":>3} {"F":>2} {"Init":>5}  '
          f'{"orders":>6} {"replen%":>8} {"cons.MJ":>7}  DL?')
    print("-" * 100)
    for r in sorted(rows, key=lambda r: (r["candidate"], r["run_idx"])):
        dl = "YES" if r["deadlock"] else ""
        print(f'{r["id"]:<18} {r["demand"]:>5.1f} {r["robots_setting"]:>3} {r["faults"]:>2} '
              f'{r["init_batt"]:>5.2f}  {r["orders"]:>6} {r["replenish_pct"]:>+7.1f}% '
              f'{r["consumed_mj"]:>6.2f}  {dl:>3}')

    # ── Per-candidate summary ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Per-candidate Taguchi S/N analysis")
    print("=" * 100)
    by_cand: dict[str, list[dict]] = {}
    for r in rows:
        by_cand.setdefault(r["candidate"], []).append(r)

    for cand, cells in sorted(by_cand.items()):
        n = len(cells)
        n_dl = sum(1 for c in cells if c["deadlock"])
        orders = [c["orders"] for c in cells]
        replen = [c["replenish_pct"] for c in cells]
        energy = [c["consumed_mj"] for c in cells]
        sn_orders = sn_larger_better(orders)
        sn_energy = sn_smaller_better(energy)
        # replenish can be negative (when consumption > gained); shift by 100
        # so all values are positive for the larger-better formula. Equivalent
        # to maximizing replenish + 100.
        replen_shifted = [v + 100 for v in replen]
        sn_replen = sn_larger_better(replen_shifted)
        print(f"\nCandidate '{cand}'  ({n} cells, {n_dl} deadlock):")
        print(f"  orders          mean={sum(orders)/n:>7.1f}  "
              f"min={min(orders):>5}  max={max(orders):>5}  S/N={sn_orders:>+6.2f} dB")
        print(f"  replenish (%)   mean={sum(replen)/n:>+7.1f}  "
              f"min={min(replen):>+5.1f}  max={max(replen):>+5.1f}  S/N={sn_replen:>+6.2f} dB")
        print(f"  consumed (MJ)   mean={sum(energy)/n:>7.2f}  "
              f"min={min(energy):>5.2f}  max={max(energy):>5.2f}  S/N={sn_energy:>+6.2f} dB")

    # ── Per-noise-factor level means (for each candidate, each response) ─
    print("\n" + "=" * 100)
    print("Per-noise-factor level means (orders), per candidate")
    print("=" * 100)
    for cand, cells in sorted(by_cand.items()):
        print(f"\n'{cand}':")
        for fname, attr, levels in [("demand", "demand", DEMAND),
                                     ("robots", "robots_setting", ROBOTS),
                                     ("faults", "faults", FAULTS),
                                     ("init_batt", "init_batt", INIT_BATT)]:
            line = f"  {fname:<10}"
            for lv in levels:
                sub = [c for c in cells if c[attr] == lv]
                m = sum(c["orders"] for c in sub) / len(sub) if sub else float("nan")
                dl = sum(1 for c in sub if c["deadlock"])
                line += f"  L={lv}: {m:>6.0f} orders ({dl} DL of {len(sub)})"
            print(line)

    # ── Verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Verdict")
    print("=" * 100)
    cells_robust = by_cand.get("robust", [])
    cells_peak = by_cand.get("peak", [])
    if cells_robust and cells_peak:
        dl_robust = sum(1 for c in cells_robust if c["deadlock"])
        dl_peak = sum(1 for c in cells_peak if c["deadlock"])
        sn_robust = sn_larger_better([c["orders"] for c in cells_robust])
        sn_peak = sn_larger_better([c["orders"] for c in cells_peak])
        print(f"  Robust  : deadlock {dl_robust}/9  orders S/N = {sn_robust:+.2f} dB")
        print(f"  Peak    : deadlock {dl_peak}/9  orders S/N = {sn_peak:+.2f} dB")
        winner = "robust" if (dl_robust < dl_peak or
                              (dl_robust == dl_peak and sn_robust > sn_peak)) else "peak"
        print(f"  Winner  : {winner}")
        if winner == "robust":
            print(f"  -> Constrained RSM optimum is more robust than Phase 1 best corner.")
            print(f"  -> Recommended (placement, policy) for the thesis: P3 + {winner}.")

    # Save CSV for later plotting / paper.
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "doe_phase3_responses.csv"
    keys = ["id", "candidate", "run_idx", "demand", "robots_setting", "faults",
            "init_batt", "orders", "replenish_pct", "consumed_mj", "deadlock",
            "last_tick", "num_robots"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in keys} for r in rows])
    print(f"\n[csv] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
