"""
eval/analyze_doe_phase1.py
──────────────────────────
Phase 1 (screening) analysis. Reads the 32 cells under
eval/runs/doe_phase1/p{P}_lo{tau_low}_fu{tau_full}_int{0|50}/ and produces:

  1. Per-cell response table (CSV + console).
  2. Per-placement main effects of (tau_low, tau_full, interrupt).
  3. Per-placement 2-factor interactions.
  4. Pareto-domination drop criterion (which placements survive into Phase 2).

Responses recorded:
  - replenish_pct (%)        gain / consumed energy
  - orders_completed         from order-finished.csv
  - total_energy_mj          consumed across the fleet
  - mean_battery_pct         end-of-run mean SoC across robots
  - dead_count               robots in 'dead' state or battery <= 0.5%
  - max_queue                from run_summary stop_and_go (proxy)

Usage:
    python eval/analyze_doe_phase1.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent
DOE_DIR = ROOT / "eval" / "runs" / "doe_phase1"
PHASE2_DIR = ROOT / "eval" / "runs" / "doe_phase2"

BATTERY_J = 6_480_000.0
BASE_DRAIN = 90.0


def parse_id(cell_id: str) -> dict | None:
    m = re.match(r"p(\d+)_lo(\d+)_fu(\d+)_int(\d+)$", cell_id)
    if not m:
        return None
    return {
        "placement": int(m.group(1)),
        "tau_low": int(m.group(2)),
        "tau_full": int(m.group(3)),
        "interrupt": int(m.group(4)),
    }


def load_cell(cell_dir: Path) -> dict | None:
    parsed = parse_id(cell_dir.name)
    if parsed is None or not (cell_dir / "run_summary.json").exists():
        return None
    s = json.loads((cell_dir / "run_summary.json").read_text())
    per = json.loads((cell_dir / "per_robot.json").read_text())
    nr = len(per)
    h = s.get("horizon", 0)
    motion = sum(r["energy_consumption_j"] for r in per)
    base = BASE_DRAIN * h * nr
    consumed = motion + base
    depleted = sum(BATTERY_J - r["battery_level_j"] for r in per)
    gained = consumed - depleted
    of = cell_dir / "order-finished.csv"
    orders = 0
    last_order_tick = 0.0
    if of.exists():
        with open(of) as f:
            import csv as _csv
            rdr = _csv.DictReader(f)
            for row in rdr:
                orders += 1
                try:
                    last_order_tick = max(last_order_tick,
                                          float(row.get("order_complete_time") or 0))
                except (TypeError, ValueError):
                    pass
    dead = sum(1 for r in per
               if r.get("current_state") == "dead"
               or r.get("battery_pct", 100) <= 0.5)
    mbatt = sum(r.get("battery_pct", 0) for r in per) / nr
    tr = s.get("tick_result", {})
    # Deadlock indicator: last order completed at < 90% of the horizon.
    # In a healthy run, the picker stations finish orders right up to the
    # horizon; if the last completion is far before the end, the fleet has
    # frozen (routing deadlock, dispatch deadlock, etc.).
    deadlock = h > 0 and last_order_tick < 0.9 * h
    parsed.update({
        "id": cell_dir.name,
        "orders": orders,
        "last_order_tick": last_order_tick,
        "deadlock": deadlock,
        "replenish_pct": gained / consumed * 100 if consumed else 0,
        "consumed_mj": consumed / 1e6,
        "gained_mj": gained / 1e6,
        "mean_battery_pct": mbatt,
        "dead_count": dead,
        "stop_and_go": tr.get("stop_and_go", 0) if isinstance(tr, dict) else 0,
    })
    return parsed


def main() -> int:
    rows: list[dict] = []
    for src in [DOE_DIR, PHASE2_DIR]:
        if not src.exists():
            continue
        for d in sorted(src.iterdir()):
            if not d.is_dir():
                continue
            cell = load_cell(d)
            if cell is not None:
                cell["phase"] = 1 if src == DOE_DIR else 2
                rows.append(cell)

    print(f"Loaded {len(rows)} cells\n")

    # ── Deadlock report ──────────────────────────────────────────────────
    deadlocks = [r for r in rows if r["deadlock"]]
    print("=" * 100)
    print(f"Deadlock cells (last order tick < 90% of horizon): {len(deadlocks)} / {len(rows)}")
    print("=" * 100)
    for r in deadlocks:
        print(f'  {r["id"]:<22}  last_order_tick={r["last_order_tick"]:>7.0f}  '
              f'orders={r["orders"]} (vs ~1750 normal)')
    if not deadlocks:
        print("  (none)")

    # ── Per-cell response table ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Per-cell responses (sorted by placement, tau_low, tau_full, interrupt)")
    print("=" * 100)
    rows.sort(key=lambda r: (r["placement"], r["tau_low"], r["tau_full"], r["interrupt"]))
    print(f'{"id":<22} {"P":>2} {"lo":>4} {"fu":>4} {"int":>4}  '
          f'{"orders":>6} {"replen%":>8} {"consMJ":>7} {"mbatt%":>6} {"dead":>4} {"DL":>3}')
    print("-" * 100)
    for r in rows:
        dl = "DL" if r["deadlock"] else ""
        print(f'{r["id"]:<22} {r["placement"]:>2} {r["tau_low"]:>4} {r["tau_full"]:>4} '
              f'{r["interrupt"]:>4}  {r["orders"]:>6} {r["replenish_pct"]:>7.1f}% '
              f'{r["consumed_mj"]:>6.2f}  {r["mean_battery_pct"]:>5.1f}% {r["dead_count"]:>4} {dl:>3}')

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = ROOT / "eval" / "results" / "doe_phase1_responses.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {csv_path}")

    # ── Per-placement main effects (deadlock cells excluded) ────────────
    print("\n" + "=" * 100)
    print("Per-placement main effects (high - low), DEADLOCK CELLS EXCLUDED")
    print("=" * 100)
    factors = ["tau_low", "tau_full", "interrupt"]
    responses = ["orders", "replenish_pct", "consumed_mj", "mean_battery_pct"]
    clean_rows = [r for r in rows if not r["deadlock"]]

    for p in sorted({r["placement"] for r in clean_rows}):
        sub = [r for r in clean_rows if r["placement"] == p]
        n_dl = sum(1 for r in rows if r["placement"] == p and r["deadlock"])
        note = f" ({n_dl} deadlock cell(s) excluded)" if n_dl else ""
        print(f"\nPlacement P{p}{note}:")
        print(f"  {'response':<18} " + " ".join(f"{f:>14}" for f in factors))
        for resp in responses:
            line = f"  {resp:<18} "
            for f in factors:
                lvls = sorted({r[f] for r in sub})
                if len(lvls) != 2:
                    line += f"{'n/a':>14}"
                    continue
                lo, hi = lvls
                lo_cells = [r for r in sub if r[f] == lo]
                hi_cells = [r for r in sub if r[f] == hi]
                if not lo_cells or not hi_cells:
                    line += f"{'imbal':>14}"
                    continue
                lo_mean = sum(r[resp] for r in lo_cells) / len(lo_cells)
                hi_mean = sum(r[resp] for r in hi_cells) / len(hi_cells)
                line += f"{hi_mean - lo_mean:>+14.2f}"
            print(line)

    # ── Per-placement best corner + Pareto-domination drop ───────────────
    print("\n" + "=" * 100)
    print("Per-placement best corner (max replenish, max orders)")
    print("=" * 100)
    best_per_p = {}
    for p in sorted({r["placement"] for r in rows}):
        sub = [r for r in rows if r["placement"] == p]
        best_replen = max(sub, key=lambda r: r["replenish_pct"])
        best_orders = max(sub, key=lambda r: r["orders"])
        print(f"\nP{p}:")
        print(f"  best replen: {best_replen['id']:<22} "
              f"replen={best_replen['replenish_pct']:.1f}% orders={best_replen['orders']} "
              f"dead={best_replen['dead_count']}")
        print(f"  best orders: {best_orders['id']:<22} "
              f"replen={best_orders['replenish_pct']:.1f}% orders={best_orders['orders']} "
              f"dead={best_orders['dead_count']}")
        best_per_p[p] = best_replen

    # Pareto-domination: drop P_i if some P_j strictly dominates it
    # across (orders, replenish, -dead).
    print("\n" + "=" * 100)
    print("Pareto-domination drop criterion (using each placement's best-replenish corner)")
    print("=" * 100)
    survivors = []
    for p, cell in best_per_p.items():
        dominated_by = None
        for q, qcell in best_per_p.items():
            if p == q:
                continue
            if (qcell["orders"] >= cell["orders"]
                    and qcell["replenish_pct"] >= cell["replenish_pct"]
                    and qcell["dead_count"] <= cell["dead_count"]
                    and (qcell["orders"] > cell["orders"]
                         or qcell["replenish_pct"] > cell["replenish_pct"]
                         or qcell["dead_count"] < cell["dead_count"])):
                dominated_by = q
                break
        if dominated_by:
            print(f"  P{p}  DROPPED — dominated by P{dominated_by}")
        else:
            print(f"  P{p}  KEPT (Pareto-front member)")
            survivors.append(p)
    print(f"\nPhase 2 survivors: {survivors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
