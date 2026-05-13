"""
eval/run_doe_screening.py
─────────────────────────
Phase 2 — Design of Experiments screening run.

Joint placement-policy screening: 4 placements (P1-P4) × 2^3 full-factorial
in (tau_low, tau_full, interrupt) at fixed |C| = 12, 20k horizon.

  Factor             Levels (low / high)
  --------------------------------------
  tau_low            15% / 25%      (BATTERY_LOW_PCT)
  tau_full           60% / 90%      (BATTERY_CHARGED_PCT)
  interrupt          off / on@50%   (BATTERY_INTERRUPT_PCT = 200 / 50)
  placement          P1 / P2 / P3 / P4 (4-level categorical block)

Total: 4 × 8 = 32 runs at 20k ticks ≈ 25 h wall-clock (sequential).

Each run writes its artifacts to:
  eval/runs/doe_phase1/p{P}_lo{tau_low}_fu{tau_full}_int{0|50}/
The override JSON is also persisted for reproducibility.

Usage:
    python eval/run_doe_screening.py --dry-run     # print design matrix only
    python eval/run_doe_screening.py               # launch sequential runs
    python eval/run_doe_screening.py --placements 3 --tau-low 25
        # filter to a subset of cells (handy for retries)
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOE_DIR = ROOT / "eval" / "runs" / "doe_phase1"

PLACEMENTS = [1, 2, 3, 4]
TAU_LOW_LEVELS = [15, 25]
TAU_FULL_LEVELS = [60, 90]
INTERRUPT_LEVELS = [0, 50]   # 0 = off (will be encoded as BATTERY_INTERRUPT_PCT=200)
                              # 50 = on at 50%

NUM_CHARGERS = 12
HORIZON = 20000
HEARTBEAT = 4000

# Canonical 12-position layout for P1 set-cover, copied from
# eval/runs/p1_100k/p1/charging_config.json. Required because the set-cover
# algorithm in apply_set_cover_layout() ignores num_chargers and produces 40
# cells; injecting an explicit charger_positions list bypasses the algorithm
# and uses these cells directly. Keeps P1 comparable to P2/P3/P4 at |C|=12.
P1_CANONICAL_LAYOUT = [
    [14, 16], [16, 37], [25, 13], [2, 35], [5, 10], [26, 36],
    [16, 10], [7, 37], [25, 10], [19, 36], [28, 35], [1, 10],
]

# Canonical 12-position layout for P3 picker-tier opportunistic, copied from
# eval/runs/p3_100k/p3/charging_config.json. Without an explicit list the
# picker-station heuristic falls back to non-selective mode and registers
# only the 5 picker-station cells themselves, which would make P3 in the
# DoE incomparable to P3 in the placement-only sweep paper. The canonical
# layout = T1 (col-2 picker rows) + T2 (col-4 picker_row+2) + 2 T3 cells.
P3_CANONICAL_LAYOUT = [
    [1, 2], [3, 4], [3, 3], [7, 2], [9, 4], [13, 2],
    [15, 4], [19, 2], [21, 4], [21, 3], [25, 2], [27, 4],
]


def cell_id(p: int, tau_low: int, tau_full: int, interrupt: int) -> str:
    return f"p{p}_lo{tau_low}_fu{tau_full}_int{interrupt}"


def design_matrix() -> list[dict]:
    """Generate the 32-row design matrix. Each row is one (placement, policy) cell."""
    rows = []
    for p, tau_low, tau_full, interrupt in itertools.product(
            PLACEMENTS, TAU_LOW_LEVELS, TAU_FULL_LEVELS, INTERRUPT_LEVELS):
        rows.append({
            "id": cell_id(p, tau_low, tau_full, interrupt),
            "placement": p,
            "tau_low": tau_low,
            "tau_full": tau_full,
            "interrupt": interrupt,
            "battery_interrupt_pct": 200 if interrupt == 0 else interrupt,
        })
    return rows


def make_override(row: dict) -> dict:
    """Build the per-cell override JSON consumed by run_one.py."""
    cfg = {
        "pipeline": row["placement"],
        "num_chargers": NUM_CHARGERS,
        "disable_active_charging": False,
        "battery_low_pct": row["tau_low"],
        "battery_charged_pct": row["tau_full"],
        "battery_interrupt_pct": row["battery_interrupt_pct"],
    }
    if row["placement"] == 1:
        cfg["charger_positions"] = list(P1_CANONICAL_LAYOUT)
    elif row["placement"] == 3:
        cfg["charger_positions"] = list(P3_CANONICAL_LAYOUT)
        cfg["selective_picker_chargers"] = True
    return cfg


def run_one_cell(row: dict, force: bool = False) -> dict:
    """Run a single DoE cell. Returns metadata."""
    cell = DOE_DIR / row["id"]
    if (cell / "run_summary.json").exists() and not force:
        print(f"[doe] {row['id']} already complete — skipping")
        return {"id": row["id"], "skipped": True}

    DOE_DIR.mkdir(parents=True, exist_ok=True)
    override = make_override(row)
    override_path = DOE_DIR / f"override_{row['id']}.json"
    override_path.write_text(json.dumps(override))

    print(f"\n[doe] === {row['id']}  P{row['placement']}  "
          f"tau_low={row['tau_low']}  tau_full={row['tau_full']}  "
          f"interrupt={'on@50%' if row['interrupt'] else 'off'} ===")

    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, str(ROOT / "eval" / "run_one.py"),
         "--pipeline", str(row["placement"]),
         "--horizon", str(HORIZON),
         "--heartbeat", str(HEARTBEAT),
         "--out-root", f"eval/runs/doe_phase1/{row['id']}_tmp",
         "--config-override", str(override_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    print(f"[doe] {row['id']} rc={rc} elapsed={elapsed/60:.1f} min")

    # Relocate run_one's snapshot into the cell's final folder.
    tmp = DOE_DIR / f"{row['id']}_tmp" / f"p{row['placement']}"
    if tmp.exists():
        cell.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            f.replace(cell / f.name)
        try:
            tmp.rmdir(); tmp.parent.rmdir()
        except OSError:
            pass

    return {"id": row["id"], "rc": rc, "elapsed_min": round(elapsed/60, 1)}


def print_design_matrix(rows: list[dict]) -> None:
    print(f"\n[doe] Design matrix — {len(rows)} cells")
    print(f"  horizon={HORIZON}  num_chargers={NUM_CHARGERS}")
    print(f"  factors: placement{{P1,P2,P3,P4}} × tau_low{{15,25}} × "
          f"tau_full{{60,90}} × interrupt{{off,on@50}}")
    print(f"  ETA: ~{len(rows) * 47:.0f} min sequential ({len(rows) * 47 / 60:.1f} h)\n")
    print(f"  {'cell_id':<25}  {'P':>2}  {'tlow':>5}  {'tfull':>6}  {'interrupt':>9}")
    print(f"  {'-'*25}  {'-'*2}  {'-'*5}  {'-'*6}  {'-'*9}")
    for r in rows:
        intr = "on@50%" if r["interrupt"] else "off"
        print(f"  {r['id']:<25}  {r['placement']:>2}  "
              f"{r['tau_low']:>4}%  {r['tau_full']:>5}%  {intr:>9}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Print the design matrix without launching runs.")
    p.add_argument("--placements", type=int, nargs="+", default=PLACEMENTS,
                   help="Subset of pipelines to run (default: all 4).")
    p.add_argument("--tau-low", type=int, nargs="+", default=TAU_LOW_LEVELS)
    p.add_argument("--tau-full", type=int, nargs="+", default=TAU_FULL_LEVELS)
    p.add_argument("--interrupt", type=int, nargs="+", default=INTERRUPT_LEVELS)
    p.add_argument("--force", action="store_true",
                   help="Re-run cells even if their run_summary.json exists.")
    args = p.parse_args()

    # Apply filters from CLI.
    rows = [r for r in design_matrix()
            if r["placement"] in args.placements
            and r["tau_low"] in args.tau_low
            and r["tau_full"] in args.tau_full
            and r["interrupt"] in args.interrupt]

    print_design_matrix(rows)

    if args.dry_run:
        print("\n[doe] --dry-run: not launching. Re-run without flag to start.")
        return 0

    DOE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for row in rows:
        results.append(run_one_cell(row, force=args.force))
    total_min = (time.time() - t0) / 60
    print(f"\n[doe] DONE — {len(results)} cells in {total_min:.1f} min")

    summary_path = DOE_DIR / "doe_phase1_summary.json"
    summary_path.write_text(json.dumps(
        {"horizon": HORIZON, "num_chargers": NUM_CHARGERS,
         "factors": {"tau_low": TAU_LOW_LEVELS,
                     "tau_full": TAU_FULL_LEVELS,
                     "interrupt": INTERRUPT_LEVELS,
                     "placements": PLACEMENTS},
         "runs": results, "total_elapsed_min": round(total_min, 1)},
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
