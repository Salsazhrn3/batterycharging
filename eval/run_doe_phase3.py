"""
eval/run_doe_phase3.py
──────────────────────
Phase 3 — Taguchi-style robustness analysis.

For each of two P3 candidate (placement, policy) configurations from
Phase 2, run a 4-factor 3-level Taguchi L9 orthogonal array over the
following noise factors:

  A. demand_load_factor   {0.7, 1.0, 1.3}    -30% / nominal / +30% load
  B. num_robots           {18,  20,  22}     -2 / nominal / +2 attrition
  C. charger_fault_count  {0,   1,   2}      number of disabled chargers
  D. initial_battery_frac {0.5, 0.7, 1.0}    start-of-shift battery level

Two P3 candidates:
  Robust : tau_low=18, tau_full=60, interrupt=on@50%   (Phase 2 constrained RSM optimum)
  Peak   : tau_low=25, tau_full=90, interrupt=off       (Phase 1 best corner)

Total: 2 candidates × 9 noise combinations = 18 cells at 20k horizon ≈ 14 h
sequential. Outputs land in eval/runs/doe_phase3/<id>/.

Standard L9 array (Box, Hunter & Hunter, p. 327; Montgomery §13.4):
    Run  A  B  C  D
     1   1  1  1  1
     2   1  2  2  2
     3   1  3  3  3
     4   2  1  2  3
     5   2  2  3  1
     6   2  3  1  2
     7   3  1  3  2
     8   3  2  1  3
     9   3  3  2  1

Usage:
    python eval/run_doe_phase3.py --dry-run
    python eval/run_doe_phase3.py
    python eval/run_doe_phase3.py --candidate robust    # single candidate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_doe_screening import P3_CANONICAL_LAYOUT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PHASE3_DIR = ROOT / "eval" / "runs" / "doe_phase3"
NUM_CHARGERS = 12
HORIZON = 20000
HEARTBEAT = 4000

# Noise factor levels (1, 2, 3 → physical values)
DEMAND_LEVELS = [0.7, 1.0, 1.3]
ROBOT_LEVELS = [18, 20, 22]
FAULT_LEVELS = [0, 1, 2]
INIT_BATT_LEVELS = [0.5, 0.7, 1.0]

# Standard Taguchi L9(3^4) orthogonal array (1-indexed levels)
L9 = [
    (1, 1, 1, 1),
    (1, 2, 2, 2),
    (1, 3, 3, 3),
    (2, 1, 2, 3),
    (2, 2, 3, 1),
    (2, 3, 1, 2),
    (3, 1, 3, 2),
    (3, 2, 1, 3),
    (3, 3, 2, 1),
]

CANDIDATES = {
    "robust": {"tau_low": 18, "tau_full": 60, "interrupt": 50},
    "peak":   {"tau_low": 25, "tau_full": 90, "interrupt":  0},
}


def make_override(candidate_name: str, run_idx: int, levels: tuple) -> dict:
    """Build the override JSON for one (candidate, L9 row) cell."""
    cand = CANDIDATES[candidate_name]
    a, b, c, d = levels
    demand = DEMAND_LEVELS[a - 1]
    robots = ROBOT_LEVELS[b - 1]
    faults = FAULT_LEVELS[c - 1]
    init_batt = INIT_BATT_LEVELS[d - 1]

    layout = list(P3_CANONICAL_LAYOUT)
    if faults > 0:
        layout = layout[:-faults]  # remove last `faults` positions

    return {
        "pipeline": 3,
        "num_chargers": len(layout),
        "disable_active_charging": False,
        "selective_picker_chargers": True,
        "charger_positions": layout,
        # Policy
        "battery_low_pct": cand["tau_low"],
        "battery_charged_pct": cand["tau_full"],
        "battery_interrupt_pct": 200 if cand["interrupt"] == 0 else cand["interrupt"],
        # Noise factors
        "demand_load_factor": demand,
        "num_robots": robots,
        "initial_battery_frac": init_batt,
        # Bookkeeping
        "_phase3_run": run_idx,
        "_candidate": candidate_name,
    }


def cell_id(candidate_name: str, run_idx: int) -> str:
    return f"{candidate_name}_run{run_idx:02d}"


def run_one_cell(candidate_name: str, run_idx: int, levels: tuple,
                 force: bool = False) -> dict:
    cid = cell_id(candidate_name, run_idx)
    cell = PHASE3_DIR / cid
    if (cell / "run_summary.json").exists() and not force:
        print(f"[doe3] {cid} already complete — skipping")
        return {"id": cid, "skipped": True}

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    override = make_override(candidate_name, run_idx, levels)
    override_path = PHASE3_DIR / f"override_{cid}.json"
    override_path.write_text(json.dumps(override))

    a, b, c, d = levels
    demand = DEMAND_LEVELS[a - 1]
    robots = ROBOT_LEVELS[b - 1]
    faults = FAULT_LEVELS[c - 1]
    init_batt = INIT_BATT_LEVELS[d - 1]
    print(f"\n[doe3] === {cid}  cand={candidate_name}  "
          f"demand={demand}  robots={robots}  faults={faults}  init={init_batt} ===")

    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, str(ROOT / "eval" / "run_one.py"),
         "--pipeline", "3",
         "--horizon", str(HORIZON),
         "--heartbeat", str(HEARTBEAT),
         "--out-root", f"eval/runs/doe_phase3/{cid}_tmp",
         "--config-override", str(override_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    print(f"[doe3] {cid} rc={rc} elapsed={elapsed/60:.1f} min")

    tmp = PHASE3_DIR / f"{cid}_tmp" / "p3"
    if tmp.exists():
        cell.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            f.replace(cell / f.name)
        try:
            tmp.rmdir(); tmp.parent.rmdir()
        except OSError:
            pass

    return {"id": cid, "rc": rc, "elapsed_min": round(elapsed/60, 1)}


def print_design(candidates: list[str]) -> None:
    rows_total = len(candidates) * 9
    print(f"\n[doe3] Phase 3 design — {rows_total} cells "
          f"({len(candidates)} candidate × L9)")
    print(f"  ETA: ~{rows_total * 47} min ({rows_total * 47 / 60:.1f} h)\n")
    for cn in candidates:
        cand = CANDIDATES[cn]
        intr_str = "off" if cand["interrupt"] == 0 else f"on@{cand['interrupt']}"
        print(f"  Candidate '{cn}': tau_low={cand['tau_low']}  "
              f"tau_full={cand['tau_full']}  interrupt={intr_str}")
    print(f"\n  L9 noise array (factors A=demand B=robots C=faults D=init_batt):")
    print(f"    {'run':>3}  {'demand':>7}  {'robots':>6}  {'faults':>6}  {'init':>5}")
    for i, lv in enumerate(L9, start=1):
        a, b, c, d = lv
        print(f"    {i:>3}  {DEMAND_LEVELS[a-1]:>7.1f}  "
              f"{ROBOT_LEVELS[b-1]:>6}  {FAULT_LEVELS[c-1]:>6}  "
              f"{INIT_BATT_LEVELS[d-1]:>5.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--candidate", choices=list(CANDIDATES.keys()) + ["all"],
                   default="all")
    p.add_argument("--smoke", action="store_true",
                   help="Run one cell only (cand=robust, run 5 — center levels)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    candidates = list(CANDIDATES.keys()) if args.candidate == "all" else [args.candidate]
    print_design(candidates)

    if args.dry_run:
        print("\n[doe3] --dry-run: not launching.")
        return 0

    if args.smoke:
        # Row 9: demand=1.3, robots=22, faults=1, init=0.5 — all 4 hooks at
        # non-default values, ideal for verifying the override pipeline.
        print("\n[doe3] SMOKE MODE: running 1 cell (robust candidate, L9 row 9)")
        run_one_cell("robust", 9, L9[8], force=args.force)
        return 0

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for cn in candidates:
        for i, lv in enumerate(L9, start=1):
            results.append(run_one_cell(cn, i, lv, force=args.force))
    total_min = (time.time() - t0) / 60
    print(f"\n[doe3] DONE — {len(results)} cells in {total_min:.1f} min")

    summary_path = PHASE3_DIR / "doe_phase3_summary.json"
    summary_path.write_text(json.dumps(
        {"horizon": HORIZON, "num_chargers_nominal": NUM_CHARGERS,
         "candidates": CANDIDATES, "L9": L9,
         "factor_levels": {
             "demand": DEMAND_LEVELS, "robots": ROBOT_LEVELS,
             "faults": FAULT_LEVELS, "init_batt": INIT_BATT_LEVELS},
         "runs": results, "total_elapsed_min": round(total_min, 1)},
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
