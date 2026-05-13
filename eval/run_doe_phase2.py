"""
eval/run_doe_phase2.py
──────────────────────
Phase 2 — Response Surface Methodology runs (face-centered CCD).

Cell composition (13 cells total, all at |C| = 12, T = 20{,}000):

  P3 CCD completion (interrupt fixed at on@50):
    4 axials  + 1 center  = 5 new cells
    (4 corners of the CCD already exist as Phase 1 cells with int=50)

  P1 confirmation (interrupt fixed at on@50):
    1 center + 2 tau_full axials = 3 new cells
    (verify flat surface in the middle of the design space)

  P2 confirmation (interrupt fixed at on@50):
    1 center + 2 tau_full axials = 3 new cells

  P2 deadlock-boundary probe (interrupt OFF):
    2 cells near (lo=25, fu=60, int=off) — the discovered deadlock zone
    (characterise how wide the deadlock region is in tau_full direction)

Each cell writes to eval/runs/doe_phase2/<id>/ and persists its override
JSON alongside for reproducibility.

Usage:
    python eval/run_doe_phase2.py --dry-run     # show cell list, don't launch
    python eval/run_doe_phase2.py               # launch sequentially
    python eval/run_doe_phase2.py --placements 3   # run only P3 cells
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Reuse the canonical layouts and naming from Phase 1.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_doe_screening import P1_CANONICAL_LAYOUT, P3_CANONICAL_LAYOUT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PHASE2_DIR = ROOT / "eval" / "runs" / "doe_phase2"

NUM_CHARGERS = 12
HORIZON = 20000
HEARTBEAT = 4000


def make_override(placement: int, tau_low: int, tau_full: int,
                  interrupt: int) -> dict:
    cfg = {
        "pipeline": placement,
        "num_chargers": NUM_CHARGERS,
        "disable_active_charging": False,
        "battery_low_pct": tau_low,
        "battery_charged_pct": tau_full,
        "battery_interrupt_pct": 200 if interrupt == 0 else interrupt,
    }
    if placement == 1:
        cfg["charger_positions"] = list(P1_CANONICAL_LAYOUT)
    elif placement == 3:
        cfg["charger_positions"] = list(P3_CANONICAL_LAYOUT)
        cfg["selective_picker_chargers"] = True
    return cfg


def cell_id(placement: int, tau_low: int, tau_full: int, interrupt: int) -> str:
    return f"p{placement}_lo{tau_low}_fu{tau_full}_int{interrupt}"


def design_matrix() -> list[dict]:
    """13-cell Phase 2 design."""
    rows = []

    # --- P3 CCD completion (interrupt = on@50%) ---
    # Axial points: one factor at center, the other at face.
    # Center point.
    p3_new = [
        (15, 75), (25, 75),    # tau_full axials
        (20, 60), (20, 90),    # tau_low axials
        (20, 75),              # center
    ]
    for lo, fu in p3_new:
        rows.append({"placement": 3, "tau_low": lo, "tau_full": fu,
                     "interrupt": 50,
                     "purpose": "P3 CCD axial/center"})

    # --- P1 confirmation (interrupt = on@50%) ---
    p1_new = [(20, 75), (20, 60), (20, 90)]
    for lo, fu in p1_new:
        rows.append({"placement": 1, "tau_low": lo, "tau_full": fu,
                     "interrupt": 50,
                     "purpose": "P1 flatness confirmation"})

    # --- P2 confirmation (interrupt = on@50%) ---
    p2_new = [(20, 75), (20, 60), (20, 90)]
    for lo, fu in p2_new:
        rows.append({"placement": 2, "tau_low": lo, "tau_full": fu,
                     "interrupt": 50,
                     "purpose": "P2 flatness confirmation"})

    # --- P2 deadlock boundary probe (interrupt = OFF) ---
    p2_deadlock = [(25, 55), (25, 65)]
    for lo, fu in p2_deadlock:
        rows.append({"placement": 2, "tau_low": lo, "tau_full": fu,
                     "interrupt": 0,
                     "purpose": "P2 deadlock boundary probe"})

    for r in rows:
        r["id"] = cell_id(r["placement"], r["tau_low"], r["tau_full"],
                          r["interrupt"])
    return rows


def run_one_cell(row: dict, force: bool = False) -> dict:
    cell = PHASE2_DIR / row["id"]
    if (cell / "run_summary.json").exists() and not force:
        print(f"[doe2] {row['id']} already complete — skipping")
        return {"id": row["id"], "skipped": True}

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    override = make_override(row["placement"], row["tau_low"],
                             row["tau_full"], row["interrupt"])
    override_path = PHASE2_DIR / f"override_{row['id']}.json"
    override_path.write_text(json.dumps(override))

    print(f"\n[doe2] === {row['id']}  ({row['purpose']}) ===")

    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, str(ROOT / "eval" / "run_one.py"),
         "--pipeline", str(row["placement"]),
         "--horizon", str(HORIZON),
         "--heartbeat", str(HEARTBEAT),
         "--out-root", f"eval/runs/doe_phase2/{row['id']}_tmp",
         "--config-override", str(override_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    print(f"[doe2] {row['id']} rc={rc} elapsed={elapsed/60:.1f} min")

    tmp = PHASE2_DIR / f"{row['id']}_tmp" / f"p{row['placement']}"
    if tmp.exists():
        cell.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            f.replace(cell / f.name)
        try:
            tmp.rmdir(); tmp.parent.rmdir()
        except OSError:
            pass

    return {"id": row["id"], "rc": rc, "elapsed_min": round(elapsed/60, 1)}


def print_design(rows: list[dict]) -> None:
    print(f"\n[doe2] Phase 2 design — {len(rows)} cells")
    print(f"  ETA: ~{len(rows) * 47} min ({len(rows) * 47 / 60:.1f} h) sequential\n")
    print(f"  {'cell_id':<22}  {'P':>2}  {'lo':>3}  {'fu':>3}  {'int':>3}  purpose")
    print(f"  {'-'*22}  {'-'*2}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*40}")
    for r in rows:
        intr = "on" if r["interrupt"] else "off"
        print(f"  {r['id']:<22}  {r['placement']:>2}  {r['tau_low']:>3}  "
              f"{r['tau_full']:>3}  {intr:>3}  {r['purpose']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--placements", type=int, nargs="+",
                   default=[1, 2, 3], help="Filter to subset of pipelines.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    rows = [r for r in design_matrix() if r["placement"] in args.placements]
    print_design(rows)
    if args.dry_run:
        print("\n[doe2] --dry-run: not launching.")
        return 0

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = [run_one_cell(r, force=args.force) for r in rows]
    total_min = (time.time() - t0) / 60
    print(f"\n[doe2] DONE — {len(results)} cells in {total_min:.1f} min")

    summary_path = PHASE2_DIR / "doe_phase2_summary.json"
    summary_path.write_text(json.dumps(
        {"horizon": HORIZON, "num_chargers": NUM_CHARGERS,
         "runs": results, "total_elapsed_min": round(total_min, 1)},
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
