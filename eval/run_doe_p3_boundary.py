"""
eval/run_doe_p3_boundary.py
───────────────────────────
Map the P3 deadlock boundary around (tau_low=20, tau_full=90, int=on@50%).

Phase 2 discovered a single-point deadlock at exactly (20, 90, on@50) for
P3, while neighbouring cells (15, 90, on) and (25, 90, on) are clean.
This script probes the immediate neighbourhood to determine whether the
deadlock is a single-point spike or a thin manifold.

Cells (4 probes, ~47 min each, ~3 h total at |C|=12, T=20{,}000):

  (lo, fu, int)         Purpose
  -------------------   ------------------------------------------
  (19, 90, on@50)       tau_low boundary - left
  (21, 90, on@50)       tau_low boundary - right
  (20, 88, on@50)       tau_full boundary - low
  (20, 92, on@50)       tau_full boundary - high

Outputs land in eval/runs/doe_phase2/<id>/ alongside other Phase 2 cells.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_doe_screening import P3_CANONICAL_LAYOUT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PHASE2_DIR = ROOT / "eval" / "runs" / "doe_phase2"
NUM_CHARGERS = 12
HORIZON = 20000
HEARTBEAT = 4000

PROBES = [
    (19, 90, 50, "tau_low boundary - left"),
    (21, 90, 50, "tau_low boundary - right"),
    (20, 88, 50, "tau_full boundary - low"),
    (20, 92, 50, "tau_full boundary - high"),
]


def make_override(tau_low: int, tau_full: int, interrupt: int) -> dict:
    return {
        "pipeline": 3,
        "num_chargers": NUM_CHARGERS,
        "disable_active_charging": False,
        "battery_low_pct": tau_low,
        "battery_charged_pct": tau_full,
        "battery_interrupt_pct": 200 if interrupt == 0 else interrupt,
        "charger_positions": list(P3_CANONICAL_LAYOUT),
        "selective_picker_chargers": True,
    }


def run_probe(tau_low: int, tau_full: int, interrupt: int, purpose: str) -> dict:
    cell_id = f"p3_lo{tau_low}_fu{tau_full}_int{interrupt}"
    cell_dir = PHASE2_DIR / cell_id
    if (cell_dir / "run_summary.json").exists():
        print(f"[probe] {cell_id} already complete — skipping")
        return {"id": cell_id, "skipped": True}

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    override_path = PHASE2_DIR / f"override_{cell_id}.json"
    override_path.write_text(json.dumps(make_override(tau_low, tau_full, interrupt)))

    print(f"\n[probe] === {cell_id}  ({purpose}) ===")
    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, str(ROOT / "eval" / "run_one.py"),
         "--pipeline", "3",
         "--horizon", str(HORIZON),
         "--heartbeat", str(HEARTBEAT),
         "--out-root", f"eval/runs/doe_phase2/{cell_id}_tmp",
         "--config-override", str(override_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    print(f"[probe] {cell_id} rc={rc} elapsed={elapsed/60:.1f} min")

    tmp = PHASE2_DIR / f"{cell_id}_tmp" / "p3"
    if tmp.exists():
        cell_dir.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            f.replace(cell_dir / f.name)
        try:
            tmp.rmdir(); tmp.parent.rmdir()
        except OSError:
            pass

    return {"id": cell_id, "rc": rc, "elapsed_min": round(elapsed/60, 1)}


def main() -> int:
    print(f"[probe] {len(PROBES)} P3 deadlock-boundary probes "
          f"(~{len(PROBES) * 47} min sequential)")
    t0 = time.time()
    results = [run_probe(*p) for p in PROBES]
    total = (time.time() - t0) / 60
    print(f"\n[probe] DONE — {len(results)} probes in {total:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
