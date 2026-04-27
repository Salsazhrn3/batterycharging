"""
eval/extend_orders.py
─────────────────────
Tile ``generated_order.csv`` so the order arrival window extends from the
original ~30,900 sim-sec out to ``--horizon`` (default 100,000 sec). This
preserves the original Poisson arrival pattern and per-order item mix —
each tile is a copy with arrival times and order_ids shifted forward.

Why
───
The 100,000-sec evaluation horizon (see project memory) was being driven
by drain physics, but the input CSV only carried demand for the first
30,900 sec — meaning ~69% of the horizon was a draw-down phase with no
new arrivals. Extending the input gives the FMS continuous demand for
the whole horizon so throughput stalls reflect fleet limits, not a dry
order queue.

Behavior
────────
* On first run, copies the existing ``generated_order.csv`` to
  ``generated_order.original.csv`` as a one-time backup.
* On every run, regenerates ``generated_order.csv`` by tiling the
  backup up to ``--horizon`` seconds.

Usage
─────
    python eval/extend_orders.py
    python eval/extend_orders.py --horizon 100000
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER_CSV = ROOT / "generated_order.csv"
BACKUP_CSV = ROOT / "generated_order.original.csv"


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=100_000,
                    help="Target sim-sec for last order arrival.")
    args = ap.parse_args()
    horizon = int(args.horizon)

    # One-time backup of the original on the very first run.
    if not BACKUP_CSV.exists():
        shutil.copy2(ORDER_CSV, BACKUP_CSV)
        print(f"[ok] backed up original to {BACKUP_CSV.name}")

    base_rows = load_rows(BACKUP_CSV)
    if not base_rows:
        print("[err] backup CSV is empty")
        return 1

    fieldnames = list(base_rows[0].keys())
    arrivals = [int(r["order_arrival"]) for r in base_rows]
    oids = [int(r["order_id"]) for r in base_rows]
    base_arrival_max = max(arrivals)
    base_id_min, base_id_max = min(oids), max(oids)
    id_span = base_id_max - base_id_min + 1
    # Use a one-second gap so consecutive tiles don't pile two orders on
    # the same sim-sec timestamp where the original ended.
    arrival_period = base_arrival_max + 1

    out: list[dict] = []
    seq = 1
    tile_idx = 0
    while True:
        arrival_offset = tile_idx * arrival_period
        id_offset = tile_idx * id_span
        if arrival_offset > horizon:
            break
        for r in base_rows:
            new_arrival = int(r["order_arrival"]) + arrival_offset
            if new_arrival > horizon:
                continue
            new_row = dict(r)
            new_row["sequence_id"] = seq
            new_row["order_id"] = int(r["order_id"]) + id_offset
            new_row["order_arrival"] = new_arrival
            out.append(new_row)
            seq += 1
        tile_idx += 1

    write_rows(ORDER_CSV, out, fieldnames)

    final_arrivals = [int(r["order_arrival"]) for r in out]
    final_ids = {int(r["order_id"]) for r in out}
    print(f"[ok] wrote {ORDER_CSV.name}: {len(out)} rows, "
          f"{len(final_ids)} unique orders, "
          f"arrival range 0..{max(final_arrivals)} sec "
          f"(tiles used: {tile_idx})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
