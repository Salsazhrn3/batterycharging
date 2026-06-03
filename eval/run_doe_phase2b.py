"""
eval/run_doe_phase2b.py
───────────────────────
Phase 2b — Joint RSM over (placement, |C|, tau_low, tau_full)
at 100,000-tick horizon.

Motivation
──────────
Phase 2 ran at 20k ticks with full initial battery. Across all 32 screening
cells the minimum per-robot battery at end-of-horizon was 55-56% — above every
tau_low level tested. The threshold trigger therefore never fired, and the
Phase-2 RSM coefficients on (tau_low, tau_full) describe a region where the
policy is inert. To find which (placement, budget, policy) combination
actually performs best when the policy is *active*, we re-do Phase 2 at a
horizon long enough for the threshold to bite (100,000 ticks).

Design — face-centred CCD per placement
───────────────────────────────────────
Continuous factors (3-factor face-centred CCD, alpha = 1):
  num_chargers : {6, 18, 30}     (|C|/N = {0.30, 0.90, 1.50}, N=20)
  tau_low      : {15, 20, 25}    %
  tau_full     : {60, 75, 90}    %

Per placement: 8 corners + 6 axial + 1 centre = 15 cells.
Categorical block: placement ∈ {P1, P2, P3, P4}.
Total: 4 × 15 = 60 cells at horizon=100,000.

ETA: ~5h/cell serial = ~300h (~12.5d), ~75h on 4-way parallel (~3.1d).

Stage A — the 60-cell CCD itself
Stage B — confirmation runs at fitted optima (driven by --custom-cells JSON)

Fixed across all stages
───────────────────────
  horizon                  = 100,000
  fleet_size               = 20
  initial_battery_frac     = 1.0
  interrupt                = on@50  (Phase 1: binary on/off, not continuous)
  disable_active_charging  = False  (active dispatch enabled everywhere)
  P3 charger_positions     explicit tier list from run_p3_charger_sweep.py
  P1/P2/P4 layout          algorithmic (set-cover / affinity-prop / picker)

Usage
─────
    # Print the 60-cell design matrix only (no compute)
    python eval/run_doe_phase2b.py --stage A --dry-run

    # Launch all 60 cells, 4 concurrent workers
    python eval/run_doe_phase2b.py --stage A --parallel 4

    # Retry / restrict to specific cells
    python eval/run_doe_phase2b.py --stage A --only p2_n18_lo20_fu75_int50

    # Confirmation runs (after fitting the response surface)
    python eval/run_doe_phase2b.py --stage B --custom-cells confirm.json
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
PHASE2B_DIR = ROOT / "eval" / "runs" / "doe_phase2b"
WORKTREE_PARENT = ROOT.parent / f"{ROOT.name}.worktrees"

# ── Constants fixed across all Phase 2b stages ─────────────────────────────
HORIZON = 100_000
HEARTBEAT = 20_000
DEFAULT_INTERRUPT = 50
INITIAL_BATTERY_FRAC = 1.0
DISABLE_ACTIVE_CHARGING = False

# ── CCD factor levels (face-centred, alpha = 1) ────────────────────────────
PLACEMENTS = [1, 2, 3, 4]
NUM_CHARGERS_LEVELS = [6, 18, 30]
TAU_LOW_LEVELS = [15, 20, 25]
TAU_FULL_LEVELS = [60, 75, 90]

# ── P1 set-cover canonical 40-cell layout ──────────────────────────────────
# Set-cover ignores num_chargers and always returns these 40 cells (greedy
# order). To honour |C| ∈ {6, 18, ...} we truncate to the first num_chargers.
# Truncation is principled because set-cover's greedy ordering places the
# highest-coverage cells first.
P1_CANONICAL_40 = [
    [10, 10], [10, 31], [22, 19], [23, 38], [1, 20], [7, 38], [23, 10], [26, 26],
    [4, 10], [16, 38], [2, 38], [28, 10], [16, 10], [28, 38], [7, 24], [11, 38],
    [19, 38], [1, 10], [20, 10], [4, 38], [5, 10], [11, 10], [25, 38], [1, 38],
    [8, 10], [8, 38], [14, 10], [14, 38], [17, 10], [20, 38], [26, 10], [29, 10],
    [29, 38], [2, 10], [5, 38], [10, 38], [17, 38], [22, 10], [22, 38], [26, 38],
]


def p1_layout(num: int) -> list[list[int]]:
    if num > len(P1_CANONICAL_40):
        raise ValueError(f"P1 num={num} exceeds {len(P1_CANONICAL_40)} cells")
    return [list(c) for c in P1_CANONICAL_40[:num]]


# ── P3 picker-tier scheme (mirrors run_p3_charger_sweep.py) ────────────────
PICKER_ROWS = [1, 7, 13, 19, 25]
P3_TIERS = [
    [[r,     2] for r in PICKER_ROWS],   # T1 processing
    [[r + 2, 4] for r in PICKER_ROWS],   # T2 queue-back
    [[r + 2, 3] for r in PICKER_ROWS],   # T3 mid-queue
    [[r + 1, 2] for r in PICKER_ROWS],   # T4 middle-gate
    [[r + 2, 2] for r in PICKER_ROWS],   # T5 corner
    [[r,     4] for r in PICKER_ROWS],   # T6 entry-deep
    [[r,     3] for r in PICKER_ROWS],   # T7 entry-mid
    [[r + 1, 3] for r in PICKER_ROWS],   # T8 gate-mid
    [[r + 1, 4] for r in PICKER_ROWS],   # T9 gate-deep
    [[r + 3, 2] for r in PICKER_ROWS],   # T10 corner+1
]


def p3_layout(num: int) -> list[list[int]]:
    cells: list[list[int]] = []
    for tier in P3_TIERS:
        cells.extend(tier)
    if num > len(cells):
        raise ValueError(f"P3 num={num} exceeds {len(cells)} tier cells")
    return cells[:num]


# ── Design generation ──────────────────────────────────────────────────────

def ccd_3factor_face_centred(
    a_levels: list[int], b_levels: list[int], c_levels: list[int],
) -> list[tuple[int, int, int]]:
    """Face-centred CCD over 3 continuous factors. Returns 15 points:
    8 corners + 6 axial + 1 centre. a/b/c_levels each = [low, mid, high]."""
    a_lo, a_mid, a_hi = a_levels
    b_lo, b_mid, b_hi = b_levels
    c_lo, c_mid, c_hi = c_levels
    corners = list(itertools.product(
        [a_lo, a_hi], [b_lo, b_hi], [c_lo, c_hi]))               # 8
    axial = [
        (a_lo, b_mid, c_mid), (a_hi, b_mid, c_mid),
        (a_mid, b_lo, c_mid), (a_mid, b_hi, c_mid),
        (a_mid, b_mid, c_lo), (a_mid, b_mid, c_hi),
    ]                                                            # 6
    centre = [(a_mid, b_mid, c_mid)]                             # 1
    return corners + axial + centre


def cell_id(placement: int, num_chargers: int,
            tau_low: int, tau_full: int, interrupt: int) -> str:
    return (f"p{placement}_n{num_chargers}"
            f"_lo{tau_low}_fu{tau_full}_int{interrupt}")


def make_override(placement: int, num_chargers: int,
                  tau_low: int, tau_full: int, interrupt: int) -> dict:
    cfg = {
        "pipeline": placement,
        "num_chargers": num_chargers,
        "disable_active_charging": DISABLE_ACTIVE_CHARGING,
        "battery_low_pct": tau_low,
        "battery_charged_pct": tau_full,
        "battery_interrupt_pct": 200 if interrupt == 0 else interrupt,
        "initial_battery_frac": INITIAL_BATTERY_FRAC,
    }
    if placement == 1:
        cfg["charger_positions"] = p1_layout(num_chargers)
    elif placement == 3:
        cfg["charger_positions"] = p3_layout(num_chargers)
        cfg["selective_picker_chargers"] = True
    return cfg


def design_stage_a() -> list[dict]:
    """4 placements × 15-point face-centred CCD = 60 cells."""
    ccd_points = ccd_3factor_face_centred(
        NUM_CHARGERS_LEVELS, TAU_LOW_LEVELS, TAU_FULL_LEVELS)
    rows = []
    for placement in PLACEMENTS:
        for n, tau_low, tau_full in ccd_points:
            rows.append({
                "id": cell_id(placement, n, tau_low, tau_full,
                              DEFAULT_INTERRUPT),
                "placement": placement, "num_chargers": n,
                "tau_low": tau_low, "tau_full": tau_full,
                "interrupt": DEFAULT_INTERRUPT,
            })
    return rows


def design_from_json(path: Path) -> list[dict]:
    """Stage B confirmation — explicit JSON list of cell specs."""
    spec = json.loads(path.read_text())
    rows = []
    for c in spec:
        intr = c.get("interrupt", DEFAULT_INTERRUPT)
        rows.append({
            "id": cell_id(c["placement"], c["num_chargers"],
                          c["tau_low"], c["tau_full"], intr),
            "placement": c["placement"], "num_chargers": c["num_chargers"],
            "tau_low": c["tau_low"], "tau_full": c["tau_full"],
            "interrupt": intr,
        })
    return rows


# ── Runner ─────────────────────────────────────────────────────────────────

def setup_worktrees(n_workers: int) -> list[Path]:
    """Create n_workers git worktrees so parallel run_one.py subprocesses
    don't trample each other's netlogo.state / output/order-finished.csv.
    Each worktree is a private checkout of HEAD; the central .git is shared,
    so disk overhead per worktree = tracked working-tree size only."""
    WORKTREE_PARENT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n_workers):
        wt = WORKTREE_PARENT / f"w{i}"
        if not (wt / "netlogo.py").exists():
            print(f"[p2b] creating worktree {wt}")
            subprocess.check_call(
                ["git", "worktree", "add", "--detach", str(wt), "HEAD"],
                cwd=str(ROOT),
            )
        (wt / "output").mkdir(exist_ok=True)
        paths.append(wt)
    return paths


def run_one_cell(row: dict, force: bool = False,
                 workdir: Path | None = None) -> dict:
    """Run one DoE cell. If workdir is given, the run_one.py subprocess
    executes inside that git worktree (isolated state); otherwise it runs
    in the main repo (serial-only mode)."""
    cell = PHASE2B_DIR / row["id"]
    if (cell / "run_summary.json").exists() and not force:
        print(f"[p2b] {row['id']} already complete — skip")
        return {"id": row["id"], "skipped": True}

    PHASE2B_DIR.mkdir(parents=True, exist_ok=True)
    override = make_override(row["placement"], row["num_chargers"],
                             row["tau_low"], row["tau_full"], row["interrupt"])
    override_path = PHASE2B_DIR / f"override_{row['id']}.json"
    override_path.write_text(json.dumps(override))

    cwd = workdir if workdir is not None else ROOT
    # Absolute --out-root keeps every cell's snapshot under the central
    # PHASE2B_DIR even when the subprocess cwd is a worktree.
    out_root_abs = PHASE2B_DIR / f"{row['id']}_tmp"

    print(f"\n[p2b] === {row['id']}  P{row['placement']}  "
          f"|C|={row['num_chargers']}  tau_lo={row['tau_low']}  "
          f"tau_fu={row['tau_full']}  int={row['interrupt']}  "
          f"[wt={cwd.name}] ===")

    t0 = time.time()
    rc = subprocess.call(
        [sys.executable, str(cwd / "eval" / "run_one.py"),
         "--pipeline", str(row["placement"]),
         "--horizon", str(HORIZON),
         "--heartbeat", str(HEARTBEAT),
         "--out-root", str(out_root_abs),
         "--config-override", str(override_path)],
        cwd=str(cwd),
    )
    elapsed = time.time() - t0
    print(f"[p2b] {row['id']} rc={rc}  elapsed={elapsed/60:.1f} min")

    tmp = out_root_abs / f"p{row['placement']}"
    if tmp.exists():
        cell.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            f.replace(cell / f.name)
        try:
            tmp.rmdir(); tmp.parent.rmdir()
        except OSError:
            pass

    return {"id": row["id"], "rc": rc, "elapsed_min": round(elapsed/60, 1)}


# ── Pretty printing ────────────────────────────────────────────────────────

def print_design(rows: list[dict], stage: str) -> None:
    eta_h = len(rows) * 5.0  # ~5h per 100k cell on this machine
    print(f"\n[p2b] Stage {stage} — {len(rows)} cells "
          f"at horizon={HORIZON:,}")
    print(f"      ETA serial: ~{eta_h:.0f}h "
          f"({eta_h/24:.1f}d); ~{eta_h/4:.0f}h on 4-way parallel "
          f"(~{eta_h/4/24:.1f}d)")
    print(f"  {'cell_id':<35} {'P':>2} {'|C|':>4} "
          f"{'lo':>4} {'fu':>4} {'int':>4}")
    print(f"  {'-'*35} {'-'*2} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
    for r in rows:
        print(f"  {r['id']:<35} {r['placement']:>2} "
              f"{r['num_chargers']:>4} {r['tau_low']:>3}% "
              f"{r['tau_full']:>3}% {r['interrupt']:>3}%")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["A", "B"], required=True,
                   help="A = 60-cell CCD, B = confirmation from JSON")
    p.add_argument("--custom-cells", type=str, default=None,
                   help="Stage B: path to JSON file listing confirmation cells")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", type=str, nargs="+", default=None,
                   help="Restrict to specific cell_ids (retry helper)")
    p.add_argument("--placements", type=int, nargs="+", default=None,
                   help="Stage A: restrict to subset of placements")
    p.add_argument("--parallel", type=int, default=1)
    args = p.parse_args()

    if args.stage == "A":
        rows = design_stage_a()
        if args.placements:
            rows = [r for r in rows if r["placement"] in set(args.placements)]
    else:
        if not args.custom_cells:
            print("[err] Stage B requires --custom-cells")
            return 2
        rows = design_from_json(Path(args.custom_cells))

    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]

    print_design(rows, args.stage)

    if args.dry_run:
        print("\n[p2b] --dry-run: not launching.")
        return 0

    PHASE2B_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.parallel > 1:
        worktrees = setup_worktrees(args.parallel)
        results = run_parallel(rows, worktrees, args.force)
    else:
        results = [run_one_cell(r, force=args.force) for r in rows]
    total_h = (time.time() - t0) / 3600
    print(f"\n[p2b] DONE — {len(results)} cells in {total_h:.2f}h")

    out = PHASE2B_DIR / f"phase2b_stage{args.stage}_summary.json"
    out.write_text(json.dumps(
        {"stage": args.stage, "horizon": HORIZON,
         "factor_levels": {
             "num_chargers": NUM_CHARGERS_LEVELS,
             "tau_low": TAU_LOW_LEVELS,
             "tau_full": TAU_FULL_LEVELS,
             "placements": PLACEMENTS,
         },
         "fixed": {"interrupt": DEFAULT_INTERRUPT,
                   "initial_battery_frac": INITIAL_BATTERY_FRAC,
                   "disable_active_charging": DISABLE_ACTIVE_CHARGING},
         "runs": results, "total_elapsed_h": round(total_h, 2)},
        indent=2, default=str))
    print(f"[p2b] summary -> {out}")
    return 0


def run_parallel(rows: list[dict], worktrees: list[Path],
                 force: bool) -> list[dict]:
    """Launch up to len(worktrees) run_one.py subprocesses concurrently,
    each pinned to its own git-worktree cwd so transient sim state stays
    isolated."""
    import queue
    import threading

    q: queue.Queue = queue.Queue()
    for r in rows:
        q.put(r)
    results: list[dict] = []
    lock = threading.Lock()

    def worker(wt: Path):
        while True:
            try:
                row = q.get_nowait()
            except queue.Empty:
                return
            res = run_one_cell(row, force=force, workdir=wt)
            with lock:
                results.append(res)
            q.task_done()

    threads = [threading.Thread(target=worker, args=(wt,)) for wt in worktrees]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


if __name__ == "__main__":
    raise SystemExit(main())
