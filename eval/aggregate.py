"""
eval/aggregate.py
─────────────────
Load eval/results/run_p{1..4}.csv and print a comparison table across the
four charging-placement pipelines.

Single-replication mode (current):
    Just stack the per-pipeline rows and print side-by-side.  No confidence
    intervals or statistical tests — the workload is deterministic, so
    these numbers are point estimates under one workload.

Future multi-seed mode (not yet used):
    If multiple rows per pipeline exist, the script would compute mean and
    95 % CI and run paired Wilcoxon tests between pipeline pairs.

Usage
─────
    python eval/aggregate.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"

PIPELINE_NAMES = {
    1: "SetCover",
    2: "AffinityProp",
    3: "PickStation",
    4: "Perimeter",
}

# Extra variants displayed alongside the core pipelines.
VARIANT_NAMES = {
    "3_opp": "P3-OppCharge",
}

# Metrics to print in the comparison (column order)
METRICS = [
    ("orders_completed",      "Orders done",        "higher better"),
    ("throughput_per_sim_hour", "Orders/hr (sim)",  "higher better"),
    ("cycle_time_mean_s",     "Mean cycle (s)",     "lower better"),
    ("cycle_time_median_s",   "Median cycle (s)",   "lower better"),
    ("cycle_time_p95_s",      "P95 cycle (s)",      "lower better"),
    ("total_energy_j",        "Total energy (J)",   "lower better"),
    ("energy_per_order_j",    "Energy/order (J)",   "lower better"),
    ("stop_and_go",           "Stop-and-go",        "lower better"),
    ("total_turning",         "Turns",              "lower better"),
    ("pending_jobs_at_end",   "Unfinished jobs",    "lower better"),
    ("battery_pct_mean",      "Batt % mean",        "higher better"),
    ("battery_pct_min",       "Batt % min",         "higher better"),
    ("robots_charging_now",   "Robots charging @T", "neutral"),
    ("robots_going_to_charge", "Robots en-route to charger", "neutral"),
    ("num_chargers",          "# chargers placed",  "neutral"),
    ("wallclock_s",           "Wall-clock (s)",     "neutral"),
]


def load_results() -> dict:
    """Load all run_p{tag}.csv rows keyed by integer or variant-string tag."""
    out: dict = {}
    for P in (1, 2, 3, 4):
        path = RESULTS_DIR / f"run_p{P}.csv"
        if path.exists():
            with open(path, newline="") as f:
                out[P] = next(csv.DictReader(f))
    for variant in VARIANT_NAMES:
        path = RESULTS_DIR / f"run_p{variant}.csv"
        if path.exists():
            with open(path, newline="") as f:
                out[variant] = next(csv.DictReader(f))
    return out


def fmt(v: str | None) -> str:
    if v is None or v == "":
        return "—"
    try:
        f = float(v)
    except ValueError:
        return v
    if f == int(f):
        return f"{int(f)}"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.2f}"


def main() -> int:
    res = load_results()
    if not res:
        print("No results found. Run the pipelines first, then run extract_kpis.py.")
        return 1

    # Integer pipelines first (1..4), then variants (3_opp, …).
    int_cols = sorted(k for k in res.keys() if isinstance(k, int))
    var_cols = sorted(k for k in res.keys() if isinstance(k, str))
    cols = int_cols + var_cols

    def col_label(k):
        if isinstance(k, int):
            return f"P{k} ({PIPELINE_NAMES[k]})"
        return f"P{k} ({VARIANT_NAMES[k]})"

    header = ["Metric"] + [col_label(k) for k in cols] + ["Direction"]
    widths = [28] + [16] * len(cols) + [14]

    def row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    print()
    print(row(header))
    print(row(["-" * (w - 1) for w in widths]))
    for key, label, direction in METRICS:
        vals = [fmt(res[P].get(key)) for P in cols]
        print(row([label] + vals + [direction]))
    print()

    # Winners per metric (where direction is meaningful and all pipelines reported)
    print("Best per metric (point estimates, single workload):")
    for key, label, direction in METRICS:
        if direction not in ("higher better", "lower better"):
            continue
        pairs = []
        for P in cols:
            v = res[P].get(key)
            try:
                pairs.append((P, float(v)))
            except (TypeError, ValueError):
                pass
        if not pairs:
            continue
        best = max(pairs, key=lambda x: x[1]) if direction == "higher better" else min(pairs, key=lambda x: x[1])
        best_key = best[0]
        best_name = (PIPELINE_NAMES[best_key] if isinstance(best_key, int)
                     else VARIANT_NAMES[best_key])
        print(f"  {label:28s} -> P{best_key} ({best_name}) = {best[1]:.2f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
