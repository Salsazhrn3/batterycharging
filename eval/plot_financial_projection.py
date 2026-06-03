"""
eval/plot_financial_projection.py
─────────────────────────────────
Generate the financial-projection figures for the paper's
§Engineering Management → Financial Projection subsection:

  - financial_cashflow_bars.png : Annual net cash flow (bars) +
    cumulative cash balance (line), Year 0..5, retrofit case.
  - financial_sensitivity_pbp.png : Payback Period and 5-yr NPV
    as functions of order margin (NT$/pick).

Inputs are baked-in: see the constants block. Outputs go to
eval/results/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "eval" / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Base inputs (all in NT$ thousands, Year-1 equivalents) ─────────────
CAPEX_RETROFIT = 775                    # 2 new chargers + install + 10 relocations
REVENUE_GAIN_FULL = 3_664.8             # 244,320 orders/yr × NT$15 (full-scale)
ENERGY_SAVING_FULL = 9.2                # 2,630 kWh × NT$3.50
BATTERY_SAVING_FULL = 0.0               # excluded: simulator does not model
                                        # battery degradation; not a policy
                                        # constraint (see Limitations).
MAINTENANCE_BASE = -28.2                # 7 % of NT$403k (new chargers only)
DEPRECIATION = -155.0                   # NT$775k / 5 yr (straight-line)
TAX_RATE      = 0.20                    # Taiwan corporate income tax
DISCOUNT_RATE = 0.10                    # WACC, industrial benchmark
LIFE_YEARS    = 5
NTD_PER_USD   = 31.0                    # 2024 avg

# ── Year-on-year dynamics ──────────────────────────────────────────────
# Throughput ramp: ops staff learning curve. Year 1 = 80 %, Year 2 = 95 %,
# Year 3+ = 100 %.
RAMP = {1: 0.80, 2: 0.95, 3: 1.00, 4: 1.00, 5: 1.00}
# Annual escalators (compounded).
INFL_REVENUE = 0.02   # Taiwan CPI pass-through to client pricing
INFL_ENERGY  = 0.03   # Taipower historical 3 %/yr trend (2018-2024 avg)
INFL_MAINT   = 0.05   # Equipment ageing premium

# ── Build year-on-year cash flow series ────────────────────────────────
years = np.arange(0, LIFE_YEARS + 1)
revenue   = np.zeros(LIFE_YEARS + 1)
energy_s  = np.zeros(LIFE_YEARS + 1)
battery_s = np.zeros(LIFE_YEARS + 1)
maint     = np.zeros(LIFE_YEARS + 1)
depr      = np.zeros(LIFE_YEARS + 1)
pretax    = np.zeros(LIFE_YEARS + 1)
tax_arr   = np.zeros(LIFE_YEARS + 1)
atcf_arr  = np.zeros(LIFE_YEARS + 1)
net_cf    = np.zeros(LIFE_YEARS + 1)

net_cf[0] = -CAPEX_RETROFIT
for y in range(1, LIFE_YEARS + 1):
    ramp = RAMP[y]
    rev_y = REVENUE_GAIN_FULL * ramp * (1 + INFL_REVENUE) ** (y - 1)
    eng_y = ENERGY_SAVING_FULL * ramp * (1 + INFL_ENERGY) ** (y - 1)
    bat_y = BATTERY_SAVING_FULL * ramp
    mnt_y = MAINTENANCE_BASE * (1 + INFL_MAINT) ** (y - 1)
    pretax_y = rev_y + eng_y + bat_y + mnt_y
    taxable_y = pretax_y + DEPRECIATION
    tax_y = -TAX_RATE * taxable_y
    atcf_y = pretax_y + tax_y       # depreciation is non-cash

    revenue[y]   = rev_y
    energy_s[y]  = eng_y
    battery_s[y] = bat_y
    maint[y]     = mnt_y
    depr[y]      = DEPRECIATION
    pretax[y]    = pretax_y
    tax_arr[y]   = tax_y
    atcf_arr[y]  = atcf_y
    net_cf[y]    = atcf_y

cum_cf = np.cumsum(net_cf)

# Console table for paste-into-paper
print()
print("Year | Revenue | Energy | Battery | Maint  | Pre-tax | Depr  | Tax    | ATCF   | Cumul.")
print("-" * 100)
for y in years:
    if y == 0:
        print(f"{y:4d} | {'---':>7} | {'---':>6} | {'---':>7} | "
              f"{'---':>6} | {'---':>7} | {'---':>5} | {'---':>6} | "
              f"{net_cf[y]:>+6.0f} | {cum_cf[y]:>+7.0f}")
    else:
        print(f"{y:4d} | {revenue[y]:>+7.0f} | {energy_s[y]:>+6.1f} | "
              f"{battery_s[y]:>+7.1f} | {maint[y]:>+6.1f} | "
              f"{pretax[y]:>+7.1f} | {depr[y]:>+5.0f} | "
              f"{tax_arr[y]:>+6.0f} | {atcf_arr[y]:>+6.0f} | "
              f"{cum_cf[y]:>+7.0f}")

atcf = atcf_arr[1]  # for sensitivity baseline use Year-1 unscaled (full ramp)

# ── Figure 1: Net cash flow + cumulative cash balance ──────────────────
fig, ax1 = plt.subplots(figsize=(8.0, 4.6))

bar_colors = ["#c0392b" if v < 0 else "#2980b9" for v in net_cf]
bars = ax1.bar(years, net_cf, color=bar_colors, edgecolor="black",
               linewidth=0.4, width=0.55, label="Net cash flow (annual)")

for x, v in zip(years, net_cf):
    label_y = v + (120 if v >= 0 else -160)
    ax1.text(x, label_y, f"{v:+,.0f}", ha="center",
             fontsize=8, fontweight="bold",
             color="#c0392b" if v < 0 else "#2980b9")

ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_xlabel("Year", fontsize=10)
ax1.set_ylabel("Net cash flow per year (NT$ thousands)", fontsize=10)
ax1.set_xticks(years)
ax1.grid(axis="y", linestyle=":", alpha=0.5)

# Cumulative cash balance on secondary axis
ax2 = ax1.twinx()
ax2.plot(years, cum_cf, color="#16a085", marker="o", linewidth=2.0,
         markersize=7, label="Cumulative cash balance")
for x, v in zip(years, cum_cf):
    ax2.text(x + 0.10, v + 350, f"{v:+,.0f}", fontsize=8,
             color="#16a085", fontweight="bold")
ax2.set_ylabel("Cumulative cash balance (NT$ thousands)",
               color="#16a085", fontsize=10)
ax2.tick_params(axis="y", colors="#16a085")

# Mark payback period (find where cumulative cash balance first crosses zero)
crossing_year = None
for y in range(LIFE_YEARS + 1):
    if cum_cf[y] >= 0 and (y == 0 or cum_cf[y - 1] < 0):
        crossing_year = y
        break
if crossing_year is not None and crossing_year > 0:
    # Linear-interpolate within the year that crosses
    before = cum_cf[crossing_year - 1]   # negative
    after = cum_cf[crossing_year]        # positive
    frac = -before / (after - before)
    pbp_year_frac = (crossing_year - 1) + frac
    ax1.axvline(pbp_year_frac, color="#8e44ad", linestyle="--",
                linewidth=1.2, alpha=0.7)
    ax1.annotate(f"PBP ≈ {pbp_year_frac*12:.1f} mo",
                 xy=(pbp_year_frac, -300),
                 xytext=(pbp_year_frac + 0.4, -550),
                 fontsize=9, color="#8e44ad",
                 arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.0))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="lower right", fontsize=9, framealpha=0.92)

plt.title("Annual Cash Flow and Cumulative Balance — Retrofit Case "
          "(NT$ thousands, Year 0–5)", fontsize=11)
plt.tight_layout()
out1 = FIG_DIR / "financial_cashflow_bars.png"
plt.savefig(out1, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {out1}")

# ── Figure 2: Sensitivity — PBP & NPV vs order margin ──────────────────
# Compute the full year-on-year cash flow series at each margin candidate
# and derive PBP (cumulative-crossing) and 5-yr discounted NPV from it.
margins = np.array([5, 10, 15, 20, 25, 30, 40])
pbp_months = np.zeros_like(margins, dtype=float)
npv_at_margin = np.zeros_like(margins, dtype=float)

for k, m in enumerate(margins):
    rev_full = 244_320 * m / 1000.0    # NT$ thousands
    series = np.zeros(LIFE_YEARS + 1)
    series[0] = -CAPEX_RETROFIT
    npv = -CAPEX_RETROFIT
    for y in range(1, LIFE_YEARS + 1):
        ramp = RAMP[y]
        rev_y = rev_full * ramp * (1 + INFL_REVENUE) ** (y - 1)
        eng_y = ENERGY_SAVING_FULL * ramp * (1 + INFL_ENERGY) ** (y - 1)
        bat_y = BATTERY_SAVING_FULL * ramp
        mnt_y = MAINTENANCE_BASE * (1 + INFL_MAINT) ** (y - 1)
        pretax_y = rev_y + eng_y + bat_y + mnt_y
        taxable_y = pretax_y + DEPRECIATION
        tax_y = -TAX_RATE * taxable_y
        atcf_y = pretax_y + tax_y
        series[y] = atcf_y
        npv += atcf_y / (1 + DISCOUNT_RATE) ** y
    cum = np.cumsum(series)
    # Linear-interpolate cumulative crossing
    pbp_y = LIFE_YEARS + 1.0
    for y in range(1, LIFE_YEARS + 1):
        if cum[y] >= 0 and cum[y - 1] < 0:
            frac = -cum[y - 1] / (cum[y] - cum[y - 1])
            pbp_y = (y - 1) + frac
            break
    pbp_months[k] = pbp_y * 12.0
    npv_at_margin[k] = npv

fig, ax1 = plt.subplots(figsize=(8.0, 4.5))
ln1 = ax1.plot(margins, pbp_months, marker="o", color="#c0392b", linewidth=2,
               label="Payback period (months)")
ax1.set_xlabel("Contribution margin per pick (NT$)", fontsize=10)
ax1.set_ylabel("Payback period (months)", color="#c0392b", fontsize=10)
ax1.tick_params(axis="y", colors="#c0392b")
ax1.set_xticks(margins)
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.axvline(15, color="grey", linestyle="--", alpha=0.6, linewidth=0.9)
ax1.text(15.4, max(pbp_months) * 0.88, "Base case\n(NT$15)", fontsize=8,
         color="grey")

ax2 = ax1.twinx()
ln2 = ax2.plot(margins, npv_at_margin, marker="s", color="#16a085",
               linewidth=2, linestyle="-", label="5-yr NPV @ 10% (NT$k)")
ax2.set_ylabel("5-yr NPV @ 10% (NT$ thousands)",
               color="#16a085", fontsize=10)
ax2.tick_params(axis="y", colors="#16a085")

lines = ln1 + ln2
ax1.legend(lines, [l.get_label() for l in lines],
           loc="center right", fontsize=9, framealpha=0.92)

plt.title("Sensitivity of PBP and 5-yr NPV to Order Margin "
          "(Retrofit, 4,800 hr/yr)", fontsize=11)
plt.tight_layout()
out2 = FIG_DIR / "financial_sensitivity_pbp.png"
plt.savefig(out2, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {out2}")

# ── Console summary ────────────────────────────────────────────────────
print()
print("=" * 60)
print("DERIVED METRICS")
print("=" * 60)
print(f"Year-1 ATCF      = NT${atcf_arr[1]:>10,.1f}k "
      f"(${atcf_arr[1] * 1000 / NTD_PER_USD:>9,.0f})")
print(f"Year-3 ATCF      = NT${atcf_arr[3]:>10,.1f}k (full ramp)")
print(f"Year-5 ATCF      = NT${atcf_arr[5]:>10,.1f}k")
print(f"5-yr undiscounted sum of ATCF = NT${atcf_arr[1:].sum():,.1f}k")
print()
# PBP recompute from the cumulative series
pbp_str = "never"
for y in range(1, LIFE_YEARS + 1):
    if cum_cf[y] >= 0 and cum_cf[y - 1] < 0:
        frac = -cum_cf[y - 1] / (cum_cf[y] - cum_cf[y - 1])
        pbp_str = f"{(y - 1) + frac:.3f} yr = {((y - 1) + frac) * 12:.1f} months"
        break
print(f"Payback period (interpolated) = {pbp_str}")
# NPV from cumulative discounted ATCF
discounted_atcf = [atcf_arr[y] / (1 + DISCOUNT_RATE) ** y for y in range(1, LIFE_YEARS + 1)]
npv_5 = -CAPEX_RETROFIT + sum(discounted_atcf)
print(f"5-yr NPV @ {DISCOUNT_RATE:.0%} = NT${npv_5:,.1f}k")
print(f"Year-1 ROI = {(atcf_arr[1] - CAPEX_RETROFIT) / CAPEX_RETROFIT:.1%}")
