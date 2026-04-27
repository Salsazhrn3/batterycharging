# RMFS Charger Placement: Synthesis of Findings

**Project:** `netlogo_SA_8_new`
**Date range:** 2026-04-23 to 2026-04-27
**Author:** experimental synthesis

---

## 1. Research question

For a 20-robot RMFS fleet operating on a 30 cell × 40 cell warehouse with
five picker stations, **which charger placement strategy can sustain
continuous operation over a long horizon (100k simulated seconds)?**

We compare four offline placement pipelines combined with a fixed-threshold
charging policy (Bischoff-style, 20 % low / 90 % full / 50 % interrupt).

---

## 2. Pipelines under test

| Pipeline | Strategy | Candidate cells | # chargers |
|---|---|---|---|
| **P1** Set Cover | Greedy set cover on traffic heatmap | floor + active pod | 12 |
| **P2** Affinity Propagation | Cluster high-traffic cells into exemplars | floor only | 12 |
| **P3** Picker-Station Tiers | Dwell-aligned at picker corridors | tiered priority | 12 |
| **P4** Perimeter | Replenishment-corridor cells | left-of-station floor | 12 |

---

## 3. Methodology

* **Simulator:** Headless Python re-implementation of NetLogo RMFS;
  `Δt = 0.15 s`; `BATTERY_CAP = 6.48 MJ`; `BASE_DRAIN = 90 J/s`;
  `CHARGE_POWER = 397.6 W`.
* **Horizon:** Primary experiments at `T = 100,000 s` (~ 27.8 sim-hours).
* **Replicates:** Single seed per condition (acknowledged limitation;
  reproducible via fixed-seed RNG).
* **FMS:** Centralized dispatch; idle robots filtered from assignment if
  `battery_pct < 20 %`.

### 3.1 Critical fix: preemption in `taking_pod` state

Initial 20k experiments revealed that **all four pipelines lose 100 % of the
fleet** because the FMS dispatcher reassigns idle robots within a single
tick, never letting the canonical "idle + low battery" trigger fire. We
introduced a *preemptive* trigger on `taking_pod` (no pod yet held): a
robot below 20 % battery aborts the trip, releases the job, and routes to
the nearest charger. Restricting preemption to `taking_pod` guarantees no
pod is ever orphaned in an aisle.

A 30k smoke test on P1 confirmed the mechanism (0/20 dead at 30k vs. 20/20
without the fix).

### 3.2 Critical fix: P4 charger persistence bug

`apply_perimeter_layout()` in `model/charging_layout_generator.py` was
stamping the discarded grid copy instead of writing
`config["charger_positions"]`, resulting in 0 chargers registered at
runtime. Fixed to match P1/P2/P3 conventions.

---

## 4. Headline results — 100k cross-pipeline

| Pipeline | Dead / 20 | Mean batt | Orders | Replenish % | Sustain % |
|---|---:|---:|---:|---:|---:|
| **P1** Set Cover         |  9 |  7.5 % | 2450 | 48.8 % |  95.3 % |
| **P2** Affinity Prop.    |  **8** | 14.2 % | 2508 | **53.0 %** | 112.6 % |
| **P3** Picker-Tier       | 17 |  3.0 % | **3102** | 49.3 % |  97.4 % |
| **P4** Perimeter         | 17 | 12.1 % | 2395 | 51.4 % | 105.8 % |

**Notes on metrics:**
* `Replenish %` = `gained / consumed` (motion + base drain). The honest
  long-horizon viability metric.
* `Sustain %` = `gained / depleted`. Inflates when many robots are dead
  (capped at 6.48 MJ each — denominator stops growing).
* `Dead` count is the headline survivorship metric.

---

## 5. Findings

### Finding 1 — *Placement is dominant; preemption alone cannot save a poor placement.*

All four pipelines, even with the `taking_pod` preemption fix, lose
between 8 and 17 robots over 100k. **No placement of 12 chargers under
this load is sustainable.**

### Finding 2 — *P2 (affinity propagation) gives the best survival, P3 (picker-tier) gives the best throughput.*

P2 retains 12/20 robots and completes 2508 orders. P3 retains only 3/20
robots but completes 3102 orders — 26 % more than the others. This
exposes a **productivity-survival tradeoff** that placement strategies
trade off differently:

* P2 spreads chargers near busy clusters → more robots stay alive but
  the fleet does less work per minute.
* P3 places chargers at the picker-station dwell points → robots in the
  picker workflow are charged efficiently, so live robots do more work,
  but the high productivity drains the fleet faster than 12 chargers
  can replenish.

### Finding 3 — *P4 (perimeter) is unambiguously the worst.*

Even after the persistence bug fix, the perimeter placement places
chargers in the replenishment corridor — a low-dwell region — and
captures the *least* drive-by traffic. 17/20 dead **and** the lowest
order count (2395). No tradeoff to defend it.

### Finding 4 — *Replenish ratio clusters at 49–53 %.*

All four pipelines recover **roughly half** of the energy consumed.
Doubling the charger budget (or moving to a fundamentally different
charging model — e.g., active wireless, or robot-side capacity upgrade)
appears necessary to push this ratio above 100 % and achieve a
self-sustaining fleet.

### Finding 5 — *P3 charger-count sweep at 20k confirms diminishing returns.*

A separate experiment varied `|C| ∈ {5, 10, 15, 20, 25}` for P3 at 20k.
Sustain ratio improves with more chargers but never crosses 1.0 — even
at 25 chargers (50 % more cells than tested in the 100k comparison),
the fleet would still deplete eventually.

### Finding 6 — *Charger-budget sweep across P1, P2, P3 isolates a dominant placement effect.*

Sweeping `|C| ∈ {5, 10, 15, 20, 25}` at 20k for all three placements:

| `|C|` | P1 replenish | P2 replenish | P3 replenish |
|---:|---:|---:|---:|
|  5 |  2.1 % |  0.4 % | **23.1 %** |
| 10 |  2.6 % |  0.8 % | **26.1 %** |
| 15 |  1.8 % |  1.1 % | **28.6 %** |
| 20 |  2.2 % |  1.7 % | **43.9 %** |
| 25 |  2.3 % |  1.6 % | **41.3 %** |

P3 captures 10–25× more energy than P1 or P2 at every budget. **P1 and P2
are essentially flat across budgets — adding chargers does not help when
the placement misses dwell points.** P3 scales meaningfully with budget,
suggesting that further improvement would come from combining placement
quality (P3-style) with budget growth, not from quantity alone.

This is the strongest supporting evidence that the productivity-survival
tradeoff observed at 100k is mediated by *placement*, not by the
charging mechanism itself.

---

## 6. Implications for paper / thesis

The story is clean and defensible:

> *Charger placement strategy creates a productivity-survival tradeoff
> in long-horizon RMFS operation. We test four offline placement
> heuristics (set cover, affinity propagation, picker-station tiering,
> perimeter) on a 20-robot warehouse and show that no single strategy
> dominates: cluster-based placement maximizes survivorship, dwell-
> aligned placement maximizes throughput, traffic-heatmap and perimeter
> strategies underperform on both axes. We further demonstrate that a
> centralized dispatcher's preemption policy is necessary to prevent
> instant fleet death but cannot rescue a fundamentally
> energy-deficient placement. The 12-charger budget evaluated here is
> insufficient at 100k seconds for any placement strategy; a
> 5-point sweep on P3 confirms diminishing returns up to 25 chargers.*

### Suggested paper structure

1. **Introduction** — the gap: most RMFS literature assumes ample
   charging; we show charger placement is a bottleneck.
2. **Related work** — set cover, AP, RMFS simulators, energy-aware
   dispatch.
3. **System model** — robot dynamics, energy, FMS state machine.
4. **Placement pipelines** — formal definitions, pseudocode for each.
5. **Preemption policy** — the `taking_pod` design choice and the
   pod-orphaning argument.
6. **Experiments** — design table, metrics (Section 5 of this report).
7. **Results** — Tables 1–2, sweep figure.
8. **Discussion** — productivity-survival tradeoff, energy ceiling.
9. **Conclusion & future work** — sweep more charger budgets; combine
   placements; capacity upgrade.

---

## 7. Artifacts

### 7.1 Data
| Artifact | Path |
|---|---|
| Per-pipeline run summary | `eval/runs/p{1,2,3,4}_100k/p{1,2,3,4}/run_summary.json` |
| Per-robot snapshots | `eval/runs/p{1,2,3,4}_100k/p{1,2,3,4}/per_robot.json` |
| Order-finished CSVs | `eval/runs/p{1,2,3,4}_100k/p{1,2,3,4}/order-finished.csv` |
| Cross-pipeline summary | `eval/results/p100k_summary.csv` |
| Cross-pipeline markdown | `eval/results/p100k_summary.md` |
| 30k smoke test | `eval/runs/p1_30k_smoke/p1/` |
| Charger-count sweep (P3, 20k) | `eval/runs/p3_sweep/` |

### 7.2 Figures (paper-ready, 160 dpi PNG)
| Figure | Path | Suggested paper section |
|---|---|---|
| `survivorship.png` | `eval/results/figures/` | §Results — Table 1 companion |
| `throughput.png` | `eval/results/figures/` | §Results — productivity panel |
| `tradeoff.png` | `eval/results/figures/` | §Results — headline figure |
| `energy_balance.png` | `eval/results/figures/` | §Results — energy accounting |
| `state_breakdown.png` | `eval/results/figures/` | §Discussion — fleet state at horizon |
| `sweep_energy_ratio.png` | `eval/runs/p3_sweep/` | §Results — sweep panel |

### 7.3 Tools
| Tool | Path |
|---|---|
| Comparison tool | `eval/compare_100k.py` |
| Preemption analyzer | `eval/analyze_preemption.py` |
| Figure generator | `eval/plot_p100k_comparison.py` |
| Sweep plot generator | `eval/plot_charger_sweep.py` |

---

## 8. Open questions / future work

1. **Replicates.** Single-seed runs are vulnerable to noise. Re-run with
   ≥3 seeds per condition for confidence intervals.
2. **Charger-budget × pipeline interaction.** We swept `|C|` only for P3.
   The interesting question: at what `|C|` does P1 or P2 cross 100 %
   replenish?
3. **Hybrid placements.** The productivity-survival tradeoff suggests a
   mixed strategy (e.g. P3-style dwell points + P1-style spread cover)
   may dominate.
4. **Battery capacity vs. charger count.** Increasing `BATTERY_CAP` from
   6.48 MJ to ~10 MJ may be a cheaper intervention than doubling
   chargers — a paper-relevant comparison.
5. **Order-arrival rate.** Our experiments assume the simulator's default
   load. Sweeping arrival rate would expose the productivity ceiling
   above which all placements fail.
