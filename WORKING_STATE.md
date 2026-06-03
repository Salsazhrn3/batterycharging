# Working State — Thesis: Joint Placement-Policy Optimisation for RMFS Charging

> **Last updated:** 2026-06-03
> **Owner:** Salsa Zahrani (nazzahraarezta@gmail.com)
> **Purpose:** Snapshot of current work-in-progress so any future session
> (this device or another) can pick up cleanly. Read this first.

---

## 1. Project status at a glance

| Artefact | Path | State |
|----------|------|-------|
| Paper draft | [eval/results/paper_full.tex](eval/results/paper_full.tex) | **34 pages**, compiles cleanly |
| Paper PDF | [eval/results/paper_full.pdf](eval/results/paper_full.pdf) | Latest build matches .tex |
| Defense slides | [eval/results/presentation.tex](eval/results/presentation.tex) | **23 slides** (1 title + 18 content + 1 closing + 3 financial) |
| Slides PDF | [eval/results/presentation.pdf](eval/results/presentation.pdf) | Latest build matches .tex |
| Financial-projection script | [eval/plot_financial_projection.py](eval/plot_financial_projection.py) | Generates 2 figures + console summary |
| RL training code | [eval/train_ppo_p1.py](eval/train_ppo_p1.py), [eval/rl_env.py](eval/rl_env.py) | **Abandoned**: 4 collapsed runs (v5–v8) |
| Defense deadline | — | ~1 month from 2026-05-29 anchor (~2026-06-29) |

---

## 2. Research narrative (one paragraph)

A phased Design of Experiments framework for joint placement-policy
optimisation of RMFS charging infrastructure. **Phase 1**: long-horizon
cross-pipeline comparison + charger-budget sweep across P1 (Set Cover),
P2 (Affinity Propagation), P3 (Picker-Tier), P4 (Perimeter) at fixed
default policy. **Phase 2**: face-centred Central Composite Design over
$(|C|, \tau_\text{low}, \tau_\text{full})$ with a discovered deadlock
manifold treated as an infeasibility constraint. **Phase 2b**: cross-
placement RSM + Michaelis-Menten saturation fit. **Phase 3**: Taguchi
$L_9$ orthogonal array over four operational noise factors. **Headline
finding:** placement quality dominates budget by ~30×; the constrained-
RSM optimum is also the Taguchi-robust optimum; initial battery state at
shift start is the dominant noise lever (not demand or fleet attrition).

**Joint optimum recommendation:** P3 placement, $|C|=12$ chargers in
picker corridor, $\tau_\text{low}=18\,\%$, $\tau_\text{full}=60\,\%$,
interrupt on @ 50%, operating practice ensuring ≥70% start-of-shift
battery.

---

## 3. Recent work (chronological, most recent first)

### Battery degradation correctly excluded from financial projection (2026-06-03)
- Added dedicated paragraph in §Limitations: simulator does not model
  cyclic/calendar aging; no policy-side constraint on battery health
- Removed battery-longevity row from operating-assumptions table
- Removed battery-saving term from year-on-year cash flow construction
- Updated all financial metrics: PBP 3.8→3.9 mo, NPV₅ 10.5→10.1 M NT$,
  IRR 315→304%, Y1 ROI 216→204%
- Updated presentation slides 16–18 with the new numbers and explicit
  "battery longevity excluded" annotation

### Glossary appendix added (2026-06-03)
- New Appendix B: 5 topic groups (Battery & Energy, DoE & Statistics,
  Robotics & Simulation, RL, Engineering Finance) with ~40 plain-language
  definitions
- 3 new bibliography entries: Keil 2016, Schmalstieg 2014, Vetter 2005
  (Li-ion calendar aging literature)

### Financial projection refined with year-on-year dynamics (2026-06-02)
- Cash-flow chart was previously flat (constant ATCF assumption);
  refined to ramp 80/95/100/100/100% + revenue inflation 2%/yr +
  energy inflation 3%/yr + maintenance escalation 5%/yr
- Sensitivity chart NPV line is correctly linear (mathematical, not
  artefact)
- 9 new bibliography entries: Bechtsis 2017, Taipower 2024,
  taiwanmof, Bartholdi 2019, Tompkins 2010, BNEF 2024, Saxena 2016,
  Sandborn 2019, Yu 2008

### Financial projection added (2026-06-01)
- Subsection §VII-J Financial Projection inserted in Discussion
- Comparator: 10 chargers random in pod area (P1-proxy) vs 12 chargers
  P3 with RSM optimum policy
- Configured for Carrefour-class Taiwan e-commerce warehouse:
  - 2-shift operation, 4,800 hr/yr
  - Taipower NT$3.50/kWh industrial tariff
  - NT$15/pick grocery margin
  - 20% corporate tax, 10% WACC
  - 5-year useful life, straight-line depreciation
- CapEx: NT$775k retrofit (10 reloc + 2 new + install)
- Throughput delta: +244k orders/yr → NT$3.66M/yr revenue gain

### Defense presentation expanded (2026-06-01 → 2026-06-03)
- From 10 → 20 → 23 slides
- Added: Problem Statement, Contributions, 4 theory slides (DoE/RSM/CCD,
  Constrained RSM, Taguchi, why CCD vs Box-Behnken/PB), Placement
  Algorithms, Energy Model two-channel, 3 deadlock mechanisms,
  Limitations + Future Work, 3 financial slides, Thank-You close

### PPO RL experiment abandoned (2026-05-25 → 2026-05-27)
- v5: ent_coef=0, full reward → collapsed iter 6
- v6: ent_coef=0.01 → collapsed iter 6-7
- v7: + battery randomisation → collapsed iter 4
- v8: β=0 reward + ent_coef=0.05 → collapsed iter 5
- All 4 archived under `eval/results/ppo_p1_{tb,checkpoints}_v{5,6,7,8}_collapsed/`
- Diagnosed structural causes: action-mask asymmetry, reward myopia
  beyond γ=0.99 horizon, decision-granularity asymmetry
- Negative result documented in paper §VII-G as a strength of the
  DoE/RSM methodology

### Paper energy-model section expanded (2026-05-29)
- Added comprehensive treatment of robot energy with formulas:
  $E_\text{accel}, E_\text{decel}, E_\text{const}, E_\text{rot},
  E_\text{lift}, E_\text{idle}^{(\Delta t)}$
- Added new Table II (manufacturer spec → simulator units derivation)
- Expanded Table I (parameters) with mechanical/geometry/battery blocks

---

## 4. Open / next-step items

1. **Multi-seed Phase 2 replication.** Phase 2 RSM currently single-seed
   per cell. Adding ≥3 seeds at the design centre would estimate
   pure-error variance and enable formal lack-of-fit F-test. Identified
   as highest-priority statistical follow-up in §Limitations.
2. **Hybrid placement exploration.** Combine P3 dwell-alignment with P1
   spatial spread; may further raise saturation ceiling beyond ~60%
   replenish.
3. **Battery degradation simulator extension.** Add a Schmalstieg 2014
   aging model to the simulator + a battery-health term to the policy
   objective. Would enable defensible monetisation of battery longevity
   in future financial projections.
4. **Defense rehearsal.** Slides exist, not yet rehearsed end-to-end
   with timing.

---

## 5. Key files & what they do

### Paper & presentation
- [eval/results/paper_full.tex](eval/results/paper_full.tex) — single-source thesis paper
- [eval/results/presentation.tex](eval/results/presentation.tex) — defense slides (Madrid theme, native beamer blocks)
- [eval/results/figures/](eval/results/figures/) — all PNG figures referenced by paper + slides

### Simulator (source of truth for empirical claims)
- [engine/universe.py](engine/universe.py) — tick loop, charger cell state
- [model/robot.py](model/robot.py) — energy model, charging policy thresholds
- [model/inventory.py](model/inventory.py) — FMS dispatcher

### RL infrastructure (abandoned but preserved for follow-up)
- [eval/rl_env.py](eval/rl_env.py) — Gymnasium env wrapper, 15-dim obs, 9-action discrete
- [eval/train_ppo_p1.py](eval/train_ppo_p1.py) — MaskablePPO training loop

### Analysis scripts
- [eval/run_doe_phase2b.py](eval/run_doe_phase2b.py) — Phase 2b RSM driver
- [eval/run_p3_taguchi.py](eval/run_p3_taguchi.py) — Phase 3 L9 driver
- [eval/plot_financial_projection.py](eval/plot_financial_projection.py) — financial figures + console summary
- [eval/analyze_doe_phase1.py](eval/analyze_doe_phase1.py), [eval/analyze_doe_phase3.py](eval/analyze_doe_phase3.py) — analysis

---

## 6. Important user preferences (also in auto-memory)

- **Exhaust existing data before claims.** Scan `eval/runs/**` for
  falsifying data before committing a paper claim or asking for new
  compute.
- **Check memory before parallel sims.** Mac is RAM-bound;
  `vm_stat`/swap before launching a 2nd long sim.
- **Charger count |C| ≤ N=20.** |C|>N is physically meaningless for
  active dispatch.

---

## 7. How to resume work on this thesis

1. `cd /Users/admin/netlogo_sa_8new`
2. `claude` → `/resume` → pick the most recent conversation
   (the JSONL with newest mtime in `~/.claude/projects/-Users-admin-netlogo-sa-8new/`)
3. Or `claude -c` to continue the latest
4. Read this file first; the chat summary gives recent dialogue but
   misses code-level state changes that only show up here.

---

## 8. Build commands

```bash
# Paper
cd eval/results
pdflatex -interaction=nonstopmode paper_full.tex
pdflatex -interaction=nonstopmode paper_full.tex   # second pass for refs

# Presentation
pdflatex -interaction=nonstopmode presentation.tex

# Financial figures (regenerates PNGs in eval/results/figures/)
cd /Users/admin/netlogo_sa_8new
/Users/admin/netlogo_sa_8new/.venv/bin/python eval/plot_financial_projection.py
```

---

## 9. Out-of-scope (intentionally NOT in the thesis)

- RL training-scale results — documented as negative result only
- Battery degradation modelling — limitation acknowledged
- Multi-seed Phase 2/3 — limitation acknowledged
- Real-world deployment validation — future work
- Hybrid placement algorithms — future work
- Capacity-vs-count comparison ($E_\text{cap}$ instead of $|C|$) — future work
