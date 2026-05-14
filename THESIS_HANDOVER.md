# Thesis Handover — Joint Placement-Policy Optimisation for RMFS Charging

**Last updated:** 2026-05-08 (handover written), updated 2026-05-13 (Phase 4 Week 1)
**Author of this handover:** Claude (working with Salsa Zahrani)
**Status:** Phases 1–3 complete and written into the paper. Phase 4 (RL extension) Week 1 complete on the current Windows laptop; Weeks 2–8 planned for the new dedicated device.

This file gives a new Claude Code session (on any device) the context it needs to continue the thesis without re-deriving everything from scratch.

---

## 1. Project at a glance

This is an Engineering Management thesis at ITB Bandung on **joint placement-policy optimisation of charging stations in a Robotic Mobile Fulfillment System (RMFS)**. The simulator is a Python re-implementation of a NetLogo RMFS model, operating on a 30×40 grid with 20 robots and 5 picker stations.

**Methodology framework:** phased Design of Experiments — Phase 1 factorial screening of 4 placement algorithms × policy corners, Phase 2 face-centred Central Composite Design with Constrained Response Surface Methodology, Phase 3 Taguchi $L_9$ robust design. References Myers/Montgomery/Anderson-Cook (RSM 4e, §7.5) and Taguchi 1986 / Phadke 1989 / Ross 1996 throughout.

**Headline findings (already established):**
1. Placement quality dominates budget by ~30× on replenish ratio (P3 wins, P1/P2/P4 are policy-immune at $|C|=12$).
2. Two empirical deadlock manifolds discovered: P2 at (lo=25, fu=60, off); P3 at (lo≈20, fu≥89, on@50).
3. Constrained-RSM optimum on P3 = $(τ_{low}=18, τ_{full}=60, \text{interrupt}=\text{on}@50\%)$.
4. Phase 3 Taguchi: Robust candidate beats Peak (CRSM optimum > Phase 1 best corner); 2/9 vs 4/9 deadlocks, +5.79 dB orders S/N.
5. Initial battery state is the dominant noise factor across the $L_9$ envelope.
6. Energy cost: P3 = 80 kJ/order vs 94–98 for P1/P2/P4 at long horizon (~17% OpEx reduction).

## 2. Paper status

- File: [eval/results/paper_full.tex](eval/results/paper_full.tex) — 21 pages
- PDF: [eval/results/paper_full.pdf](eval/results/paper_full.pdf)
- 28 references, 11 figures, Appendix A with per-cell $L_9$ tables
- Compiles cleanly with `pdflatex` (MiKTeX bin must be on PATH)
- Spelling: British (-ise, -our, -re) throughout

Section structure:
```
§1   Introduction (with methodology flow Fig. 1)
§2   Related Work
§3   Proposed Placement Algorithms (P1-P4 with algorithms)
§4   Experimental Setup
§5   Phase 1 Results (100k cross-pipeline + 20k sweep)
§6   Phase 2: Constrained RSM
§7   Phase 3: Taguchi L9
§8   Discussion (incl. §8.5 "role of RL", §8.7 cost-per-order, §8.8 mechanism analysis)
§9   Limitations
§10  Conclusion
Appendix A — L9 per-cell results
References
```

## 3. Codebase orientation

Critical paths (all relative to repo root):

| Path | Purpose |
|---|---|
| `netlogo.py` | Main simulator entry; `setup()` and `console_tick()` |
| `model/robot.py` | Robot class with battery, motion, charging logic |
| `model/inventory.py` | Universe / fleet manager; ticking |
| `model/charging_layout_generator.py` | Generates P1/P2/P3/P4 layouts |
| `model/order_generator.py` | Order generation; `order_cycle_time` is the demand scaler |
| `engine/landscape.py` | Spatial map; out-of-bounds bug already fixed (lines 86, 59) |
| `eval/run_one.py` | Headless single-run driver with config-override hook |
| `eval/run_doe_screening.py` | Phase 1 DoE launcher (32 cells) |
| `eval/run_doe_phase2.py` | Phase 2 CCD launcher |
| `eval/run_doe_phase3.py` | Phase 3 Taguchi L9 launcher |
| `eval/analyze_doe_phase1.py` | Phase 1 analysis |
| `eval/fit_p3_rsm.py` | Phase 2 RSM fit + constrained optimum |
| `eval/analyze_doe_phase3.py` | Phase 3 Taguchi S/N analysis |
| `eval/runs/` | All raw run artefacts (run_summary.json, per_robot.json, order-finished.csv per cell) |
| `eval/results/` | Aggregated CSVs, figures, paper.tex, paper.pdf |
| `warehouse.db` | **64 MB SQLite file, Git LFS tracked** — see §6 below |

## 4. Existing override / hook mechanism

`run_one.py` reads a config-override JSON via `--config-override` and applies it to:
- Charger placement (via `charger_positions` if provided, else algorithm generates)
- Policy parameters: `battery_low_pct`, `battery_charged_pct`, `battery_interrupt_pct`, `initial_battery_frac` — applied to `Robot` class attributes
- Noise factors: `num_robots`, `demand_load_factor` — applied to `netlogo` module attributes

Adding a new override key requires editing the loop in `run_one.py` around line 195-210.

## 5. The proposed Phase 4 — Reinforcement Learning Extension

**Research question:** Can a PPO-learned policy eliminate the depletion trade-off on placements P1/P2/P4 at the 100,000-second horizon, or is depletion structural at $|C|=12$?

**Outcomes are all publishable:**
- If RL achieves ≥95% replenish on all 4 placements → re-frame the thesis: policy can save any placement.
- If RL on P3 reaches ≥80% but P1/P2/P4 still deplete → confirm placement-quality dominance under best policy.
- If RL gives modest improvements on all 4 but depletion persists → confirm energy-deficit at $|C|=12$ is structural.

**Methodology (mirror Bischoff et al. 2025 \cite{bischoff2025}):**
- Algorithm: PPO with action masking (`MaskablePPO` from `sb3-contrib`)
- Action space: discrete charge targets `[0, 30, 40, 50, 60, 70, 80, 90, 100]` (0 = no charge)
- Observation space: ~15 per-AMR features (battery, distance to nearest free charger, queue lengths, fleet utilisation, etc.)
- Reward: start with Bischoff's `FullyShaped` (queue penalty + free-AMR bonus + charge-action penalties)
- Training: 2-4 M steps per placement; multi-layer perceptron, 2 layers × 64 neurons
- Evaluation: 100k-second horizon × 2-3 seeds per (placement, agent) for replication

**Concrete week-by-week plan (8 weeks):**

| Week | Deliverable | Status |
|---|---|---|
| 1 | `gym.Env` wrapper around the simulator; smoke test | **DONE** 2026-05-13 |
| 2 | Patch `deliver_quantity` `NoneType` simulator bug; migrate `rl_env.py` to per-AMR-per-event decision granularity (full Bischoff design) | pending |
| 3 | Train PPO on P3 with CRSM optimum as warm-start | pending |
| 4 | Train PPO on P1 and P2 (sequential) | pending |
| 5 | Train PPO on P4; rerun any failed-training attempts | pending |
| 6 | Evaluate trained agents at 100k × 2-3 seeds × 4 placements (40-60h compute) | pending |
| 7 | Analysis + new §11 "Phase 4: RL Refinement" section | pending |
| 8 | Paper update + buffer for slippage; defence prep | pending |

**Week 1 (completed 2026-05-13) artefacts:**

- `eval/rl_env.py` — `RLChargingEnv(gym.Env)`. Fleet-level decision granularity (action applied to all robots every `decision_every` ticks); 15-dim Bischoff Table 2 observation; `Discrete(9)` action space over `CHARGE_TARGETS = [0, 30, 40, 50, 60, 70, 80, 90, 100]`; Bischoff-style shaped reward (`Δorders - 5.0·deadlock_indicator - 0.5·n_free/N_r`); action masking via `action_masks()` for `MaskablePPO` compatibility. Reuses the existing `run_one.py` override mechanism so noise factors / initial-battery / placement choice all work through the same config-override JSON.
- `eval/verify_rl_env.py` — runs `stable_baselines3.check_env()` then a tiny `MaskablePPO` training (32 timesteps, 2 gradient updates) to validate the training loop end-to-end. Both checks passed on 2026-05-13 in 35.9 s; toy model saved to `eval/results/ppo_toy.zip`.
- Dependencies installed in the current Windows venv: `gymnasium 1.2.3`, `stable-baselines3 2.8.0`, `sb3-contrib 2.8.0` (torch + numpy were already present).

**Critical Week 2 task: patch the `deliver_quantity` `NoneType` bug.** During the toy training, random-action exploration triggered an existing simulator exception:
```
[ERROR] finish_picking_task for job <built-in function id>
[ERROR] for pod Pod(19) location (2,18)
[rl_env] simulator exception: 'NoneType' object has no attribute 'deliver_quantity'
```
The `_tick_until_manual` wrapper in `rl_env.py` already catches this and ends the episode gracefully, so training continues — but premature episode termination slows learning. Locate the offending code path (likely in `model/inventory.py` near `finish_picking_task`) and patch the `None` dereference before scaling up PPO training in Week 3.

**Out of scope (deliberately):**
- Multiple non-RL policy classes
- Full Taguchi noise envelope on the RL agents (108 long-horizon runs, slips schedule)
- RL hyperparameter sweeping (use Bischoff's defaults)
- Comparing different RL algorithms (just PPO)

**Risk factors:**
- RL training instability — plan for 1-2 retraining attempts per placement
- Reward shaping iteration — start with Bischoff's `FullyShaped` and don't deviate unless clearly broken
- Action masking is mandatory (without it PPO requests infeasible actions)
- Long compute runs vulnerable to laptop crashes — see §7 below

## 6. Critical operational gotchas

1. **`warehouse.db` is Git LFS-tracked.** After any `git checkout`/`pull`/branch-switch, run `file warehouse.db` to verify it's `SQLite 3.x database` and not a 132-byte ASCII pointer. If pointer, run `git lfs checkout` before any simulation.
2. **Landscape bugs are fixed** but the patches are in the codebase only — don't revert `engine/landscape.py:86` (upper-bound check) or `engine/landscape.py:59` (defensive setObject with WARN logging).
3. **P1's set-cover algorithm ignores `num_chargers` unless `charger_positions` is supplied explicitly.** P3's `apply_picking_station_layout` requires `selective_picker_chargers: True` + explicit `charger_positions` to use the canonical 12-cell layout. Both are documented in the launcher scripts; just be aware when extending.
4. **The simulator has no seed parameter.** All experiments to date are single-seed. This is a known limitation. Extending the simulator with a seed-controlled RNG is the highest-priority methodological follow-up (also noted in §9 of the paper).
5. **Run pace: ~5 ticks/s on the current laptop.** A 100k-tick run takes ~5 hours. A 20k-tick run takes ~47 min. Plan compute budget accordingly.

## 7. Hardware / device guidance

The current laptop has frozen multiple times during long runs. For a 2-month RL extension, recommendations:

| Setup | Suitability | Notes |
|---|---|---|
| **Current laptop** | Marginal | Use for development only; not for unattended training. Battery + thermal issues already caused 2 freezes. |
| **Desktop workstation** | Best for local | Reliable thermal envelope, can run 24/7. Recommended: i7-13700K or Ryzen 9 7900X, 32 GB RAM, NVMe SSD. CPU-only is fine — the PPO networks are tiny (2×64 MLP), GPU adds marginal benefit at this network scale. |
| **Cloud VM** | Best for parallelism | AWS `c6i.4xlarge` (16 vCPU, 32 GB RAM) at $0.68/h on-demand or ~$0.25/h spot. For 8 weeks × 12 h/day × 0.25 = ~$170. Allows parallel environment vectorisation in PPO — 8 parallel envs gives ~8× simulator throughput, cutting training time from ~10 days to ~1.5 days per placement. |
| **Free-tier GPU (Colab Pro)** | Optional supplement | Tiny network doesn't need GPU, but free CPU access on Colab works as a backup workstation for unattended runs. |

The simulator is CPU-bound; **the bottleneck is simulator throughput, not neural-network compute**. Faster CPU + parallel environments matter far more than GPU. If the user can rent a single high-core-count cloud VM for the training phase (Weeks 3-5), it pays for itself in schedule risk reduction.

## 8. Memory file index

Detailed notes on prior decisions live in `C:\Users\salsa\.claude\projects\c--TA-netlogo-SA-8-new\memory\` on the current device:

- `project_eval_design.md` — original placement evaluation design
- `project_p3_opportunistic_death.md` — Phase 1 P3 finding (100k pure opportunistic → 20/20 dead)
- `project_lfs_db_pointers.md` — warehouse.db LFS gotcha
- `project_research_scope_placement_x_policy.md` — research scope clarification
- `project_phase2_phase3_design.md` — Phase 2 + Phase 3 design parameters
- `project_p1_charger_count_fix.md` — P1 set-cover override fix
- `project_phase1_doe_findings.md` — Phase 1 DoE findings
- `project_phase2_methodology_crsm.md` — Constrained RSM methodology choice
- `project_phase3_doe_findings.md` — Phase 3 Taguchi findings

If switching devices: copy this directory across, OR rely on this handover file as the authoritative starting point.

## 9. Where to start on a new device

1. Verify the codebase is intact: `git status`, `file warehouse.db` (must report "SQLite 3.x database").
2. If `warehouse.db` is a 132-byte text pointer, run `git lfs checkout`.
3. Verify the latest paper still compiles: `cd eval/results && pdflatex paper_full.tex` (twice).
4. Verify existing experiment artefacts are in place: `ls eval/runs/doe_phase1 eval/runs/doe_phase2 eval/runs/doe_phase3`.
5. Read this handover file and `eval/results/paper_full.pdf` end-to-end.
6. Install Phase 4 dependencies: `pip install gymnasium stable-baselines3 sb3-contrib` (torch + numpy should already be present).
7. Validate the Phase 4 wrapper runs on the new device: `python eval/verify_rl_env.py` (expect `check_env` PASS + toy training PASS in under 1 min on a healthy machine).
8. Benchmark the simulator's per-tick speed on the new device: `python eval/run_one.py --pipeline 3 --horizon 5000 --heartbeat 1000` and read `ticks_per_second` from the generated `run_summary.json`. Above ~5 ticks/s = healthy; lower means the new device is slower than the Windows laptop.
9. Begin Phase 4 Week 2: patch the `deliver_quantity` `NoneType` bug (see §5), then migrate `eval/rl_env.py` from fleet-level to per-AMR-per-event decision granularity (the proper Bischoff design).

## 10. Final defence-ready claim (one-sentence summary)

> *This thesis presents a phased Design-of-Experiments framework for joint placement-policy optimisation of charging stations in an RMFS, discovers and characterises two operational deadlock manifolds as empirical infeasibility constraints, identifies a constrained-RSM optimum at (P3, $τ_{low}=18\%$, $τ_{full}=60\%$, interrupt=on@50%) that is simultaneously the most productive and most robust configuration across a Taguchi $L_9$ noise envelope, and articulates a CapEx-OpEx-procedural recommendation in which start-of-shift battery state is identified as the dominant operational lever.*
