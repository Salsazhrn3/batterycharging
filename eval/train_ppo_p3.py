"""
eval/train_ppo_p3.py
────────────────────
Phase 4 Week 3 launcher: train MaskablePPO on the P3 placement with the
Phase-2 CRSM optimum as the warm-start policy.

  Placement   : P3 (picker-station opportunistic, 12 chargers)
  Policy init : tau_low=18, tau_full=60, interrupt=on@50  (CRSM optimum)
  Algo        : MaskablePPO from sb3-contrib (Bischoff 2025 baseline)
  Net         : 2-layer MLP, 64 units (Bischoff Table 3)
  Action      : Discrete(9), A_full = {0, 30..100}
  Decisions   : per-AMR-per-event (rl_env.py)

Quick benchmark mode
────────────────────
Run with `--benchmark` to do a short rollout (default total_timesteps=2048)
and print env-steps/sec, so we can extrapolate the budget for the full
2-4 M step training run before committing to the long compute.

Full-training mode
──────────────────
Run with `--total-timesteps N` (e.g., 2_000_000) for the real run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: F401  (kept for downstream eval extensions)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback

from eval.rl_env import RLChargingEnv


# Phase-2 Constrained-RSM optimum on P3.
CRSM_OVERRIDE = {
    "battery_low_pct": 18.0,
    "battery_charged_pct": 60.0,
    "battery_interrupt_pct": 50.0,
    "initial_battery_frac": 1.0,
}

# Bischoff Table 3 baseline hyperparameters (sb3 defaults where unspecified).
BISCHOFF_HP = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)


def mask_fn(env):
    return env.action_masks()


def build_env(horizon: int, verbose: bool):
    env = RLChargingEnv(
        pipeline=3,
        horizon=horizon,
        decision_every=500,  # safety cap on ticks-per-step when no event fires
        config_override=CRSM_OVERRIDE,
        verbose=verbose,
    )
    return ActionMasker(env, mask_fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=2048,
                   help="Total env steps for PPO. 2048 = short benchmark; "
                        "2_000_000 = Bischoff-style full training.")
    p.add_argument("--horizon", type=int, default=20000,
                   help="Sim ticks per episode (default 20k = Phase-1 short).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benchmark", action="store_true",
                   help="Bench mode: print steps/s, skip checkpoint saving.")
    p.add_argument("--verbose-env", action="store_true",
                   help="Print rl_env per-decision diagnostics.")
    p.add_argument("--out", type=str,
                   default=str(ROOT / "eval" / "results" / "ppo_p3.zip"))
    args = p.parse_args()

    print("=" * 70)
    print("MaskablePPO training on P3 with CRSM-optimum warm start")
    print("=" * 70)
    print(f"  total_timesteps = {args.total_timesteps:,}")
    print(f"  horizon         = {args.horizon:,} sim ticks/episode")
    print(f"  seed            = {args.seed}")
    print(f"  CRSM_OVERRIDE   = {CRSM_OVERRIDE}")
    print(f"  HP              = {BISCHOFF_HP}")
    print()

    env = build_env(horizon=args.horizon, verbose=args.verbose_env)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=args.seed,
        verbose=1,
        **BISCHOFF_HP,
    )

    # Checkpoint every ~10% of training (skip in benchmark mode).
    callbacks = None
    if not args.benchmark:
        ckpt_every = max(args.total_timesteps // 10, BISCHOFF_HP["n_steps"])
        ckpt_dir = ROOT / "eval" / "results" / "ppo_p3_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks = CheckpointCallback(
            save_freq=ckpt_every,
            save_path=str(ckpt_dir),
            name_prefix="ppo_p3",
        )
        print(f"[train] checkpointing every {ckpt_every} steps "
              f"to {ckpt_dir}")

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    elapsed = time.time() - t0

    sps = args.total_timesteps / max(elapsed, 1e-9)
    print()
    print("=" * 70)
    print(f"TRAINING DONE: {args.total_timesteps:,} env steps "
          f"in {elapsed:.1f} s")
    print(f"  rate = {sps:.2f} env steps/s")
    print(f"  extrapolated 2.0 M-step run: "
          f"{2_000_000 / sps / 3600:.2f} h "
          f"({2_000_000 / sps / 60:.1f} min)")
    print(f"  extrapolated 4.0 M-step run: "
          f"{4_000_000 / sps / 3600:.2f} h")
    print("=" * 70)

    if not args.benchmark:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(out)
        print(f"[train] model saved to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
