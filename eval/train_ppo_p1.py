"""
eval/train_ppo_p1.py
────────────────────
Phase 4 Week 3-onwards: train MaskablePPO on the P1 placement.

P1 (Set Cover Approximation) places chargers spatially distant from picker
stations — robots must make *active* go-to-charge decisions. This is the
placement family where RL has the most decisions to make, in contrast to
P3's drive-by/opportunistic mechanism.

  Placement   : P1 (set-cover, |C|=12 explicit positions per launcher convention)
  Policy init : Phase-1 best corner (tau_low=20, tau_full=80, interrupt=off)
                — P1 was never analysed in Phase 2 RSM so we use the Phase-1
                screening best corner as the warm-start prior.
  Algo        : MaskablePPO (sb3-contrib)
  Net         : 2-layer MLP, 64 units
  Decisions   : per-AMR-per-event (rl_env.py)

Usage
─────
    # Short benchmark
    python eval/train_ppo_p1.py --benchmark --total-timesteps 2048
    # Full Bischoff-scale training (multi-day)
    python eval/train_ppo_p1.py --total-timesteps 2000000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gymnasium.wrappers import TimeLimit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from eval.rl_env import RLChargingEnv

EPISODE_ENV_STEP_CAP = 10_000


# Phase-1 best corner for P1 (interrupt-off threshold policy).
P1_OVERRIDE = {
    "battery_low_pct": 20.0,
    "battery_charged_pct": 80.0,
    "battery_interrupt_pct": 50.0,
    "initial_battery_frac": 1.0,
}

BISCHOFF_HP = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
)


def mask_fn(env):
    return env.unwrapped.action_masks()


def build_env(horizon: int, verbose: bool):
    env = RLChargingEnv(
        pipeline=1,
        horizon=horizon,
        decision_every=500,
        config_override=P1_OVERRIDE,
        verbose=verbose,
    )
    env = TimeLimit(env, max_episode_steps=EPISODE_ENV_STEP_CAP)
    env = Monitor(env)
    return ActionMasker(env, mask_fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--verbose-env", action="store_true")
    p.add_argument("--out", type=str,
                   default=str(ROOT / "eval" / "results" / "ppo_p1.zip"))
    p.add_argument("--tensorboard-log", type=str,
                   default=str(ROOT / "eval" / "results" / "ppo_p1_tb"),
                   help="Directory for tensorboard event files. "
                        "Run `tensorboard --logdir <this_dir>` to view.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a checkpoint .zip to resume from. "
                        "Already-completed steps are not re-counted.")
    args = p.parse_args()

    print("=" * 70)
    print("MaskablePPO training on P1 (set-cover)")
    print("=" * 70)
    print(f"  total_timesteps = {args.total_timesteps:,}")
    print(f"  horizon         = {args.horizon:,} sim ticks/episode")
    print(f"  seed            = {args.seed}")
    print(f"  P1_OVERRIDE     = {P1_OVERRIDE}")
    print()

    env = build_env(horizon=args.horizon, verbose=args.verbose_env)

    tb_dir = Path(args.tensorboard_log)
    tb_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from:
        print(f"[train] resuming from checkpoint {args.resume_from}")
        model = MaskablePPO.load(
            args.resume_from, env=env, tensorboard_log=str(tb_dir))
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            policy_kwargs=dict(net_arch=[64, 64]),
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(tb_dir),
            **BISCHOFF_HP,
        )

    callbacks = None
    if not args.benchmark:
        ckpt_every = max(args.total_timesteps // 10, BISCHOFF_HP["n_steps"])
        ckpt_dir = ROOT / "eval" / "results" / "ppo_p1_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks = CheckpointCallback(
            save_freq=ckpt_every,
            save_path=str(ckpt_dir),
            name_prefix="ppo_p1",
        )
        print(f"[train] checkpointing every {ckpt_every} steps -> {ckpt_dir}")

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps,
                callback=callbacks,
                reset_num_timesteps=(args.resume_from is None),
                tb_log_name="ppo_p1")
    elapsed = time.time() - t0
    sps = args.total_timesteps / max(elapsed, 1e-9)
    print()
    print("=" * 70)
    print(f"TRAINING DONE: {args.total_timesteps:,} env steps in {elapsed:.1f}s")
    print(f"  rate = {sps:.2f} env steps/s")
    print(f"  extrapolated 2.0M: {2_000_000 / sps / 3600:.2f} h")
    print("=" * 70)

    if not args.benchmark:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(out)
        print(f"[train] model saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
