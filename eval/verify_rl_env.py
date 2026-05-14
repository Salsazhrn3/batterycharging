"""
eval/verify_rl_env.py
─────────────────────
Two-step validation of the Gym wrapper from eval/rl_env.py:

  Step A — stable_baselines3.common.env_checker.check_env(env)
           verifies the env satisfies the Gymnasium contract (obs/action
           types, return shapes, reset semantics, etc.).

  Step B — MaskablePPO toy training for a small number of timesteps
           verifies the actual training loop (rollout buffer fill →
           gradient update → policy update) runs without errors.

Both are smoke tests: the toy training uses too few timesteps to learn
anything useful; the goal is to surface integration bugs early.

Tunables:
    decision_every  smaller => more RL steps per sim second; faster smoke
    total_timesteps total RL env steps for the toy training
    n_steps         PPO rollout buffer size (must <= total_timesteps)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from eval.rl_env import RLChargingEnv


# Toy-run sizing — tuned so the whole run completes in ~10 min on our box.
DECISION_EVERY = 200       # 200 sim ticks per RL step (vs 500 in smoke test)
TOTAL_TIMESTEPS = 32       # tiny: just enough to trigger one gradient update
N_STEPS = 16               # PPO rollout buffer fill size
BATCH_SIZE = 8             # mini-batch for the gradient update
HORIZON = 4000             # short episode so an episode boundary lands inside the run


def mask_fn(env):
    """ActionMasker callback that pulls the mask from the underlying env."""
    return env.action_masks()


def main() -> int:
    # ── Step A: check_env on the bare environment ───────────────────────
    print("=" * 70)
    print("Step A: stable_baselines3.check_env()")
    print("=" * 70)
    bare = RLChargingEnv(pipeline=3, horizon=HORIZON,
                         decision_every=DECISION_EVERY, verbose=False)
    t0 = time.time()
    try:
        check_env(bare, warn=True, skip_render_check=True)
        print(f"check_env PASS in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"check_env FAIL after {time.time() - t0:.1f}s: {e}")
        return 1

    # ── Step B: Toy MaskablePPO training ─────────────────────────────────
    print()
    print("=" * 70)
    print(f"Step B: MaskablePPO toy training "
          f"({TOTAL_TIMESTEPS} env steps, n_steps={N_STEPS})")
    print("=" * 70)
    print(f"  decision_every = {DECISION_EVERY} sim ticks per RL step")
    print(f"  horizon        = {HORIZON} sim ticks per episode")
    print(f"  expected wall-clock: ~{TOTAL_TIMESTEPS * DECISION_EVERY / 5.5 / 60:.1f} min "
          f"@ ~5.5 sim ticks/s")
    print()

    # Build a fresh env wrapped with ActionMasker so MaskablePPO sees masks.
    env = RLChargingEnv(pipeline=3, horizon=HORIZON,
                        decision_every=DECISION_EVERY, verbose=True)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=2,
        learning_rate=3e-4,
        verbose=1,
    )
    print()
    print("[verify] starting model.learn()…")
    t1 = time.time()
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except Exception as e:
        print(f"[verify] toy training FAIL after {time.time() - t1:.1f}s: {e}")
        import traceback; traceback.print_exc()
        return 2
    elapsed = time.time() - t1
    print(f"[verify] toy training PASS in {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")

    # Save the toy model so we can reload it later.
    out = ROOT / "eval" / "results" / "ppo_toy.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"[verify] toy model saved to {out}")

    print()
    print("=" * 70)
    print("BOTH STEPS PASSED — RL infrastructure is sound.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
