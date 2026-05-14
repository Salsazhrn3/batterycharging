"""
eval/rl_env.py
──────────────
Gymnasium environment wrapping the RMFS simulator for PPO-style charging-
policy training. Phase 4 Week 1 deliverable.

Design notes
────────────
This file implements a SIMPLIFIED first-iteration RL environment with a
fleet-level decision granularity (one action per N ticks, applied to all
robots) rather than the per-AMR-per-event granularity of Bischoff et al.
2025.  Reasons:

  - Per-event RL decisions require hooks into the FMS state machine to
    notify the agent after each completed pod delivery; that integration
    is week-2 work.
  - Fleet-level periodic decisions are sufficient to validate the Gym
    wrapper, reward shaping, action masking, and PPO training loop.
  - The action semantics (charge-target threshold) are identical to
    Bischoff's; only the decision granularity differs.

State features adapted from Bischoff et al. 2025, Table 2 (page 11):
  - Mean battery level (fleet average)
  - Mean battery level (busy AMRs only)
  - Min / max battery level
  - Fraction of fleet depleted / free / busy
  - Number of queued jobs (pending orders)
  - Number of free charger cells
  - Fleet utilisation
  - Time-of-day proxy (current_tick / horizon)

Action space (Bischoff A_full):
  Discrete(9) -> charge target ∈ {0, 30, 40, 50, 60, 70, 80, 90, 100}
  - 0 means "no active charging" (sets BATTERY_CHARGED_PCT very high,
    so the fleet only opportunistically charges)
  - others set the fleet-wide BATTERY_CHARGED_PCT

Reward (Bischoff's shaped reward, simplified):
  R = Δorders - α·deadlock_indicator + β·n_free_amrs / N_r
  with α = 5.0, β = 0.5 (initial guesses; will tune in week 2-3)

Episode termination:
  - max_ticks reached (truncated=True)
  - or all robots dead (terminated=True)

Usage
─────
    from eval.rl_env import RLChargingEnv
    env = RLChargingEnv(pipeline=3, horizon=20000, decision_every=1000)
    obs, info = env.reset()
    for _ in range(40):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Make project root importable so we can do `import netlogo`, etc.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import netlogo  # noqa: E402
from model.robot import Robot  # noqa: E402

# Bischoff's full action space A_full.
CHARGE_TARGETS = [0, 30, 40, 50, 60, 70, 80, 90, 100]


class RLChargingEnv(gym.Env):
    """Gymnasium environment for fleet-level charge-target decisions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        pipeline: int = 3,
        horizon: int = 20000,
        decision_every: int = 500,
        config_override: dict | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.horizon = horizon
        self.decision_every = decision_every  # simulator ticks per RL step
        self.config_override = config_override or {}
        self.verbose = verbose

        # Observation: 15-dim normalised float vector (per Bischoff Table 2)
        # All features scaled to [0, 1] except utilisation/queue-length
        # which are normalised against their maxima.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )

        # Action: Discrete(9) over Bischoff's A_full charge targets.
        self.action_space = spaces.Discrete(len(CHARGE_TARGETS))

        # State (filled by reset/step):
        self.universe = None
        self.current_tick: float = 0.0
        self.prev_orders: int = 0
        self.episode_reward: float = 0.0
        self.deadlock_streak: int = 0  # n consecutive decision steps with 0 new orders
        self.last_action_target: int = 90  # remember for observation

    # ── Required Gym API ──────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Apply policy + noise overrides via class attribute set
        cfg = self._build_config()
        self._apply_overrides(cfg)
        # Run the simulator setup — produces netlogo.state pickle on disk
        netlogo.setup()
        # Load universe in-memory and re-establish object back-references
        # (mirror of the pattern in netlogo.console_tick lines 1006-1011)
        import pickle
        with open(ROOT / "netlogo.state", "rb") as f:
            self.universe = pickle.load(f)
        for obj in self.universe._objects:
            obj.setUniverse(self.universe)
        self.current_tick = 0.0
        self.prev_orders = 0
        self.episode_reward = 0.0
        self.deadlock_streak = 0
        self.last_action_target = 90
        obs = self._extract_observation()
        info = {"current_tick": self.current_tick, "prev_orders": 0}
        return obs, info

    def step(self, action: int):
        # 1. Apply the action: set fleet-wide BATTERY_CHARGED_PCT
        target = CHARGE_TARGETS[action]
        # Action 0 = "no active charging" — push target very high so robots
        # never reach it; they will only gain charge opportunistically.
        Robot.BATTERY_CHARGED_PCT = float(target) if target > 0 else 200.0
        self.last_action_target = target

        # 2. Advance the simulator for `decision_every` ticks (in-memory)
        target_tick = min(self.current_tick + self.decision_every, self.horizon)
        self._tick_until_manual(target_tick)
        self.current_tick = float(self.universe._tick)

        # 3. Compute reward
        new_orders = self._count_orders_completed()
        delta_orders = new_orders - self.prev_orders
        self.prev_orders = new_orders

        n_dead = self._count_dead_robots()
        n_free = self._count_free_robots()
        N_r = self._fleet_size()

        # Deadlock indicator: no progress in 3 consecutive decision steps
        if delta_orders == 0:
            self.deadlock_streak += 1
        else:
            self.deadlock_streak = 0
        deadlock = 1.0 if self.deadlock_streak >= 3 else 0.0

        # Bischoff-style shaped reward (simplified, fleet-level):
        # +Δorders rewards productivity (the main objective)
        # -α·deadlock_indicator penalises stalls
        # +β·n_free/N_r encourages keeping fleet available
        alpha = 5.0
        beta = 0.5
        reward = (
            float(delta_orders)
            - alpha * deadlock
            + beta * (n_free / max(N_r, 1))
        )
        self.episode_reward += reward

        # 4. Termination checks
        terminated = (n_dead >= N_r)  # all dead → end
        truncated = (self.current_tick >= self.horizon - 1)

        # 5. Build obs and info
        obs = self._extract_observation()
        info = {
            "current_tick": float(self.current_tick),
            "orders": int(new_orders),
            "delta_orders": int(delta_orders),
            "n_dead": int(n_dead),
            "n_free": int(n_free),
            "deadlock_streak": int(self.deadlock_streak),
            "episode_reward": float(self.episode_reward),
            "action_target": int(target),
        }
        if self.verbose:
            print(f"[rl_env] t={self.current_tick:.0f}  act={target}  "
                  f"d_ord={delta_orders}  ord={new_orders}  "
                  f"dead={n_dead}  free={n_free}  r={reward:+.2f}", flush=True)

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Bischoff's action-mask: target below current mean battery is illegal.

        Returns a bool array of shape (9,) — True if action i is legal.
        At fleet-level decision granularity, we use the fleet mean battery
        as the "current state" reference; per-AMR masking is week-2 work.
        """
        mean_batt = self._fleet_mean_battery_pct()
        return np.array([t > mean_batt for t in CHARGE_TARGETS], dtype=bool)

    def close(self):
        # netlogo.setup() doesn't return anything to clean up explicitly;
        # subsequent calls to reset() reinitialise.
        pass

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_config(self) -> dict:
        """Build the full override config for one episode."""
        return {
            "pipeline": self.pipeline,
            "num_chargers": 12,
            "disable_active_charging": False,
            "selective_picker_chargers": (self.pipeline == 3),
            **self.config_override,
        }

    def _apply_overrides(self, cfg: dict) -> None:
        """Replicate run_one.py's override mechanism."""
        # Charger overlay: clear transient state + generate config
        from eval.run_one import write_charger_overlay, clear_transient_state
        clear_transient_state(ROOT)
        # Persist cfg so write_charger_overlay can read it as override
        tmp = ROOT / ".rl_env_override.json"
        tmp.write_text(json.dumps(cfg))
        write_charger_overlay(ROOT, self.pipeline, override_path=tmp)
        # Apply policy + noise overrides
        for key, attr in [
            ("battery_low_pct", "BATTERY_LOW_PCT"),
            ("battery_charged_pct", "BATTERY_CHARGED_PCT"),
            ("battery_interrupt_pct", "BATTERY_INTERRUPT_PCT"),
            ("initial_battery_frac", "INITIAL_BATTERY_FRAC"),
        ]:
            if key in cfg:
                setattr(Robot, attr, float(cfg[key]))
        if "num_robots" in cfg:
            netlogo.NUM_ROBOTS_OVERRIDE = int(cfg["num_robots"])
        if "demand_load_factor" in cfg:
            netlogo.DEMAND_LOAD_FACTOR_OVERRIDE = float(cfg["demand_load_factor"])

    def _tick_until_manual(self, target_tick: float) -> None:
        """Advance the simulator until current_tick >= target_tick.

        Uses universe.tick() directly because netlogo.console_tick() runs
        a full horizon, which is the wrong granularity for RL stepping.
        """
        while self.universe._tick < target_tick:
            try:
                self.universe.tick()
            except Exception as e:
                # Simulator crashed mid-tick — surface as termination
                if self.verbose:
                    print(f"[rl_env] simulator exception: {e}")
                break

    def _extract_observation(self) -> np.ndarray:
        """Build the 15-dim normalised feature vector.

        Indices map to:
          0. Mean battery level (fleet)
          1. Mean battery level (busy AMRs)
          2. Min battery level
          3. Max battery level
          4. Fraction depleted
          5. Fraction free (idle)
          6. Fraction busy
          7. Fleet utilisation
          8. Job queue length (normalised by 50)
          9. Number of free chargers (normalised by total)
         10. Number of dead robots (normalised by fleet)
         11. Current tick / horizon  (time-of-episode proxy)
         12. Last action target (normalised by 100)
         13. Deadlock streak (clipped at 5, normalised)
         14. Episode-progress reward (cumulative reward / 1000, clipped)
        """
        robots = self._get_robots()
        N = max(len(robots), 1)
        b = np.array([r.battery_pct for r in robots])
        busy = np.array([r.current_state in ("taking_pod", "delivering_pod",
                                              "returning_pod", "station_processing")
                         for r in robots])
        free = np.array([r.current_state == "idle" for r in robots])
        dead = np.array([r.current_state == "dead" or r.battery_pct <= 0.5
                         for r in robots])
        mean_b = float(b.mean() / 100.0) if N else 0.0
        mean_b_busy = float(b[busy].mean() / 100.0) if busy.any() else mean_b
        min_b = float(b.min() / 100.0) if N else 0.0
        max_b = float(b.max() / 100.0) if N else 0.0
        frac_dead = float(dead.sum() / N)
        frac_free = float(free.sum() / N)
        frac_busy = float(busy.sum() / N)
        utilisation = frac_busy

        qlen = len(getattr(self.universe, "job_queue", []))
        n_chargers_total = len(getattr(self.universe, "charger_cells", set()))
        # "Free chargers" = those not currently claimed by any robot.
        n_free_chargers = max(0, n_chargers_total -
                              sum(1 for r in robots
                                  if getattr(r, "_claimed_charger", None)))
        time_frac = min(1.0, self.current_tick / max(self.horizon, 1))

        obs = np.array([
            mean_b,
            mean_b_busy,
            min_b,
            max_b,
            frac_dead,
            frac_free,
            frac_busy,
            utilisation,
            min(qlen / 50.0, 1.0),
            float(n_free_chargers) / max(n_chargers_total, 1),
            frac_dead,
            time_frac,
            self.last_action_target / 100.0,
            min(self.deadlock_streak / 5.0, 1.0),
            min(max(self.episode_reward, 0) / 1000.0, 1.0),
        ], dtype=np.float32)
        return obs

    def _get_robots(self) -> list:
        if self.universe is None:
            return []
        return [o for o in getattr(self.universe, "_objects", [])
                if getattr(o, "object_type", None) == "robot"]

    def _fleet_size(self) -> int:
        return len(self._get_robots())

    def _fleet_mean_battery_pct(self) -> float:
        robots = self._get_robots()
        if not robots:
            return 100.0
        return float(np.mean([r.battery_pct for r in robots]))

    def _count_dead_robots(self) -> int:
        return sum(1 for r in self._get_robots()
                   if r.current_state == "dead" or r.battery_pct <= 0.5)

    def _count_free_robots(self) -> int:
        return sum(1 for r in self._get_robots()
                   if r.current_state == "idle")

    def _count_orders_completed(self) -> int:
        of = ROOT / "output" / "order-finished.csv"
        if not of.exists():
            return 0
        with open(of) as f:
            return max(0, sum(1 for _ in f) - 1)


# Quick smoke-test entry point.
if __name__ == "__main__":
    env = RLChargingEnv(pipeline=3, horizon=2000,
                        decision_every=500, verbose=True)
    print("[rl_env] action_space:", env.action_space)
    print("[rl_env] obs_space:", env.observation_space)
    print("[rl_env] running 1 random-action episode (horizon 2000)…")
    obs, info = env.reset()
    print("[rl_env] reset OK; initial obs shape:", obs.shape)
    print("[rl_env] initial obs:", obs)
    total_steps = 0
    while True:
        mask = env.action_masks()
        # Pick a random legal action
        legal = np.where(mask)[0]
        if not len(legal):
            action = 0
        else:
            action = int(np.random.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated or truncated:
            break
    print(f"[rl_env] episode done in {total_steps} steps; "
          f"final tick={env.current_tick:.0f}, "
          f"orders={info['orders']}, dead={info['n_dead']}, "
          f"reward={env.episode_reward:+.2f}")
