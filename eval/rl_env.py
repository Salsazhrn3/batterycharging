"""
eval/rl_env.py
──────────────
Gymnasium environment wrapping the RMFS simulator for PPO-style charging-
policy training.

Phase 4 Week 2: per-AMR-per-event decision granularity (Bischoff et al. 2025
design). Each Gym step corresponds to a single robot finishing a job and
returning to idle; the agent picks a charge target for that one robot.

Decision-event detection
────────────────────────
After every `universe.tick()` we scan the fleet and queue any robot whose
state transitioned `busy -> idle` this tick. Busy = any of
{taking_pod, delivering_pod, returning_pod, station_processing, going_to_charge}.
The transition `delivering_pod -> station_processing -> returning_pod -> idle`
ends a delivery job; the transition `going_to_charge -> idle` ends a charging
trip. Both are decision points: the agent re-picks a charge target.

Action semantics (per-AMR override of class attributes via instance attrs)
─────────────────────────────────────────────────────────────────────────
The action `a ∈ {0..8}` maps to `CHARGE_TARGETS[a] ∈ {0, 30..100}`.
  - a = 0 ("no charge"): set `robot.BATTERY_LOW_PCT = 0` so the simulator
    will not auto-trigger a charging trip for this robot. The robot stays
    in service until the next idle transition or until battery is exhausted.
  - a > 0 ("charge to k%"): set `robot.BATTERY_CHARGED_PCT = k`, call
    `robot._start_charging_trip()` to dispatch immediately, then set
    `robot.BATTERY_LOW_PCT = 0` so the next idle transition (after the
    charging trip ends) defers to the agent rather than auto-charging.

Action mask: targets ≤ current robot battery are illegal (no point charging
below current level). When all positive targets are masked the agent must
pick action 0.

Observation (15-dim, mixed per-AMR + global, normalised to roughly [0, 1])
────────────────────────────────────────────────────────────────────────
   0  Focus AMR battery %
   1  Focus AMR distance to nearest free charger (Manhattan / 70)
   2  Focus AMR is currently "dead" (battery ≤ 0.5%) → 0/1
   3  Fraction of fleet depleted
   4  Fraction of fleet idle
   5  Fraction of fleet busy
   6  Fleet mean battery %
   7  Fleet min battery %
   8  Fleet max battery %
   9  Mean battery of busy AMRs
  10  Job queue length / 50
  11  Free charger count / total
  12  Current tick / horizon
  13  Last action target for this AMR / 100
  14  Δorders since previous decision / 10 (clipped)

Reward (Bischoff shaped, per-decision):
  R = Δorders_since_last_decision
    - α · I(focus_robot_just_died)
    + β · n_free / N
With α = 5.0, β = 0.5 (kept from Week 1 toy run).
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import netlogo  # noqa: E402
from model.robot import Robot  # noqa: E402

CHARGE_TARGETS = [0, 30, 40, 50, 60, 70, 80, 90, 100]
BUSY_STATES = frozenset(
    {"taking_pod", "delivering_pod", "returning_pod",
     "station_processing", "going_to_charge"}
)
MAX_DIST_NORM = 70.0  # 30+40 grid diagonal upper bound (Manhattan)


class RLChargingEnv(gym.Env):
    """Per-AMR-per-event Gymnasium env for charge-target decisions."""

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
        # max_step_ticks: safety cap on simulator ticks per RL step when no
        # decision event arrives (e.g., everyone is busy or dead). Reuses the
        # `decision_every` kw so verify_rl_env.py stays compatible.
        self.max_step_ticks = max(int(decision_every), 50)
        self.config_override = config_override or {}
        self.verbose = verbose

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(CHARGE_TARGETS))

        # Per-episode state
        self.universe = None
        self.current_tick: float = 0.0
        self.prev_orders: int = 0
        self.episode_reward: float = 0.0
        self.pending: deque = deque()  # robots awaiting agent decision
        self.prev_states: dict = {}    # robot.id -> previous state
        self.last_action_per_robot: dict = {}  # robot.id -> last target
        self.focus_robot = None        # head of pending queue
        self.last_delta_orders: int = 0
        self.prev_alive: dict = {}     # robot.id -> was alive last decision

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self._build_config()
        self._apply_overrides(cfg)
        netlogo.setup()
        import pickle
        with open(ROOT / "netlogo.state", "rb") as f:
            self.universe = pickle.load(f)
        for obj in self.universe._objects:
            obj.setUniverse(self.universe)

        self.current_tick = 0.0
        self.prev_orders = 0
        self.episode_reward = 0.0
        self.pending.clear()
        self.last_delta_orders = 0
        robots = self._get_robots()
        # Randomise initial battery per robot to break the t=0 action-0
        # attractor (all-100% start makes action 0 the only legal mask choice
        # at the first decision event, collapsing the policy). battery_pct is
        # a derived property; set the underlying joule level instead.
        for r in robots:
            pct = float(self.np_random.uniform(60.0, 100.0))
            r.battery_level_j = float(r.BATTERY_CAPACITY_J) * (pct / 100.0)
        self.prev_states = {r.id: r.current_state for r in robots}
        self.last_action_per_robot = {r.id: 90 for r in robots}
        self.prev_alive = {r.id: True for r in robots}
        # Seed the queue: any robot already idle at t=0 is a decision point.
        for r in robots:
            if r.current_state == "idle":
                self.pending.append(r.id)
        # If nothing is idle yet (rare), spin a few ticks until something is.
        if not self.pending:
            self._advance_until_event()
        self.focus_robot = self._head_robot()
        obs = self._extract_observation()
        info = {"current_tick": float(self.current_tick),
                "focus_robot_id": self._focus_id()}
        return obs, info

    def step(self, action: int):
        focus = self.focus_robot
        target = CHARGE_TARGETS[int(action)]
        if focus is not None:
            self._apply_action_to_robot(focus, target)
            self.last_action_per_robot[focus.id] = target

        # Pop the just-handled robot.
        if self.pending:
            self.pending.popleft()

        # If more decisions are queued, return next without advancing time.
        if not self.pending:
            self._advance_until_event()
        self.focus_robot = self._head_robot()

        # Compute reward against the decision-event window.
        new_orders = self._count_orders_completed()
        delta_orders = new_orders - self.prev_orders
        self.prev_orders = new_orders
        self.last_delta_orders = delta_orders

        n_dead = self._count_dead_robots()
        n_free = self._count_free_robots()
        N_r = max(self._fleet_size(), 1)

        # Per-AMR death penalty: did the focus robot die during this window?
        focus_just_died = 0.0
        if focus is not None:
            was_alive = self.prev_alive.get(focus.id, True)
            is_alive_now = not self._is_dead(focus)
            if was_alive and not is_alive_now:
                focus_just_died = 1.0
            self.prev_alive[focus.id] = is_alive_now

        # β·n_free/N_r term removed: it rewards idleness, which structurally
        # favoured action 0 ("don't charge") and collapsed the policy across
        # v5/v6/v7. Reward is now Δorders − α·focus_just_died only.
        alpha = 5.0
        reward = float(delta_orders) - alpha * focus_just_died
        self.episode_reward += reward

        terminated = (n_dead >= N_r)
        truncated = (self.current_tick >= self.horizon - 1)

        obs = self._extract_observation()
        info = {
            "current_tick": float(self.current_tick),
            "orders": int(new_orders),
            "delta_orders": int(delta_orders),
            "n_dead": int(n_dead),
            "n_free": int(n_free),
            "focus_robot_id": self._focus_id(),
            "action_target": int(target),
            "episode_reward": float(self.episode_reward),
        }
        if self.verbose:
            fid = info["focus_robot_id"]
            print(f"[rl_env] t={self.current_tick:.0f}  amr={fid}  "
                  f"act={target}  d_ord={delta_orders}  "
                  f"dead={n_dead}/{N_r}  r={reward:+.2f}", flush=True)

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Targets at or below focus robot's current battery are illegal.

        Action 0 ("no charge") is always legal. With this rule the agent can
        only request a charge level higher than the robot already has.
        """
        if self.focus_robot is None:
            mask = np.zeros(len(CHARGE_TARGETS), dtype=bool)
            mask[0] = True
            return mask
        b = float(getattr(self.focus_robot, "battery_pct", 0.0))
        mask = np.array([(t == 0) or (t > b) for t in CHARGE_TARGETS],
                        dtype=bool)
        if not mask.any():
            mask[0] = True
        return mask

    def close(self):
        pass

    # ── Action application ────────────────────────────────────────────────

    def _apply_action_to_robot(self, robot, target: int) -> None:
        """Apply Bischoff charge-target action via per-instance attrs."""
        if target == 0:
            # Don't auto-trigger a charge trip for this robot.
            robot.BATTERY_LOW_PCT = 0.0
            return
        # Active charge action: dispatch the robot now, charge until `target`%.
        robot.BATTERY_CHARGED_PCT = float(target)
        if robot.current_state == "idle":
            try:
                robot._start_charging_trip()
            except Exception as e:
                if self.verbose:
                    print(f"[rl_env] _start_charging_trip failed "
                          f"for robot {robot.id}: {e}")
        # After dispatch, suppress auto-charge on the *next* idle transition;
        # the agent will get another decision event there.
        robot.BATTERY_LOW_PCT = 0.0

    # ── Event detection / advancing ───────────────────────────────────────

    def _advance_until_event(self) -> None:
        """Tick the simulator until a busy→idle transition is detected,
        the horizon is reached, all robots die, or max_step_ticks elapse."""
        start = float(self.universe._tick)
        cap = start + self.max_step_ticks
        while not self.pending:
            if self.universe._tick >= self.horizon:
                break
            if self.universe._tick >= cap:
                break
            if self._count_dead_robots() >= self._fleet_size():
                break
            try:
                self.universe.tick()
            except Exception as e:
                if self.verbose:
                    print(f"[rl_env] simulator exception: {e}")
                break
            self._collect_events()
        self.current_tick = float(self.universe._tick)

    def _collect_events(self) -> None:
        """Queue robots that just went busy→idle this tick."""
        for r in self._get_robots():
            prev = self.prev_states.get(r.id)
            curr = r.current_state
            if prev in BUSY_STATES and curr == "idle":
                # Avoid double-queueing.
                if r.id not in (rid for rid in self.pending):
                    self.pending.append(r.id)
            self.prev_states[r.id] = curr

    def _head_robot(self):
        if not self.pending:
            return None
        rid = self.pending[0]
        for r in self._get_robots():
            if r.id == rid:
                return r
        # Robot vanished (shouldn't happen) — drop and recurse.
        self.pending.popleft()
        return self._head_robot()

    def _focus_id(self) -> int:
        if self.focus_robot is None:
            return -1
        return int(getattr(self.focus_robot, "id", -1))

    # ── Observation ───────────────────────────────────────────────────────

    def _extract_observation(self) -> np.ndarray:
        robots = self._get_robots()
        N = max(len(robots), 1)
        b = np.array([r.battery_pct for r in robots], dtype=np.float64)
        busy = np.array([r.current_state in BUSY_STATES for r in robots])
        idle = np.array([r.current_state == "idle" for r in robots])
        dead = np.array([self._is_dead(r) for r in robots])
        mean_b = float(b.mean()) if N else 0.0
        min_b = float(b.min()) if N else 0.0
        max_b = float(b.max()) if N else 0.0
        mean_b_busy = float(b[busy].mean()) if busy.any() else mean_b

        focus = self.focus_robot
        if focus is not None:
            focus_b = float(getattr(focus, "battery_pct", 0.0))
            focus_dist = self._nearest_free_charger_distance(focus)
            focus_dead = 1.0 if self._is_dead(focus) else 0.0
            last_tgt = self.last_action_per_robot.get(focus.id, 90)
        else:
            focus_b = mean_b
            focus_dist = MAX_DIST_NORM
            focus_dead = 0.0
            last_tgt = 90

        qlen = len(getattr(self.universe, "job_queue", []))
        n_chargers_total = len(getattr(self.universe, "charger_cells", set()))
        occupied = getattr(self.universe, "occupied_chargers", {})
        n_free_chargers = max(0, n_chargers_total - len(occupied))
        time_frac = min(1.0, self.current_tick / max(self.horizon, 1))

        obs = np.array([
            focus_b / 100.0,
            min(focus_dist / MAX_DIST_NORM, 1.0),
            focus_dead,
            float(dead.sum()) / N,
            float(idle.sum()) / N,
            float(busy.sum()) / N,
            mean_b / 100.0,
            min_b / 100.0,
            max_b / 100.0,
            mean_b_busy / 100.0,
            min(qlen / 50.0, 1.0),
            float(n_free_chargers) / max(n_chargers_total, 1),
            time_frac,
            last_tgt / 100.0,
            min(max(self.last_delta_orders, 0) / 10.0, 1.0),
        ], dtype=np.float32)
        np.clip(obs, 0.0, 1.0, out=obs)
        return obs

    def _nearest_free_charger_distance(self, robot) -> float:
        cells = getattr(self.universe, "charger_cells", set())
        if not cells:
            return MAX_DIST_NORM
        occupied = getattr(self.universe, "occupied_chargers", {})
        free = [c for c in cells if c not in occupied]
        pool = free if free else list(cells)
        rx, ry = float(robot.pos_x), float(robot.pos_y)
        return min(abs(cx - rx) + abs(cy - ry) for (cx, cy) in pool)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_config(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "num_chargers": 12,
            "disable_active_charging": False,
            "selective_picker_chargers": (self.pipeline == 3),
            **self.config_override,
        }

    def _apply_overrides(self, cfg: dict) -> None:
        from eval.run_one import write_charger_overlay, clear_transient_state
        clear_transient_state(ROOT)
        tmp = ROOT / ".rl_env_override.json"
        tmp.write_text(json.dumps(cfg))
        write_charger_overlay(ROOT, self.pipeline, override_path=tmp)
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

    def _get_robots(self) -> list:
        if self.universe is None:
            return []
        return [o for o in getattr(self.universe, "_objects", [])
                if getattr(o, "object_type", None) == "robot"]

    def _fleet_size(self) -> int:
        return len(self._get_robots())

    def _is_dead(self, robot) -> bool:
        return (getattr(robot, "current_state", None) == "dead"
                or float(getattr(robot, "battery_pct", 0.0)) <= 0.5)

    def _count_dead_robots(self) -> int:
        return sum(1 for r in self._get_robots() if self._is_dead(r))

    def _count_free_robots(self) -> int:
        return sum(1 for r in self._get_robots()
                   if r.current_state == "idle")

    def _count_orders_completed(self) -> int:
        of = ROOT / "output" / "order-finished.csv"
        if not of.exists():
            return 0
        with open(of) as f:
            return max(0, sum(1 for _ in f) - 1)


# Smoke-test entry point.
if __name__ == "__main__":
    env = RLChargingEnv(pipeline=3, horizon=2000,
                        decision_every=500, verbose=True)
    print("[rl_env] action_space:", env.action_space)
    print("[rl_env] obs_space:", env.observation_space)
    obs, info = env.reset()
    print("[rl_env] reset OK; initial obs:", obs)
    print("[rl_env] initial focus_robot_id:", info["focus_robot_id"])
    total_steps = 0
    while True:
        mask = env.action_masks()
        legal = np.where(mask)[0]
        action = int(np.random.choice(legal)) if len(legal) else 0
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated or truncated:
            break
    print(f"[rl_env] episode done in {total_steps} decisions; "
          f"final tick={env.current_tick:.0f}, "
          f"orders={info['orders']}, dead={info['n_dead']}, "
          f"reward={env.episode_reward:+.2f}")
