"""Gymnasium environment for single-room combat in The Binding of Isaac.

Each episode restarts the run (optionally with a fixed seed), teleports into
one of the configured rooms, and ends when the room is cleared, the player
dies, or the player leaves the room (terminated), or after ``max_steps``
(truncated).

The game must be running with the TBOI-RL Bridge mod and ``--luadebug``. The
environment opens its server when constructed and waits for the game on the
first ``reset()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tboi_rl import obs
from tboi_rl.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    NUM_MOVES,
    NUM_SHOOTS,
    BridgeError,
    BridgeServer,
    action_message,
    reset_message,
)
from tboi_rl.reward import RewardConfig, compute_reward


class IsaacEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        rooms: Sequence[str],
        run_seed: str | None = None,
        max_steps: int = 900,
        reward_config: RewardConfig = RewardConfig(),
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        connect_timeout: float | None = None,
        step_timeout: float = 30.0,
    ) -> None:
        """
        rooms:     rooms to sample from each episode, in debug ``goto`` syntax
                   (e.g. ``["d.12", "d.40"]``).
        run_seed:  seed for every restart (e.g. ``"JPCX AJ4R"``); ``None`` lets
                   the game pick one, which varies the run between episodes.
        max_steps: agent steps per episode before truncation (15 steps = 1 s).
        """
        if not rooms:
            raise ValueError("at least one room is required")
        self.rooms = list(rooms)
        self.run_seed = run_seed
        self.max_steps = max_steps
        self.reward_config = reward_config
        self.connect_timeout = connect_timeout
        self.step_timeout = step_timeout

        self.action_space = spaces.MultiDiscrete([NUM_MOVES, NUM_SHOOTS])
        self.observation_space = obs.observation_space()

        self.server = BridgeServer(host, port)
        self._connected = False
        self._awaiting_reply = False  # the game has sent a state and is waiting for us
        self._state: dict[str, Any] | None = None
        self._start_room_index: int | None = None
        self._steps = 0

    @property
    def last_state(self) -> dict[str, Any] | None:
        """The most recent raw state message, for scripted policies and debugging."""
        return self._state

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        room = options.get("room") or self.rooms[int(self.np_random.integers(len(self.rooms)))]
        run_seed = options.get("run_seed", self.run_seed)

        if not self._connected:
            self.server.wait_for_game(timeout=self.connect_timeout)
            self._connected = True
        if not self._awaiting_reply:
            # The game only listens between steps, right after sending a state.
            self._recv_state(timeout=None)

        self.server.send(reset_message(run_seed, room))
        self._awaiting_reply = False
        state = self._recv_state(timeout=self.step_timeout)
        if not state["new_run"]:
            raise BridgeError("expected the first state of a new episode after a reset")

        self._state = state
        self._start_room_index = state["room"]["index"]
        self._steps = 0
        info = {"room": room, "seed": state["seed"], "enemies": state["room"]["enemies"]}
        if state["room"]["clear"]:
            info["warning"] = f"room {room} has no enemies; the episode will end immediately"
        return obs.encode(state), info

    def step(
        self, action: np.ndarray | Sequence[int]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._state is None or not self._awaiting_reply:
            raise RuntimeError("call reset() before step()")
        move, shoot = int(action[0]), int(action[1])
        prev = self._state

        self.server.send(action_message(move, shoot))
        self._awaiting_reply = False
        state = self._recv_state(timeout=self.step_timeout)
        if state["step"] != prev["step"] + 1:
            raise BridgeError(f"desync: step {prev['step']} followed by {state['step']}")
        self._state = state
        self._steps += 1

        reward, terms = compute_reward(prev, state, self.reward_config)
        dead = state["player"]["dead"]
        cleared = state["room"]["clear"]
        left_room = state["room"]["index"] != self._start_room_index
        terminated = dead or cleared or left_room
        truncated = not terminated and self._steps >= self.max_steps
        info = {
            "reward_terms": terms,
            "dead": dead,
            "room_clear": cleared,
            "left_room": left_room,
            "steps": self._steps,
        }
        return obs.encode(state), reward, terminated, truncated, info

    def close(self) -> None:
        self.server.close()
        self._connected = self._awaiting_reply = False

    def _recv_state(self, timeout: float | None) -> dict[str, Any]:
        while True:
            msg = self.server.recv(timeout=timeout)
            if msg["type"] == "state":
                self._awaiting_reply = True
                return msg
