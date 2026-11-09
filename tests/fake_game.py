"""A stand-in for the game mod that speaks the bridge protocol over TCP.

Simulates one 1x1 room with simple, deterministic combat: shooting damages
the nearest enemy, killing every enemy clears the room, and the player loses
half a heart every ``hit_every`` steps.
"""

from __future__ import annotations

import copy
import json
import socket
import threading
from typing import Any

GRID_W, GRID_H = 15, 9
ORIGIN = (40.0, 120.0)  # center of grid tile 0 in a 1x1 room
TEAR_DAMAGE = 3.5
MOVE_DIRS = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def room_grid(width: int = GRID_W, height: int = GRID_H) -> str:
    """Walls around the edge, a closed door on top, one rock in the middle."""
    rows = []
    for r in range(height):
        row = []
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                row.append("5" if (r == 0 and c == width // 2) else "1")
            elif (r, c) == (height // 2, width // 2 - 3):
                row.append("2")
            else:
                row.append("0")
        rows.append("".join(row))
    return "".join(rows)


def make_state(**overrides: Any) -> dict[str, Any]:
    """A state message shaped like the mod's, with sensible defaults."""
    state: dict[str, Any] = {
        "type": "state",
        "step": 0,
        "frame": 0,
        "new_run": False,
        "seed": "JPCX AJ4R",
        "stage": 1,
        "counters": {"kills": 0, "damage_dealt": 0.0, "hits_taken": 0},
        "player": {
            "pos": [320.0, 280.0],
            "vel": [0.0, 0.0],
            "hearts": 6,
            "max_hearts": 6,
            "soul_hearts": 0,
            "dead": False,
            "damage": 3.5,
            "fire_delay": 10.0,
            "shot_speed": 1.0,
            "move_speed": 1.0,
            "range": 260.0,
            "coins": 0,
            "bombs": 1,
            "keys": 0,
            "collectibles": 0,
        },
        "room": {
            "index": -3,
            "type": 1,
            "shape": 1,
            "clear": False,
            "frame": 0,
            "enemies": 2,
            "grid_width": GRID_W,
            "grid_height": GRID_H,
            "grid_origin": list(ORIGIN),
            "grid": room_grid(),
        },
        "enemies": [
            [10, 0, 200.0, 200.0, 1.0, 0.0, 10.0, 10.0, 1, 0],
            [10, 0, 440.0, 360.0, 0.0, -1.0, 10.0, 10.0, 1, 0],
        ],
        "projectiles": [[290.0, 280.0, 2.0, 0.0]],
        "pickups": [[20, 1, 120.0, 160.0]],
        "hazards": [],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            state[key].update(value)
        else:
            state[key] = value
    return state


class FakeGame(threading.Thread):
    def __init__(self, port: int, hit_every: int = 40, enemy_hp: float = 10.0) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.hit_every = hit_every
        self.enemy_hp = enemy_hp
        self.received: list[dict[str, Any]] = []
        self.step = 0
        self.state = make_state()
        self.error: BaseException | None = None

    def _new_episode(self) -> None:
        state = make_state(new_run=True)
        for e in state["enemies"]:
            e[6] = e[7] = self.enemy_hp
        state["counters"] = copy.deepcopy(self.state["counters"])  # counters run across episodes
        self.state = state
        self.episode_steps = 0

    def _apply_action(self, move: int, shoot: int) -> None:
        s = self.state
        s["new_run"] = False
        self.episode_steps += 1
        dx, dy = MOVE_DIRS[move]
        s["player"]["pos"][0] += 5.0 * dx
        s["player"]["pos"][1] += 5.0 * dy
        s["room"]["frame"] += 2

        if shoot and s["enemies"]:
            target = s["enemies"][0]
            dealt = min(TEAR_DAMAGE, target[6])
            target[6] -= dealt
            s["counters"]["damage_dealt"] += dealt
            if target[6] <= 0:
                s["enemies"].pop(0)
                s["counters"]["kills"] += 1
        s["room"]["enemies"] = len(s["enemies"])
        s["room"]["clear"] = not s["enemies"]

        if self.episode_steps % self.hit_every == 0 and not s["player"]["dead"]:
            s["counters"]["hits_taken"] += 1
            s["player"]["hearts"] = max(s["player"]["hearts"] - 1, 0)
            s["player"]["dead"] = s["player"]["hearts"] == 0

    def run(self) -> None:
        try:
            with socket.create_connection(("127.0.0.1", self.port)) as conn:
                reader = conn.makefile("rb")

                def send(msg: dict[str, Any]) -> None:
                    conn.sendall((json.dumps(msg) + "\n").encode())

                send({"type": "hello", "protocol": 2, "frame_skip": 2})
                self.episode_steps = 0
                while True:
                    self.state["step"] = self.step
                    send(self.state)
                    self.step += 1
                    line = reader.readline()
                    if not line:
                        return
                    msg = json.loads(line)
                    self.received.append(msg)
                    if msg["type"] == "reset":
                        if msg.get("room") == "d.999":
                            send({"type": "log", "message": "goto d.999: room not found"})
                            send({"type": "error", "message": "reset timed out during goto"})
                            continue
                        self._new_episode()
                    else:
                        self._apply_action(msg["move"], msg["shoot"])
        except (ConnectionError, OSError):
            return
        except BaseException as exc:  # surfaced by tests
            self.error = exc
