"""Hand-written baseline policies that act on raw state messages.

They give reference scores for the learned agent: a uniformly random policy
as the floor, and a simple scripted fighter that lines up with the nearest
enemy, shoots at it, keeps its distance, and sidesteps incoming projectiles.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tboi_rl.protocol import NUM_MOVES, NUM_SHOOTS

# (sign x, sign y) -> movement index used by the mod
_DIR_TO_MOVE = {
    (0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4,
    (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8,
}
SHOOT_UP, SHOOT_RIGHT, SHOOT_DOWN, SHOOT_LEFT = 1, 2, 3, 4


def direction_to_move(dx: float, dy: float) -> int:
    """Nearest of the 8 movement directions to (dx, dy), or no-op for a zero vector."""
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0
    # A component counts if it is at least sin(22.5 deg) of the vector, giving 8 even sectors.
    threshold = 0.383 * norm
    sx = 0 if abs(dx) < threshold else (1 if dx > 0 else -1)
    sy = 0 if abs(dy) < threshold else (1 if dy > 0 else -1)
    return _DIR_TO_MOVE[(sx, sy)]


def shoot_towards(dx: float, dy: float) -> int:
    if abs(dx) >= abs(dy):
        return SHOOT_RIGHT if dx > 0 else SHOOT_LEFT
    return SHOOT_DOWN if dy > 0 else SHOOT_UP


class RandomPolicy:
    def __init__(self, seed: int | None = None, hold: int = 5) -> None:
        self.rng = np.random.default_rng(seed)
        self.hold = hold
        self._t = 0
        self._action = (0, 0)

    def __call__(self, state: dict[str, Any]) -> tuple[int, int]:
        if self._t % self.hold == 0:
            self._action = (int(self.rng.integers(NUM_MOVES)), int(self.rng.integers(NUM_SHOOTS)))
        self._t += 1
        return self._action


class ScriptedPolicy:
    def __init__(
        self,
        min_distance: float = 120.0,
        align_tolerance: float = 20.0,
        dodge_radius: float = 110.0,
    ) -> None:
        self.min_distance = min_distance
        self.align_tolerance = align_tolerance
        self.dodge_radius = dodge_radius

    def __call__(self, state: dict[str, Any]) -> tuple[int, int]:
        px, py = state["player"]["pos"]
        move_x = move_y = 0.0
        shoot = 0

        targets = [e for e in state["enemies"] if e[8]]  # vulnerable enemies, nearest first
        if targets:
            dx, dy = targets[0][2] - px, targets[0][3] - py
            shoot = shoot_towards(dx, dy)
            if math.hypot(dx, dy) < self.min_distance:
                move_x, move_y = -dx, -dy  # back off
            elif abs(dx) >= abs(dy):
                # Shooting horizontally: line up vertically with the target.
                move_y = dy if abs(dy) > self.align_tolerance else 0.0
            else:
                move_x = dx if abs(dx) > self.align_tolerance else 0.0

        for p in state["projectiles"]:  # nearest first
            rx, ry = px - p[0], py - p[1]
            if math.hypot(rx, ry) > self.dodge_radius:
                break
            vx, vy = p[2], p[3]
            if vx * rx + vy * ry <= 0:
                continue  # moving away from the player
            # Step sideways, perpendicular to the projectile's path, on the side the player is already on.
            perp_x, perp_y = -vy, vx
            if perp_x * rx + perp_y * ry < 0:
                perp_x, perp_y = -perp_x, -perp_y
            move_x, move_y = perp_x, perp_y
            break

        return direction_to_move(move_x, move_y), shoot
