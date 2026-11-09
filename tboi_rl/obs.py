"""Turns ``state`` messages from the game into fixed-size observation arrays.

The observation is a dict with two parts:

``grid`` (float32, shape ``(GRID_CHANNELS, GRID_H, GRID_W)``)
    The room as an image, one cell per 40x40 game grid tile, padded to the
    largest room shape (2x2 rooms are 28x16 tiles including walls). Smaller
    rooms occupy the top-left corner; the ``inside`` channel marks real cells.

``vec`` (float32, shape ``(VEC_SIZE,)``)
    Player stats and room status, followed by the nearest enemies and enemy
    projectiles relative to the player. Missing entries are zero with their
    ``present`` flag cleared.

Entity rows sent by the mod (see ``mod/tboi_rl_bridge/main.lua``):
    enemies:     [type, variant, x, y, vx, vy, hp, max_hp, vulnerable, boss]
    projectiles: [x, y, vx, vy]
    pickups:     [variant, subtype, x, y]
    hazards:     [type, variant, x, y]
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import spaces

TILE = 40.0  # game units per grid tile
GRID_H, GRID_W = 16, 28

# Grid channels
CH_WALL = 0       # walls and closed doors
CH_OBSTACLE = 1   # rocks, poop, blocks
CH_PIT = 2
CH_HAZARD = 3     # spikes and fires
CH_DOOR_OPEN = 4
CH_ENEMY = 5      # HP-weighted
CH_PROJECTILE = 6
CH_PICKUP = 7
CH_PLAYER = 8
CH_INSIDE = 9     # 1 for cells that belong to the room
GRID_CHANNELS = 10

# Cell codes in the mod's grid string
_CELL_TO_CHANNEL = {
    "1": CH_WALL,
    "2": CH_OBSTACLE,
    "3": CH_PIT,
    "4": CH_HAZARD,
    "5": CH_WALL,  # closed door
    "6": CH_DOOR_OPEN,
}

ENEMY_HP_SCALE = 20.0  # enemy cell value = hp / (hp + scale), so tougher enemies read brighter

N_ENEMIES = 8
N_PROJECTILES = 8
PLAYER_FEATURES = 18
ENEMY_FEATURES = 8       # dx, dy, vx, vy, hp fraction, vulnerable, boss, present
PROJECTILE_FEATURES = 5  # dx, dy, vx, vy, present
VEC_SIZE = PLAYER_FEATURES + N_ENEMIES * ENEMY_FEATURES + N_PROJECTILES * PROJECTILE_FEATURES

POS_SCALE = 400.0   # relative distances, roughly half a 1x1 room width
VEL_SCALE = 10.0
VEC_CLIP = 5.0


def observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(0.0, 1.0, (GRID_CHANNELS, GRID_H, GRID_W), np.float32),
            "vec": spaces.Box(-VEC_CLIP, VEC_CLIP, (VEC_SIZE,), np.float32),
        }
    )


def _to_cell(x: float, y: float, origin: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    # origin is the center of tile 0, so the nearest tile center is the floor of +0.5
    col = math.floor((x - origin[0]) / TILE + 0.5)
    row = math.floor((y - origin[1]) / TILE + 0.5)
    return min(max(row, 0), height - 1), min(max(col, 0), width - 1)


def encode_grid(state: dict[str, Any]) -> np.ndarray:
    room = state["room"]
    width, height = room["grid_width"], room["grid_height"]
    if width > GRID_W or height > GRID_H:
        raise ValueError(f"room grid {width}x{height} exceeds {GRID_W}x{GRID_H}")
    cells = room["grid"]
    if len(cells) != width * height:
        raise ValueError(f"grid string has {len(cells)} cells, expected {width * height}")

    grid = np.zeros((GRID_CHANNELS, GRID_H, GRID_W), np.float32)
    grid[CH_INSIDE, :height, :width] = 1.0
    codes = np.frombuffer(cells.encode("ascii"), np.uint8).reshape(height, width)
    for code, channel in _CELL_TO_CHANNEL.items():
        grid[channel, :height, :width][codes == ord(code)] = 1.0

    origin = tuple(room["grid_origin"])

    def place(channel: int, x: float, y: float, value: float = 1.0) -> None:
        r, c = _to_cell(x, y, origin, width, height)
        grid[channel, r, c] = min(1.0, grid[channel, r, c] + value)

    for e in state["enemies"]:
        hp = max(e[6], 0.0)
        place(CH_ENEMY, e[2], e[3], hp / (hp + ENEMY_HP_SCALE) if e[8] else 1.0)
    for p in state["projectiles"]:
        place(CH_PROJECTILE, p[0], p[1])
    for p in state["pickups"]:
        place(CH_PICKUP, p[2], p[3])
    for h in state["hazards"]:
        place(CH_HAZARD, h[2], h[3])
    px, py = state["player"]["pos"]
    place(CH_PLAYER, px, py)
    return grid


def encode_vec(state: dict[str, Any]) -> np.ndarray:
    player, room = state["player"], state["room"]
    px, py = player["pos"]
    ox, oy = room["grid_origin"]
    span_x = TILE * max(room["grid_width"] - 1, 1)
    span_y = TILE * max(room["grid_height"] - 1, 1)
    tears_per_second = 30.0 / (player["fire_delay"] + 1.0)

    features = [
        (px - ox) / span_x,
        (py - oy) / span_y,
        player["vel"][0] / VEL_SCALE,
        player["vel"][1] / VEL_SCALE,
        player["hearts"] / 24.0,
        player["max_hearts"] / 24.0,
        player["soul_hearts"] / 24.0,
        player["damage"] / 10.0,
        tears_per_second / 10.0,
        player["shot_speed"] / 2.0,
        player["move_speed"] / 2.0,
        player["range"] / 400.0,
        player["coins"] / 99.0,
        player["bombs"] / 99.0,
        player["keys"] / 99.0,
        1.0 if room["clear"] else 0.0,
        min(room["frame"] / 1800.0, 1.0),  # time in room, saturating at 60 s
        room["enemies"] / 20.0,
    ]
    assert len(features) == PLAYER_FEATURES

    vec = np.zeros(VEC_SIZE, np.float32)
    vec[:PLAYER_FEATURES] = features

    offset = PLAYER_FEATURES
    for i, e in enumerate(state["enemies"][:N_ENEMIES]):
        base = offset + i * ENEMY_FEATURES
        vec[base : base + ENEMY_FEATURES] = [
            (e[2] - px) / POS_SCALE,
            (e[3] - py) / POS_SCALE,
            e[4] / VEL_SCALE,
            e[5] / VEL_SCALE,
            e[6] / max(e[7], 1.0),
            e[8],
            e[9],
            1.0,
        ]

    offset += N_ENEMIES * ENEMY_FEATURES
    for i, p in enumerate(state["projectiles"][:N_PROJECTILES]):
        base = offset + i * PROJECTILE_FEATURES
        vec[base : base + PROJECTILE_FEATURES] = [
            (p[0] - px) / POS_SCALE,
            (p[1] - py) / POS_SCALE,
            p[2] / VEL_SCALE,
            p[3] / VEL_SCALE,
            1.0,
        ]

    return np.clip(vec, -VEC_CLIP, VEC_CLIP)


def encode(state: dict[str, Any]) -> dict[str, np.ndarray]:
    return {"grid": encode_grid(state), "vec": encode_vec(state)}
