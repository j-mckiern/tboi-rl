import numpy as np
import pytest
from fake_game import GRID_H, GRID_W, make_state

from tboi_rl import obs


def test_shapes_match_observation_space():
    encoded = obs.encode(make_state())
    space = obs.observation_space()
    assert encoded["grid"].shape == space["grid"].shape
    assert encoded["vec"].shape == space["vec"].shape
    assert space.contains(encoded)


def test_grid_layout_and_padding():
    grid = obs.encode_grid(make_state())
    inside = grid[obs.CH_INSIDE]
    assert inside[:GRID_H, :GRID_W].all()
    assert not inside[GRID_H:, :].any() and not inside[:, GRID_W:].any()

    walls = grid[obs.CH_WALL]
    assert walls[0, :GRID_W].all() and walls[GRID_H - 1, :GRID_W].all()  # includes the closed door
    assert not walls[1 : GRID_H - 1, 1 : GRID_W - 1].any()
    assert grid[obs.CH_OBSTACLE, GRID_H // 2, GRID_W // 2 - 3] == 1.0
    assert grid[obs.CH_OBSTACLE].sum() == 1.0


def test_entities_land_in_the_right_cells():
    grid = obs.encode_grid(make_state())
    # (x, y) -> (row, col) with tile 0 centered at (40, 120) and 40-unit tiles
    assert grid[obs.CH_PLAYER, 4, 7] == 1.0      # (320, 280): room center
    assert grid[obs.CH_PLAYER].sum() == 1.0
    assert grid[obs.CH_ENEMY, 2, 4] == pytest.approx(10 / 30)  # 10 HP enemy at (200, 200)
    assert grid[obs.CH_ENEMY, 6, 10] > 0          # (440, 360)
    assert grid[obs.CH_PROJECTILE, 4, 6] == 1.0  # (290, 280)
    assert grid[obs.CH_PICKUP, 1, 2] == 1.0      # (120, 160)


def test_open_doors_and_out_of_bounds_entities():
    state = make_state()
    state["room"]["grid"] = state["room"]["grid"].replace("5", "6")
    state["enemies"].append([10, 0, -500.0, 9000.0, 0, 0, 5.0, 5.0, 1, 0])  # clamps to an edge cell
    grid = obs.encode_grid(state)
    assert grid[obs.CH_DOOR_OPEN, 0, GRID_W // 2] == 1.0
    assert grid[obs.CH_ENEMY, GRID_H - 1, 0] > 0


def test_vector_features():
    vec = obs.encode_vec(make_state())
    assert vec[obs.PLAYER_FEATURES - 1] == pytest.approx(2 / 20)  # enemy count

    first_enemy = vec[obs.PLAYER_FEATURES : obs.PLAYER_FEATURES + obs.ENEMY_FEATURES]
    np.testing.assert_allclose(first_enemy, [-0.3, -0.2, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0], atol=1e-6)
    third_enemy_present = obs.PLAYER_FEATURES + 3 * obs.ENEMY_FEATURES - 1
    assert vec[third_enemy_present] == 0.0

    proj = obs.PLAYER_FEATURES + obs.N_ENEMIES * obs.ENEMY_FEATURES
    np.testing.assert_allclose(vec[proj : proj + 5], [-0.075, 0.0, 0.2, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(vec[:2], [0.5, 0.5], atol=1e-6)  # player at the room center


def test_rejects_malformed_grid():
    state = make_state()
    state["room"]["grid"] = state["room"]["grid"][:-1]
    with pytest.raises(ValueError):
        obs.encode_grid(state)
