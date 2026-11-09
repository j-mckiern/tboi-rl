import pytest
from fake_game import make_state

from tboi_rl.policies import (
    SHOOT_DOWN,
    SHOOT_LEFT,
    SHOOT_RIGHT,
    SHOOT_UP,
    RandomPolicy,
    ScriptedPolicy,
    direction_to_move,
    shoot_towards,
)


@pytest.mark.parametrize(
    "dx,dy,move",
    [(0, 0, 0), (0, -1, 1), (1, -1, 2), (1, 0, 3), (1, 1, 4), (0, 1, 5), (-1, 1, 6), (-1, 0, 7),
     (-1, -1, 8), (10, -1, 3), (10, -6, 2)],
)
def test_direction_to_move(dx, dy, move):
    assert direction_to_move(dx, dy) == move


def test_shoot_towards():
    assert shoot_towards(5, 1) == SHOOT_RIGHT
    assert shoot_towards(-5, 1) == SHOOT_LEFT
    assert shoot_towards(1, 5) == SHOOT_DOWN
    assert shoot_towards(1, -5) == SHOOT_UP


def _state(enemies, projectiles=()):
    return make_state(enemies=[list(e) for e in enemies], projectiles=[list(p) for p in projectiles])


def test_scripted_lines_up_and_shoots():
    # Enemy far to the right and a bit below: shoot right, move down to line up.
    state = _state([[10, 0, 600, 320, 0, 0, 10, 10, 1, 0]])
    assert ScriptedPolicy()(state) == (5, SHOOT_RIGHT)


def test_scripted_backs_off_when_close():
    state = _state([[10, 0, 360, 280, 0, 0, 10, 10, 1, 0]])  # 40 units to the right
    assert ScriptedPolicy()(state) == (7, SHOOT_RIGHT)


def test_scripted_ignores_invulnerable_enemies():
    state = _state([[44, 0, 360, 280, 0, 0, 10, 10, 0, 0]])
    assert ScriptedPolicy()(state) == (0, 0)


def test_scripted_dodges_incoming_projectile():
    # Projectile left of the player flying right at it, player slightly below its path.
    state = _state([], projectiles=[[240, 270, 8, 0]])
    move, _ = ScriptedPolicy()(state)
    assert move == 5  # sidestep down


def test_scripted_ignores_projectiles_moving_away():
    state = _state([], projectiles=[[240, 280, -8, 0]])
    assert ScriptedPolicy()(state) == (0, 0)


def test_random_policy_holds_actions():
    policy = RandomPolicy(seed=0, hold=4)
    actions = [policy(make_state()) for _ in range(8)]
    assert actions[0] == actions[1] == actions[2] == actions[3]
    assert all(0 <= m < 9 and 0 <= s < 5 for m, s in actions)
