import pytest
from fake_game import FakeGame
from gymnasium.utils.env_checker import check_env

from tboi_rl.env import IsaacEnv
from tboi_rl.protocol import BridgeError


@pytest.fixture
def env_and_game():
    env = IsaacEnv(rooms=["d.1", "d.2"], run_seed="JPCX AJ4R", max_steps=50, port=0,
                   connect_timeout=5, step_timeout=5)
    game = FakeGame(env.server.address[1])
    game.start()
    yield env, game
    env.close()
    game.join(timeout=5)
    assert game.error is None


def test_passes_gymnasium_checker(env_and_game):
    env, _ = env_and_game
    check_env(env, skip_render_check=True)


def test_shooting_clears_the_room(env_and_game):
    env, game = env_and_game
    _, info = env.reset(seed=0)
    assert info["room"] in ("d.1", "d.2") and info["enemies"] == 2

    total, terms_sum = 0.0, {}
    for _ in range(20):
        _, reward, terminated, truncated, info = env.step([0, 1])
        total += reward
        for k, v in info["reward_terms"].items():
            terms_sum[k] = terms_sum.get(k, 0.0) + v
        if terminated or truncated:
            break

    assert terminated and not truncated and info["room_clear"]
    assert terms_sum["kill"] == 2.0
    assert terms_sum["room_clear"] == 10.0
    assert terms_sum["damage_dealt"] == pytest.approx(20.0 * 0.05)

    reset_msgs = [m for m in game.received if m["type"] == "reset"]
    assert len(reset_msgs) == 1
    assert reset_msgs[0]["seed"] == "JPCX AJ4R" and reset_msgs[0]["room"] in ("d.1", "d.2")


def test_truncates_and_penalizes_damage(env_and_game):
    env, _ = env_and_game
    env.reset(seed=1)
    damage = 0.0
    for _ in range(50):
        _, _, terminated, truncated, info = env.step([3, 0])  # never shoots
        damage += info["reward_terms"]["damage_taken"]
    assert truncated and not terminated
    assert damage == -1.0  # one hit every 40 steps


def test_counters_carry_across_episodes(env_and_game):
    env, _ = env_and_game
    for _ in range(2):
        env.reset()
        _, reward, *_ = env.step([0, 1])
        assert reward == pytest.approx(3.5 * 0.05 - 0.005)


def test_unknown_room_raises(env_and_game):
    env, _ = env_and_game
    with pytest.raises(BridgeError, match="timed out"):
        env.reset(options={"room": "d.999"})
    assert env.server.last_log == "goto d.999: room not found"


def test_step_before_reset_fails():
    env = IsaacEnv(rooms=["d.1"], port=0)
    try:
        with pytest.raises(RuntimeError):
            env.step([0, 0])
    finally:
        env.close()
