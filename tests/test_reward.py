import pytest
from fake_game import make_state

from tboi_rl.reward import RewardConfig, compute_reward


def test_idle_step_only_pays_step_penalty():
    state = make_state()
    total, terms = compute_reward(state, make_state())
    assert total == pytest.approx(RewardConfig().step)
    assert {k for k, v in terms.items() if v} == {"step"}


def test_combat_terms():
    prev = make_state()
    curr = make_state(
        counters={"kills": 1, "damage_dealt": 10.0, "hits_taken": 1},
        player={"hearts": 5, "soul_hearts": 0},
    )
    total, terms = compute_reward(prev, curr)
    assert terms["damage_dealt"] == pytest.approx(0.5)
    assert terms["kill"] == 1.0
    assert terms["damage_taken"] == -1.0
    assert total == pytest.approx(sum(terms.values()))


def test_soul_hearts_count_as_health():
    prev = make_state(player={"hearts": 6, "soul_hearts": 2})
    curr = make_state(player={"hearts": 6, "soul_hearts": 1})
    assert compute_reward(prev, curr)[1]["damage_taken"] == -1.0


def test_healing_is_not_rewarded():
    prev = make_state(player={"hearts": 3})
    curr = make_state(player={"hearts": 6})
    assert compute_reward(prev, curr)[1]["damage_taken"] == 0.0


def test_one_off_events_fire_once():
    cfg = RewardConfig()
    prev = make_state()
    clear = make_state(room={"clear": True})
    assert compute_reward(prev, clear, cfg)[1]["room_clear"] == cfg.room_clear
    assert compute_reward(clear, clear, cfg)[1]["room_clear"] == 0.0

    dead = make_state(player={"dead": True, "hearts": 0})
    assert compute_reward(prev, dead, cfg)[1]["death"] == cfg.death
    assert compute_reward(dead, dead, cfg)[1]["death"] == 0.0


def test_custom_weights():
    cfg = RewardConfig(kill=5.0, step=0.0)
    curr = make_state(counters={"kills": 2, "damage_dealt": 0.0, "hits_taken": 0})
    total, _ = compute_reward(make_state(), curr, cfg)
    assert total == pytest.approx(10.0)
