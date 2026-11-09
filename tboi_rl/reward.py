"""Reward shaping for room combat.

Each term is computed and returned separately so they can be logged
individually; reward hacking usually shows up as one term behaving oddly
long before it shows up in the total.

Damage, kills and hits come from running counters kept by the mod, so a
reward is only paid for events that actually happened in the game.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    damage_dealt: float = 0.05    # per enemy HP removed (a base tear does 3.5)
    kill: float = 1.0             # per enemy killed
    room_clear: float = 10.0      # once, when the room becomes clear
    damage_taken: float = -1.0    # per half heart lost (red or soul)
    death: float = -10.0          # once, on death
    step: float = -0.005          # every step, discourages stalling

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _health(state: dict[str, Any]) -> int:
    player = state["player"]
    return player["hearts"] + player["soul_hearts"]


def compute_reward(
    prev: dict[str, Any], curr: dict[str, Any], config: RewardConfig = RewardConfig()
) -> tuple[float, dict[str, float]]:
    """Reward for the transition ``prev -> curr``, plus each term on its own."""
    pc, cc = prev["counters"], curr["counters"]
    newly_clear = curr["room"]["clear"] and not prev["room"]["clear"]
    died = curr["player"]["dead"] and not prev["player"]["dead"]

    terms = {
        "damage_dealt": config.damage_dealt * max(cc["damage_dealt"] - pc["damage_dealt"], 0.0),
        "kill": config.kill * max(cc["kills"] - pc["kills"], 0),
        "room_clear": config.room_clear if newly_clear else 0.0,
        "damage_taken": config.damage_taken * max(_health(prev) - _health(curr), 0),
        "death": config.death if died else 0.0,
        "step": config.step,
    }
    return float(sum(terms.values())), terms
