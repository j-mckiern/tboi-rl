"""Evaluate a baseline policy on combat rooms and report reference scores.

    uv run python scripts/baselines.py --policy scripted --episodes 10 --seed "JPCX AJ4R"
    uv run python scripts/baselines.py --policy random --rooms d.5 d.12

Rooms default to the combat rooms in configs/basement_rooms.json (written by
scripts/scan_rooms.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tboi_rl.env import IsaacEnv
from tboi_rl.policies import RandomPolicy, ScriptedPolicy


def load_rooms(path: Path, limit: int | None) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rooms = [r["room"] for r in data["rooms"] if r["combat"]]
    return rooms[:limit] if limit else rooms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=["random", "scripted"], default="scripted")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--rooms", nargs="*", help="rooms in goto syntax, e.g. d.5 d.12")
    parser.add_argument("--rooms-file", type=Path, default=Path("configs/basement_rooms.json"))
    parser.add_argument("--num-rooms", type=int, default=None, help="use only the first N rooms from the file")
    parser.add_argument("--seed", default=None, help="run seed for every episode (default: random)")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--port", type=int, default=5577)
    args = parser.parse_args()

    rooms = args.rooms or load_rooms(args.rooms_file, args.num_rooms)
    env = IsaacEnv(rooms, run_seed=args.seed, max_steps=args.max_steps, port=args.port)
    policy = RandomPolicy(seed=0) if args.policy == "random" else ScriptedPolicy()
    print(f"Evaluating {args.policy} policy on {len(rooms)} rooms; waiting for the game ...")

    returns, clears, deaths, damage, lengths = [], [], [], [], []
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=episode)
            room = info["room"]
            total, hurt, done = 0.0, 0.0, False
            while not done:
                _, reward, terminated, truncated, step_info = env.step(policy(env.last_state))
                total += reward
                hurt -= step_info["reward_terms"]["damage_taken"]
                done = terminated or truncated

            returns.append(total)
            clears.append(step_info["room_clear"])
            deaths.append(step_info["dead"])
            damage.append(hurt)
            lengths.append(step_info["steps"])
            outcome = "cleared" if step_info["room_clear"] else "died" if step_info["dead"] else (
                "left room" if step_info["left_room"] else "timed out")
            print(f"  episode {episode + 1:>3} {room:>7}: return {total:7.2f}, {outcome}, "
                  f"{step_info['steps']} steps, {hurt:.0f} half-hearts lost")
    finally:
        env.close()

    print(
        f"\n{args.policy}: mean return {np.mean(returns):.2f} ± {np.std(returns):.2f} | "
        f"clear rate {np.mean(clears):.0%} | death rate {np.mean(deaths):.0%} | "
        f"half-hearts lost {np.mean(damage):.1f} | mean length {np.mean(lengths):.0f} steps"
    )


if __name__ == "__main__":
    main()
