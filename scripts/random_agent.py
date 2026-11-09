"""Drive the game with uniformly random actions to test the bridge.

Start this script, then launch the game with --luadebug and the TBOI-RL Bridge
mod enabled, and start a run. The agent restarts the run on death or after a
fixed number of steps, and checks that the step counter from the game never
skips (which would mean the two sides are out of sync).

    uv run python scripts/random_agent.py --episodes 3 --seed "JPCX AJ4R"
"""

from __future__ import annotations

import argparse
import random
import time

from tboi_rl.protocol import (
    NUM_MOVES,
    NUM_SHOOTS,
    BridgeError,
    BridgeServer,
    action_message,
    reset_message,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600, help="steps per episode before a reset")
    parser.add_argument("--seed", default=None, help="run seed, e.g. 'JPCX AJ4R' (default: random)")
    parser.add_argument("--hold", type=int, default=5, help="repeat each random action for this many steps")
    args = parser.parse_args()

    rng = random.Random(0)
    with BridgeServer(port=args.port) as server:
        print(f"Waiting for the game on {server.address[0]}:{server.address[1]} ...")
        hello = server.wait_for_game()
        print(f"Game connected: {hello}")

        # Episodes only count from a reset requested here, so whatever run was
        # already in progress when the game connected is discarded.
        episode, ep_steps, last_step = 0, 0, None
        in_episode = reset_requested = False
        move = shoot = 0
        t0, steps_since_report = time.perf_counter(), 0

        while True:
            msg = server.recv(timeout=30)
            if msg["type"] != "state":
                print(f"Ignoring message: {msg}")
                continue

            if last_step is not None and msg["step"] != last_step + 1:
                raise BridgeError(f"desync: step {last_step} followed by {msg['step']}")
            last_step = msg["step"]

            if not in_episode:
                if not reset_requested:
                    server.send(reset_message(args.seed))
                    reset_requested = True
                    continue
                if not msg["new_run"]:
                    raise BridgeError("expected the first state of a new run after a reset")
                episode += 1
                ep_steps = 0
                in_episode, reset_requested = True, False
                print(f"Episode {episode} started (seed {msg['seed']})")

            player = msg["player"]
            if player["dead"] or ep_steps >= args.max_steps:
                print(
                    f"Episode {episode} ended after {ep_steps} steps "
                    f"(dead={player['dead']}, hearts={player['hearts']}, room={msg['room']['index']})"
                )
                in_episode = False
                if episode >= args.episodes:
                    server.send(action_message(0, 0))
                    break
                server.send(reset_message(args.seed))
                reset_requested = True
                continue

            if ep_steps % args.hold == 0:
                move, shoot = rng.randrange(NUM_MOVES), rng.randrange(NUM_SHOOTS)
            server.send(action_message(move, shoot))
            ep_steps += 1
            steps_since_report += 1

            elapsed = time.perf_counter() - t0
            if elapsed >= 5:
                print(
                    f"  {steps_since_report / elapsed:5.1f} steps/s | step {msg['step']} | "
                    f"pos ({player['pos'][0]:.0f}, {player['pos'][1]:.0f}) | hearts {player['hearts']}"
                )
                t0, steps_since_report = time.perf_counter(), 0

        print("Done.")


if __name__ == "__main__":
    main()
