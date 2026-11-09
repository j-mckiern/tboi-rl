"""Find Basement rooms that are usable for combat training.

Teleports through normal room variants ``d.<start>`` .. ``d.<end>`` on the
first floor of a fixed seed, records what each room contains, and writes the
results to a JSON file. Variants that don't exist on the floor time out after
about a second and are skipped.

    uv run python scripts/scan_rooms.py --seed "JPCX AJ4R" --start 1 --end 300
"""

from __future__ import annotations

import argparse
import json
import socket
from collections import Counter
from pathlib import Path

from tboi_rl.protocol import BridgeError, BridgeServer, reset_message


def recv_state(server: BridgeServer) -> dict:
    while True:
        msg = server.recv(timeout=30)
        if msg["type"] == "state":
            return msg


def format_result(result: dict) -> str:
    """JSON with one room per line, so the file stays short and diffs stay readable."""
    header = {k: v for k, v in result.items() if k != "rooms"}
    lines = ["{"]
    lines += [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in header.items()]
    lines.append('  "rooms": [')
    lines.append(",\n".join(f"    {json.dumps(room)}" for room in result["rooms"]))
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", required=True, help="run seed for the scan, e.g. 'JPCX AJ4R'")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("configs/basement_rooms.json"))
    parser.add_argument("--port", type=int, default=5577)
    args = parser.parse_args()

    rooms = []
    with BridgeServer(port=args.port) as server:
        print(f"Waiting for the game on port {args.port} (start a run) ...")
        server.wait_for_game()
        recv_state(server)

        first = True
        for variant in range(args.start, args.end + 1):
            name = f"d.{variant}"
            # Restart once with the seed, then only teleport (much faster).
            server.send(reset_message(args.seed, name, restart=first))
            try:
                state = recv_state(server)
            except (BridgeError, socket.timeout) as exc:
                print(f"{name:>7}: skipped ({exc})")
                recv_state(server)  # the game resumes sending states after an error
                continue
            first = False

            room = state["room"]
            enemy_types = Counter(f"{e[0]}.{e[1]}" for e in state["enemies"])
            entry = {
                "room": name,
                "shape": room["shape"],
                "grid": [room["grid_width"], room["grid_height"]],
                "enemies": room["enemies"],
                "enemy_types": dict(sorted(enemy_types.items())),
                "hazards": len(state["hazards"]),
                "combat": room["enemies"] > 0 and not room["clear"],
            }
            rooms.append(entry)
            print(f"{name:>7}: shape {entry['shape']:>2}, {entry['enemies']:>2} enemies {entry['enemy_types']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"seed": args.seed, "stage": 1, "rooms": rooms}
    args.out.write_text(format_result(result), encoding="utf-8")
    combat = sum(r["combat"] for r in rooms)
    print(f"Found {len(rooms)} rooms ({combat} with enemies); wrote {args.out}")


if __name__ == "__main__":
    main()
