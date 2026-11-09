import json
import socket
import threading

import pytest

from tboi_rl.protocol import (
    BridgeError,
    BridgeServer,
    action_message,
    decode,
    encode,
    reset_message,
)


def test_encode_decode_round_trip():
    msg = {"type": "state", "step": 3, "player": {"pos": [1.5, 2.0]}}
    line = encode(msg)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert decode(line) == msg


def test_decode_rejects_untyped_message():
    with pytest.raises(BridgeError):
        decode(b'{"step": 1}\n')


@pytest.mark.parametrize("move,shoot", [(-1, 0), (9, 0), (0, 5)])
def test_action_message_validates_range(move, shoot):
    with pytest.raises(ValueError):
        action_message(move, shoot)


def test_reset_message_normalizes_seed():
    assert reset_message("abcd 1234") == {"type": "reset", "seed": "ABCD 1234"}
    assert reset_message() == {"type": "reset"}
    with pytest.raises(ValueError):
        reset_message("ABCD1234")


def _fake_game(port: int, received: list[dict]) -> None:
    """Behaves like the Lua mod: hello, then one state/reply exchange."""
    with socket.create_connection(("127.0.0.1", port)) as conn:
        reader = conn.makefile("rb")
        conn.sendall(encode({"type": "hello", "protocol": 2, "frame_skip": 2}))
        conn.sendall(encode({"type": "state", "step": 0, "new_run": True}))
        received.append(json.loads(reader.readline()))


def test_server_lock_step_exchange():
    with BridgeServer(port=0) as server:
        received: list[dict] = []
        game = threading.Thread(target=_fake_game, args=(server.address[1], received))
        game.start()

        hello = server.wait_for_game(timeout=5)
        assert hello["frame_skip"] == 2
        state = server.recv(timeout=5)
        assert state["type"] == "state" and state["new_run"] is True
        server.send(action_message(3, 1))

        game.join(timeout=5)
        assert received == [{"type": "action", "move": 3, "shoot": 1}]

        with pytest.raises(BridgeError):
            server.recv(timeout=5)  # game hung up


def test_server_raises_on_game_error():
    with BridgeServer(port=0) as server:
        def game():
            with socket.create_connection(("127.0.0.1", server.address[1])) as conn:
                conn.sendall(encode({"type": "hello", "protocol": 2, "frame_skip": 2}))
                conn.sendall(encode({"type": "error", "message": "invalid seed: XXXX XXXX"}))
                conn.recv(1)  # stay connected until the server closes

        thread = threading.Thread(target=game)
        thread.start()
        server.wait_for_game(timeout=5)
        with pytest.raises(BridgeError, match="invalid seed"):
            server.recv(timeout=5)
        server.close()
        thread.join(timeout=5)
