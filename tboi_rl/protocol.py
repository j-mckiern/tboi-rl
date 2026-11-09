"""TCP bridge between Python and the TBOI-RL Lua mod.

The game connects to a server opened here and exchanges newline-delimited
JSON messages in lock-step: the game sends a ``state`` message, then waits for
an ``action`` or ``reset`` reply before advancing.
"""

from __future__ import annotations

import json
import re
import socket
import time
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5577
PROTOCOL_VERSION = 1

NUM_MOVES = 9   # no-op + 8 directions, clockwise from up
NUM_SHOOTS = 5  # no-shoot + up, right, down, left

_SEED_RE = re.compile(r"^[A-Z0-9]{4} [A-Z0-9]{4}$")


class BridgeError(RuntimeError):
    """Raised when the connection to the game is lost or misbehaves."""


def encode(msg: dict[str, Any]) -> bytes:
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes) -> dict[str, Any]:
    msg = json.loads(line.decode("utf-8"))
    if not isinstance(msg, dict) or "type" not in msg:
        raise BridgeError(f"malformed message: {line!r}")
    return msg


def action_message(move: int, shoot: int) -> dict[str, Any]:
    if not 0 <= move < NUM_MOVES:
        raise ValueError(f"move must be in [0, {NUM_MOVES}), got {move}")
    if not 0 <= shoot < NUM_SHOOTS:
        raise ValueError(f"shoot must be in [0, {NUM_SHOOTS}), got {shoot}")
    return {"type": "action", "move": int(move), "shoot": int(shoot)}


def reset_message(seed: str | None = None) -> dict[str, Any]:
    """Restart the run. ``seed`` uses the in-game format, e.g. ``"JPCX AJ4R"``.

    Seeds include a checksum, so arbitrary strings are rejected by the game;
    use seeds shown in-game or reported in ``state`` messages.
    """
    msg: dict[str, Any] = {"type": "reset"}
    if seed is not None:
        seed = seed.strip().upper()
        if not _SEED_RE.match(seed):
            raise ValueError(f"seed must look like 'ABCD 1234', got {seed!r}")
        msg["seed"] = seed
    return msg


class BridgeServer:
    """Listens for the game and exchanges messages with it."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self._listener = socket.create_server((host, port))
        self._conn: socket.socket | None = None
        self._reader = None
        self.address = self._listener.getsockname()

    def wait_for_game(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the game connects and return its ``hello`` message.

        Waits in short slices so Ctrl+C is handled promptly on Windows, where
        a blocking ``accept()`` ignores it.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        self._listener.settimeout(0.25)
        while True:
            try:
                conn, _ = self._listener.accept()
                break
            except socket.timeout:
                if deadline is not None and time.monotonic() >= deadline:
                    raise
        conn.settimeout(None)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn = conn
        self._reader = conn.makefile("rb")
        hello = self.recv()
        if hello.get("type") != "hello":
            raise BridgeError(f"expected hello, got {hello!r}")
        if hello.get("protocol") != PROTOCOL_VERSION:
            raise BridgeError(
                f"protocol mismatch: game={hello.get('protocol')} python={PROTOCOL_VERSION}"
            )
        return hello

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        if self._conn is None or self._reader is None:
            raise BridgeError("not connected")
        self._conn.settimeout(timeout)
        line = self._reader.readline()
        if not line:
            raise BridgeError("game closed the connection")
        msg = decode(line)
        if msg["type"] == "error":
            raise BridgeError(f"game reported an error: {msg.get('message')}")
        return msg

    def send(self, msg: dict[str, Any]) -> None:
        if self._conn is None:
            raise BridgeError("not connected")
        self._conn.sendall(encode(msg))

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        if self._conn is not None:
            self._conn.close()
        self._listener.close()
        self._conn = self._reader = None

    def __enter__(self) -> BridgeServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
