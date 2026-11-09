"""Compiles the mod's Lua so syntax errors are caught without launching the game."""

from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

MOD_DIR = Path(__file__).resolve().parent.parent / "mod" / "tboi_rl_bridge"


@pytest.mark.parametrize("path", sorted(MOD_DIR.rglob("*.lua")), ids=lambda p: p.name)
def test_lua_compiles(path):
    lua = lupa.LuaRuntime()
    load = lua.eval("function(src, name) local f, err = load(src, name) return f ~= nil, err end")
    ok, err = load(path.read_text(encoding="utf-8"), "@" + path.name)
    assert ok, err
