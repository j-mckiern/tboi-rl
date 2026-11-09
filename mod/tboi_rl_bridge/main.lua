-- TBOI-RL Bridge
--
-- Streams game state to an external Python process over TCP and applies the
-- actions it sends back. The game acts as the client; Python listens on
-- HOST:PORT. Messages are newline-delimited JSON.
--
-- Lock-step protocol: every FRAME_SKIP game updates the mod sends a "state"
-- message and blocks until Python replies with an "action" or "reset". The
-- action is held for the following FRAME_SKIP updates. Because the game waits
-- for the agent, the environment is deterministic for a given seed and action
-- sequence, regardless of how long the agent takes to decide.
--
-- Requires the game to be launched with --luadebug, which makes the bundled
-- LuaSocket library available to mods.

local mod = RegisterMod("TBOI-RL Bridge", 1)
local json = require("json")

local HOST = "127.0.0.1"
local PORT = 5577
local PROTOCOL_VERSION = 2
local FRAME_SKIP = 2            -- game updates (30 per second) per agent step
local RECV_TIMEOUT = 300        -- seconds to wait for the agent (covers PPO updates) before dropping it
local RECONNECT_INTERVAL = 60   -- render frames (60 per second) between connection attempts
local RESET_TIMEOUT = 300       -- render frames to wait for a requested restart to finish
local GOTO_TIMEOUT = 60         -- render frames to wait for a teleport (fails fast on unknown rooms)

-- Per-message caps on entity lists, nearest to the player first.
local MAX_ENEMIES = 32
local MAX_PROJECTILES = 64
local MAX_PICKUPS = 16
local MAX_HAZARDS = 16

local hasSocket, socket = pcall(require, "socket")
if hasSocket then
    Isaac.DebugString("[TBOI-RL] loaded, LuaSocket " .. tostring(socket._VERSION))
else
    Isaac.DebugString("[TBOI-RL] loaded, LuaSocket unavailable: " .. tostring(socket))
end

local client = nil
local lastConnectAttempt = -RECONNECT_INTERVAL
local lastConnectError = nil
local updateCounter = 0
local stepCount = 0
local statusText = ""

-- Reset in progress: { phase = "restart" | "goto", room = string|nil, startedAt = frame }
local reset = nil
local newRun = false         -- next state message is the first of a new episode
local pendingCommand = nil   -- console command to run on the next render frame

-- Running totals since the mod loaded. Python takes differences between
-- consecutive states, so these never need resetting.
local counters = { kills = 0, damage_dealt = 0, hits_taken = 0 }

-- Current action, held between agent steps.
local currentMove = 0
local currentShoot = 0

-- Movement index -> (x, y) direction. 0 = no movement, then clockwise from up.
local MOVE_DIRS = {
    [0] = { 0, 0 },
    [1] = { 0, -1 },  -- up
    [2] = { 1, -1 },  -- up-right
    [3] = { 1, 0 },   -- right
    [4] = { 1, 1 },   -- down-right
    [5] = { 0, 1 },   -- down
    [6] = { -1, 1 },  -- down-left
    [7] = { -1, 0 },  -- left
    [8] = { -1, -1 }, -- up-left
}

-- Shooting index -> ButtonAction. 0 = not shooting.
local SHOOT_ACTIONS = {
    [1] = ButtonAction.ACTION_SHOOTUP,
    [2] = ButtonAction.ACTION_SHOOTRIGHT,
    [3] = ButtonAction.ACTION_SHOOTDOWN,
    [4] = ButtonAction.ACTION_SHOOTLEFT,
}

local CONTROLLED_ACTIONS = {
    [ButtonAction.ACTION_LEFT] = true,
    [ButtonAction.ACTION_RIGHT] = true,
    [ButtonAction.ACTION_UP] = true,
    [ButtonAction.ACTION_DOWN] = true,
    [ButtonAction.ACTION_SHOOTLEFT] = true,
    [ButtonAction.ACTION_SHOOTRIGHT] = true,
    [ButtonAction.ACTION_SHOOTUP] = true,
    [ButtonAction.ACTION_SHOOTDOWN] = true,
}

-- One character per grid cell, row-major. Decoded by tboi_rl/obs.py.
local CELL_EMPTY = "0"
local CELL_WALL = "1"
local CELL_OBSTACLE = "2"
local CELL_PIT = "3"
local CELL_SPIKES = "4"
local CELL_DOOR_CLOSED = "5"
local CELL_DOOR_OPEN = "6"

---------------------------------------------------------------------------
-- Connection handling
---------------------------------------------------------------------------

local function disconnect(reason)
    if client then
        client:close()
        client = nil
    end
    currentMove, currentShoot = 0, 0
    reset = nil
    statusText = "disconnected (" .. tostring(reason) .. ")"
    Isaac.DebugString("[TBOI-RL] disconnected: " .. tostring(reason))
end

local function send(msg)
    if not client then return false end
    local ok, err = client:send(json.encode(msg) .. "\n")
    if not ok then
        disconnect("send failed: " .. tostring(err))
        return false
    end
    return true
end

local function tryConnect()
    local now = Isaac.GetFrameCount()
    if now - lastConnectAttempt < RECONNECT_INTERVAL then return end
    lastConnectAttempt = now

    local c, err = socket.tcp()
    local ok = false
    if c then
        c:settimeout(0.05)
        ok, err = c:connect(HOST, PORT)
    end
    if not ok then
        if c then c:close() end
        statusText = "waiting for agent on " .. HOST .. ":" .. PORT .. " (" .. tostring(err) .. ")"
        if err ~= lastConnectError then
            lastConnectError = err
            Isaac.DebugString("[TBOI-RL] connect failed: " .. tostring(err))
        end
        return
    end
    lastConnectError = nil
    c:settimeout(RECV_TIMEOUT)
    c:setoption("tcp-nodelay", true)
    client = c
    stepCount = 0
    statusText = "connected"
    Isaac.DebugString("[TBOI-RL] connected to agent")
    send({ type = "hello", protocol = PROTOCOL_VERSION, frame_skip = FRAME_SKIP })
end

---------------------------------------------------------------------------
-- State serialization
---------------------------------------------------------------------------

local function round(v)
    return math.floor(v * 100 + 0.5) / 100
end

-- Encodes the room layout as one character per grid cell (see CELL_*).
local function encodeGrid(room)
    local cells = {}
    for i = 0, room:GetGridSize() - 1 do
        local cell = CELL_EMPTY
        local ge = room:GetGridEntity(i)
        local gtype = ge and ge:GetType() or nil
        if gtype == GridEntityType.GRID_DOOR then
            local door = ge:ToDoor()
            cell = (door and door:IsOpen()) and CELL_DOOR_OPEN or CELL_DOOR_CLOSED
        elseif gtype == GridEntityType.GRID_SPIKES or gtype == GridEntityType.GRID_SPIKES_ONOFF then
            cell = CELL_SPIKES
        else
            local coll = room:GetGridCollision(i)
            if coll == GridCollisionClass.COLLISION_WALL or coll == GridCollisionClass.COLLISION_WALL_EXCEPT_PLAYER then
                cell = CELL_WALL
            elseif coll == GridCollisionClass.COLLISION_PIT then
                cell = CELL_PIT
            elseif coll == GridCollisionClass.COLLISION_OBJECT or coll == GridCollisionClass.COLLISION_SOLID then
                cell = CELL_OBSTACLE
            end
        end
        cells[#cells + 1] = cell
    end
    return table.concat(cells)
end

-- Keeps the `limit` entries closest to the player.
local function nearest(list, limit)
    table.sort(list, function(a, b) return a.dist < b.dist end)
    local out = {}
    for i = 1, math.min(#list, limit) do
        out[i] = list[i].data
    end
    return out
end

local function collectEntities(playerPos)
    local enemies, projectiles, pickups, hazards = {}, {}, {}, {}
    for _, e in ipairs(Isaac.GetRoomEntities()) do
        local p, v = e.Position, e.Velocity
        local dist = p:Distance(playerPos)
        local friendly = e:HasEntityFlags(EntityFlag.FLAG_FRIENDLY)
        if e.Type == EntityType.ENTITY_PROJECTILE then
            projectiles[#projectiles + 1] = { dist = dist, data = { round(p.X), round(p.Y), round(v.X), round(v.Y) } }
        elseif e.Type == EntityType.ENTITY_PICKUP then
            pickups[#pickups + 1] = { dist = dist, data = { e.Variant, e.SubType, round(p.X), round(p.Y) } }
        elseif e.Type == EntityType.ENTITY_FIREPLACE then
            if e.HitPoints > 1 then  -- extinguished fires keep 1 HP
                hazards[#hazards + 1] = { dist = dist, data = { e.Type, e.Variant, round(p.X), round(p.Y) } }
            end
        elseif e:IsActiveEnemy(false) and not friendly then
            enemies[#enemies + 1] = { dist = dist, data = {
                e.Type, e.Variant, round(p.X), round(p.Y), round(v.X), round(v.Y),
                round(e.HitPoints), round(e.MaxHitPoints),
                e:IsVulnerableEnemy() and 1 or 0, e:IsBoss() and 1 or 0,
            } }
        end
    end
    return nearest(enemies, MAX_ENEMIES), nearest(projectiles, MAX_PROJECTILES),
        nearest(pickups, MAX_PICKUPS), nearest(hazards, MAX_HAZARDS)
end

local function getState()
    local game = Game()
    local level = game:GetLevel()
    local room = game:GetRoom()
    local player = Isaac.GetPlayer(0)
    local origin = room:GetGridPosition(0)
    local enemies, projectiles, pickups, hazards = collectEntities(player.Position)

    return {
        type = "state",
        step = stepCount,
        frame = game:GetFrameCount(),
        new_run = newRun,
        seed = game:GetSeeds():GetStartSeedString(),
        stage = level:GetStage(),
        counters = { kills = counters.kills, damage_dealt = round(counters.damage_dealt), hits_taken = counters.hits_taken },
        player = {
            pos = { round(player.Position.X), round(player.Position.Y) },
            vel = { round(player.Velocity.X), round(player.Velocity.Y) },
            hearts = player:GetHearts(),
            max_hearts = player:GetMaxHearts(),
            soul_hearts = player:GetSoulHearts(),
            dead = player:IsDead(),
            damage = round(player.Damage),
            fire_delay = round(player.MaxFireDelay),
            shot_speed = round(player.ShotSpeed),
            move_speed = round(player.MoveSpeed),
            range = round(player.TearRange),
            coins = player:GetNumCoins(),
            bombs = player:GetNumBombs(),
            keys = player:GetNumKeys(),
            collectibles = player:GetCollectibleCount(),
        },
        room = {
            index = level:GetCurrentRoomIndex(),
            type = room:GetType(),
            shape = room:GetRoomShape(),
            clear = room:IsClear(),
            frame = room:GetFrameCount(),
            enemies = room:GetAliveEnemiesCount(),
            grid_width = room:GetGridWidth(),
            grid_height = room:GetGridHeight(),
            grid_origin = { round(origin.X), round(origin.Y) },
            grid = encodeGrid(room),
        },
        -- Entity rows; field order is documented in tboi_rl/obs.py.
        enemies = enemies,
        projectiles = projectiles,
        pickups = pickups,
        hazards = hazards,
    }
end

---------------------------------------------------------------------------
-- Resets
---------------------------------------------------------------------------

local function finishReset()
    reset = nil
    newRun = true
    updateCounter = 0
    currentMove, currentShoot = 0, 0
end

-- reset message fields:
--   seed    (optional) restart with this seed, e.g. "JPCX AJ4R"
--   room    (optional) then teleport to this room, e.g. "d.12" (debug "goto" syntax)
--   restart (optional, default true) false = only teleport, keep the current run
local function startReset(msg)
    currentMove, currentShoot = 0, 0
    local seed = msg.seed
    if seed ~= nil and seed ~= "" and not Seeds.IsStringValidSeed(tostring(seed)) then
        send({ type = "error", message = "invalid seed: " .. tostring(seed) })
        return
    end
    local room = msg.room
    if room ~= nil and (type(room) ~= "string" or room:find("[%s;]")) then
        send({ type = "error", message = "invalid room: " .. tostring(room) })
        return
    end

    reset = { room = room, startedAt = Isaac.GetFrameCount() }
    if msg.restart == false then
        if not room then
            reset = nil
            send({ type = "error", message = "reset with restart=false needs a room" })
            return
        end
        reset.phase = "goto"
        pendingCommand = "goto " .. room
    else
        reset.phase = "restart"
        pendingCommand = (seed and seed ~= "") and ("seed " .. seed) or "restart"
    end
end

---------------------------------------------------------------------------
-- Message handling
---------------------------------------------------------------------------

local function clampIndex(v, maxValue)
    v = math.floor(tonumber(v) or 0)
    if v < 0 or v > maxValue then return 0 end
    return v
end

local function handleMessage(msg)
    if msg.type == "action" then
        currentMove = clampIndex(msg.move, 8)
        currentShoot = clampIndex(msg.shoot, 4)
    elseif msg.type == "reset" then
        startReset(msg)
    else
        Isaac.DebugString("[TBOI-RL] unknown message type: " .. tostring(msg.type))
    end
end

local function exchange()
    if not send(getState()) then return end
    newRun = false
    stepCount = stepCount + 1

    local line, err = client:receive("*l")
    if not line then
        disconnect("receive failed: " .. tostring(err))
        return
    end
    local ok, msg = pcall(json.decode, line)
    if not ok or type(msg) ~= "table" then
        disconnect("bad message")
        return
    end
    handleMessage(msg)
end

---------------------------------------------------------------------------
-- Callbacks
---------------------------------------------------------------------------

function mod:onUpdate()
    if not hasSocket or not client or reset then return end
    updateCounter = updateCounter + 1
    if updateCounter % FRAME_SKIP ~= 0 then return end
    exchange()
end

function mod:onGameStarted()
    if not reset or reset.phase ~= "restart" then return end
    if reset.room then
        reset.phase = "goto"
        reset.startedAt = Isaac.GetFrameCount()
        pendingCommand = "goto " .. reset.room
    else
        finishReset()
    end
end

function mod:onNewRoom()
    if reset and reset.phase == "goto"
        and Game():GetLevel():GetCurrentRoomIndex() == GridRooms.ROOM_DEBUG_IDX then
        finishReset()
    end
end

function mod:onEntityTakeDamage(entity, amount)
    if entity.Type == EntityType.ENTITY_PLAYER then
        counters.hits_taken = counters.hits_taken + 1
    elseif entity:IsActiveEnemy(false) and not entity:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
        counters.damage_dealt = counters.damage_dealt + math.min(amount, math.max(entity.HitPoints, 0))
    end
end

function mod:onEntityKill(entity)
    if entity:IsEnemy() and not entity:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
        counters.kills = counters.kills + 1
    end
end

function mod:onInputAction(entity, inputHook, buttonAction)
    if not client or not CONTROLLED_ACTIONS[buttonAction] then return nil end
    if entity == nil or entity:ToPlayer() == nil then return nil end

    local dir = MOVE_DIRS[currentMove]
    local pressed
    if buttonAction == ButtonAction.ACTION_LEFT then
        pressed = dir[1] < 0
    elseif buttonAction == ButtonAction.ACTION_RIGHT then
        pressed = dir[1] > 0
    elseif buttonAction == ButtonAction.ACTION_UP then
        pressed = dir[2] < 0
    elseif buttonAction == ButtonAction.ACTION_DOWN then
        pressed = dir[2] > 0
    else
        pressed = SHOOT_ACTIONS[currentShoot] == buttonAction
    end

    if inputHook == InputHook.GET_ACTION_VALUE then
        return pressed and 1.0 or 0.0
    end
    return pressed
end

function mod:onRender()
    if pendingCommand then
        local cmd = pendingCommand
        pendingCommand = nil
        local output = Isaac.ExecuteCommand(cmd)
        Isaac.DebugString("[TBOI-RL] ran '" .. cmd .. "': " .. tostring(output))
        if output and output ~= "" then
            send({ type = "log", message = cmd .. ": " .. tostring(output) })
        end
    end

    local timeout = (reset and reset.phase == "goto") and GOTO_TIMEOUT or RESET_TIMEOUT
    if reset and Isaac.GetFrameCount() - reset.startedAt > timeout then
        local phase = reset.phase
        reset = nil
        send({ type = "error", message = "reset timed out during " .. phase
            .. (phase == "goto" and " (does the room exist on this floor?)" or "") })
    end

    if not hasSocket then
        Isaac.RenderText("TBOI-RL: LuaSocket unavailable, launch the game with --luadebug", 40, 240, 1, 0.4, 0.4, 1)
        return
    end
    if not client then tryConnect() end

    local text = "TBOI-RL: " .. statusText
    if client then
        text = text .. string.format(" | step %d | move %d shoot %d", stepCount, currentMove, currentShoot)
    end
    Isaac.RenderText(text, 40, 250, 1, 1, 1, 0.8)
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.onUpdate)
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, mod.onGameStarted)
mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, mod.onNewRoom)
mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, mod.onEntityTakeDamage)
mod:AddCallback(ModCallbacks.MC_POST_ENTITY_KILL, mod.onEntityKill)
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, mod.onInputAction)
mod:AddCallback(ModCallbacks.MC_POST_RENDER, mod.onRender)
