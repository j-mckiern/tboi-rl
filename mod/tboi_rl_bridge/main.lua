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
local FRAME_SKIP = 2            -- game updates (30 per second) per agent step
local RECV_TIMEOUT = 10         -- seconds to wait for the agent before dropping the connection
local RECONNECT_INTERVAL = 60   -- render frames (60 per second) between connection attempts
local PROTOCOL_VERSION = 1

local hasSocket, socket = pcall(require, "socket")
if hasSocket then
    Isaac.DebugString("[TBOI-RL] loaded, LuaSocket " .. tostring(socket._VERSION))
else
    Isaac.DebugString("[TBOI-RL] loaded, LuaSocket unavailable: " .. tostring(socket))
end
local lastConnectError = nil

local client = nil
local lastConnectAttempt = -RECONNECT_INTERVAL
local updateCounter = 0
local stepCount = 0
local resetting = false      -- waiting for a requested restart to finish
local newRun = false         -- next state message is the first of a new run
local pendingCommand = nil   -- console command to run on the next render frame
local resetStartedAt = 0
local RESET_TIMEOUT = 300    -- render frames to wait for a requested restart
local statusText = ""

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

---------------------------------------------------------------------------
-- Connection handling
---------------------------------------------------------------------------

local function disconnect(reason)
    if client then
        client:close()
        client = nil
    end
    currentMove, currentShoot = 0, 0
    resetting = false
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

local function getState()
    local game = Game()
    local level = game:GetLevel()
    local room = game:GetRoom()
    local player = Isaac.GetPlayer(0)

    return {
        type = "state",
        step = stepCount,
        frame = game:GetFrameCount(),
        new_run = newRun,
        seed = game:GetSeeds():GetStartSeedString(),
        stage = level:GetStage(),
        player = {
            pos = { player.Position.X, player.Position.Y },
            vel = { player.Velocity.X, player.Velocity.Y },
            hearts = player:GetHearts(),
            max_hearts = player:GetMaxHearts(),
            soul_hearts = player:GetSoulHearts(),
            dead = player:IsDead(),
        },
        room = {
            index = level:GetCurrentRoomIndex(),
            clear = room:IsClear(),
            enemies = room:GetAliveEnemiesCount(),
        },
    }
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
        currentMove, currentShoot = 0, 0
        if msg.seed and msg.seed ~= "" then
            local seed = tostring(msg.seed)
            if not Seeds.IsStringValidSeed(seed) then
                send({ type = "error", message = "invalid seed: " .. seed })
                return
            end
            pendingCommand = "seed " .. seed
        else
            pendingCommand = "restart"
        end
        resetting = true
        resetStartedAt = Isaac.GetFrameCount()
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
    if not hasSocket or not client or resetting then return end
    updateCounter = updateCounter + 1
    if updateCounter % FRAME_SKIP ~= 0 then return end
    exchange()
end

function mod:onGameStarted()
    updateCounter = 0
    newRun = true
    resetting = false
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
    end

    if resetting and Isaac.GetFrameCount() - resetStartedAt > RESET_TIMEOUT then
        resetting = false
        send({ type = "error", message = "reset timed out (the run did not restart)" })
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
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, mod.onInputAction)
mod:AddCallback(ModCallbacks.MC_POST_RENDER, mod.onRender)
