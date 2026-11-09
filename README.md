# TBOI-RL

A reinforcement learning agent that learns to play *The Binding of Isaac: Repentance*, with the goal of clearing entire floors on its own.

> **Status: early development.** The game bridge is under construction. Training results, gameplay videos, and learning curves will be added as milestones are completed.

<!-- Hero GIF goes here once the agent can clear a room -->

## Overview

*The Binding of Isaac* is a hard environment for reinforcement learning. Every run uses procedurally generated floors, rooms are full of different enemy types and projectiles, and the player's abilities change with each item picked up. This project trains an agent to:

1. **Fight:** survive and clear combat rooms by dodging projectiles and aiming at enemies.
2. **Beat bosses:** win the boss fight at the end of each floor.
3. **Explore:** move through a procedurally generated floor, find the boss, and go down to the next floor.

## How it works

```
 The Binding of Isaac: Repentance               Python
┌──────────────────────────────┐   TCP    ┌───────────────────────────────┐
│ Lua bridge mod               │ ◄──────► │ Gymnasium environment         │
│  • reads game state each step│  JSON    │  • encodes observations       │
│  • applies agent actions     │          │  • computes rewards           │
│  • draws a debug/stats HUD   │          │ PPO agent (Stable-Baselines3) │
└──────────────────────────────┘          └───────────────────────────────┘
```

- **Game interface:** a Lua mod uses the game's official modding API to send structured state to Python each step: player stats, enemies, projectiles, and room layout. It applies the agent's move and shoot actions straight to the game's input, so no screen capture or simulated key presses are involved.
- **Observations:** a multi-channel grid of the current room (obstacles, hazards, enemies, projectiles, pickups) plus a vector of player stats and a floor map for navigation.
- **Actions:** 8-direction movement and 4-direction shooting (`MultiDiscrete([9, 5])`).
- **Learning:** Proximal Policy Optimization (PPO) with a curriculum. Training starts with single rooms, moves on to boss fights, and ends with full floors.

## Roadmap

| Milestone | Goal | Status |
|---|---|---|
| M0 | Game ↔ Python bridge, random agent | In progress |
| M1 | Gym environment, single-room combat | Planned |
| M2 | Combat that generalizes to unseen rooms | Planned |
| M3 | Chapter 1 boss fights | Planned |
| M4 | **Clear Basement I end to end** | Planned |
| M5 | Results write-up and progress video | Planned |

## Results

*Coming soon:* clear rate on held-out seeds, training curves, and a video of the agent at different stages of training.

## Requirements

- *The Binding of Isaac: Rebirth* with the **Repentance** DLC (Steam). Game files are **not** included in this repository.
- Windows, Python 3.11+, NVIDIA GPU recommended.

Setup instructions will be added once the bridge is complete.

## Disclaimer

This is an unofficial fan project for research and education. It is not affiliated with or endorsed by Edmund McMillen or Nicalis. *The Binding of Isaac* and all related assets belong to their respective owners.
