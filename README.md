# RL 3D Map PPO Viewer

Lightweight reinforcement learning project for training a PPO agent to navigate randomly generated obstacle maps with a live 3D-style viewer.

## Recommended architecture

This repo now follows a simple split that is friendly for both training speed and live visualization:

- `env_maze.py`: 2D obstacle-map generation, environment stepping, reward logic, episode metrics, trace tracking, and visualization snapshots.
- `train.py`: PPO training runner that opens a live Tkinter UI by default.
- `maze_ui.py`: desktop UI for drawing the current map as an interactive 3D-style scene with obstacles, start, goal, agent, and trace path.
- `test_env.py`: quick evaluation loop with ASCII rendering using the same environment state as the trainer.
- `agent/`: PPO policy, rollout storage, and optimization code.

## Training

Default training with live UI:

```bash
python train.py
```

The viewer supports mouse-drag orbit, Shift-drag pan, mouse-wheel zoom, and camera buttons in the sidebar. For smoother training, the UI receives compact sparse visual updates and redraws only the moving layer unless the map or camera changes.

By default the trainer keeps running after it reaches the target success rate. It saves a checkpoint, generates a fresh map, and continues improving the policy. Use `--updates N` if you want a finite run.

Headless training:

```bash
python train.py --headless
```

## Current behavior

- The map is kept stable during training.
- A new obstacle map is generated only when you press the `Generate New Map` button in the UI.
- After the target success rate is reached, a new map is generated automatically and training continues.
- The goal is always reachable from the start.
- Obstacles are rectangles of different sizes placed randomly on each map.
- The robot has a footprint: obstacles are inflated internally before collision checks and shortest-path hints are computed.
- The observation includes shortest-path distance hints computed from the current map, so the same policy can adapt immediately when a new map is generated.
- The UI shows an amber trace line of cells the agent has visited.

## Evaluation

```bash
python test_env.py --model-path map_policy.pt
```
