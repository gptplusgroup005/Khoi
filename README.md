# RL Maze PPO

Lightweight reinforcement learning project for training a PPO agent to solve randomly generated mazes.

## Recommended architecture

This repo now follows a simple split that is friendly for both training speed and live visualization:

- `env_maze.py`: maze generation, environment stepping, reward logic, episode metrics, trace tracking, and visualization snapshots.
- `train.py`: PPO training runner that opens a live Tkinter UI by default.
- `maze_ui.py`: desktop UI for drawing the current maze, start, goal, agent, and trace path.
- `test_env.py`: quick evaluation loop with ASCII rendering using the same environment state as the trainer.
- `agent/`: PPO policy, rollout storage, and optimization code.

## Training

Default training with live UI:

```bash
python train.py
```

Headless training:

```bash
python train.py --headless
```

## Current behavior

- The maze is kept stable during training.
- A new maze is generated only when you press the `Generate New Maze` button in the UI.
- The goal is always reachable from the start.
- The UI shows an amber trace line of cells the agent has visited.

## Evaluation

```bash
python test_env.py --model-path maze_policy.pt
```
