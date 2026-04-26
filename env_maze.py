import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

class MazeEnv:
    """Vectorized 2D obstacle-map environment for discrete-action RL."""

    def __init__(
        self,
        num_envs: int = 16,
        size: int = 40,
        max_steps: int = 400,
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.size = size
        self.max_steps = max_steps
        self.device = device
        self.obs_dim = 15
        self.act_dim = 4
        self.robot_radius_cells = max(1, round(size * 0.035))
        self.position_scale = 1.0 / max(1, size - 1)
        self.distance_scale = 1.0 / max(1, size)

        self.grids = torch.zeros((num_envs, size, size), dtype=torch.bool, device=device)
        self.navigation_grids = torch.zeros((num_envs, size, size), dtype=torch.bool, device=device)
        self.distance_maps = torch.full((num_envs, size, size), size * size, dtype=torch.float32, device=device)
        self.agent_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.start_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.goal_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episodes_on_maze = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.map_versions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.trace_map = torch.zeros((num_envs, size, size), dtype=torch.float32, device=device)
        self.env_indices = torch.arange(num_envs, device=device)
        self.action_deltas = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]],
            dtype=torch.long,
            device=device,
        )

    def _inflate_obstacles(self, grid: torch.Tensor) -> torch.Tensor:
        radius = self.robot_radius_cells
        kernel_size = radius * 2 + 1
        blocked = (~grid).float().view(1, 1, self.size, self.size)
        inflated = F.max_pool2d(blocked, kernel_size=kernel_size, stride=1, padding=radius)
        return inflated.view(self.size, self.size) <= 0.0

    def _bfs_distances(self, grid: torch.Tensor, start: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        start_x, start_y = start
        queue = deque([(start_x, start_y)])
        distances = {(start_x, start_y): 0}
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            current_distance = distances[(x, y)]
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if not bool(grid[nx, ny].item()):
                    continue
                if (nx, ny) in distances:
                    continue
                distances[(nx, ny)] = current_distance + 1
                queue.append((nx, ny))

        return distances

    def _generate_obstacle_map(self) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        n = self.size
        min_path_length = max(6, int(n * 0.7))

        for _ in range(200):
            grid = torch.ones((n, n), dtype=torch.bool)
            grid[0, :] = False
            grid[-1, :] = False
            grid[:, 0] = False
            grid[:, -1] = False

            obstacle_count = random.randint(max(5, n // 5), max(8, n // 3))
            for _ in range(obstacle_count):
                width = random.randint(max(2, n // 14), max(3, n // 5))
                height = random.randint(max(2, n // 14), max(3, n // 5))
                x = random.randint(1, max(1, n - height - 1))
                y = random.randint(1, max(1, n - width - 1))
                grid[x : x + height, y : y + width] = False

            navigation_grid = self._inflate_obstacles(grid)
            free_cells = torch.where(navigation_grid)
            if free_cells[0].numel() < n:
                continue

            candidate_index = random.randrange(free_cells[0].numel())
            start = (int(free_cells[0][candidate_index].item()), int(free_cells[1][candidate_index].item()))
            distances = self._bfs_distances(navigation_grid, start)
            if len(distances) < n:
                continue

            goal, path_length = max(distances.items(), key=lambda item: item[1])
            if path_length >= min_path_length:
                return grid, navigation_grid, start, goal

        grid = torch.ones((n, n), dtype=torch.bool)
        grid[0, :] = False
        grid[-1, :] = False
        grid[:, 0] = False
        grid[:, -1] = False
        navigation_grid = self._inflate_obstacles(grid)
        return grid, navigation_grid, (1 + self.robot_radius_cells, 1 + self.robot_radius_cells), (
            n - 2 - self.robot_radius_cells,
            n - 2 - self.robot_radius_cells,
        )

    def _compute_distance_map(self, grid: torch.Tensor, goal: Tuple[int, int]) -> torch.Tensor:
        goal_x, goal_y = goal
        max_distance = self.size * self.size
        distances = [[max_distance for _ in range(self.size)] for _ in range(self.size)]
        distances[goal_x][goal_y] = 0
        queue = deque([(goal_x, goal_y)])
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            next_distance = distances[x][y] + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if not bool(grid[nx, ny].item()):
                    continue
                if distances[nx][ny] <= next_distance:
                    continue
                distances[nx][ny] = next_distance
                queue.append((nx, ny))

        return torch.tensor(distances, dtype=torch.float32, device=self.device)

    def _reset_trace(self, idx: int):
        self.trace_map[idx].zero_()
        start_x, start_y = self.start_pos[idx].tolist()
        self.trace_map[idx, start_x, start_y] = 1.0

    def _generate_new_maze(self, idx: int):
        grid, navigation_grid, start, goal = self._generate_obstacle_map()

        self.grids[idx] = grid.to(self.device)
        self.navigation_grids[idx] = navigation_grid.to(self.device)
        self.start_pos[idx] = torch.tensor(start, dtype=torch.long, device=self.device)
        self.goal_pos[idx] = torch.tensor(goal, dtype=torch.long, device=self.device)
        self.distance_maps[idx] = self._compute_distance_map(navigation_grid, goal)
        self.episodes_on_maze[idx] = 0
        self.map_versions[idx] += 1

    def _reset_episode(self, idx: int, regenerate_maze: bool):
        if regenerate_maze:
            self._generate_new_maze(idx)

        self.agent_pos[idx] = self.start_pos[idx]
        self.step_count[idx] = 0
        self.episode_return[idx] = 0.0
        self.episode_length[idx] = 0
        self._reset_trace(idx)

    def reset(self, env_ids: Optional[torch.Tensor] = None, regenerate_maze: bool = True):
        if env_ids is None:
            env_ids = self.env_indices

        for idx in env_ids.tolist():
            self._reset_episode(idx, regenerate_maze=regenerate_maze)

        return self._get_obs()

    def regenerate_maze(self, env_ids: Optional[torch.Tensor] = None):
        return self.reset(env_ids=env_ids, regenerate_maze=True)

    def regenerate_map(self, env_ids: Optional[torch.Tensor] = None):
        return self.regenerate_maze(env_ids=env_ids)

    def step(self, action: torch.Tensor):
        action = action.view(-1).long()
        action = torch.clamp(action, 0, self.act_dim - 1)
        move = self.action_deltas[action]

        prev_pos = self.agent_pos.clone()
        proposed = prev_pos + move
        proposed[:, 0] = torch.clamp(proposed[:, 0], 0, self.size - 1)
        proposed[:, 1] = torch.clamp(proposed[:, 1], 0, self.size - 1)

        free_mask = self.navigation_grids[self.env_indices, proposed[:, 0], proposed[:, 1]]
        hit_wall = ~free_mask

        self.agent_pos[free_mask] = proposed[free_mask]
        self.step_count += 1
        self.episode_length += 1

        current_x = self.agent_pos[:, 0]
        current_y = self.agent_pos[:, 1]
        self.trace_map[self.env_indices, current_x, current_y] += 1.0

        reached_goal = (self.agent_pos == self.goal_pos).all(dim=1)
        timeout = self.step_count >= self.max_steps
        done = reached_goal | timeout

        prev_dist = self.distance_maps[self.env_indices, prev_pos[:, 0], prev_pos[:, 1]]
        new_dist = self.distance_maps[self.env_indices, self.agent_pos[:, 0], self.agent_pos[:, 1]]
        progress = (prev_dist - new_dist) * 0.05

        reward = -0.005 + progress
        reward[hit_wall] -= 0.10
        reward[reached_goal] += 10.0
        reward[timeout & ~reached_goal] -= 1.0
        self.episode_return += reward

        completed_return = torch.zeros_like(self.episode_return)
        completed_length = torch.zeros_like(self.episode_length)
        completed_success = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        maze_refresh = torch.zeros_like(self.episode_return)

        for idx in torch.where(done)[0].tolist():
            completed_return[idx] = self.episode_return[idx]
            completed_length[idx] = self.episode_length[idx]
            completed_success[idx] = reached_goal[idx].float()
            self.episodes_on_maze[idx] += 1
            self._reset_episode(idx, regenerate_maze=False)

        info = {
            "reached_goal": reached_goal.float(),
            "timeout": timeout.float(),
            "hit_wall": hit_wall.float(),
            "episode_return": completed_return,
            "episode_length": completed_length.float(),
            "episode_success": completed_success,
            "done_count": done.float().sum().unsqueeze(0),
            "maze_refresh": maze_refresh,
            "map_refresh": maze_refresh,
            "episodes_on_maze": self.episodes_on_maze.float(),
            "episodes_on_map": self.episodes_on_maze.float(),
        }

        return self._get_obs(), reward, done.float(), info

    def get_visual_state(self, env_index: int = 0, dense: bool = False) -> Dict[str, object]:
        grid_device = self.grids[env_index].detach()
        trace_device = self.trace_map[env_index].detach()
        agent = self.agent_pos[env_index].detach().to("cpu")
        start = self.start_pos[env_index].detach().to("cpu")
        goal = self.goal_pos[env_index].detach().to("cpu")
        start_x, start_y = start.tolist()
        blocked_x, blocked_y = torch.where(~grid_device)
        trace_x, trace_y = torch.where(trace_device > 0)
        trace_values = trace_device[trace_x, trace_y]

        state = {
            "env_index": env_index,
            "size": self.size,
            "agent": agent.tolist(),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "map_version": int(self.map_versions[env_index].item()),
            "obstacle_cells": list(zip(blocked_x.to("cpu").tolist(), blocked_y.to("cpu").tolist())),
            "trace_cells": [
                (int(row), int(col), float(visits))
                for row, col, visits in zip(
                    trace_x.to("cpu").tolist(),
                    trace_y.to("cpu").tolist(),
                    trace_values.to("cpu").tolist(),
                )
            ],
            "shortest_path_length": int(self.distance_maps[env_index, start_x, start_y].item()),
            "step_count": int(self.step_count[env_index].item()),
            "episode_return": float(self.episode_return[env_index].item()),
            "episode_length": int(self.episode_length[env_index].item()),
            "episodes_on_maze": int(self.episodes_on_maze[env_index].item()),
            "episodes_on_map": int(self.episodes_on_maze[env_index].item()),
            "robot_radius_cells": self.robot_radius_cells,
        }
        if dense:
            state["grid"] = grid_device.to("cpu").int().tolist()
            state["trace"] = trace_device.to("cpu").tolist()
        return state

    def render_ascii(self, env_index: int = 0) -> str:
        state = self.get_visual_state(env_index=env_index, dense=True)
        grid: List[List[int]] = state["grid"]
        trace: List[List[float]] = state["trace"]
        agent_x, agent_y = state["agent"]
        goal_x, goal_y = state["goal"]
        start_x, start_y = state["start"]

        rows = []
        for row_idx, row in enumerate(grid):
            chars = []
            for col_idx, cell in enumerate(row):
                if row_idx == agent_x and col_idx == agent_y:
                    chars.append("A")
                elif row_idx == goal_x and col_idx == goal_y:
                    chars.append("G")
                elif row_idx == start_x and col_idx == start_y:
                    chars.append("S")
                elif trace[row_idx][col_idx] > 0:
                    chars.append("*")
                else:
                    chars.append("." if cell else "#")
            rows.append("".join(chars))

        return "\n".join(rows)

    def _get_obs(self):
        agent_x = self.agent_pos[:, 0]
        agent_y = self.agent_pos[:, 1]
        goal_x = self.goal_pos[:, 0]
        goal_y = self.goal_pos[:, 1]
        agent_x_float = agent_x.float()
        agent_y_float = agent_y.float()
        goal_x_float = goal_x.float()
        goal_y_float = goal_y.float()

        up = torch.clamp(agent_x - 1, 0, self.size - 1)
        down = torch.clamp(agent_x + 1, 0, self.size - 1)
        left = torch.clamp(agent_y - 1, 0, self.size - 1)
        right = torch.clamp(agent_y + 1, 0, self.size - 1)

        obs = torch.empty((self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        obs[:, 0] = agent_x_float * self.position_scale
        obs[:, 1] = agent_y_float * self.position_scale
        obs[:, 2] = goal_x_float * self.position_scale
        obs[:, 3] = goal_y_float * self.position_scale
        obs[:, 4] = (goal_x_float - agent_x_float) * self.position_scale
        obs[:, 5] = (goal_y_float - agent_y_float) * self.position_scale
        obs[:, 6] = (~self.navigation_grids[self.env_indices, up, agent_y]).float()
        obs[:, 7] = (~self.navigation_grids[self.env_indices, down, agent_y]).float()
        obs[:, 8] = (~self.navigation_grids[self.env_indices, agent_x, left]).float()
        obs[:, 9] = (~self.navigation_grids[self.env_indices, agent_x, right]).float()
        obs[:, 10] = self.distance_maps[self.env_indices, agent_x, agent_y] * self.distance_scale
        obs[:, 11] = self.distance_maps[self.env_indices, up, agent_y] * self.distance_scale
        obs[:, 12] = self.distance_maps[self.env_indices, down, agent_y] * self.distance_scale
        obs[:, 13] = self.distance_maps[self.env_indices, agent_x, left] * self.distance_scale
        obs[:, 14] = self.distance_maps[self.env_indices, agent_x, right] * self.distance_scale
        return obs
