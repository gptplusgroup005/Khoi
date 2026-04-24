import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch

class MazeEnv:
    """Vectorized maze environment for discrete-action RL."""

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

        self.grids = torch.zeros((num_envs, size, size), dtype=torch.bool, device=device)
        self.distance_maps = torch.full((num_envs, size, size), size * size, dtype=torch.float32, device=device)
        self.agent_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.start_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.goal_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episodes_on_maze = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.trace_map = torch.zeros((num_envs, size, size), dtype=torch.float32, device=device)

    def _generate_maze(self) -> torch.Tensor:
        n = self.size
        grid = [[False for _ in range(n)] for _ in range(n)]

        stack = [(1, 1)]
        grid[1][1] = True
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < n - 1 and 1 <= ny < n - 1 and not grid[nx][ny]:
                    neighbors.append((nx, ny, dx, dy))

            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                grid[x + dx // 2][y + dy // 2] = True
                grid[nx][ny] = True
                stack.append((nx, ny))
            else:
                stack.pop()

        return torch.tensor(grid, dtype=torch.bool)

    def _find_goal(self, grid: torch.Tensor, start: Tuple[int, int]) -> Tuple[int, int]:
        start_x, start_y = start
        queue = deque([(start_x, start_y)])
        distances = {(start_x, start_y): 0}
        farthest = (start_x, start_y)
        farthest_distance = 0
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            current_distance = distances[(x, y)]
            if current_distance > farthest_distance:
                farthest = (x, y)
                farthest_distance = current_distance

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

        return farthest

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
        grid = self._generate_maze()
        start = (1, 1)
        goal = self._find_goal(grid, start)

        self.grids[idx] = grid
        self.start_pos[idx] = torch.tensor(start, dtype=torch.long, device=self.device)
        self.goal_pos[idx] = torch.tensor(goal, dtype=torch.long, device=self.device)
        self.distance_maps[idx] = self._compute_distance_map(grid, goal)
        self.episodes_on_maze[idx] = 0

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
            env_ids = torch.arange(self.num_envs, device=self.device)

        for idx in env_ids.tolist():
            self._reset_episode(idx, regenerate_maze=regenerate_maze)

        return self._get_obs()

    def regenerate_maze(self, env_ids: Optional[torch.Tensor] = None):
        return self.reset(env_ids=env_ids, regenerate_maze=True)

    def step(self, action: torch.Tensor):
        action = action.view(-1).long()
        action = torch.clamp(action, 0, self.act_dim - 1)

        deltas = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]],
            dtype=torch.long,
            device=self.device,
        )
        move = deltas[action]

        prev_pos = self.agent_pos.clone()
        proposed = prev_pos + move
        proposed[:, 0] = torch.clamp(proposed[:, 0], 0, self.size - 1)
        proposed[:, 1] = torch.clamp(proposed[:, 1], 0, self.size - 1)

        env_indices = torch.arange(self.num_envs, device=self.device)
        free_mask = self.grids[env_indices, proposed[:, 0], proposed[:, 1]]
        hit_wall = ~free_mask

        self.agent_pos[free_mask] = proposed[free_mask]
        self.step_count += 1
        self.episode_length += 1

        current_x = self.agent_pos[:, 0]
        current_y = self.agent_pos[:, 1]
        self.trace_map[env_indices, current_x, current_y] += 1.0

        reached_goal = (self.agent_pos == self.goal_pos).all(dim=1)
        timeout = self.step_count >= self.max_steps
        done = reached_goal | timeout

        prev_dist = self.distance_maps[env_indices, prev_pos[:, 0], prev_pos[:, 1]]
        new_dist = self.distance_maps[env_indices, self.agent_pos[:, 0], self.agent_pos[:, 1]]
        progress = (prev_dist - new_dist) * 0.05

        reward = -0.005 + progress
        reward[hit_wall] -= 0.10
        reward[reached_goal] += 10.0
        reward[timeout & ~reached_goal] -= 1.0
        self.episode_return += reward

        completed_return = torch.zeros_like(self.episode_return)
        completed_length = torch.zeros_like(self.episode_length)
        completed_success = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        maze_refresh = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

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
            "episodes_on_maze": self.episodes_on_maze.float(),
        }

        return self._get_obs(), reward, done.float(), info

    def get_visual_state(self, env_index: int = 0) -> Dict[str, object]:
        grid = self.grids[env_index].detach().to("cpu")
        agent = self.agent_pos[env_index].detach().to("cpu")
        start = self.start_pos[env_index].detach().to("cpu")
        goal = self.goal_pos[env_index].detach().to("cpu")
        trace = self.trace_map[env_index].detach().to("cpu")
        distance = self.distance_maps[env_index].detach().to("cpu")
        start_x, start_y = start.tolist()

        return {
            "env_index": env_index,
            "size": self.size,
            "grid": grid.int().tolist(),
            "agent": agent.tolist(),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "trace": trace.tolist(),
            "shortest_path_length": int(distance[start_x, start_y].item()),
            "step_count": int(self.step_count[env_index].item()),
            "episode_return": float(self.episode_return[env_index].item()),
            "episode_length": int(self.episode_length[env_index].item()),
            "episodes_on_maze": int(self.episodes_on_maze[env_index].item()),
        }

    def render_ascii(self, env_index: int = 0) -> str:
        state = self.get_visual_state(env_index=env_index)
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
        env_indices = torch.arange(self.num_envs, device=self.device)
        x = self.agent_pos[:, 0].float() / (self.size - 1)
        y = self.agent_pos[:, 1].float() / (self.size - 1)
        gx = self.goal_pos[:, 0].float() / (self.size - 1)
        gy = self.goal_pos[:, 1].float() / (self.size - 1)
        dx = (self.goal_pos[:, 0].float() - self.agent_pos[:, 0].float()) / (self.size - 1)
        dy = (self.goal_pos[:, 1].float() - self.agent_pos[:, 1].float()) / (self.size - 1)

        up = torch.clamp(self.agent_pos[:, 0] - 1, 0, self.size - 1)
        down = torch.clamp(self.agent_pos[:, 0] + 1, 0, self.size - 1)
        left = torch.clamp(self.agent_pos[:, 1] - 1, 0, self.size - 1)
        right = torch.clamp(self.agent_pos[:, 1] + 1, 0, self.size - 1)

        wall_up = (~self.grids[env_indices, up, self.agent_pos[:, 1]]).float()
        wall_down = (~self.grids[env_indices, down, self.agent_pos[:, 1]]).float()
        wall_left = (~self.grids[env_indices, self.agent_pos[:, 0], left]).float()
        wall_right = (~self.grids[env_indices, self.agent_pos[:, 0], right]).float()

        normalizer = max(1, self.size)
        current_distance = self.distance_maps[env_indices, self.agent_pos[:, 0], self.agent_pos[:, 1]] / normalizer
        up_distance = self.distance_maps[env_indices, up, self.agent_pos[:, 1]] / normalizer
        down_distance = self.distance_maps[env_indices, down, self.agent_pos[:, 1]] / normalizer
        left_distance = self.distance_maps[env_indices, self.agent_pos[:, 0], left] / normalizer
        right_distance = self.distance_maps[env_indices, self.agent_pos[:, 0], right] / normalizer

        return torch.stack(
            [
                x,
                y,
                gx,
                gy,
                dx,
                dy,
                wall_up,
                wall_down,
                wall_left,
                wall_right,
                current_distance,
                up_distance,
                down_distance,
                left_distance,
                right_distance,
            ],
            dim=-1,
        )
