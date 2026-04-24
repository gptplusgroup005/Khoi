import queue
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Tuple

class MazeTrainingUI:
    def __init__(self, config, event_queue: "queue.Queue[Dict[str, object]]", command_queue: "queue.Queue[str]", stop_event):
        self.config = config
        self.event_queue = event_queue
        self.command_queue = command_queue
        self.stop_event = stop_event
        self.latest_visual_state: Optional[Dict[str, object]] = None
        self.latest_summary: Dict[str, object] = {}
        self.current_maze_signature: Optional[Tuple[Tuple[int, ...], ...]] = None
        self.start_item = None
        self.goal_item = None
        self.agent_item = None
        self.trace_items = []

        self.root = tk.Tk()
        self.root.title("Maze RL Trainer")
        self.root.geometry("1100x900")
        self.root.configure(bg="#f2efe8")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Preparing trainer...")
        self.update_var = tk.StringVar(value="Update: -")
        self.reward_var = tk.StringVar(value="Avg reward: -")
        self.success_var = tk.StringVar(value="Success rate: -")
        self.episode_var = tk.StringVar(value="Episode return: -")
        self.length_var = tk.StringVar(value="Episode length: -")
        self.loss_var = tk.StringVar(value="Losses: -")
        self.fps_var = tk.StringVar(value="FPS: -")
        self.maze_info_var = tk.StringVar(value="Maze: waiting for first random 40x40 maze...")

        self.cell_size = 16
        self.canvas_size = self.config.maze_size * self.cell_size
        self.canvas = None

        self._build_layout()
        self.show_placeholder()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Panel.TFrame", background="#f2efe8")
        style.configure("Card.TFrame", background="#fffaf0")
        style.configure("Title.TLabel", background="#f2efe8", foreground="#2b2b2b", font=("Segoe UI Semibold", 20))
        style.configure("Body.TLabel", background="#fffaf0", foreground="#2b2b2b", font=("Consolas", 11))
        style.configure("Status.TLabel", background="#f2efe8", foreground="#5a4b3f", font=("Segoe UI", 11))

        container.configure(style="Panel.TFrame")

        header = ttk.Frame(container, style="Panel.TFrame")
        header.pack(fill="x")

        ttk.Label(header, text="Maze RL Training", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(4, 12))
        ttk.Button(header, text="Generate New Maze", command=self.request_new_maze).pack(anchor="w", pady=(0, 8))

        body = ttk.Frame(container, style="Panel.TFrame")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, style="Card.TFrame", padding=12)
        left.pack(side="left", fill="both", expand=False)

        right = ttk.Frame(body, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self.canvas = tk.Canvas(
            left,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#f7f4ec",
            highlightthickness=1,
            highlightbackground="#d7cbb8",
        )
        self.canvas.pack()

        stats_card = ttk.Frame(right, style="Card.TFrame", padding=16)
        stats_card.pack(fill="x")

        for text_var in [
            self.update_var,
            self.reward_var,
            self.success_var,
            self.episode_var,
            self.length_var,
            self.loss_var,
            self.fps_var,
            self.maze_info_var,
        ]:
            ttk.Label(stats_card, textvariable=text_var, style="Body.TLabel", wraplength=360, justify="left").pack(
                anchor="w",
                pady=4,
            )

        legend_card = ttk.Frame(right, style="Card.TFrame", padding=16)
        legend_card.pack(fill="x", pady=(16, 0))
        ttk.Label(
            legend_card,
            text="Legend\nWall: dark block\nPath: light tile\nTrace: amber\nStart: blue\nGoal: green\nAgent: orange",
            style="Body.TLabel",
            justify="left",
        ).pack(anchor="w")

    def on_close(self):
        self.status_var.set("Stopping trainer...")
        self.stop_event.set()
        self.root.after(150, self.root.destroy)

    def push_status(self, text: str):
        self.status_var.set(text)

    def request_new_maze(self):
        while True:
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.command_queue.put_nowait("generate_maze")
            self.push_status("Manual maze regeneration requested...")
        except queue.Full:
            self.push_status("Maze request queued. Trainer will pick it up shortly.")

    def show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0,
            0,
            self.canvas_size,
            self.canvas_size,
            fill="#f7f4ec",
            outline="",
        )
        self.canvas.create_text(
            self.canvas_size / 2,
            self.canvas_size / 2 - 12,
            text="Waiting for maze render",
            fill="#5a4b3f",
            font=("Segoe UI Semibold", 20),
        )
        self.canvas.create_text(
            self.canvas_size / 2,
            self.canvas_size / 2 + 18,
            text="The first random 40x40 maze will appear here.",
            fill="#7b6a58",
            font=("Segoe UI", 12),
        )

    def _cell_bounds(self, row: int, col: int, pad: int = 0):
        x0 = col * self.cell_size + pad
        y0 = row * self.cell_size + pad
        x1 = (col + 1) * self.cell_size - pad
        y1 = (row + 1) * self.cell_size - pad
        return x0, y0, x1, y1

    def _draw_marker(self, row: int, col: int, color: str, pad: int):
        return self.canvas.create_oval(*self._cell_bounds(row, col, pad=pad), fill=color, outline="")

    def _redraw_static_maze(self, state: Dict[str, object]):
        grid = state["grid"]
        size = state["size"]
        trace = state["trace"]

        self.canvas.delete("all")
        self.trace_items = []
        for row_idx in range(size):
            for col_idx in range(size):
                x0, y0, x1, y1 = self._cell_bounds(row_idx, col_idx)

                if grid[row_idx][col_idx]:
                    fill = "#f4ede1"
                    outline = "#e8dcc8"
                else:
                    fill = "#2c3639"
                    outline = "#2c3639"

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline)
                if trace[row_idx][col_idx] > 0:
                    self.trace_items.append(
                        self.canvas.create_rectangle(
                            x0 + 4,
                            y0 + 4,
                            x1 - 4,
                            y1 - 4,
                            fill="#f59e0b",
                            outline="",
                        )
                    )

        start_x, start_y = state["start"]
        goal_x, goal_y = state["goal"]
        agent_x, agent_y = state["agent"]
        self.start_item = self._draw_marker(start_x, start_y, "#3b82f6", pad=3)
        self.goal_item = self._draw_marker(goal_x, goal_y, "#16a34a", pad=3)
        self.agent_item = self._draw_marker(agent_x, agent_y, "#f97316", pad=2)

    def draw_maze(self, state: Dict[str, object]):
        grid = state["grid"]
        trace = state["trace"]
        maze_signature = tuple(tuple(row) for row in grid)
        if maze_signature != self.current_maze_signature:
            self.current_maze_signature = maze_signature
            self._redraw_static_maze(state)
        else:
            for item_id in self.trace_items:
                self.canvas.delete(item_id)
            self.trace_items = []

            for row_idx, row in enumerate(trace):
                for col_idx, visits in enumerate(row):
                    if visits > 0:
                        x0, y0, x1, y1 = self._cell_bounds(row_idx, col_idx)
                        self.trace_items.append(
                            self.canvas.create_rectangle(
                                x0 + 4,
                                y0 + 4,
                                x1 - 4,
                                y1 - 4,
                                fill="#f59e0b",
                                outline="",
                            )
                        )

        if self.agent_item is not None:
            agent_x, agent_y = state["agent"]
            self.canvas.coords(self.agent_item, *self._cell_bounds(agent_x, agent_y, pad=2))

        self.maze_info_var.set(
            f"Maze: {state['size']}x{state['size']} random maze | visual env={state['env_index']} | "
            f"step={state['step_count']} | ep_return={state['episode_return']:+.3f} | "
            f"episodes_on_this_maze={state['episodes_on_maze']}"
        )
        self.canvas.update_idletasks()

    def update_summary_labels(self, summary: Dict[str, object]):
        self.update_var.set(f"Update: {int(summary.get('update', -1))}")
        self.reward_var.set(f"Avg reward: {summary.get('avg_reward', 0.0):+.4f}")
        self.success_var.set(f"Success rate: {summary.get('success_rate', 0.0):.4f} | Wall hit rate: {summary.get('wall_hit_rate', 0.0):.4f}")
        self.episode_var.set(
            f"Episode return: {summary.get('mean_episode_return', 0.0):+.4f} | "
            f"Episode success: {summary.get('episode_success_rate', 0.0):.3f} | "
            f"Done: {int(summary.get('completed_episodes', 0.0))} | "
            f"Maze refresh: {int(summary.get('maze_refreshes', 0.0))}"
        )
        self.length_var.set(f"Episode length: {summary.get('mean_episode_length', 0.0):.1f}")
        self.loss_var.set(
            f"Losses: pi={summary.get('policy_loss', 0.0):.4f} | "
            f"v={summary.get('value_loss', 0.0):.4f} | "
            f"entropy={summary.get('entropy', 0.0):.4f} | "
            f"kl={summary.get('approx_kl', 0.0):.4f}"
        )
        self.fps_var.set(f"FPS: {summary.get('fps', 0.0):.1f}")

    def process_events(self):
        latest_visual = None
        latest_summary = None
        latest_status = None

        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event.get("type")
            payload = event.get("payload", {})

            if event_type == "run_started":
                latest_status = (
                    f"Training on {payload.get('device', 'cpu')} | "
                    f"maze {payload.get('config', {}).get('maze_size', self.config.maze_size)}x"
                    f"{payload.get('config', {}).get('maze_size', self.config.maze_size)}"
                )
            elif event_type == "bootstrap_started":
                latest_status = "Bootstrapping policy..."
            elif event_type == "bootstrap_finished":
                latest_status = f"Bootstrap accuracy {payload.get('bootstrap_accuracy', 0.0):.3f}"
            elif event_type == "visual_state":
                latest_visual = payload
            elif event_type == "update_summary":
                latest_summary = payload
                latest_status = "Training in progress..."
            elif event_type == "checkpoint_saved":
                latest_status = f"Checkpoint saved at update {payload.get('update', '?')}"
            elif event_type == "target_reached":
                latest_status = (
                    f"Target reached at update {payload.get('update', '?')} | "
                    f"episode success {payload.get('episode_success_rate', 0.0):.3f} | "
                    f"continuing on maze #{payload.get('target_hits', 1) + 1}"
                )
            elif event_type == "maze_regenerated":
                latest_status = f"New maze generated ({payload.get('source', 'manual')})."
            elif event_type == "run_finished":
                latest_status = "Training complete."
            elif event_type == "run_error":
                latest_status = f"Training error: {payload.get('message', 'unknown error')}"
            elif event_type == "trainer_stopped":
                latest_status = "Training stopped by user."

        if latest_visual is not None:
            self.latest_visual_state = latest_visual
            self.draw_maze(latest_visual)
        if latest_summary is not None:
            self.latest_summary = latest_summary
            self.update_summary_labels(latest_summary)
        if latest_status is not None:
            self.push_status(latest_status)

        self.root.after(60, self.process_events)

    def run(self):
        self.process_events()
        self.root.mainloop()
