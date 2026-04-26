import math
import queue
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple


class MazeTrainingUI:
    def __init__(self, config, event_queue: "queue.Queue[Dict[str, object]]", command_queue: "queue.Queue[str]", stop_event):
        self.config = config
        self.event_queue = event_queue
        self.command_queue = command_queue
        self.stop_event = stop_event
        self.latest_visual_state: Optional[Dict[str, object]] = None
        self.latest_summary: Dict[str, object] = {}

        self.yaw_degrees = 42.0
        self.pitch_degrees = 58.0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 8.0
        self.drag_start: Optional[Tuple[int, int]] = None
        self.camera_start: Optional[Tuple[float, float, float]] = None
        self.static_scene_key: Optional[Tuple[object, ...]] = None

        self.root = tk.Tk()
        self.root.title("3D Map RL Trainer")
        self.root.geometry("1280x860")
        self.root.minsize(1040, 720)
        self.root.configure(bg="#0b1020")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Preparing trainer...")
        self.update_var = tk.StringVar(value="Update: -")
        self.reward_var = tk.StringVar(value="Avg reward: -")
        self.success_var = tk.StringVar(value="Success rate: -")
        self.episode_var = tk.StringVar(value="Episode return: -")
        self.length_var = tk.StringVar(value="Episode length: -")
        self.loss_var = tk.StringVar(value="Losses: -")
        self.fps_var = tk.StringVar(value="FPS: -")
        self.maze_info_var = tk.StringVar(value="Map: waiting for first random 40x40 obstacle map...")

        self.canvas_width = 880
        self.canvas_height = 760
        self.canvas = None

        self._build_layout()
        self.show_placeholder()

    def _build_layout(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#0b1020")
        style.configure("Sidebar.TFrame", background="#111827")
        style.configure("Panel.TFrame", background="#0b1020")
        style.configure("Card.TFrame", background="#151f32")
        style.configure("Title.TLabel", background="#111827", foreground="#e5eefc", font=("Segoe UI Semibold", 18))
        style.configure("Body.TLabel", background="#151f32", foreground="#d7e3f5", font=("Consolas", 10))
        style.configure("Status.TLabel", background="#111827", foreground="#93a4bd", font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background="#151f32", foreground="#93a4bd", font=("Segoe UI", 9))
        style.configure("Accent.TButton", background="#243145", foreground="#e5eefc", font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", "#31415c")])

        root_frame = ttk.Frame(self.root, style="Root.TFrame")
        root_frame.pack(fill="both", expand=True)

        sidebar = ttk.Frame(root_frame, style="Sidebar.TFrame", padding=18)
        sidebar.pack(side="left", fill="y")

        main = ttk.Frame(root_frame, style="Panel.TFrame", padding=(12, 12, 12, 12))
        main.pack(side="left", fill="both", expand=True)

        ttk.Label(sidebar, text="3D Map RL Training", style="Title.TLabel").pack(anchor="w")
        ttk.Label(sidebar, textvariable=self.status_var, style="Status.TLabel", wraplength=260).pack(anchor="w", pady=(6, 18))

        ttk.Button(sidebar, text="Generate New Map", command=self.request_new_maze, style="Accent.TButton").pack(
            anchor="w",
            fill="x",
            pady=(0, 14),
        )

        camera_card = ttk.Frame(sidebar, style="Card.TFrame", padding=14)
        camera_card.pack(fill="x", pady=(0, 14))
        ttk.Label(camera_card, text="Camera", style="Body.TLabel").pack(anchor="w")
        ttk.Button(camera_card, text="Rotate Left", command=lambda: self.rotate_camera(-12), style="Accent.TButton").pack(
            fill="x",
            pady=(10, 4),
        )
        ttk.Button(camera_card, text="Rotate Right", command=lambda: self.rotate_camera(12), style="Accent.TButton").pack(
            fill="x",
            pady=4,
        )
        ttk.Button(camera_card, text="Zoom In", command=lambda: self.zoom_camera(1.12), style="Accent.TButton").pack(
            fill="x",
            pady=4,
        )
        ttk.Button(camera_card, text="Zoom Out", command=lambda: self.zoom_camera(0.88), style="Accent.TButton").pack(
            fill="x",
            pady=4,
        )
        ttk.Button(camera_card, text="Reset View", command=self.reset_camera, style="Accent.TButton").pack(fill="x", pady=4)
        ttk.Label(camera_card, text="Drag scene to orbit. Shift-drag to pan. Mouse wheel zooms.", style="Hint.TLabel", wraplength=250).pack(
            anchor="w",
            pady=(8, 0),
        )

        stats_card = ttk.Frame(sidebar, style="Card.TFrame", padding=14)
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
            ttk.Label(stats_card, textvariable=text_var, style="Body.TLabel", wraplength=270, justify="left").pack(
                anchor="w",
                pady=4,
            )

        self.canvas = tk.Canvas(
            main,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#07101f",
            highlightthickness=1,
            highlightbackground="#243145",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self.zoom_camera(1.08))
        self.canvas.bind("<Button-5>", lambda _event: self.zoom_camera(0.92))

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
            self.command_queue.put_nowait("generate_map")
            self.push_status("Manual map regeneration requested...")
        except queue.Full:
            self.push_status("Map request queued. Trainer will pick it up shortly.")

    def rotate_camera(self, delta_degrees: float):
        self.yaw_degrees = (self.yaw_degrees + delta_degrees) % 360
        self.static_scene_key = None
        self.redraw_latest()

    def zoom_camera(self, factor: float):
        self.zoom = min(2.4, max(0.45, self.zoom * factor))
        self.static_scene_key = None
        self.redraw_latest()

    def reset_camera(self):
        self.yaw_degrees = 42.0
        self.pitch_degrees = 58.0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 8.0
        self.static_scene_key = None
        self.redraw_latest()

    def on_canvas_resize(self, event):
        self.canvas_width = max(480, event.width)
        self.canvas_height = max(420, event.height)
        self.static_scene_key = None
        self.redraw_latest()

    def on_drag_start(self, event):
        self.drag_start = (event.x, event.y)
        self.camera_start = (self.yaw_degrees, self.pan_x, self.pan_y)

    def on_drag(self, event):
        if self.drag_start is None or self.camera_start is None:
            return

        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        start_yaw, start_pan_x, start_pan_y = self.camera_start
        if event.state & 0x0001:
            self.pan_x = start_pan_x + dx
            self.pan_y = start_pan_y + dy
        else:
            self.yaw_degrees = (start_yaw + dx * 0.35) % 360
            self.pitch_degrees = min(70.0, max(38.0, self.pitch_degrees + dy * 0.12))
        self.static_scene_key = None
        self.redraw_latest()

    def on_mouse_wheel(self, event):
        self.zoom_camera(1.08 if event.delta > 0 else 0.92)

    def redraw_latest(self):
        if self.latest_visual_state is None:
            self.show_placeholder()
        else:
            self.draw_maze(self.latest_visual_state)

    def show_placeholder(self):
        self.static_scene_key = None
        self.canvas.delete("all")
        self._draw_background()
        self.canvas.create_text(
            self.canvas_width / 2,
            self.canvas_height / 2 - 18,
            text="Waiting for 3D map render",
            fill="#d7e3f5",
            font=("Segoe UI Semibold", 22),
        )
        self.canvas.create_text(
            self.canvas_width / 2,
            self.canvas_height / 2 + 18,
            text="The first random obstacle map will appear as a 3D scene.",
            fill="#93a4bd",
            font=("Segoe UI", 12),
        )

    def _draw_background(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_width, self.canvas_height, fill="#07101f", outline="")
        horizon = int(self.canvas_height * 0.58)
        self.canvas.create_rectangle(0, 0, self.canvas_width, horizon, fill="#07101f", outline="")
        self.canvas.create_rectangle(0, horizon, self.canvas_width, self.canvas_height, fill="#0d1728", outline="")
        for idx in range(9):
            y = horizon + idx * 24
            shade = "#111c2f" if idx % 2 == 0 else "#0f192a"
            self.canvas.create_line(0, y, self.canvas_width, y, fill=shade)
        self.canvas.create_text(
            self.canvas_width - 18,
            18,
            text="CAD VIEW",
            anchor="ne",
            fill="#52647f",
            font=("Segoe UI Semibold", 9),
        )

    def _project(self, row: float, col: float, z: float, size: int) -> Tuple[float, float]:
        cx = (size - 1) / 2.0
        cy = (size - 1) / 2.0
        x = col - cy
        y = row - cx
        yaw = math.radians(self.yaw_degrees)
        pitch = math.radians(self.pitch_degrees)

        rx = x * math.cos(yaw) - y * math.sin(yaw)
        ry = x * math.sin(yaw) + y * math.cos(yaw)

        scale = min(self.canvas_width * 0.86, self.canvas_height * 1.08) / max(1, size) * self.zoom
        sx = self.canvas_width / 2 + self.pan_x + rx * scale
        sy = self.canvas_height * 0.58 + self.pan_y + ry * scale * math.cos(pitch) - z * scale * math.sin(pitch)
        return sx, sy

    def _cell_corners(self, row: float, col: float, z: float, size: int) -> List[Tuple[float, float]]:
        return [
            self._project(row, col, z, size),
            self._project(row, col + 1, z, size),
            self._project(row + 1, col + 1, z, size),
            self._project(row + 1, col, z, size),
        ]

    def _polygon(self, points: List[Tuple[float, float]], fill: str, outline: str = "", width: int = 1, tags: str = ""):
        flat = [coord for point in points for coord in point]
        self.canvas.create_polygon(*flat, fill=fill, outline=outline, width=width, tags=tags)

    def _draw_block(self, row: int, col: int, size: int, height: float = 0.92):
        top = self._cell_corners(row, col, height, size)
        base = self._cell_corners(row, col, 0.02, size)
        faces = [
            ([base[0], base[1], top[1], top[0]], "#273247"),
            ([base[1], base[2], top[2], top[1]], "#1d2638"),
            ([base[2], base[3], top[3], top[2]], "#151e2f"),
            ([base[3], base[0], top[0], top[3]], "#202a3d"),
        ]
        for points, color in faces:
            self._polygon(points, color, "#324159")
        self._polygon(top, "#8c98aa", "#c7d1df", width=1)

    def _draw_floor_cell(self, row: int, col: int, size: int, fill: str, outline: str = "#17253b"):
        self._polygon(self._cell_corners(row, col, 0.0, size), fill, outline)

    def _draw_trace_cell(self, row: int, col: int, size: int, visits: float):
        intensity = min(1.0, 0.35 + visits * 0.12)
        fill = "#f59e0b" if intensity < 0.8 else "#facc15"
        points = [
            self._project(row + 0.24, col + 0.24, 0.035, size),
            self._project(row + 0.76, col + 0.24, 0.035, size),
            self._project(row + 0.76, col + 0.76, 0.035, size),
            self._project(row + 0.24, col + 0.76, 0.035, size),
        ]
        self._polygon(points, fill, "", tags="dynamic")

    def _draw_marker_column(self, row: int, col: int, size: int, color: str, label: str):
        center = self._project(row + 0.5, col + 0.5, 0.78, size)
        base = self._project(row + 0.5, col + 0.5, 0.08, size)
        radius = max(5.0, min(self.canvas_width, self.canvas_height) / max(80.0, size * 2.2) * self.zoom)
        self.canvas.create_line(base[0], base[1], center[0], center[1], fill=color, width=2)
        self.canvas.create_oval(
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
            fill=color,
            outline="#e5eefc",
            width=1,
        )
        self.canvas.create_text(center[0], center[1] - radius - 10, text=label, fill="#d7e3f5", font=("Segoe UI Semibold", 9))

    def _draw_robot(self, row: int, col: int, size: int, radius_cells: int):
        body_center = self._project(row + 0.5, col + 0.5, 0.72, size)
        shadow_center = self._project(row + 0.5, col + 0.5, 0.03, size)
        body_radius = max(8.0, min(self.canvas_width, self.canvas_height) / max(64.0, size * 1.7) * max(1, radius_cells) * self.zoom)
        self.canvas.create_oval(
            shadow_center[0] - body_radius * 1.15,
            shadow_center[1] - body_radius * 0.45,
            shadow_center[0] + body_radius * 1.15,
            shadow_center[1] + body_radius * 0.45,
            fill="#020617",
            outline="",
            tags="dynamic",
        )
        self.canvas.create_line(
            shadow_center[0],
            shadow_center[1],
            body_center[0],
            body_center[1],
            fill="#fb923c",
            width=3,
            tags="dynamic",
        )
        self.canvas.create_oval(
            body_center[0] - body_radius,
            body_center[1] - body_radius,
            body_center[0] + body_radius,
            body_center[1] + body_radius,
            fill="#f97316",
            outline="#fed7aa",
            width=2,
            tags="dynamic",
        )
        nose = self._project(row + 0.18, col + 0.5, 0.86, size)
        self.canvas.create_line(body_center[0], body_center[1], nose[0], nose[1], fill="#ffedd5", width=2, tags="dynamic")

    def _obstacle_cells(self, state: Dict[str, object]) -> List[Tuple[int, int]]:
        if "obstacle_cells" in state:
            return [(int(row), int(col)) for row, col in state["obstacle_cells"]]

        grid = state["grid"]
        return [
            (row_idx, col_idx)
            for row_idx, row in enumerate(grid)
            for col_idx, cell in enumerate(row)
            if not bool(cell)
        ]

    def _trace_cells(self, state: Dict[str, object]) -> List[Tuple[int, int, float]]:
        if "trace_cells" in state:
            return [(int(row), int(col), float(visits)) for row, col, visits in state["trace_cells"]]

        trace = state.get("trace", [])
        return [
            (row_idx, col_idx, float(visits))
            for row_idx, row in enumerate(trace)
            for col_idx, visits in enumerate(row)
            if visits > 0
        ]

    def _camera_key(self) -> Tuple[float, float, float, float, float, int, int]:
        return (
            round(self.yaw_degrees, 2),
            round(self.pitch_degrees, 2),
            round(self.zoom, 3),
            round(self.pan_x, 1),
            round(self.pan_y, 1),
            self.canvas_width,
            self.canvas_height,
        )

    def _static_key(self, state: Dict[str, object], obstacle_cells: List[Tuple[int, int]]) -> Tuple[object, ...]:
        map_key = state.get("map_version")
        if map_key is None:
            map_key = tuple(obstacle_cells)
        return (state.get("env_index"), int(state["size"]), map_key, tuple(state["start"]), tuple(state["goal"]), self._camera_key())

    def _draw_static_scene(self, state: Dict[str, object], obstacle_cells: List[Tuple[int, int]]):
        size = int(state["size"])
        obstacle_set = set(obstacle_cells)
        start_x, start_y = state["start"]
        goal_x, goal_y = state["goal"]

        self._draw_background()

        draw_items = []
        for row_idx in range(size):
            for col_idx in range(size):
                depth = row_idx + col_idx
                is_obstacle = (row_idx, col_idx) in obstacle_set
                draw_items.append((depth, "floor", row_idx, col_idx, is_obstacle))
                if is_obstacle:
                    draw_items.append((depth + 0.35, "block", row_idx, col_idx, is_obstacle))

        draw_items.sort(key=lambda item: item[0])

        for _, item_type, row_idx, col_idx, is_obstacle in draw_items:
            if item_type == "floor":
                fill = "#0b1220" if is_obstacle else "#111d31"
                self._draw_floor_cell(row_idx, col_idx, size, fill)
            else:
                self._draw_block(row_idx, col_idx, size)

        self._draw_marker_column(start_x, start_y, size, "#38bdf8", "START")
        self._draw_marker_column(goal_x, goal_y, size, "#22c55e", "GOAL")

    def draw_maze(self, state: Dict[str, object]):
        size = int(state["size"])
        agent_x, agent_y = state["agent"]
        robot_radius = int(state.get("robot_radius_cells", 1))
        obstacle_cells = self._obstacle_cells(state)
        trace_cells = self._trace_cells(state)
        static_key = self._static_key(state, obstacle_cells)

        if static_key != self.static_scene_key:
            self._draw_static_scene(state, obstacle_cells)
            self.static_scene_key = static_key
        else:
            self.canvas.delete("dynamic")

        for row_idx, col_idx, visits in trace_cells:
            self._draw_trace_cell(row_idx, col_idx, size, visits)
        self._draw_robot(agent_x, agent_y, size, robot_radius)

        self.maze_info_var.set(
            f"Map: {state['size']}x{state['size']} 3D obstacle map | "
            f"robot radius={state.get('robot_radius_cells', 1)} cell(s) | visual env={state['env_index']} | "
            f"step={state['step_count']} | ep_return={state['episode_return']:+.3f} | "
            f"episodes_on_this_map={state['episodes_on_maze']}"
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
            f"Map refresh: {int(summary.get('map_refreshes', summary.get('maze_refreshes', 0.0)))}"
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
                    f"map {payload.get('config', {}).get('maze_size', self.config.maze_size)}x"
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
                    f"continuing on map #{payload.get('target_hits', 1) + 1}"
                )
            elif event_type in {"maze_regenerated", "map_regenerated"}:
                latest_status = f"New map generated ({payload.get('source', 'manual')})."
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
