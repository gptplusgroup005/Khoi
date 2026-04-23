import argparse
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from agent.policy import ActorCritic
from agent.ppo_algorithm import PpoAlgorithm
from agent.storage import RolloutStorage
from env_maze import MazeEnv
from maze_ui import MazeTrainingUI


@dataclass
class TrainConfig:
    num_envs: int = 16
    maze_size: int = 40
    max_steps: int = 400
    num_steps: int = 600
    updates: int = 1500
    learning_rate: float = 3e-4
    log_every: int = 32
    checkpoint_every: int = 50
    checkpoint_path: str = "maze_policy.pt"
    metrics_path: str = "runs/train_metrics.jsonl"
    visual_env_index: int = 0
    visual_log_every: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    headless: bool = False


class JsonlEventWriter:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.handle = None

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a", encoding="utf-8")

    def write(self, event_type: str, payload: Dict[str, object]):
        if self.handle is None:
            return

        record = {
            "timestamp": time.time(),
            "type": event_type,
            "payload": payload,
        }
        self.handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train PPO for maze navigation with live UI.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--maze-size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-path", type=str, default="maze_policy.pt")
    parser.add_argument("--metrics-path", type=str, default="runs/train_metrics.jsonl")
    parser.add_argument("--visual-env-index", type=int, default=0)
    parser.add_argument("--visual-log-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--headless", action="store_true", help="Disable the Tkinter UI and train in terminal only.")
    args = parser.parse_args()
    return TrainConfig(
        num_envs=args.num_envs,
        maze_size=args.maze_size,
        max_steps=args.max_steps,
        num_steps=args.num_steps,
        updates=args.updates,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=args.checkpoint_path,
        metrics_path=args.metrics_path,
        visual_env_index=args.visual_env_index,
        visual_log_every=args.visual_log_every,
        seed=args.seed,
        device=args.device,
        headless=args.headless,
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_payload(policy: ActorCritic, config: TrainConfig, update: int) -> Dict[str, object]:
    return {
        "model_state_dict": policy.state_dict(),
        "config": asdict(config),
        "update": update,
    }


def save_checkpoint(policy: ActorCritic, config: TrainConfig, update: int):
    path = Path(config.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(policy, config, update), path)


def emit_event(
    event_type: str,
    payload: Dict[str, object],
    writer: JsonlEventWriter,
    ui_callback: Optional[Callable[[str, Dict[str, object]], None]],
):
    writer.write(event_type, payload)
    if ui_callback is not None:
        ui_callback(event_type, payload)


def train(
    config: TrainConfig,
    *,
    ui_callback: Optional[Callable[[str, Dict[str, object]], None]] = None,
    command_queue: Optional["queue.Queue[str]"] = None,
    stop_event: Optional[threading.Event] = None,
):
    set_seed(config.seed)
    writer = JsonlEventWriter(config.metrics_path)

    env = MazeEnv(
        num_envs=config.num_envs,
        size=config.maze_size,
        max_steps=config.max_steps,
        device=config.device,
    )
    obs = env.reset()

    policy = ActorCritic(env.obs_dim, env.act_dim).to(config.device)
    ppo = PpoAlgorithm(policy, learning_rate=config.learning_rate)
    storage = RolloutStorage(
        num_steps=config.num_steps,
        obs_dim=env.obs_dim,
        num_envs=config.num_envs,
        device=config.device,
    )

    emit_event(
        "run_started",
        {
            "device": config.device,
            "config": asdict(config),
            "obs_dim": env.obs_dim,
            "act_dim": env.act_dim,
        },
        writer,
        ui_callback,
    )
    emit_event("visual_state", env.get_visual_state(env_index=config.visual_env_index), writer, ui_callback)

    print(f"=== START MAZE TRAINING ON {config.device.upper()} ===")
    print(
        f"envs={config.num_envs} | maze={config.maze_size}x{config.maze_size} | "
        f"rollout_steps={config.num_steps} | updates={config.updates}"
    )

    try:
        for update in range(config.updates):
            if stop_event is not None and stop_event.is_set():
                emit_event("trainer_stopped", {"update": update}, writer, ui_callback)
                break

            storage.clear()
            update_reward = 0.0
            goals_reached = 0.0
            wall_hits = 0.0
            completed_episodes = 0.0
            episode_return_sum = 0.0
            episode_length_sum = 0.0
            maze_refreshes = 0.0
            started_at = time.perf_counter()

            for step in range(config.num_steps):
                if stop_event is not None and stop_event.is_set():
                    break

                while command_queue is not None:
                    try:
                        command = command_queue.get_nowait()
                    except queue.Empty:
                        break

                    if command == "generate_maze":
                        obs = env.regenerate_maze()
                        maze_refreshes += 1.0
                        emit_event(
                            "visual_state",
                            env.get_visual_state(env_index=config.visual_env_index),
                            writer,
                            ui_callback,
                        )
                        emit_event(
                            "maze_regenerated",
                            {"source": "manual", "update": update, "step": step},
                            writer,
                            ui_callback,
                        )

                action, logp, _, value = policy.act(obs)
                next_obs, reward, done, info = env.step(action)

                storage.add(obs, action, logp, reward, done, value)

                update_reward += reward.mean().item()
                goals_reached += info["reached_goal"].sum().item()
                wall_hits += info["hit_wall"].sum().item()
                completed_episodes += done.sum().item()
                episode_return_sum += info["episode_return"].sum().item()
                episode_length_sum += info["episode_length"].sum().item()
                maze_refreshes += info["maze_refresh"].sum().item()

                if step % config.log_every == 0 or step == config.num_steps - 1:
                    print(
                        f"UPD {update:04d} | step {step:03d}/{config.num_steps - 1:03d} | "
                        f"rew_mean={reward.mean().item():+.4f} | "
                        f"goal_hits={int(info['reached_goal'].sum().item())} | "
                        f"done={int(done.sum().item())}"
                    )

                if step % config.visual_log_every == 0 or done.any():
                    emit_event(
                        "visual_state",
                        env.get_visual_state(env_index=config.visual_env_index),
                        writer,
                        ui_callback,
                    )

                obs = next_obs

            if storage.step == 0:
                emit_event("trainer_stopped", {"update": update}, writer, ui_callback)
                break

            with torch.no_grad():
                _, _, _, next_value = policy.act(obs)

            storage.compute_returns(next_value)
            ppo_info = ppo.update(storage)

            avg_reward = update_reward / max(1, storage.step)
            success_rate = goals_reached / max(1.0, storage.step * config.num_envs)
            mean_episode_return = episode_return_sum / max(1.0, completed_episodes)
            mean_episode_length = episode_length_sum / max(1.0, completed_episodes)
            wall_hit_rate = wall_hits / max(1.0, storage.step * config.num_envs)
            fps = (storage.step * config.num_envs) / max(1e-6, time.perf_counter() - started_at)

            summary = {
                "update": update,
                "avg_reward": avg_reward,
                "success_rate": success_rate,
                "completed_episodes": completed_episodes,
                "mean_episode_return": mean_episode_return,
                "mean_episode_length": mean_episode_length,
                "wall_hit_rate": wall_hit_rate,
                "maze_refreshes": maze_refreshes,
                "fps": fps,
                **ppo_info,
            }

            print("-" * 100)
            print(
                f"UPDATE {update:04d} | avg_rew={avg_reward:+.4f} | succ_rate={success_rate:.4f} | "
                f"ep_done={int(completed_episodes)} | ep_rew={mean_episode_return:+.4f} | "
                f"ep_len={mean_episode_length:.1f} | wall={wall_hit_rate:.4f} | "
                f"maze_refresh={int(maze_refreshes)} | fps={fps:.1f}"
            )
            print(
                f"ppo: pi_loss={ppo_info['policy_loss']:.4f} | v_loss={ppo_info['value_loss']:.4f} | "
                f"entropy={ppo_info['entropy']:.4f} | approx_kl={ppo_info['approx_kl']:.4f}"
            )
            print("-" * 100)

            emit_event("update_summary", summary, writer, ui_callback)

            if update % config.checkpoint_every == 0 or update == config.updates - 1:
                save_checkpoint(policy, config, update)
                emit_event(
                    "checkpoint_saved",
                    {
                        "update": update,
                        "checkpoint_path": str(Path(config.checkpoint_path)),
                    },
                    writer,
                    ui_callback,
                )

        else:
            emit_event(
                "run_finished",
                {
                    "final_checkpoint": str(Path(config.checkpoint_path)),
                    "updates": config.updates,
                },
                writer,
                ui_callback,
            )
            print("=== TRAINING COMPLETE ===")
            writer.close()
            return

        emit_event(
            "run_finished",
            {
                "final_checkpoint": str(Path(config.checkpoint_path)),
                "updates": config.updates,
            },
            writer,
            ui_callback,
        )
        print("=== TRAINING STOPPED ===")
    except Exception as exc:
        emit_event("run_error", {"message": str(exc)}, writer, ui_callback)
        raise
    finally:
        writer.close()


def run_with_ui(config: TrainConfig):
    event_queue: "queue.Queue[Dict[str, object]]" = queue.Queue(maxsize=8)
    command_queue: "queue.Queue[str]" = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    def ui_callback(event_type: str, payload: Dict[str, object]):
        event = {"type": event_type, "payload": payload}
        if event_type == "visual_state":
            try:
                event_queue.put_nowait(event)
            except queue.Full:
                pass
            return

        try:
            event_queue.put_nowait(event)
        except queue.Full:
            drained_visual = []
            while True:
                try:
                    queued_event = event_queue.get_nowait()
                except queue.Empty:
                    break
                if queued_event.get("type") != "visual_state":
                    drained_visual.append(queued_event)

            for queued_event in drained_visual[-2:]:
                try:
                    event_queue.put_nowait(queued_event)
                except queue.Full:
                    break

            try:
                event_queue.put_nowait(event)
            except queue.Full:
                pass

    ui = MazeTrainingUI(config, event_queue, command_queue, stop_event)

    trainer_thread = threading.Thread(
        target=train,
        kwargs={
            "config": config,
            "ui_callback": ui_callback,
            "command_queue": command_queue,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    trainer_thread.start()
    ui.run()
    trainer_thread.join(timeout=1.0)


if __name__ == "__main__":
    config = parse_args()
    if config.headless:
        train(config)
    else:
        run_with_ui(config)

import torch

from agent.policy import ActorCritic
from agent.storage import RolloutStorage
from agent.ppo_algorithm import PpoAlgorithm
from env_maze import MazeEnv

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== HYPERPARAMETERS =====
num_envs = 16
num_steps = 256
updates = 1500
lr = 3e-4  
log_every = 20

# ===== ENV + POLICY =====
env = MazeEnv(num_envs=num_envs, size=40, max_steps=400, device=device)
obs_dim = env.obs_dim
act_dim = env.act_dim

policy = ActorCritic(obs_dim, act_dim).to(device)
ppo = PpoAlgorithm(policy, learning_rate=lr)
storage = RolloutStorage(num_steps=num_steps, obs_dim=obs_dim, num_envs=num_envs, device=device)

obs = env.reset()

print(f"=== START MAZE TRAINING ON {device.upper()} ===")
print(f"obs_dim={obs_dim}, act_dim={act_dim}, num_envs={num_envs}, maze=40x40")

for update in range(updates):
    storage.clear()
    update_reward = 0.0
    goals_reached = 0.0

    for step in range(num_steps):
        action, logp, _, value = policy.act(obs)
        
        next_obs, reward, done, info = env.step(action)
        
        storage.add(obs, action, logp, reward, done, value)
        
        update_reward += reward.mean().item()
        goals_reached += info["reached_goal"].sum().item()

        if step % log_every == 0:
            print(
                f"UPD {update:04d} | step {step:03d} | "
                f"rew_mean={reward.mean().item():.4f} | "
                f"goal_hits={int(info['reached_goal'].sum().item())}"
            )
        
        obs = next_obs

    with torch.no_grad():
        _, _, _, next_value = policy.act(obs)
    
    ppo_info = ppo.update(storage)

    avg_reward = update_reward / num_steps
    success_rate = goals_reached / (num_steps * num_envs)

    print("-" * 90)
    print(
        f"UPDATE {update:04d} | avg_rew={avg_reward:.4f} | succ_rate={success_rate:.4f} | "
        f"pi_loss={ppo_info['policy_loss']:.4f} | v_loss={ppo_info['value_loss']:.4f} | "
        f"entropy={ppo_info['entropy']:.4f}"
    )
    print("-" * 90)

    if update % 50 == 0:
        torch.save(policy.state_dict(), "maze_policy.pt")

print("=== TRAINING COMPLETE ===")