import argparse
import sys
import time
from pathlib import Path

import torch

from agent.policy import ActorCritic
from env_maze import MazeEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained PPO 2D map-navigation policy.")
    parser.add_argument("--model-path", type=str, default="map_policy.pt")
    parser.add_argument("--maze-size", "--map-size", dest="maze_size", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--render-every", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_policy(model_path: str, device: str, obs_dim: int, act_dim: int) -> ActorCritic:
    policy = ActorCritic(obs_dim, act_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    else:
        policy.load_state_dict(checkpoint)

    policy.eval()
    return policy

def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f">>> ERROR: model '{model_path}' not found")
        sys.exit(1)

    env = MazeEnv(num_envs=1, size=args.maze_size, max_steps=args.max_steps, device=args.device)
    policy = load_policy(str(model_path), args.device, env.obs_dim, env.act_dim)

    print(f">>> Loaded model: {model_path}")
    print("\n=== TEST 2D MAP RL ===")
    print("Ctrl+C to stop")

    obs = env.reset()

    try:
        for step_idx in range(args.steps):
            with torch.no_grad():
                action, _, _, _ = policy.act(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            if step_idx % args.render_every == 0 or done.item():
                print("\n" + "=" * 70)
                print(
                    f"step={step_idx} | reward={reward.item():+.4f} | "
                    f"goal={int(info['reached_goal'].item())} | "
                    f"timeout={int(info['timeout'].item())} | "
                    f"done={int(done.item())}"
                )
                print(env.render_ascii())
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n>>> Stop test")

if __name__ == "__main__":
    main()
