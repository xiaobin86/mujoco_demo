"""Evaluate a trained PPO model on PandaPickEnv with a live MuJoCo 3D viewer.

The task is pick-and-place: the trained policy must pick up a small red cube and
place it in the target tray. The MuJoCo 3D viewer opens and the policy runs
deterministically for the requested number of episodes.

--- How to run ---

    cd /mnt/d/work/jaka_zu35_mujoco_rl
    pip install -e ".[rl]"
    python examples/evaluate_ppo.py --model checkpoints/ppo_panda_final.zip --episodes 5

The terminal prints the final cube-to-tray distance and success flag for each
episode. Close the window to stop evaluation early.

Controls inside the viewer window:
- Left drag  : rotate camera
- Right drag : pan camera
- Scroll     : zoom
- Esc / close window : stop the demo
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import mujoco.viewer
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from jaka_zu35_mujoco_rl.envs import make_env


def make_vec_env(robot: str):
    env = make_env(robot=robot)
    env = RecordEpisodeStatistics(env)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on PandaPickEnv or SO101PickEnv")
    parser.add_argument("--robot", type=str, default="panda", choices=["panda", "so101"], help="Robot to evaluate")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained PPO model (default: checkpoints/ppo_<robot>_final.zip)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.005,
        help="Seconds between viewer syncs (lower = faster simulation)",
    )
    args = parser.parse_args()

    robot = args.robot
    model_path = Path(args.model) if args.model else Path(f"checkpoints/ppo_{robot}_final.zip")

    env = DummyVecEnv([lambda: make_vec_env(robot=robot)])
    model = PPO.load(str(model_path), env=env, device="cpu")

    print(f"Loading model from {model_path}")
    print("Opening MuJoCo viewer... Close the window to stop evaluation.")

    with mujoco.viewer.launch_passive(env.envs[0].unwrapped.model, env.envs[0].unwrapped.data) as viewer:
        for episode in range(args.episodes):
            if not viewer.is_running():
                break

            obs = env.reset()
            done = False
            step = 0

            while viewer.is_running() and not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                done = bool(done[0])
                step += 1
                viewer.sync()
                time.sleep(args.sleep)

            if viewer.is_running():
                info0 = info[0]
                truncated = info0.get("TimeLimit.truncated", False)
                success = done and not truncated
                print(
                    f"Episode {episode + 1}: "
                    f"steps={step}, reward={float(reward[0]):.4f}, "
                    f"success={success}, final_distance={info0.get('distance', float('nan')):.4f}"
                )

    env.close()


if __name__ == "__main__":
    main()
