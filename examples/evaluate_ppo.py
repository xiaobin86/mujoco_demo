"""Evaluate a trained PPO model on PandaReachEnv with a live MuJoCo 3D viewer.

--- How to run ---

Make sure the project is installed in editable mode with the RL extras, then:

    cd /mnt/d/work/jaka_zu35_mujoco_rl
    pip install -e ".[rl]"
    python examples/evaluate_ppo.py --model checkpoints/ppo_panda_final.zip --episodes 5

The MuJoCo 3D viewer opens and the trained policy runs deterministically for the
requested number of episodes. The terminal prints the final distance and success flag
for each episode. Close the window to stop evaluation early.

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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from jaka_zu35_mujoco_rl import PandaReachEnv


def make_env() -> PandaReachEnv:
    """Factory for the vectorized environment."""
    return PandaReachEnv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on PandaReachEnv")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/ppo_panda_final.zip",
        help="Path to the trained PPO model",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.005,
        help="Seconds between viewer syncs (lower = faster simulation)",
    )
    args = parser.parse_args()

    env = DummyVecEnv([make_env])
    model = PPO.load(args.model, env=env)

    print(f"Loading model from {args.model}")
    print("Opening MuJoCo viewer... Close the window to stop evaluation.")

    with mujoco.viewer.launch_passive(env.envs[0].model, env.envs[0].data) as viewer:
        for episode in range(args.episodes):
            if not viewer.is_running():
                break

            obs, info = env.reset()
            terminated = False
            truncated = False
            step = 0

            while viewer.is_running() and not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                viewer.sync()
                time.sleep(args.sleep)

            if viewer.is_running():
                print(
                    f"Episode {episode + 1}: "
                    f"steps={step}, reward={reward:.4f}, "
                    f"success={terminated}, final_distance={info['distance']:.4f}"
                )

    env.close()


if __name__ == "__main__":
    main()
