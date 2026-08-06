"""Launch an interactive MuJoCo 3D viewer and watch JakaReachEnv run.

--- How to run ---

Make sure the project is installed in editable mode, then:

    cd /mnt/d/work/jaka_zu35_mujoco_rl
    python examples/viewer_demo.py

Controls inside the MuJoCo viewer window:
- Left drag  : rotate camera
- Right drag : pan camera
- Scroll     : zoom
- Space      : pause/resume (the script still steps, but you can inspect)
- Esc / close window : stop the demo

The demo runs a random policy, so the arm will jitter around but you can see the
current end-effector to box-top distance printed in the terminal after each episode.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import mujoco.viewer

from jaka_zu35_mujoco_rl import JakaReachEnv


def run_episodes(env: JakaReachEnv, viewer: Any, num_episodes: int, sleep_dt: float) -> None:
    """Step a random policy while the viewer window is open."""
    for episode in range(num_episodes):
        if not viewer.is_running():
            break

        obs, info = env.reset(seed=episode)
        terminated = False
        truncated = False
        step = 0

        while viewer.is_running() and not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            viewer.sync()
            time.sleep(sleep_dt)

        if viewer.is_running():
            print(
                f"Episode {episode + 1}: "
                f"steps={step}, reward={reward:.4f}, "
                f"success={terminated}, final_distance={info['distance']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo viewer for JakaReachEnv")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.005,
        help="Seconds between viewer syncs (lower = faster simulation)",
    )
    args = parser.parse_args()

    env = JakaReachEnv(seed=args.seed)
    env.reset(seed=args.seed)

    print("Opening MuJoCo viewer... Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        run_episodes(env, viewer, args.episodes, args.sleep)

    env.close()


if __name__ == "__main__":
    main()
