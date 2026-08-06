"""Render one frame of the JakaReachEnv and save it as an image."""

import argparse
from pathlib import Path

import imageio

from jaka_zu35_mujoco_rl import JakaReachEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="jaka_reach_scene.png")
    args = parser.parse_args()

    env = JakaReachEnv(render_mode="rgb_array")
    env.reset(seed=0)

    # Let the arm settle for a few steps with random actions.
    for _ in range(20):
        env.step(env.action_space.sample())

    frame = env.render()
    if frame is None:
        raise RuntimeError("render() returned None")

    imageio.v3.imwrite(args.output, frame)
    print(f"Saved render to {Path(args.output).resolve()}")
    env.close()


if __name__ == "__main__":
    main()
