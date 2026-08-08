from typing import Any

from jaka_zu35_mujoco_rl.envs.panda_pick_env import JakaReachEnv, PandaPickEnv, PandaReachEnv
from jaka_zu35_mujoco_rl.envs.so101_pick_env import SO101PickEnv

__all__ = ["PandaPickEnv", "PandaReachEnv", "JakaReachEnv", "SO101PickEnv", "make_env"]


def make_env(robot: str = "panda", **kwargs: Any) -> PandaPickEnv | SO101PickEnv:
    robot = robot.lower()
    if robot == "panda":
        return PandaPickEnv(**kwargs)
    if robot == "so101":
        return SO101PickEnv(**kwargs)
    raise ValueError(f"Unknown robot: {robot}. Supported: panda, so101")

