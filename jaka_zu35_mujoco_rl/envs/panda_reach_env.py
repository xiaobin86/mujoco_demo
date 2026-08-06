"""Backward-compatibility re-export for the pick-place environment.

The project goal has shifted from reaching a box to picking and placing a small
cube. The original ``PandaReachEnv`` and ``JakaReachEnv`` names now refer to the
new :class:`PandaPickEnv` implementation so that existing imports continue to work.
"""

from jaka_zu35_mujoco_rl.envs.panda_pick_env import (
    JakaReachEnv,
    PandaPickEnv,
    PandaReachEnv,
)

__all__ = ["PandaPickEnv", "PandaReachEnv", "JakaReachEnv"]
