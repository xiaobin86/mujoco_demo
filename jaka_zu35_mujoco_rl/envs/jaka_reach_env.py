"""Backward-compatibility shim for the old JAKA environment module.

The actual implementation now lives in ``panda_reach_env.py``. Imports from this
module are redirected to the new Panda implementation while preserving the old
``JakaReachEnv`` name.
"""

from jaka_zu35_mujoco_rl.envs.panda_reach_env import JakaReachEnv, PandaReachEnv

__all__ = ["JakaReachEnv", "PandaReachEnv"]
