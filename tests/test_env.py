import numpy as np
import mujoco
from gymnasium.utils.env_checker import check_env
from pathlib import Path

import jaka_zu35_mujoco_rl
from jaka_zu35_mujoco_rl import PandaPickEnv, PandaReachEnv


def test_model_loads():
    model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
    xml_path = model_dir / "panda_pick_scene.xml"
    assert xml_path.exists()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    # Panda arm: 7 joints + 8 gripper joints + 7 cube freejoint qpos = 22 nq.
    assert model.nq == 22
    # 7 arm + 8 gripper + 6 cube freejoint velocity dofs = 21 nv.
    assert model.nv == 21
    # 7 arm motor actuators + 1 gripper tendon actuator = 8 nu.
    assert model.nu == 8


def test_env_reset_and_step():
    env = PandaPickEnv(seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (32,)
    assert obs.dtype == np.float64
    assert isinstance(info, dict)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (32,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance_cube_to_tray" in info
    assert "distance_ee_to_cube" in info
    assert "steps" in info
    env.close()


def test_gymnasium_api():
    env = PandaPickEnv(seed=0)
    check_env(env, skip_render_check=True)
    env.close()


def test_backward_reach_alias_still_works():
    """Old imports (PandaReachEnv, JakaReachEnv) should still instantiate the pick env."""
    from jaka_zu35_mujoco_rl import JakaReachEnv, PandaReachEnv

    env_reach = PandaReachEnv(seed=0)
    env_jaka = JakaReachEnv(seed=0)
    assert env_reach.action_space.shape == (8,)
    assert env_jaka.action_space.shape == (8,)
    env_reach.close()
    env_jaka.close()
