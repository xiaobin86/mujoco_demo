import numpy as np
import pytest
import mujoco
from gymnasium.utils.env_checker import check_env

import jaka_zu35_mujoco_rl
from jaka_zu35_mujoco_rl import JakaReachEnv
from pathlib import Path


def test_model_loads():
    model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
    xml_path = model_dir / "jaka_zu35.xml"
    assert xml_path.exists()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nq == 13  # 6 arm joints + 7 freejoint box
    assert model.nv == 12  # 6 arm dofs + 6 box dofs
    assert model.nu == 6   # 6 actuators


def test_env_reset_and_step():
    env = JakaReachEnv(seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (18,)
    assert obs.dtype == np.float64
    assert isinstance(info, dict)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (18,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info
    assert "steps" in info
    env.close()


def test_gymnasium_api():
    env = JakaReachEnv(seed=0)
    check_env(env, skip_render_check=True)
    env.close()
