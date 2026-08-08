import cv2
import numpy as np
import pytest
import torch

from jaka_zu35_mujoco_rl.act import ACTPolicy
from jaka_zu35_mujoco_rl.envs import make_env
from jaka_zu35_mujoco_rl.teleop_utils import (
    JOINT_NAMES,
    WRIST_ROLL_OFFSET_DEG,
    flip_image_if_needed,
    master_to_action,
    reset_wrist_roll_unwrap,
    set_env_cube_position,
    sync_env_to_action,
)


@pytest.fixture(autouse=True)
def _reset_wrist_roll_unwrap_state():
    # master_to_action unwraps wrist_roll across calls (stateful); each test
    # starts from a clean state so isolated assertions stay deterministic.
    reset_wrist_roll_unwrap()
    yield


def test_master_to_action_zero():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    master = {f"{name}.pos": 0.0 for name in JOINT_NAMES[:5]}
    # Cancel the hardcoded wrist_roll offset so that the resulting action is 0.
    master["wrist_roll.pos"] = -WRIST_ROLL_OFFSET_DEG
    master["gripper.pos"] = 50.0
    action = master_to_action(master, env)
    assert action.shape == (6,)
    assert np.allclose(action, 0.0, atol=1e-6)
    env.close()


def test_master_to_action_full_range():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    half_ranges_deg = np.rad2deg((env._joint_high - env._joint_low) / 2.0)
    master = {f"{name}.pos": half_ranges_deg[i] for i, name in enumerate(JOINT_NAMES[:5])}
    # Wrist_roll offset must be cancelled to reach the positive limit.
    master["wrist_roll.pos"] = half_ranges_deg[4] - WRIST_ROLL_OFFSET_DEG
    master["gripper.pos"] = 100.0
    action = master_to_action(master, env)
    expected = np.ones(6, dtype=np.float64)
    assert np.allclose(action, expected, atol=1e-6)

    master = {f"{name}.pos": -half_ranges_deg[i] for i, name in enumerate(JOINT_NAMES[:5])}
    master["wrist_roll.pos"] = -half_ranges_deg[4] - WRIST_ROLL_OFFSET_DEG
    master["gripper.pos"] = 0.0
    reset_wrist_roll_unwrap()
    action = master_to_action(master, env)
    expected = -np.ones(6, dtype=np.float64)
    assert np.allclose(action, expected, atol=1e-6)
    env.close()


def test_master_to_action_wrist_roll_offset():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    half = float(np.rad2deg((env._joint_high[4] - env._joint_low[4]) / 2.0))
    master = {f"{name}.pos": 0.0 for name in JOINT_NAMES[:5]}
    master["gripper.pos"] = 50.0
    master["wrist_roll.pos"] = 0.0
    action = master_to_action(master, env)
    expected = WRIST_ROLL_OFFSET_DEG / half
    assert np.isclose(action[4], expected, atol=1e-6)
    env.close()


def test_master_to_action_wrist_roll_clamps_to_limits():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    master = {f"{name}.pos": 0.0 for name in JOINT_NAMES[:5]}
    master["gripper.pos"] = 50.0
    master["wrist_roll.pos"] = 300.0
    action = master_to_action(master, env)
    assert action[4] == 1.0
    env.close()


def test_master_to_action_gripper_maps_to_normalized_range():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    master = {f"{name}.pos": 0.0 for name in JOINT_NAMES[:5]}
    master["gripper.pos"] = 0.0
    assert np.isclose(master_to_action(master, env)[-1], -1.0, atol=1e-6)
    master["gripper.pos"] = 100.0
    assert np.isclose(master_to_action(master, env)[-1], 1.0, atol=1e-6)
    env.close()


def test_act_forward_shape():
    model = ACTPolicy(proprio_dim=11, action_dim=6, encoder_pretrained=False)
    img = torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)
    prop = torch.randn(2, 11)
    out = model(img, prop)
    assert out.shape == (2, 8, 6)
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6


def test_sync_env_to_action_sets_qpos_and_ctrl():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    action = np.ones(6, dtype=np.float64)
    sync_env_to_action(env, action)

    arm_high = env.model.jnt_range[env._arm_qpos_ids, 1]
    assert np.allclose(env.data.qpos[env._arm_qpos_ids], arm_high, atol=1e-6)
    assert np.allclose(env.data.ctrl[env._arm_actuator_ids], arm_high, atol=1e-6)
    assert env.data.qpos[env._gripper_qpos_id] == env._gripper_ctrl_high


def test_set_env_cube_position_moves_cube():
    env = make_env("so101", render_mode="rgb_array")
    env.reset(seed=0)
    target = np.array([0.2, 0.05, 0.05], dtype=np.float64)
    set_env_cube_position(env, target)
    assert np.allclose(env._get_cube_position(), target, atol=1e-6)


def test_flip_image_if_needed_flips_wrist_cam_horizontally():
    image = np.arange(12).reshape(2, 2, 3)
    flipped = flip_image_if_needed(image, "wrist_cam", flip=True)
    assert np.array_equal(flipped, cv2.flip(image, 1))
    assert np.array_equal(flip_image_if_needed(image, "overhead", flip=True), image)
    assert np.array_equal(flip_image_if_needed(image, "wrist_cam", flip=False), image)
