"""Utilities for mapping a LeRobot SO101 master arm to the simulated slave arm."""

from __future__ import annotations

from typing import Any

import cv2
import mujoco
import numpy as np

JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

_BODY_JOINT_NAMES: list[str] = JOINT_NAMES[:5]

# The LeRobot SO101 leader uses MotorNormMode.DEGREES for body joints by default
# and MotorNormMode.RANGE_0_100 for the gripper.
_GRIPPER_NORM_RANGE = 50.0

# The body joints of the SO101 master arm use the same positive direction as the
# MuJoCo model by default, so all joint signs are +1.
_BODY_JOINT_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

# The physical gripper/wrist camera is rotated -90 degrees relative to the MuJoCo
# model when viewed from the +x direction, so add a fixed offset to wrist_roll.
WRIST_ROLL_OFFSET_DEG = -90.0

_BODY_JOINT_OFFSETS_DEG = np.array([0.0, 0.0, 0.0, 0.0, WRIST_ROLL_OFFSET_DEG], dtype=np.float64)

# wrist_roll is a continuous-rotation joint, but the leader servo reports a
# single-turn absolute angle that wraps at +/-180 deg. Without unwrapping, the
# mapped slave target jumps by a full turn every time the master crosses the
# wrap point. We track cumulative rotation here so the slave follows smoothly.
_WRIST_ROLL_INDEX = 4
_wrist_roll_unwrap_state: dict[str, float | None] = {"prev_reported": None, "cumulative": 0.0}


def reset_wrist_roll_unwrap() -> None:
    """Clear the wrist_roll unwrap state (call at the start of a teleop session)."""
    _wrist_roll_unwrap_state["prev_reported"] = None
    _wrist_roll_unwrap_state["cumulative"] = 0.0


def _unwrap_wrist_roll(reported_deg: float) -> float:
    """Convert the leader's wrapping wrist_roll reading into a continuous angle."""
    state = _wrist_roll_unwrap_state
    prev = state["prev_reported"]
    if prev is None:
        state["prev_reported"] = reported_deg
        state["cumulative"] = reported_deg
        return reported_deg
    delta = reported_deg - prev
    if delta > 180.0:
        delta -= 360.0
    elif delta < -180.0:
        delta += 360.0
    state["cumulative"] += delta
    state["prev_reported"] = reported_deg
    return state["cumulative"]


def master_to_action(master_obs: dict[str, float], env: Any) -> np.ndarray:
    """Map a LeRobot SO101 master arm observation to the slave env's action.

    The SO101 leader uses ``SO101LeaderConfig(use_degrees=True)``, so the 5 body
    joints arrive as degrees (relative to the calibration zero) and the gripper
    arrives in ``[0, 100]``.  We map degrees to the simulation joint ranges,
    normalized to ``[-1, 1]``, and the gripper to ``[-1, 1]`` where ``-1`` is
    closed and ``+1`` is open.
    """
    arm_half_range_deg = np.rad2deg((env._joint_high - env._joint_low) / 2.0)
    body_pos = np.array([float(master_obs[f"{name}.pos"]) for name in _BODY_JOINT_NAMES])
    body_pos[_WRIST_ROLL_INDEX] = _unwrap_wrist_roll(body_pos[_WRIST_ROLL_INDEX])
    body_action = np.clip(
        (body_pos + _BODY_JOINT_OFFSETS_DEG) / arm_half_range_deg * _BODY_JOINT_SIGNS,
        -1.0,
        1.0,
    )
    gripper_action = np.clip(
        (float(master_obs["gripper.pos"]) - _GRIPPER_NORM_RANGE) / _GRIPPER_NORM_RANGE,
        -1.0,
        1.0,
    )
    return np.concatenate([body_action, np.array([gripper_action])])


def sync_env_to_action(env: Any, action: np.ndarray) -> None:
    """Set the simulation arm and gripper to the pose described by a [-1, 1] action.

    This is used both during teleoperation and during replay so that the
    simulation starts exactly at the commanded pose without actuator lag.
    """
    action = np.clip(action, -1.0, 1.0)
    arm_action = action[: len(env._arm_qpos_ids)]
    arm_low = env.model.jnt_range[env._arm_qpos_ids, 0]
    arm_high = env.model.jnt_range[env._arm_qpos_ids, 1]
    arm_qpos = arm_low + (arm_action + 1.0) / 2.0 * (arm_high - arm_low)
    env.data.qpos[env._arm_qpos_ids] = arm_qpos
    env.data.ctrl[env._arm_actuator_ids] = np.clip(arm_qpos, arm_low, arm_high)

    gripper_low = env._gripper_ctrl_low
    gripper_high = env._gripper_ctrl_high
    gripper_action = action[-1]
    gripper_qpos = gripper_low + (gripper_action + 1.0) / 2.0 * (gripper_high - gripper_low)
    env.data.qpos[env._gripper_qpos_id] = gripper_qpos
    env.data.ctrl[env._gripper_actuator_id] = gripper_qpos

    mujoco.mj_forward(env.model, env.data)


def set_env_cube_position(env: Any, position: np.ndarray) -> None:
    """Reset the simulated cube to a known world position (for replay)."""
    position = np.asarray(position, dtype=np.float64)
    cube_qpos = np.zeros(7, dtype=np.float64)
    cube_qpos[:3] = position
    cube_qpos[3] = 1.0
    env.data.qpos[env._cube_qpos_ids] = cube_qpos
    mujoco.mj_forward(env.model, env.data)


def flip_image_if_needed(
    image: np.ndarray,
    camera: str,
    flip: bool,
) -> np.ndarray:
    """Horizontally flip a wrist camera image when the physical mount is mirrored."""
    if flip and camera == "wrist_cam":
        return cv2.flip(image, 1)
    return image
