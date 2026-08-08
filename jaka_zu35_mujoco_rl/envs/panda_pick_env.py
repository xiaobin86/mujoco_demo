"""Minimal MuJoCo RL environment for Franka Panda pick-and-place of a small cube."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

import jaka_zu35_mujoco_rl
from jaka_zu35_mujoco_rl.vision import CubeDetector


_DEFAULT_REWARD_CONFIG: dict[str, Any] = {
    "version": "1.1",
    "description": "Default reward configuration for PandaPickEnv.",
    "rewards": {
        "time_penalty": -0.05,
        "approach": {"scale": 2.0, "distance": 0.1},
        "grasp": {"bonus": 2.0, "distance": 0.05},
        "lift": {"scale": 5.0, "height": 0.1, "reference_height": 0.015, "min_lifted_height": 0.08},
        "transport": {"scale": 2.0, "distance": 0.1, "ee_max_distance": 0.2},
        "success": {
            "bonus": 500.0,
            "threshold": 0.05,
            "hold_steps": 50,
            "min_height": 0.08,
            "require_lifted": True,
        },
        "drop": {"penalty": -50.0, "height": -0.05},
        "action": {"penalty": -0.0001},
        "push": {"penalty": -10.0, "distance": 0.06, "min_displacement": 0.001},
        "open_gripper": {"penalty": -0.5, "distance": 0.04},
    },
    "logging": {
        "log_reward_components": True,
        "log_success_rate": True,
        "success_rate_window": 100,
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_reward_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        return dict(_DEFAULT_REWARD_CONFIG)
    if isinstance(config, dict):
        user_config = config
    else:
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Reward config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
    merged = dict(_DEFAULT_REWARD_CONFIG)
    _deep_update(merged, user_config)
    return merged


def _load_reward_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    return load_reward_config(config)


class PandaPickEnv(gym.Env):
    """Gymnasium environment where a Franka Panda arm picks and places a red cube.

    The Panda model is loaded from ``panda_pick_scene.xml``, which composes the
    Franka Emika Panda arm + Robotiq 2F85 gripper, a table, a 5 cm red cube, and
    a target tray. A simulated overhead camera provides RGB images that are fed
    to a color-based cube detector to estimate the cube's 3D position.

    The Panda uses ``motor`` actuators (torque control). To keep the RL interface
    simple, actions are normalized desired joint positions and a fixed-gain PD
    controller computes the motor torques that track those positions. The 8th
    action dimension controls the gripper opening.

    Observation (32-dim):
        - arm joint positions (7)
        - arm joint velocities (7)
        - end-effector position (3)
        - gripper opening width (1)
        - estimated cube 3D position (3)
        - target tray position (3)
        - cube relative to end effector (3)
        - cube relative to target (3)
        - detector confidence (1)
        - normalized elapsed steps (1)

    Action (8-dim):
        - normalized joint position targets in [-1, 1] for the 7 arm joints
        - normalized gripper command in [-1, 1], mapped to actuator range

    Reward configuration:
        All reward coefficients and thresholds can be supplied via the
        ``reward_config`` argument (a dictionary or a path to a YAML file).
        See ``config/reward_panda_pick.yaml`` for the default structure.
    """

    # Offscreen rendering of named fixed cameras returns black images in this
    # environment, so we use a free camera positioned overhead for RGB capture
    # and vision-based cube detection.
    _OVERHEAD_LOOKAT = np.array([0.5, 0.0, 0.0])
    _OVERHEAD_DISTANCE = 1.2
    _OVERHEAD_AZIMUTH = 90.0
    _OVERHEAD_ELEVATION = -90.0

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        success_threshold: float = 0.05,
        success_hold_steps: int = 50,
        action_penalty: float = 0.005,
        pd_kp: float = 200.0,
        pd_kd: float = 40.0,
        seed: int | None = None,
        camera_fovy: float = 45.0,
        detector_noise_std: float = 0.005,
        reward_config: dict[str, Any] | str | Path | None = None,
        fast_observation: bool = False,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode {render_mode}. Must be one of {self.metadata['render_modes']}")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._fast_observation = bool(fast_observation)

        self._reward_config = _load_reward_config(reward_config)
        if reward_config is None:
            self._reward_config["rewards"]["success"]["threshold"] = success_threshold
            self._reward_config["rewards"]["success"]["hold_steps"] = success_hold_steps
            self._reward_config["rewards"]["action"]["penalty"] = action_penalty

        self.success_threshold = self._reward_config["rewards"]["success"]["threshold"]
        self.success_hold_steps = self._reward_config["rewards"]["success"]["hold_steps"]
        self.action_penalty = self._reward_config["rewards"]["action"]["penalty"]
        self._pd_kp = pd_kp
        self._pd_kd = pd_kd

        # Load model and data.
        model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
        xml_path = model_dir / "panda_pick_scene.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Model file not found: {xml_path}")

        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)

        # Arm joint indices (first 7 qpos/qvel; the Panda model places arm joints
        # before the 8 gripper joints).
        self._num_arm_joints = 7
        self._arm_qpos_ids = np.arange(0, self._num_arm_joints)
        self._arm_qvel_ids = np.arange(0, self._num_arm_joints)

        # Joint limits from the model.
        self._joint_low = self._model.jnt_range[:self._num_arm_joints, 0].copy()
        self._joint_high = self._model.jnt_range[:self._num_arm_joints, 1].copy()

        # Arm actuator indices and torque limits (motor actuators).
        self._arm_actuator_ids = np.arange(0, self._num_arm_joints)
        self._torque_low = self._model.actuator_ctrlrange[:self._num_arm_joints, 0].copy()
        self._torque_high = self._model.actuator_ctrlrange[:self._num_arm_joints, 1].copy()

        # Gripper actuator index (single tendon actuator, after the 7 arm motors).
        self._gripper_actuator_id = self._num_arm_joints
        self._gripper_ctrl_low = float(self._model.actuator_ctrlrange[self._gripper_actuator_id, 0])
        self._gripper_ctrl_high = float(self._model.actuator_ctrlrange[self._gripper_actuator_id, 1])

        # End-effector site id: the "pinch" site sits between the gripper fingers.
        self._ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

        cube_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self._cube_half_size = float(self._model.geom_size[cube_geom_id, 0])
        self._cube_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qpos_start = self._model.jnt_qposadr[cube_joint_id]
        self._cube_qpos_ids = np.arange(self._cube_qpos_start, self._cube_qpos_start + 7)

        self._tray_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "tray")

        # Use a free camera for offscreen rendering; model fixed cameras render
        # black in the current MuJoCo/Python setup.
        self._overhead_camera = mujoco.MjvCamera()
        self._overhead_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._overhead_camera.lookat[:] = self._OVERHEAD_LOOKAT
        self._overhead_camera.distance = self._OVERHEAD_DISTANCE
        self._overhead_camera.azimuth = self._OVERHEAD_AZIMUTH
        self._overhead_camera.elevation = self._OVERHEAD_ELEVATION

        camera_position = self._OVERHEAD_LOOKAT + np.array([0.0, 0.0, self._OVERHEAD_DISTANCE])
        camera_rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
        )

        self._table_height = 0.0

        self._detector = CubeDetector(
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            table_height=self._table_height,
            cube_half_size=self._cube_half_size,
            fovy=camera_fovy,
            image_size=(480, 640),
            noise_std=detector_noise_std,
        )

        # Spaces.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._num_arm_joints + 1,),
            dtype=np.float64,
        )

        self._steps = 0
        self._success_hold = 0
        self._cube_position = np.zeros(3, dtype=np.float64)
        self._tray_position = np.zeros(3, dtype=np.float64)
        self._gripper_opening = 0.0
        self._renderer: mujoco.Renderer | None = None

        self._np_random = np.random.default_rng(seed)

    @property
    def _np_random(self) -> np.random.Generator:
        return self.__np_random

    @_np_random.setter
    def _np_random(self, value: np.random.Generator | int | None) -> None:
        if isinstance(value, int):
            self.__np_random = np.random.default_rng(value)
        elif isinstance(value, np.random.Generator):
            self.__np_random = value
        elif value is None:
            self.__np_random = np.random.default_rng()
        else:
            raise TypeError("seed must be int, Generator, or None")

    def _get_tray_position(self) -> np.ndarray:
        """Return the tray target position in world frame (centre of tray floor)."""
        return self._data.xpos[self._tray_body_id].copy()

    def _get_cube_position(self) -> np.ndarray:
        """Return the cube centre position in world frame."""
        return self._data.xpos[self._cube_body_id].copy()

    def _get_gripper_opening(self) -> float:
        """Estimate gripper opening width from the pad geoms.

        Uses the absolute y-positions of the left and right pad bodies in the
        gripper frame. A rough approximation is sufficient for the RL policy.
        """
        try:
            left_pad_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "left_pad")
            right_pad_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "right_pad")
            left_y = self._data.xpos[left_pad_id, 1]
            right_y = self._data.xpos[right_pad_id, 1]
            return float(abs(left_y - right_y))
        except Exception:
            return 0.0

    def set_cube_position(self, position: np.ndarray) -> None:
        """Place the cube at a known world position for replay / debugging."""
        position = np.asarray(position, dtype=np.float64)
        cube_qpos = np.zeros(7, dtype=np.float64)
        cube_qpos[:3] = position
        cube_qpos[3] = 1.0
        self._data.qpos[self._cube_qpos_ids] = cube_qpos
        mujoco.mj_forward(self._model, self._data)

    def _detect_cube(self) -> dict[str, Any]:
        """Render an RGB image and run the cube detector."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        self._renderer.update_scene(self._data, camera=self._overhead_camera)
        rgb = self._renderer.render()
        return self._detector.detect(rgb, rng=self.__np_random)

    def _fast_detect_cube(self) -> dict[str, Any]:
        """Return the ground-truth cube position with detector noise, skipping offscreen rendering."""
        position = self._get_cube_position()
        if self._detector.noise_std > 0.0:
            position = position + self._np_random.normal(0.0, self._detector.noise_std, size=3)
        return {"detected": True, "position": position, "confidence": 1.0, "bbox": None}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._np_random = seed

        # Reset simulation.
        mujoco.mj_resetData(self._model, self._data)
        self._steps = 0
        self._success_hold = 0
        self._cube_lifted = False

        # Randomize arm joint positions within safe ranges.
        arm_qpos = self._np_random.uniform(
            low=self._joint_low * 0.7,
            high=self._joint_high * 0.7,
        )
        self._data.qpos[self._arm_qpos_ids] = arm_qpos

        # Keep the gripper open by setting its driver joints to their open limits.
        # The gripper occupies qpos indices 7..14 (8 joints). Setting the driver
        # joints to 0 opens the fingers; the follower joints are coupled by the
        # tendon and equality constraints, so we reset all gripper joints to 0.
        gripper_qpos_start = self._num_arm_joints
        gripper_qpos_end = self._num_arm_joints + 8
        self._data.qpos[gripper_qpos_start:gripper_qpos_end] = 0.0

        # Open gripper actuator.
        self._data.ctrl[self._gripper_actuator_id] = self._gripper_ctrl_low

        # Randomize cube position on the table.
        cube_x = self._np_random.uniform(0.30, 0.60)
        cube_y = self._np_random.uniform(-0.25, 0.25)
        cube_z = self._cube_half_size + self._table_height
        cube_pos = np.array([cube_x, cube_y, cube_z], dtype=np.float64)

        cube_qpos = np.zeros(7, dtype=np.float64)
        cube_qpos[:3] = cube_pos
        cube_qpos[3] = 1.0
        self._data.qpos[self._cube_qpos_ids] = cube_qpos

        # Forward kinematics to update site positions.
        mujoco.mj_forward(self._model, self._data)

        self._cube_position = self._get_cube_position()
        self._tray_position = self._get_tray_position()
        self._gripper_opening = self._get_gripper_opening()
        self._prev_cube_position = self._cube_position[:2].copy()

        obs = self._get_obs()
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)

        arm_action = action[:self._num_arm_joints]
        gripper_action = action[self._num_arm_joints]

        # Map normalized action to desired arm joint positions.
        desired_qpos = 0.5 * (arm_action + 1.0) * (self._joint_high - self._joint_low) + self._joint_low

        # PD controller: compute motor torques that track the desired positions.
        current_qpos = self._data.qpos[self._arm_qpos_ids].copy()
        current_qvel = self._data.qvel[self._arm_qvel_ids].copy()
        torque = self._pd_kp * (desired_qpos - current_qpos) - self._pd_kd * current_qvel
        torque = np.clip(torque, self._torque_low, self._torque_high)

        # Map gripper action to actuator range.
        gripper_ctrl = 0.5 * (gripper_action + 1.0) * (self._gripper_ctrl_high - self._gripper_ctrl_low) + self._gripper_ctrl_low

        self._data.ctrl[self._arm_actuator_ids] = torque
        self._data.ctrl[self._gripper_actuator_id] = gripper_ctrl

        # Step simulation.
        mujoco.mj_step(self._model, self._data)
        self._steps += 1

        # Detect simulation instability.
        if not (np.isfinite(self._data.qpos).all() and np.isfinite(self._data.qacc).all()):
            print("WARNING: Simulation unstable, resetting episode.")
            obs, reset_info = self.reset()
            reset_info["unstable"] = True
            reset_info["distance"] = float("inf")
            return obs, -100.0, False, True, reset_info

        # Update state estimates.
        detection = self._fast_detect_cube() if self._fast_observation else self._detect_cube()
        self._cube_position = detection["position"] if detection["detected"] else self._get_cube_position()
        self._tray_position = self._get_tray_position()
        self._gripper_opening = self._get_gripper_opening()

        cube_pos = self._get_cube_position()
        ee_pos = self._data.site_xpos[self._ee_site_id].copy()
        tray_pos = self._tray_position

        distance_ee_to_cube = float(np.linalg.norm(ee_pos - cube_pos))
        distance_cube_to_tray = float(np.linalg.norm(cube_pos - tray_pos))
        cube_height = cube_pos[2] - self._table_height

        rc = self._reward_config["rewards"]

        cube_horizontal_displacement = float(np.linalg.norm(cube_pos[:2] - self._prev_cube_position))
        if cube_height >= rc["lift"]["min_lifted_height"]:
            self._cube_lifted = True

        time_reward = float(rc["time_penalty"])
        approach_reward = float(
            rc["approach"]["scale"] * np.exp(-distance_ee_to_cube / rc["approach"]["distance"])
        )

        gripper_command = float(action[self._num_arm_joints])
        grasp_reward = 0.0
        if distance_ee_to_cube < rc["grasp"]["distance"] and gripper_command > 0.0:
            grasp_reward = float(rc["grasp"]["bonus"])

        lift_height = cube_height - rc["lift"]["reference_height"]
        lift_reward = float(rc["lift"]["scale"] * np.clip(lift_height / rc["lift"]["height"], 0.0, 1.0))

        transport_reward = 0.0
        if self._cube_lifted and distance_ee_to_cube < rc["transport"]["ee_max_distance"]:
            transport_reward = float(
                rc["transport"]["scale"] * np.exp(-distance_cube_to_tray / rc["transport"]["distance"])
            )

        terminated = False
        success = False
        success_min_height = rc["success"]["min_height"]
        success_require_lifted = rc["success"].get("require_lifted", False)
        success_condition = (
            distance_cube_to_tray < self.success_threshold
            and cube_height > success_min_height
            and (not success_require_lifted or self._cube_lifted)
        )
        if success_condition:
            self._success_hold += 1
            if self._success_hold >= self.success_hold_steps:
                success = True
                terminated = True
        else:
            self._success_hold = 0

        success_reward = float(rc["success"]["bonus"]) if success else 0.0

        drop_reward = 0.0
        if cube_height < rc["drop"]["height"]:
            drop_reward = float(rc["drop"]["penalty"])

        action_reward = float(rc["action"]["penalty"] * np.sum(action ** 2))

        push_reward = 0.0
        if cube_height < rc["lift"]["min_lifted_height"] and distance_ee_to_cube < rc["push"]["distance"]:
            if cube_horizontal_displacement > rc["push"]["min_displacement"]:
                push_reward = float(rc["push"]["penalty"] * cube_horizontal_displacement)

        open_gripper_reward = 0.0
        if (
            cube_height < rc["lift"]["min_lifted_height"]
            and distance_ee_to_cube < rc["open_gripper"]["distance"]
            and gripper_command > 0.0
        ):
            open_gripper_reward = float(rc["open_gripper"]["penalty"])

        reward = (
            time_reward
            + approach_reward
            + grasp_reward
            + lift_reward
            + transport_reward
            + success_reward
            + drop_reward
            + action_reward
            + push_reward
            + open_gripper_reward
        )

        self._prev_cube_position = cube_pos[:2].copy()

        truncated = self._steps >= self.max_episode_steps

        obs = self._get_obs()
        info = {
            "distance_ee_to_cube": distance_ee_to_cube,
            "distance_cube_to_tray": distance_cube_to_tray,
            "cube_height": cube_height,
            "cube_lifted": self._cube_lifted,
            "detected": detection["detected"],
            "confidence": detection["confidence"],
            "steps": self._steps,
            "success_hold": self._success_hold,
            "is_success": success,
        }
        if self._reward_config["logging"]["log_reward_components"]:
            info["reward_components"] = {
                "time": time_reward,
                "approach": approach_reward,
                "grasp": grasp_reward,
                "lift": lift_reward,
                "transport": transport_reward,
                "success": success_reward,
                "drop": drop_reward,
                "action": action_reward,
                "push": push_reward,
                "open_gripper": open_gripper_reward,
            }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self) -> np.ndarray:
        qpos = self._data.qpos[self._arm_qpos_ids].copy()
        qvel = self._data.qvel[self._arm_qvel_ids].copy()
        ee_pos = self._data.site_xpos[self._ee_site_id].copy()
        cube_pos = self._cube_position
        tray_pos = self._tray_position
        confidence = 1.0  # always trust position estimate in simulation
        elapsed = float(self._steps) / float(self.max_episode_steps)

        return np.concatenate(
            [
                qpos,
                qvel,
                ee_pos,
                np.array([self._gripper_opening]),
                cube_pos,
                tray_pos,
                cube_pos - ee_pos,
                cube_pos - tray_pos,
                np.array([confidence]),
                np.array([elapsed]),
            ]
        )

    # Optional: expose MuJoCo internals for advanced users.
    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data


# Backward-compatibility aliases: old code importing the reach class names will
# receive the new pick-place implementation. This keeps the training scripts and
# tests working while the project focus has shifted to grasping.
PandaReachEnv = PandaPickEnv
JakaReachEnv = PandaPickEnv
