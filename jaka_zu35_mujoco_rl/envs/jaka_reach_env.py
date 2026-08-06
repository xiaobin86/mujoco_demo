"""Minimal MuJoCo RL environment for JAKA Zu35 reaching a single box."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import jaka_zu35_mujoco_rl


class JakaReachEnv(gym.Env):
    """Gymnasium environment where a simplified JAKA Zu35 arm reaches a cola_24 box.

    Observation (18-dim):
        - joint positions (6)
        - joint velocities (6)
        - end-effector position (3)
        - target box top position (3)

    Action (6-dim):
        - normalized joint position targets in [-1, 1], mapped to each joint's ctrl range.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        success_threshold: float = 0.05,
        action_penalty: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode {render_mode}. Must be one of {self.metadata['render_modes']}")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.action_penalty = action_penalty

        # Load model and data.
        model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
        xml_path = model_dir / "jaka_zu35.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Model file not found: {xml_path}")

        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)

        # Arm joint indices (first 6 qpos/qvel after excluding box freejoint).
        self._arm_qpos_ids = np.arange(0, 6)
        self._arm_qvel_ids = np.arange(0, 6)
        self._arm_actuator_ids = np.arange(0, 6)

        # Joint limits from the model.
        self._joint_low = self._model.jnt_range[:6, 0].copy()
        self._joint_high = self._model.jnt_range[:6, 1].copy()

        # End-effector site id.
        self._ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self._box_top_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "box_top")
        self._box_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "target_box")

        # Box qpos indices (freejoint, 7 dofs: pos + quat).
        box_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self._box_qpos_start = self._model.jnt_qposadr[box_joint_id]
        self._box_qpos_ids = np.arange(self._box_qpos_start, self._box_qpos_start + 7)

        # Spaces.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float64,
        )

        self._steps = 0
        self._target_box_top_pos = np.zeros(3, dtype=np.float64)
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

        # Randomize arm joint positions within safe ranges.
        # Use 70% of the full range to avoid extreme postures near singularities.
        arm_qpos = self._np_random.uniform(
            low=self._joint_low * 0.7,
            high=self._joint_high * 0.7,
        )
        self._data.qpos[self._arm_qpos_ids] = arm_qpos

        # Randomize target box position on the table.
        # Box size is 0.4 x 0.27 x 0.24; half-height is 0.12.
        box_x = self._np_random.uniform(0.30, 0.70)
        box_y = self._np_random.uniform(-0.30, 0.30)
        box_z = 0.12
        box_pos = np.array([box_x, box_y, box_z], dtype=np.float64)

        # Set box qpos: position + identity quaternion (w, x, y, z).
        box_qpos = np.zeros(7, dtype=np.float64)
        box_qpos[:3] = box_pos
        box_qpos[3] = 1.0  # quaternion w
        self._data.qpos[self._box_qpos_ids] = box_qpos
        self._target_box_top_pos = box_pos + np.array([0.0, 0.0, 0.12])

        # Forward kinematics to update site positions.
        mujoco.mj_forward(self._model, self._data)

        obs = self._get_obs()
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)

        # Map normalized action to joint ctrl range.
        ctrl = 0.5 * (action + 1.0) * (self._joint_high - self._joint_low) + self._joint_low
        self._data.ctrl[:6] = ctrl

        # Step simulation.
        mujoco.mj_step(self._model, self._data)
        self._steps += 1

        obs = self._get_obs()
        ee_pos = self._data.site_xpos[self._ee_site_id].copy()
        distance = float(np.linalg.norm(ee_pos - self._target_box_top_pos))

        reward = -distance - self.action_penalty * float(np.sum(action ** 2))

        terminated = distance < self.success_threshold
        truncated = self._steps >= self.max_episode_steps

        info = {"distance": distance, "steps": self._steps}
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
        return np.concatenate([qpos, qvel, ee_pos, self._target_box_top_pos])

    # Optional: expose MuJoCo internals for advanced users.
    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data
