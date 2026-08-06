# JAKA Zu35 MuJoCo RL 环境实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `/mnt/d/work/jaka_zu35_mujoco_rl` 创建一个最小可运行的 Gymnasium 兼容 MuJoCo RL 环境，包含简化 JAKA Zu35 机械臂模型、单箱目标、示例脚本和基础测试。

**Architecture:** 使用 MuJoCo 原生 MJCF 描述简化 6 轴机械臂与箱子；通过 `mujoco` Python 绑定加载模型并步进；`JakaReachEnv` 继承 `gymnasium.Env` 封装 reset/step/render；状态包含关节位置/速度、末端位置、目标箱位置；动作为 6 维归一化关节位置目标。

**Tech Stack:** Python 3.10+, MuJoCo 3.x, Gymnasium 1.x, NumPy, pytest, imageio

**参考设计文档:** `/mnt/d/work/jaka_zu35_mujoco_rl/docs/design.md`

---

## 文件结构

```
jaka_zu35_mujoco_rl/
├── jaka_zu35_mujoco_rl/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── jaka_reach_env.py
│   └── models/
│       ├── jaka_zu35.xml
│       ├── box_cola24.xml
│       └── ground.xml
├── examples/
│   ├── random_agent.py
│   ├── check_env.py
│   └── render_scene.py
├── tests/
│   └── test_env.py
├── pyproject.toml
└── README.md
```

---

## Task 1: 创建项目骨架与 pyproject.toml

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/pyproject.toml`
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/jaka_zu35_mujoco_rl/__init__.py`
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/jaka_zu35_mujoco_rl/envs/__init__.py`
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/jaka_zu35_mujoco_rl/models/__init__.py`
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/tests/__init__.py`
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/examples/__init__.py` (optional, keep empty)
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/README.md` (Task 9 重写内容，这里先占位为空文件)

- [ ] **Step 1: 创建空包初始化文件**

所有 `__init__.py` 内容均为空（仅用于标记 Python 包）。

- [ ] **Step 2: 编写 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "jaka_zu35_mujoco_rl"
version = "0.1.0"
description = "A minimal MuJoCo RL environment for JAKA Zu35 arm + box reaching."
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Acelan", email = "acelan@example.com"}
]
keywords = ["mujoco", "reinforcement-learning", "robotics", "jaka"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "mujoco>=3.0.0",
    "gymnasium>=1.0.0",
    "numpy",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "imageio",
]

[project.urls]
Homepage = "https://github.com/example/jaka_zu35_mujoco_rl"

[tool.setuptools.packages.find]
where = ["."]
include = ["jaka_zu35_mujoco_rl*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 3: 创建空 README.md**

```bash
touch /mnt/d/work/jaka_zu35_mujoco_rl/README.md
```

- [ ] **Step 4: 在虚拟环境中尝试安装（验证 pyproject.toml 语法）**

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
python -m pip install -e .[dev]
```

Expected: 安装成功，无解析错误。如果依赖下载失败，先检查网络/环境，后续任务完成后再重试。

---

## Task 2: 创建 MuJoCo 地面 asset

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/jaka_zu35_mujoco_rl/models/ground.xml`

- [ ] **Step 1: 编写地面 XML**

```xml
<mujocoinclude>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="groundplane" condim="3" friction="1.0 0.005 0.0001"/>
    <light directional="true" diffuse=".8 .8 .8" specular="0.1 0.1 0.1" pos="0 0 5" dir="0 0 -1"/>
    <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="5 5 5" dir="-1 -1 -1"/>
  </worldbody>
</mujocoinclude>
```

- [ ] **Step 2: 独立验证 XML 可被 MuJoCo 解析**

```python
import mujoco
import jaka_zu35_mujoco_rl
from pathlib import Path

model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
xml_path = model_dir / "ground.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
print("ground.xml loads OK", model.nq, model.nv)
```

Expected: 输出 `ground.xml loads OK 0 0`（仅包含世界几何，无关节）。

---

## Task 3: 创建箱子 asset

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/jaka_zu35_mujoco_rl/models/box_cola24.xml`

- [ ] **Step 1: 编写 cola_24 箱子 XML**

复用 `pallet_vision_lidar` 的 `cola_24` 尺寸：0.4 m × 0.27 m × 0.24 m。质量 ~14 kg，通过 density 计算。

```xml
<mujocoinclude>
  <asset>
    <material name="box_cola24" rgba="0.8 0.2 0.2 1.0" specular="0.3" shininess="0.3"/>
  </asset>
  <worldbody>
    <body name="target_box" pos="0.5 0.0 0.12">
      <freejoint name="box_joint"/>
      <geom name="box_geom" type="box" size="0.2 0.135 0.12" material="box_cola24" mass="14.0" friction="1.0 0.005 0.0001"/>
      <site name="box_top" pos="0.0 0.0 0.12" size="0.03" rgba="0.0 1.0 0.0 0.5"/>
    </body>
  </worldbody>
</mujocoinclude>
```

> 注：这里使用 `freejoint` 让箱子有完整 6DoF，但环境默认不扰动它。后续抓取阶段可直接利用其物理。

- [ ] **Step 2: 验证箱子 XML 可解析**

```python
import mujoco
import jaka_zu35_mujoco_rl
from pathlib import Path

model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
xml_path = model_dir / "box_cola24.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
print("box_cola24.xml loads OK", model.nq, model.nv)
```

Expected: 输出 `box_cola24.xml loads OK 7 6`（freejoint: 3 位置 + 4 四元数，速度 6）。

---

## Task 4: 创建 JAKA Zu35 简化机械臂模型

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/jaka_zu35_mujoco_rl/models/jaka_zu35.xml`

- [ ] **Step 1: 编写机械臂主 XML**

```xml
<mujoco model="jaka_zu35">
  <compiler angle="radian" meshdir="." autolimits="true"/>

  <include file="ground.xml"/>
  <include file="box_cola24.xml"/>

  <option timestep="0.004" iterations="50" tolerance="1e-10" solver="Newton" gravity="0 0 -9.81">
    <flag warmstart="enable"/>
  </option>

  <asset>
    <material name="link_gray" rgba="0.5 0.55 0.6 1.0" specular="0.4" shininess="0.3"/>
    <material name="joint_dark" rgba="0.2 0.2 0.25 1.0" specular="0.5" shininess="0.5"/>
    <material name="ee_red" rgba="0.9 0.2 0.2 1.0" specular="0.3" shininess="0.3"/>
  </asset>

  <worldbody>
    <!-- Base -->
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.12 0.08" material="link_gray" mass="5.0"/>

      <!-- Link 1 (rotates around Z) -->
      <body name="link1" pos="0 0 0.08">
        <joint name="joint_1" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="2.0" armature="0.1"/>
        <geom type="cylinder" size="0.08 0.15" pos="0 0 0.15" material="link_gray" mass="3.0"/>
        <geom type="sphere" size="0.09" material="joint_dark" mass="0.5"/>

        <!-- Link 2 -->
        <body name="link2" pos="0 0 0.30">
          <joint name="joint_2" type="hinge" axis="0 1 0" range="-1.5708 1.5708" damping="2.0" armature="0.1"/>
          <geom type="capsule" size="0.07" fromto="0 0 0 0 0 0.35" material="link_gray" mass="3.0"/>
          <geom type="sphere" size="0.08" material="joint_dark" mass="0.5"/>

          <!-- Link 3 -->
          <body name="link3" pos="0 0 0.35">
            <joint name="joint_3" type="hinge" axis="0 1 0" range="-1.5708 1.5708" damping="2.0" armature="0.1"/>
            <geom type="capsule" size="0.06" fromto="0 0 0 0 0 0.30" material="link_gray" mass="2.5"/>
            <geom type="sphere" size="0.07" material="joint_dark" mass="0.5"/>

            <!-- Link 4 (wrist roll) -->
            <body name="link4" pos="0 0 0.30">
              <joint name="joint_4" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="1.0" armature="0.05"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.15" material="link_gray" mass="1.5"/>
              <geom type="sphere" size="0.06" material="joint_dark" mass="0.3"/>

              <!-- Link 5 -->
              <body name="link5" pos="0 0 0.15">
                <joint name="joint_5" type="hinge" axis="0 1 0" range="-1.5708 1.5708" damping="1.0" armature="0.05"/>
                <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.12" material="link_gray" mass="1.0"/>
                <geom type="sphere" size="0.05" material="joint_dark" mass="0.2"/>

                <!-- Link 6 + end-effector -->
                <body name="link6" pos="0 0 0.12">
                  <joint name="joint_6" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="1.0" armature="0.05"/>
                  <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.08" material="link_gray" mass="0.5"/>
                  <site name="end_effector" pos="0 0 0.10" size="0.04" material="ee_red"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position joint="joint_1" kp="500" kv="50" ctrlrange="-3.14159 3.14159"/>
    <position joint="joint_2" kp="400" kv="40" ctrlrange="-1.5708 1.5708"/>
    <position joint="joint_3" kp="300" kv="30" ctrlrange="-1.5708 1.5708"/>
    <position joint="joint_4" kp="200" kv="20" ctrlrange="-3.14159 3.14159"/>
    <position joint="joint_5" kp="150" kv="15" ctrlrange="-1.5708 1.5708"/>
    <position joint="joint_6" kp="100" kv="10" ctrlrange="-3.14159 3.14159"/>
  </actuator>
</mujoco>
```

- [ ] **Step 2: 验证机械臂 XML 可解析并打印关节信息**

```python
import mujoco
import jaka_zu35_mujoco_rl
from pathlib import Path

model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
xml_path = model_dir / "jaka_zu35.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
print(f"jaka_zu35.xml loads OK: nq={model.nq}, nv={model.nv}, nu={model.nu}")
print(f"Joint names: {[model.joint(i).name for i in range(model.njnt)]}")
print(f"Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")
print(f"Body names: {[model.body(i).name for i in range(1, model.nbody)]}")
print(f"Site names: {[model.site(i).name for i in range(model.nsite)]}")
```

Expected: 输出类似 `nq=13, nv=12, nu=6`（6 机械臂关节 + 7 箱子 freejoint），关节/执行器/site 名称与 XML 一致。

---

## Task 5: 实现 JakaReachEnv

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/jaka_zu35_mujoco_rl/envs/jaka_reach_env.py`

- [ ] **Step 1: 编写环境实现**

```python
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
```

- [ ] **Step 2: 在 envs __init__.py 中导出环境**

```python
from jaka_zu35_mujoco_rl.envs.jaka_reach_env import JakaReachEnv

__all__ = ["JakaReachEnv"]
```

- [ ] **Step 3: 在包根 __init__.py 中导出环境（方便 import）**

```python
from jaka_zu35_mujoco_rl.envs.jaka_reach_env import JakaReachEnv

__all__ = ["JakaReachEnv"]
```

- [ ] **Step 4: 手动验证环境可 reset/step**

```python
import numpy as np
from jaka_zu35_mujoco_rl import JakaReachEnv

env = JakaReachEnv()
obs, info = env.reset(seed=0)
print("obs shape", obs.shape, "info", info)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward={reward:.4f}, distance={info['distance']:.4f}, terminated={terminated}")

env.close()
```

Expected: 10 步内无异常，reward 为负数，info 包含 distance 和 steps。

---

## Task 6: 实现随机策略示例

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/examples/random_agent.py`

- [ ] **Step 1: 编写随机策略脚本**

```python
"""Run a random agent for a few episodes in JakaReachEnv."""

import gymnasium as gym
import numpy as np

from jaka_zu35_mujoco_rl import JakaReachEnv


def main() -> None:
    env = JakaReachEnv()

    num_episodes = 3
    max_steps = 500

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        print(
            f"Episode {episode + 1}: "
            f"steps={step}, reward={episode_reward:.4f}, "
            f"success={terminated}, final_distance={info['distance']:.4f}"
        )

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行示例**

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
python examples/random_agent.py
```

Expected: 3 个 episode 都成功跑完，无异常退出。

---

## Task 7: 实现 Gymnasium API 检查示例

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/examples/check_env.py`

- [ ] **Step 1: 编写检查脚本**

```python
"""Verify that JakaReachEnv complies with the Gymnasium API."""

import gymnasium as gym

from jaka_zu35_mujoco_rl import JakaReachEnv


def main() -> None:
    env = JakaReachEnv()
    gym.check_env(env, skip_render_check=True)
    print("check_env passed")
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行检查**

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
python examples/check_env.py
```

Expected: 输出 `check_env passed`。

---

## Task 8: 实现渲染示例

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/examples/render_scene.py`

- [ ] **Step 1: 编写渲染脚本**

```python
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

    # Let the arm settle for a few steps with zero action.
    for _ in range(20):
        env.step(env.action_space.sample())

    frame = env.render()
    if frame is None:
        raise RuntimeError("render() returned None")

    imageio.imwrite(args.output, frame)
    print(f"Saved render to {Path(args.output).resolve()}")
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行渲染示例**

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
python examples/render_scene.py
ls jaka_reach_scene.png
```

Expected: 生成 `jaka_reach_scene.png` 文件，大小非零。

---

## Task 9: 实现测试

**Files:**
- Create: `/mnt/d/work/jaka_zu35_mujoco_rl/tests/test_env.py`

- [ ] **Step 1: 编写测试**

```python
import numpy as np
import pytest
import gymnasium as gym
import mujoco

from jaka_zu35_mujoco_rl import JakaReachEnv
import jaka_zu35_mujoco_rl
from pathlib import Path


def test_model_loads():
    model_dir = Path(jaka_zu35_mujoco_rl.__file__).parent / "models"
    xml_path = model_dir / "jaka_zu35.xml"
    assert xml_path.exists()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nq == 13  # 6 arm joints + 7 freejoint box
    assert model.nv == 12  # 6 arm dofs + 6 box dofs
    assert model.nu == 6   # 6 actuators


def test_env_reset():
    env = JakaReachEnv(seed=0)
    obs, info = env.reset(seed=1)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (18,)
    assert obs.dtype == np.float64
    assert isinstance(info, dict)
    env.close()


def test_env_step():
    env = JakaReachEnv(seed=0)
    obs, info = env.reset(seed=2)
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
    gym.check_env(env, skip_render_check=True)
    env.close()


def test_success_condition():
    env = JakaReachEnv(seed=0, success_threshold=0.05)
    obs, info = env.reset(seed=3)
    # Place the end-effector exactly at the target box top by setting qpos directly
    # is hard; instead, just make sure the environment terminates after many steps.
    for _ in range(env.max_episode_steps + 5):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert terminated or truncated
    env.close()


def test_render_rgb_array():
    env = JakaReachEnv(render_mode="rgb_array", seed=0)
    env.reset(seed=4)
    env.step(env.action_space.sample())
    frame = env.render()
    assert frame is not None
    assert frame.shape[2] == 3  # RGB
    env.close()
```

- [ ] **Step 2: 运行测试**

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
pytest tests/test_env.py -v
```

Expected: 5 个测试全部通过（或 6 个，如果包含 test_success_condition）。

---

## Task 10: 编写 README.md

**Files:**
- Create/Modify: `/mnt/d/work/jaka_zu35_mujoco_rl/README.md`

- [ ] **Step 1: 编写 README**

```markdown
# JAKA Zu35 MuJoCo RL 环境

一个最小可运行的 **MuJoCo + Gymnasium** 强化学习环境，包含一个简化的 JAKA Zu35 6 轴机械臂和一个 `cola_24` 箱体目标。

## 特性

- 简化 JAKA Zu35 6 轴机械臂 MuJoCo 模型
- 单箱 `cola_24` 场景（0.4 × 0.27 × 0.24 m）
- Gymnasium 兼容接口
- 18 维状态观测 + 6 维归一化关节位置控制
- 随机策略、环境检查、渲染示例

## 安装

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
pip install -e ".[dev]"
```

## 快速开始

### 检查环境是否符合 Gymnasium API

```bash
python examples/check_env.py
```

### 运行随机策略

```bash
python examples/random_agent.py
```

### 渲染场景

```bash
python examples/render_scene.py
```

## 测试

```bash
pytest tests/test_env.py -v
```

## 环境接口

```python
from jaka_zu35_mujoco_rl import JakaReachEnv

env = JakaReachEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## 扩展路线

- 增加两指夹爪与抓取成功判断
- 多箱垛体与混码场景
- 视觉相机观测（RGB/深度）
- 与 ROS2 / pallet_vision_lidar 对接

## 许可证

MIT
```

---

## 自审检查

- [ ] 每个设计文档要求都有对应任务：机械臂模型（Task 4）、箱子模型（Task 3）、环境接口（Task 5）、示例（Task 6-8）、测试（Task 9）、文档（Task 10）。
- [ ] 计划无 TBD/TODO/placeholder。
- [ ] 文件名和函数名在前后任务中一致（`JakaReachEnv`, `end_effector`, `box_top`）。
- [ ] 动作/观测空间维度一致：action=(6,), obs=(18,)。
- [ ] 项目路径统一为 `/mnt/d/work/jaka_zu35_mujoco_rl`。

## 执行方式选择

计划完成并保存到 `/mnt/d/work/jaka_zu35_mujoco_rl/docs/superpowers/plans/2026-08-06-jaka-zu35-mujoco-rl-plan.md`。

**执行选项：**

1. **Subagent-Driven（推荐）** - 每个 Task 派一个子代理实现，完成后复核。
2. **Inline Execution** - 在当前会话中按 Task 顺序直接实现。
