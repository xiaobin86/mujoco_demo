# JAKA Zu35 MuJoCo RL 环境设计文档

**项目路径**: `/mnt/d/work/jaka_zu35_mujoco_rl`  
**文档日期**: 2026-08-06  
**版本**: 1.0  
**关联项目**: `pallet_vision_lidar`（仅复用箱体尺寸先验，本项目独立）

## 1. 目标

在 `/mnt/d/work/` 下创建一个新的独立项目 `jaka_zu35_mujoco_rl`，提供一个**最小可运行**的 MuJoCo 强化学习环境示例：

- 一个简化的 **JAKA Zu35 6 轴机械臂** MuJoCo 模型。
- 一个 **单箱目标**（复用 `pallet_vision_lidar` 的 `cola_24` 尺寸先验）。
- 一个 **Gymnasium 兼容**的 `JakaReachEnv`。
- 可运行的示例脚本和基础测试。

本项目的核心目的是验证"MuJoCo 机械臂 + 箱子 + Gymnasium 接口"这条链路可跑通，为后续抓取、多箱、视觉输入等扩展打下基础。

## 2. 范围

### 2.1 本阶段做（MVP）

- 简化 6 轴 JAKA Zu35 运动学模型（圆柱体/胶囊近似）。
- 单箱 `cola_24` 场景（0.4 m × 0.27 m × 0.24 m）。
- 6 维关节位置控制。
- 18 维低维状态观测。
- 基于末端到目标点距离的奖励函数。
- Gymnasium API 兼容环境。
- 随机策略示例、环境检查示例、渲染示例。
- `pytest` 基础测试。

### 2.2 本阶段不做

- 真实夹爪模型与闭合抓取物理。
- 多箱垛体与混码场景。
- 视觉/相机图像观测。
- 与 ROS2 / `pallet_vision_lidar` 的运行时对接。
- 真实 URDF 导入与高保真外观。
- 训练脚本（只提供环境，训练由用户后续自行接入）。

## 3. 项目结构

```
jaka_zu35_mujoco_rl/
├── jaka_zu35_mujoco_rl/          # Python 包
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── jaka_reach_env.py     # JakaReachEnv 实现
│   └── models/                   # MuJoCo MJCF 模型文件
│       ├── jaka_zu35.xml         # 机械臂主模型
│       ├── box_cola24.xml        # 箱子 asset
│       └── ground.xml            # 地面 asset
├── examples/                     # 可运行示例
│   ├── random_agent.py           # 随机策略跑 episode
│   ├── check_env.py              # gymnasium.check_env 验证
│   └── render_scene.py           # 渲染并保存场景图片
├── tests/                        # 单元测试
│   └── test_env.py
├── pyproject.toml                # 包配置与依赖
└── README.md                     # 快速开始
```

## 4. 机械臂模型

### 4.1 设计原则

- **简化几何**：使用 MuJoCo 的 `cylinder` 和 `capsule` 几何体近似连杆与关节，不导入真实 URDF。
- **运动学近似**：参考 JAKA Zu35 的典型 6 轴串联布局（底座 → 肩部 → 肘部 → 腕部三轴），关节数 6，各关节限位参考公开典型值。
- **无夹爪**：末端为一个固定工具点（site），用于计算距离奖励。夹爪在后续阶段加入。

### 4.2 关节配置（暂定）

| 关节 | 类型 | 范围（示例） | 说明 |
|------|------|-------------|------|
| `joint_1` | hinge | [-π, π] | 底座旋转 |
| `joint_2` | hinge | [-π/2, π/2] | 肩部俯仰 |
| `joint_3` | hinge | [-π/2, π/2] | 肘部俯仰 |
| `joint_4` | hinge | [-π, π] | 腕部旋转 |
| `joint_5` | hinge | [-π/2, π/2] | 腕部俯仰 |
| `joint_6` | hinge | [-π, π] | 腕部旋转 |

> 注：具体关节限位和连杆长度在实现时根据可工作空间微调，不需要与真实 JAKA Zu35 完全一致。

## 5. 环境接口

### 5.1 类定义

```python
class JakaReachEnv(gymnasium.Env):
    ...
```

### 5.2 动作空间

- `gymnasium.spaces.Box`
- `shape = (6,)`
- `low = -1.0`, `high = 1.0`
- 每一维对应一个关节的位置目标，内部线性映射到该关节的实际范围。
- 使用 PD 位置控制（`position` actuator）驱动。

### 5.3 观测空间

- `gymnasium.spaces.Box`
- `shape = (18,)`
- `low/high = -inf / inf`
- 观测向量：
  - 关节位置：6 维
  - 关节速度：6 维
  - 末端执行器位置（site `end_effector`）：3 维
  - 目标箱顶面中心位置：3 维

### 5.4 奖励函数

```
r = -dist(endeffector, box_top_center) - 0.01 * ||action||^2
```

- 距离项：鼓励末端靠近目标箱顶面中心。
- 动作惩罚项：抑制过大动作，增加平滑性。

### 5.5 终止条件

- **成功**：`dist(endeffector, box_top_center) < 0.05 m`，返回 `terminated=True`。
- **截断**：达到最大步数（默认 500 步），返回 `truncated=True`。
- **失败/异常**：关节或物体状态出现 NaN/异常（可选，默认不启用）。

### 5.6 重置行为

- 恢复 MuJoCo 模型到默认配置。
- 随机初始化关节位置（在关节范围内采样，避免奇异姿态）。
- 随机初始化目标箱位置（在固定桌面上一定区域内）。
- 返回初始观测和空信息字典。

## 6. 箱子模型

- 使用 `pallet_vision_lidar` 中的 `cola_24` 尺寸：0.4 m × 0.27 m × 0.24 m。
- 简化为一个 MuJoCo `box` 几何体。
- 质量参考 `~14 kg`，密度根据体积反推。
- 初始状态：固定放置在地面/桌面上，不引入自由关节（简化 reach 任务）。

## 7. 依赖

```toml
[project]
name = "jaka_zu35_mujoco_rl"
version = "0.1.0"
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
```

## 8. 示例脚本

### 8.1 `examples/random_agent.py`

跑 3 个 episode 的随机策略，每步从动作空间采样，打印累计奖励和是否成功。

### 8.2 `examples/check_env.py`

调用 `gymnasium.check_env(env)` 验证环境是否符合 Gymnasium API 规范。

### 8.3 `examples/render_scene.py`

加载环境，渲染一帧 RGB 和深度，保存为 PNG 文件。

## 9. 测试

### 9.1 测试文件

`tests/test_env.py`

### 9.2 测试内容

- `test_model_loads()`：MuJoCo 模型可以从 XML 文件加载。
- `test_env_reset()`：`env.reset()` 返回正确的观测形状和信息字典。
- `test_env_step()`：`env.step(action)` 返回 `(obs, reward, terminated, truncated, info)`，且观测形状正确。
- `test_gymnasium_api()`：`gymnasium.check_env` 通过。

## 10. 扩展路线（未来工作）

1. **夹爪与抓取**：加入两指夹爪，动作空间增加夹爪开合，奖励基于是否抓取并抬起箱子。
2. **多箱垛体**：支持多个箱子、不同 class（`cola_24` / `water_12`）、按目标选择抓取。
3. **视觉观测**：加入相机渲染，输出 RGB/深度，与 `pallet_vision_lidar` 的 YOLO 检测输出对齐。
4. **ROS2 对接**：可选提供 ROS2 topic 输出，与真实流水线对接。

## 11. 验收标准

- `pip install -e .` 成功安装。
- `python examples/check_env.py` 无错误通过。
- `python examples/random_agent.py` 可以跑完 3 个 episode。
- `pytest tests/test_env.py` 全部通过。
- 用户可以在 README 指引下 5 分钟内跑通第一个 episode。
