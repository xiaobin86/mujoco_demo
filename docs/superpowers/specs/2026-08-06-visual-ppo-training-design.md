# 带 3D 可视化的 PPO 训练设计文档

**项目路径**: `/mnt/d/work/jaka_zu35_mujoco_rl`  
**文档日期**: 2026-08-06  
**版本**: 1.0  
**关联设计**: `jaka_zu35_mujoco_rl/docs/design.md`

## 1. 目标

在 `jaka_zu35_mujoco_rl` 中新增一个**带实时 3D 可视化的 PPO 训练流程**：

- 打开 MuJoCo 3D viewer 窗口即可看到机械臂持续训练。
- 训练使用 Stable-Baselines3 的 PPO 算法。
- 终端和 TensorBoard 实时显示训练进展（reward、episode length、success 率、最终距离等）。
- 关闭 viewer 窗口即停止训练。
- 训练好的模型保存到 `checkpoints/`，可加载继续评估。

## 2. 范围

### 2.1 本阶段做

- 新增 `examples/train_ppo.py`：启动 viewer、训练 PPO、同步 3D 画面、保存模型。
- 新增 `examples/evaluate_ppo.py`：加载保存的模型，在 viewer 中跑 deterministic 评估。
- 新增 `ViewerSyncCallback`：在 SB3 训练回调中同步 viewer 并检测窗口关闭。
- 更新 `pyproject.toml`：新增 `rl` optional dependency（`stable-baselines3`, `tensorboard`）。
- 更新 `README.md`：说明训练、TensorBoard、评估命令。
- 模型保存到 `checkpoints/ppo_jaka_final.zip`，TensorBoard 日志写到 `logs/`。

### 2.2 本阶段不做

- 自定义网络架构（使用 SB3 默认 `MlpPolicy`）。
- 多环境并行（只使用 `DummyVecEnv` 包装一个环境）。
- 超参数网格搜索或自动调参。
- 在 3D 窗口内直接叠加文字 HUD（只通过终端/TB 输出进度）。
- 与真实机械臂或 ROS2 对接。

## 3. 新增文件结构

```
jaka_zu35_mujoco_rl/
├── examples/
│   ├── train_ppo.py          # PPO 训练 + 3D viewer
│   ├── evaluate_ppo.py       # 加载模型 + 3D 评估
│   ├── viewer_demo.py        # 已存在：随机策略演示
│   ├── random_agent.py       # 已存在
│   ├── check_env.py          # 已存在
│   └── render_scene.py       # 已存在
├── checkpoints/              # 模型保存目录（运行时自动创建）
├── logs/                     # TensorBoard 日志目录（运行时自动创建）
├── pyproject.toml            # 新增 rl optional dependency
└── README.md                 # 更新训练相关说明
```

## 4. 训练流程

### 4.1 环境包装

```python
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: JakaReachEnv()])
```

`JakaReachEnv` 继承 `gymnasium.Env`，`DummyVecEnv` 会处理接口兼容。

### 4.2 模型创建

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)
```

### 4.3 训练回调 `ViewerSyncCallback`

继承 `stable_baselines3.common.callbacks.BaseCallback`：

- 在 `_on_step()` 中：
  - 如果 `viewer.is_running()` 为 False，返回 `False` 停止训练。
  - 每 `sync_every` 步调用 `viewer.sync()`，把 MuJoCo 数据同步到 3D 窗口。
- 在 `_on_training_end()` 中：
  - 保存最终模型到 `checkpoints/ppo_jaka_final.zip`。
- 可选：在 `_on_rollout_end()` 中打印每 rollout 的统计信息。

### 4.4 主训练循环

```python
with mujoco.viewer.launch_passive(env.envs[0].model, env.envs[0].data) as viewer:
    callback = ViewerSyncCallback(viewer, sync_every=100)
    model.learn(
        total_timesteps=200_000,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=True,
    )
```

说明：
- `mujoco.viewer.launch_passive` 在 background thread 渲染窗口。
- 主线程调用 `model.learn()`，SB3 callback 中调用 `viewer.sync()`。
- `viewer.sync()` 线程安全，用于更新渲染数据。
- 关闭窗口后 `viewer.is_running()` 变 False，callback 返回 False 停止训练。

### 4.5 中间检查点

可选：每 50,000 步保存一次模型，文件名为 `checkpoints/ppo_jaka_{steps}.zip`。

## 5. 评估流程

### 5.1 `examples/evaluate_ppo.py`

```python
model = PPO.load("checkpoints/ppo_jaka_final.zip", env=env)

with mujoco.viewer.launch_passive(env.envs[0].model, env.envs[0].data) as viewer:
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        while viewer.is_running() and not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(0.005)
        print(f"Episode {episode + 1}: final_distance={info['distance']:.4f}, success={terminated}")
```

支持命令行参数：`--model`, `--episodes`, `--sleep`。

## 6. 进度显示

### 6.1 终端

`PPO(verbose=1)` 和 `progress_bar=True` 会自动输出：

```text
| rollout/           |          |  ep_len_mean     |  500        |
|                    |          |  ep_rew_mean     |  -120.5     |
|                    |          |  ep_success_rate |  0.05       |
| time/              |          |  fps             |  1200       |
|                    |          |  iterations      |  5          |
|                    |          |  total_timesteps |  10240      |
```

### 6.2 TensorBoard

启动：

```bash
tensorboard --logdir logs/
```

访问 `http://localhost:6006`，查看：
- `rollout/ep_rew_mean`
- `rollout/ep_len_mean`
- `train/loss`, `train/value_loss`, `train/policy_gradient_loss`

## 7. 依赖更新

在 `pyproject.toml` 中新增 `rl` optional dependency：

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "imageio",
]
rl = [
    "stable-baselines3",
    "tensorboard",
]
```

安装命令：

```bash
pip install -e ".[rl]"
```

## 8. 使用命令

### 8.1 训练

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
pip install -e ".[rl]"
python examples/train_ppo.py --total-timesteps 200000 --sync-every 100
```

### 8.2 TensorBoard

```bash
tensorboard --logdir logs/
```

### 8.3 评估

```bash
python examples/evaluate_ppo.py --model checkpoints/ppo_jaka_final.zip --episodes 5
```

## 9. 验收标准

- `pip install -e ".[rl]"` 成功安装依赖。
- `python examples/train_ppo.py` 打开 3D viewer，机械臂持续动作。
- 终端显示训练进度条和 SB3 日志。
- 关闭 viewer 窗口后训练停止，不崩溃。
- `checkpoints/ppo_jaka_final.zip` 保存成功。
- `tensorboard --logdir logs/` 能读取到训练数据。
- `python examples/evaluate_ppo.py` 加载模型并跑 5 个 episode，终端打印 final_distance 和 success。

## 10. 风险与限制

- **MuJoCo viewer 需要图形界面**：headless 服务器或 WSL 无 X server 时无法打开窗口，只能在本地桌面/带 X11 转发的环境运行。
- **训练速度受 viewer sync 影响**：`sync_every` 越小画面越流畅但训练 FPS 越低；默认 100 步同步一次。
- **PPO 默认超参数不一定最优**：本阶段只保证能跑通，后续需根据收敛情况调整 `n_steps`, `learning_rate`, `batch_size` 等。
