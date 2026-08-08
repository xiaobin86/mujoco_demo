# AGENTS.md —— jaka_zu35_mujoco_rl 项目上下文

> 本文件用于长期记忆。每次进入本项目时，请先阅读本文件。
> 最后更新：2026-08-07

## 1. 项目定位

- **项目代号**：`jaka_zu35_mujoco_rl`
- **目标**：在 MuJoCo 仿真中训练 JAKA Zu35 / SO101 机械臂的拆/码垛策略，支持遥操作 + ACT + PPO 两阶段训练。
- **核心路线**：实体 SO101 主臂遥操作 → 仿真从臂 → 录制演示 → ACT 训练 → rollout 生成更多数据 → BC 预热 PPO actor → PPO fine-tune。
- **工作分支**：`feature/so101-teleop-act-rl`

## 2. 环境与路径

- **Conda 环境**：`mujoco`
- **操作系统**：原生 Ubuntu 24.04（非 WSL）
- **LeRobot 源码路径**：`/mnt/d/work/lerobot`（已 editable 安装）
- **本项目路径**：`/mnt/d/work/jaka_zu35_mujoco_rl`
- **硬件依赖**：`feetech-servo-sdk`（提供 `scservo_sdk`）

## 3. 实体 SO101 主臂配置

- **串口**：`/dev/ttyACM0`
- **标定 ID**：`07252802`
- **当前读取配置**：脚本 `scripts/teleop_record.py` 使用 `SO101LeaderConfig` 读取主臂位置，并通过 `--joint-signs` 提供方向修正。
- **首次标定**（二选一）：
  1. 使用 LeRobot 内置标定脚本（注意主臂属于 teleop，不是 robot）：
     ```bash
     conda activate mujoco
     python /mnt/d/work/lerobot/src/lerobot/scripts/lerobot_calibrate.py \
       --teleop.type so101_leader \
       --teleop.port /dev/ttyACM0 \
       --teleop.id 07252802
     ```
  2. 运行 `scripts/teleop_record.py` 时由 `teleop.connect(calibrate=True)` 自动触发：
     ```bash
     python scripts/teleop_record.py --port /dev/ttyACM0 --id 07252802
     ```

## 4. 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/teleop_record.py` | 实体主臂遥操作仿真从臂并录制演示 |
| `scripts/train_act.py` | 阶段一：ACT 训练 |
| `scripts/act_rollout.py` | 用 ACT 在仿真中生成更多演示数据 |
| `scripts/pretrain_ppo_with_bc.py` | 阶段二第一步：BC 预热 PPO actor |
| `examples/train_ppo.py` | 阶段二第二步：PPO fine-tune |
| `docs/so101_teleop_act_rl_guide.md` | 详细操作指南 |

## 5. 已知注意事项

- 主臂连接后脚本会自动关闭扭矩，便于手动拖动。
- 标定文件保存在 LeRobot 默认缓存目录，与 `id` 绑定；更换 `id` 会重新标定。
- 遥操作使用 `SO101LeaderConfig`；主臂关节输出范围 `[-100, 100]`，夹爪 `[0, 100]`。
- `scripts/teleop_record.py` 参考 LeRobot 录制逻辑，按时间分回合：复位 `reset_time_s` + 录制 `episode_time_s`；右箭头结束保存，左箭头丢弃重录，ESC 停止保存。
- 运行与训练在**原生 Ubuntu 24.04**（非 WSL）的 `mujoco` conda 环境中进行。
- MuJoCo 当前环境下命名固定相机（named fixed camera）离屏渲染会全黑，因此 `SO101PickEnv` / `PandaPickEnv` 改用自由相机（free camera）俯视场景。
