# SO101 遥操作 + 双相机 ACT + RL 两阶段训练用户指南

> 适用于分支 `feature/so101-teleop-act-rl`  
> 目标：用实体 SO101 主臂遥操作仿真从臂完成抓取-放置，录制双相机演示数据，训练 ACT 策略，再用 ACT rollout 生成大量数据，BC 预热 PPO 后继续 fine-tune。

---

## 0. 前置条件

1. 进入 `mujoco` conda 环境：
   ```bash
   conda activate mujoco
   ```
2. 确认 LeRobot 已 editable 安装：
   ```bash
   cd /mnt/d/work/lerobot
   pip install -e .
   ```
3. 确认本仓库已安装 teleop 与 RL 依赖：
   ```bash
   cd /mnt/d/work/jaka_zu35_mujoco_rl
   pip install -e ".[teleop,rl]"
   ```
4. 本项目在**原生 Ubuntu 24.04**（非 WSL）上运行；实体 SO101 主臂通过 USB 串口连接，本机端口为 `/dev/ttyACM0`，标定 ID 为 `07252802`。
5. 已完成 SO101 主臂标定（若未标定，见第 8 节）。
6. **RTX 50 系列（Blackwell / sm_120）注意**：当前环境若安装的是 `torch 2.7.1+cu126`，运行 CUDA 训练会报错 `CUDA error: no kernel image is available for execution on the device`。请切换到 CUDA 12.8 构建：
   ```bash
   pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

---

## 1. 脚本参数速查

所有脚本统一使用 `--device auto` 在可用时自动选择 CUDA，也可显式指定 `--device cpu` 或 `--device cuda`。

### 1.1 `scripts/teleop_record.py` — 实体主臂遥操作 + 双相机录制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--port` | str | `/dev/ttyACM0` | 主臂串口 |
| `--id` | str | `07252802` | 标定文件 ID |
| `--output` | str | `None` | 输出文件；默认 `data/teleop_demos_<时间戳>.npz` |
| `--episodes` | int | `10` | 录制回合数 |
| `--image-size` | int | `128` | 保存的图像尺寸（正方形，双相机都按此尺寸保存） |
| `--display-size` | str | `1280,480` | OpenCV 预览窗口尺寸 `宽,高` |
| `--record-fps` | float | `20.0` | 控制/录制频率（Hz） |
| `--episode-time-s` | float | `60.0` | 每回合最大录制时长 |
| `--reset-time-s` | float | `10.0` | 每回合前复位/摆位时间 |
| `--no-calibrate` | flag | `False` | 跳过自动标定提示 |
| `--no-viewer` | flag | `False` | 禁用 MuJoCo 3D viewer |
| `--camera` | str | `both` | 预览布局：`both` 双相机并排，`wrist_cam` 手眼，`overhead` 俯视 |
| `--flip-wrist-camera` | flag | `False` | 水平翻转手眼相机画面 |

默认已同时保存 `images_wrist`（手眼相机）和 `images_overhead`（俯视相机），即双相机数据集。

### 1.2 `scripts/replay_teleop.py` — 回放录制的演示数据

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | `data/teleop_demos.npz` | 输入演示文件 |
| `--robot` | str | `so101` | 仿真机器人：`so101` / `panda` |
| `--fps` | float | `20.0` | 回放频率 |
| `--display-size` | str | `640,480` | OpenCV 窗口尺寸 |
| `--headless` | flag | `False` | 无 OpenCV 无 3D viewer，纯后台验证 |
| `--no-viewer` | flag | `False` | 禁用 3D viewer，保留 OpenCV 相机窗口 |
| `--camera` | str | `wrist_cam` | 回放时显示的相机视角 |
| `--flip-wrist-camera` | flag | `False` | 水平翻转手眼画面 |

### 1.3 `scripts/train_act.py` — 训练 ACT 策略

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str 列表 | `None` | 输入 `.npz` 文件，可传多个；默认取 `data/*.npz` 中最新修改的文件 |
| `--output` | str | `checkpoints/act_so101.pt` | 输出 checkpoint |
| `--resume` | str | `None` | 从已有 ACT checkpoint 加载权重继续训练 |
| `--epochs` | int | `200` | 训练 epoch 数 |
| `--batch-size` | int | `64` | 训练 batch size |
| `--lr` | float | `1e-4` | 学习率 |
| `--weight-decay` | float | `1e-4` | AdamW 权重衰减 |
| `--chunk-size` | int | `8` | ACT 动作块长度；**必须与 rollout 一致** |
| `--hidden-dim` | int | `128` | Transformer 隐藏维度 |
| `--val-split` | float | `0.1` | 验证集比例（按 episode 划分） |
| `--seed` | int | `None` | 随机种子 |
| `--device` | str | `auto` | 计算设备 |
| `--tensorboard-log` | str | `logs/` | TensorBoard 日志目录 |
| `--no-compile` | flag | `False` | 禁用 `torch.compile`（CUDA 下默认开启） |
| `--no-amp` | flag | `False` | 禁用 `bfloat16` 自动混合精度（CUDA 下默认开启） |

当前 ACT 使用 **ResNet-18 图像编码器**（ImageNet 预训练）作为 backbone，双相机输入会自动拼接成多视角 memory token。

### 1.4 `scripts/act_rollout.py` — 用 ACT 在仿真中生成更多数据

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `checkpoints/act_so101.pt` | 训练好的 ACT checkpoint |
| `--output` | str | `data/act_demos.npz` | 输出演示文件 |
| `--episodes` | int | `50` | rollout 回合数 |
| `--max-steps` | int | `300` | 每回合最大步数 |
| `--chunk-size` | int | `8` | 动作块长度，**必须与训练时一致** |
| `--single-camera` | flag | `False` | 只用手眼相机；默认使用双相机（wrist + overhead） |
| `--temporal-ensemble-coeff` | float | `0.01` | 时序集成系数；`<=0` 表示开环执行整个 chunk |
| `--device` | str | `cuda` | 计算设备（默认 `cuda`，可选 `auto` / `cpu`） |
| `--image-size` | int | `128` | 输入图像尺寸 |

### 1.5 `scripts/pretrain_ppo_with_bc.py` — BC 预热 PPO actor

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--demos` | str | `data/act_demos.npz` | 演示数据（ACT rollout 或遥操作数据） |
| `--output` | str | `checkpoints/ppo_so101_bc_init.zip` | 输出 PPO checkpoint |
| `--epochs` | int | `100` | BC 训练 epoch 数 |
| `--batch-size` | int | `256` | batch size |
| `--lr` | float | `1e-4` | 学习率 |
| `--val-split` | float | `0.1` | 验证集比例 |
| `--device` | str | `auto` | 计算设备 |

### 1.6 `examples/train_ppo.py` — PPO fine-tune

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robot` | str | `panda` | 机器人：`panda` / `so101` |
| `--total-timesteps` | int | `2000000` | 总训练步数 |
| `--sync-every` | int | `1` | 3D viewer 同步频率 |
| `--checkpoint-every` | int | `50000` | 自动保存 checkpoint 间隔 |
| `--seed` | int | `0` | 随机种子 |
| `--no-viewer` | flag | `False` | 无 viewer 训练 |
| `--device` | str | `auto` | 计算设备 |
| `--reward-config` | str | `None` | 奖励 YAML 配置文件 |
| `--resume` | str | `None` | 从 checkpoint 恢复（如 BC 预热后的 `ppo_so101_bc_init.zip`） |
| `--learning-rate` | float | `1e-4` | PPO 学习率 |
| `--n-steps` | int | `4096` | 每次更新前的环境步数 |
| `--batch-size` | int | `256` | PPO minibatch size |
| `--n-epochs` | int | `10` | 每次 rollout 重复训练的 epoch 数 |
| `--gamma` | float | `0.99` | 折扣因子 |
| `--gae-lambda` | float | `0.95` | GAE lambda |
| `--clip-range` | float | `0.1` | PPO clip 范围 |
| `--ent-coef` | float | `0.01` | 熵系数 |
| `--vf-coef` | float | `0.5` | value 函数 loss 系数 |
| `--fast-obs` | flag | `False` | 跳过相机渲染，直接用带噪声的 ground-truth 方块位置（训练更快） |
| `--n-envs` | int | `1` | 并行环境数；`>1` 时使用 SubprocVecEnv |

### 1.7 `examples/evaluate_ppo.py` — 评估训练好的 PPO

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robot` | str | `panda` | 机器人：`panda` / `so101` |
| `--model` | str | `None` | 模型路径；默认 `checkpoints/ppo_<robot>_final.zip` |
| `--episodes` | int | `5` | 评估回合数 |
| `--sleep` | float | `0.005` | viewer 每步同步间隔 |

---

## 2. 完整训练流程（从遥操作到 RL）

### 2.1 录制双相机遥操作数据

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
conda activate mujoco
python scripts/teleop_record.py \
  --port /dev/ttyACM0 \
  --id 07252802 \
  --episodes 30 \
  --camera both \
  --image-size 128
```

- 默认输出到 `data/teleop_demos_YYYYMMDD_HHMMSS.npz`。
- `--camera both` 是默认值，可省略；这里显式写出是为了强调双相机录制。
- 录制时按 `→` 提前结束并保存当前回合，`←` 丢弃重录，`ESC` 退出并保存已完成回合。
- 建议至少录制 **20-30 条成功**的抓取-放置轨迹。

### 2.2 回放检查数据

```bash
python scripts/replay_teleop.py \
  --input data/teleop_demos_YYYYMMDD_HHMMSS.npz \
  --robot so101 \
  --camera both
```

- 检查动作、图像、方块位置是否一致。
- 若有多段录制，建议每段都回放一次。

### 2.3 训练 ACT 策略（阶段一）

```bash
python scripts/train_act.py \
  --input data/teleop_demos_*.npz \
  --output checkpoints/act_so101.pt \
  --epochs 500 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --chunk-size 8 \
  --hidden-dim 128 \
  --device cuda
```

- `--input` 可传多个文件合并训练，也可省略让它自动选最新的 `data/*.npz`。
- 默认启用 `torch.compile` + `bfloat16 AMP`，在 CUDA 上训练最快；若出错可加 `--no-compile` 或 `--no-amp` 排查。
- 每 10 epoch 打印 train/val loss，验证 loss 最低时自动保存 `checkpoints/act_so101.pt`。

### 2.4 用 ACT rollout 生成更多仿真数据

```bash
python scripts/act_rollout.py \
  --model checkpoints/act_so101.pt \
  --output data/act_demos.npz \
  --episodes 200 \
  --max-steps 500 \
  --temporal-ensemble-coeff 0.01 \
  --device cuda
```

- 默认已使用双相机（wrist + overhead），与 `teleop_record.py` / `train_act.py` 的双相机数据一致。
- `--chunk-size` 默认 8，且脚本会从 checkpoint 中自动推断并匹配实际的 `chunk_size`，不再需要手动与训练时对齐。
- 输出 `data/act_demos.npz` 格式与遥操作数据一致，用于 BC 预热。

### 2.5 BC 预热 PPO actor（阶段二第一步）

```bash
python scripts/pretrain_ppo_with_bc.py \
  --demos data/act_demos.npz \
  --output checkpoints/ppo_so101_bc_init.zip \
  --epochs 200 \
  --batch-size 256 \
  --lr 1e-4 \
  --device cuda
```

- 只更新 PPO 的 actor（策略网络），value 网络不动。
- 输出 `checkpoints/ppo_so101_bc_init.zip` 作为后续 PPO fine-tune 的初始权重。

### 2.6 PPO fine-tune（阶段二第二步）

```bash
python examples/train_ppo.py \
  --robot so101 \
  --resume checkpoints/ppo_so101_bc_init.zip \
  --no-viewer \
  --fast-obs \
  --total-timesteps 500000 \
  --n-steps 2048 \
  --batch-size 256 \
  --n-epochs 10 \
  --learning-rate 1e-4 \
  --ent-coef 0.01 \
  --device cuda
```

- `--resume` 加载 BC 预热后的模型。
- `--fast-obs` 跳过 offscreen 相机渲染，训练速度显著提升。
- `--no-viewer` 适合无显示器或长时间训练；需要 viewer 可去掉该参数。
- 最终模型保存为 `checkpoints/ppo_so101_final.zip`。
- 可用 TensorBoard 查看曲线：
  ```bash
  tensorboard --logdir logs/
  ```

### 2.7 评估模型

```bash
python examples/evaluate_ppo.py \
  --robot so101 \
  --model checkpoints/ppo_so101_final.zip \
  --episodes 10
```

---

## 3. 简化默认命令集（双相机 ACT + 最优训练配置）

如果你已经标定好主臂、环境可用，并且想要**用默认参数即可跑通双相机 ACT → BC → PPO** 的最简流程，直接按顺序执行下面这组命令：

```bash
# 1. 进入环境
cd /mnt/d/work/jaka_zu35_mujoco_rl
conda activate mujoco

# 2. 录制 30 条双相机遥操作轨迹（默认 already 双相机，--camera both 可省略）
python scripts/teleop_record.py \
  --port /dev/ttyACM0 \
  --id 07252802 \
  --episodes 30 \
  --image-size 128

# 3. 训练 ACT（默认 batch-size 64, chunk-size 8, hidden-dim 128, epochs 200）
python scripts/train_act.py \
  --output checkpoints/act_so101.pt \
  --epochs 500 \
  --device cuda

# 4. 用 ACT 在仿真中生成更多数据（默认双相机，chunk_size 自动推断）
python scripts/act_rollout.py \
  --model checkpoints/act_so101.pt \
  --output data/act_demos.npz \
  --episodes 200 \
  --max-steps 500

# 5. BC 预热 PPO actor
python scripts/pretrain_ppo_with_bc.py \
  --demos data/act_demos.npz \
  --output checkpoints/ppo_so101_bc_init.zip \
  --epochs 200 \
  --device cuda

# 6. PPO fine-tune（SO101，关闭 viewer，fast obs 加速）
python examples/train_ppo.py \
  --robot so101 \
  --resume checkpoints/ppo_so101_bc_init.zip \
  --no-viewer \
  --fast-obs \
  --total-timesteps 500000 \
  --device cuda

# 7. 评估
python examples/evaluate_ppo.py \
  --robot so101 \
  --model checkpoints/ppo_so101_final.zip \
  --episodes 10
```

> **关键提醒**：
> - `act_rollout.py` 默认使用双相机输入，与 `teleop_record.py` / `train_act.py` 一致；如需单相机，请加 `--single-camera`。
> - `act_rollout.py` 会自动从 checkpoint 推断 `--chunk-size`，一般无需手动指定。
> - `train_ppo.py` 的 `--device` 仍默认 `auto`；`act_rollout.py` 已改为默认 `cuda`。
> - 若显存不足，可在 `train_act.py` 加 `--batch-size 32` 或 `--no-amp`。
> - 若训练时提示 PyTorch 不支持 sm_120，请按前置条件切换到 `cu128` 版本。

---

## 4. 首次标定实体 SO101 主臂

如果还没做过标定：

```bash
conda activate mujoco
python -c "
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
cfg = SO101LeaderConfig(port='/dev/ttyACM0', id='07252802')
teleop = make_teleoperator_from_config(cfg)
teleop.connect(calibrate=True)
teleop.disconnect()
"
```

按终端提示：
1. 把主臂摆到各关节中间位置，按回车。
2. 依次把每个关节活动到最大/最小范围，按回车结束。
3. 标定文件会自动保存并与 `--id` 绑定。

标定完成后，主臂 5 个身体关节以**角度（degrees）**输出，夹爪在 `[0, 100]` 输出（`0=闭合，100=张开`）。

---

## 5. 常见问题与注意事项

### Q1: 串口权限错误 / `PermissionError: /dev/ttyACM0`

- 检查设备是否存在：`ls -l /dev/ttyACM0`。
- 若当前用户不在 `dialout` 组：
  ```bash
  sudo usermod -a -G dialout $USER
  # 重新登录或执行 newgrp dialout
  ```
- 临时测试：`sudo chmod 666 /dev/ttyACM0`。

### Q2: `CUDA error: no kernel image is available for execution on the device`

- 你当前的 PyTorch 可能是 `+cu126` 构建，不支持 Blackwell（sm_120）。切换到 CUDA 12.8 构建：
  ```bash
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```

### Q3: ACT rollout 时报 `size mismatch for action_queries.weight`

- 脚本现在会从 checkpoint 自动推断 `chunk_size`；若出现此错误，请确认你运行的是最新版本的 `act_rollout.py`。
- 旧版本需要手动保证 `train_act.py` 和 `act_rollout.py` 的 `--chunk-size` 一致。

### Q4: ACT rollout 完全失败 / 奖励很低

- 默认已使用双相机，无需额外参数；若手动加了 `--single-camera` 却用双相机模型训练，则行为会退化。
- 增加 `--episodes` 和训练 epoch。
- 检查录制数据中成功轨迹比例是否足够高。
- 若仍怀疑 chunk_size 不匹配，可检查脚本启动日志中打印的 checkpoint `chunk_size`。

### Q5: PPO resume 时报维度错误

- 确保 BC 预热用的是与 `SO101PickEnv` 相同的 28 维 observation（或当前环境默认维度）。
- 若改动过 observation，需要重新 rollout 生成 `act_demos.npz`。

### Q6: 训练太慢

- 确认 `--device cuda` 已生效。
- 在 `train_ppo.py` 中加 `--fast-obs` 跳过相机渲染。
- 在 `train_act.py` 中可减小 `--batch-size` 或加 `--no-compile` 排查 compile 开销。
- 录制时降低 `--record-fps` 或缩短 `--episode-time-s` 可减少数据量。

### Q7: 录制时如何看俯视相机

- `teleop_record.py` 默认双相机预览窗口左侧是手眼、右侧是俯视。若只想看俯视：
  ```bash
  python scripts/teleop_record.py --camera overhead
  ```

### Q8: 录制时窗口是黑色

- 检查 MuJoCo 与 GPU/GL 驱动；或临时用 `--no-viewer` 只保留 OpenCV 窗口。
- 若 OpenCV 窗口也黑，确认 `opencv-python` 版本在 `4.x` 且带 GUI 后端。
