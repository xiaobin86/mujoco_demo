# 奖励函数调优指南

本项目的奖励函数已经从 `PandaPickEnv` 的源码中抽出，集中到 `config/reward_panda_pick.yaml`。你可以通过修改 YAML 文件快速对比不同奖励规则，不需要重新改代码。

---

## 1. 文件结构

```text
config/reward_panda_pick.yaml   # 奖励系数配置文件
jaka_zu35_mujoco_rl/envs/panda_pick_env.py   # 读取 YAML 并计算奖励
jaka_zu35_mujoco_rl/callbacks.py             # 把奖励分量写到 TensorBoard
examples/train_ppo.py                        # 支持 --reward-config 参数
```

---

## 2. 配置文件说明

`config/reward_panda_pick.yaml` 默认内容：

```yaml
rewards:
  time_penalty: -0.05          # 每步时间惩罚，鼓励快速完成

  approach:
    scale: 2.0                 # 接近 cube 奖励系数
    distance: 0.1              # 距离缩放（越小，奖励越集中在近处）

  grasp:
    bonus: 2.0                 # 夹取意图奖励
    distance: 0.05             # EE 必须在 cube 多近才算

  lift:
    scale: 5.0                 # 抬升奖励系数
    height: 0.1                # 抬升多高给满奖励
    reference_height: 0.025     # 从 cube 底面以上开始算

  transport:
    scale: 2.0                 # 运输到 tray 奖励系数
    distance: 0.1               # 距离缩放

  success:
    bonus: 500.0                # 成功奖励
    threshold: 0.05             # cube 离 tray 多近算成功
    hold_steps: 50              # 保持多少步才触发成功
    min_height: 0.03            # 成功时 cube 必须高于桌面

  drop:
    penalty: -50.0              # cube 掉下桌子惩罚
    height: -0.05               # 低于这个高度算掉下

  action:
    penalty: -0.0001            # 动作惩罚，抑制抖动

logging:
  log_reward_components: true   # 是否记录每个奖励分量
  log_success_rate: true        # 是否记录成功率
  success_rate_window: 100        # 成功率滑动窗口（最近多少 episode）
```

---

## 3. 如何运行实验

### 3.1 使用默认配置

```bash
cd /mnt/d/work/jaka_zu35_mujoco_rl
source /home/acelan/yes/bin/activate mujoco
python examples/train_ppo.py --total-timesteps 200000 --no-viewer --device cpu
```

### 3.2 使用自定义奖励配置

复制默认配置，修改后运行：

```bash
cp config/reward_panda_pick.yaml config/reward_my_v1.yaml
# 编辑 config/reward_my_v1.yaml
python examples/train_ppo.py \
  --total-timesteps 200000 \
  --no-viewer --device cpu \
  --reward-config config/reward_my_v1.yaml
```

### 3.3 对比多个实验

每次运行都会在 `logs/` 生成一个 `PPO_X` 目录。要对比多个奖励配置，分别跑：

```bash
python examples/train_ppo.py --total-timesteps 50000 --no-viewer --device cpu \
  --reward-config config/reward_panda_pick.yaml

python examples/train_ppo.py --total-timesteps 50000 --no-viewer --device cpu \
  --reward-config config/reward_my_v1.yaml

python examples/train_ppo.py --total-timesteps 50000 --no-viewer --device cpu \
  --reward-config config/reward_my_v2.yaml
```

然后启动 TensorBoard：

```bash
tensorboard --logdir logs/
```

在 TensorBoard 的 SCALARS 页面，同时勾选 `PPO_1`、`PPO_2`、`PPO_3`，对比曲线。

---

## 4. 关键指标怎么看

### 4.1 必看指标

| TensorBoard 标签 | 含义 | 健康趋势 |
|---|---|---|
| `rollout/ep_rew_mean` | 每 episode 平均总奖励 | 持续上升 |
| `rollout/success_rate` | 最近 100 个 episode 的成功率 | 从 0 逐渐涨到接近 1 |
| `rollout/ep_len_mean` | 每 episode 平均长度 | 逐渐下降 |
| `train/explained_variance` | 价值函数解释方差 | 接近 1.0 |

### 4.2 奖励分量指标

`reward/xxx_mean` 是每个奖励分量的**每步平均值**。

| 标签 | 含义 | 正常学习时的表现 |
|---|---|---|
| `reward/time_mean` | 时间惩罚 | 恒为 -0.05 |
| `reward/approach_mean` | 接近 cube 奖励 | 早期快速上升，后期饱和 |
| `reward/grasp_mean` | 夹取意图奖励 | 学会夹取后显著大于 0 |
| `reward/lift_mean` | 抬升奖励 | 学会抬升后显著大于 0 |
| `reward/transport_mean` | 运输到 tray 奖励 | 学会运输后显著大于 0 |
| `reward/success_mean` | 成功奖励 | 成功后给 500/1000，数值会很高 |
| `reward/drop_mean` | 掉落惩罚 | 最好恒为 0 |
| `reward/action_mean` | 动作惩罚 | 接近 0 |

### 4.3 典型分析场景

**场景 A：奖励曲线上升，但成功率不动**

- 表现：`ep_rew_mean` 涨到 300+，`success_rate` 还是 0
- 原因：dense reward 太厚，智能体靠“摸鱼”也能拿高分
- 修复：
  - 把 `success.bonus` 加大到 2000-5000
  - 把 `approach.scale`、`transport.scale` 调小
  - 缩短 `max_episode_steps` 或增加 `time_penalty`

**场景 B：approach 奖励饱和，但 lift/grasp 一直为 0**

- 原因：智能体只学会了靠近 cube，没学会夹取和抬升
- 修复：
  - 提高 `grasp.bonus` 和 `lift.scale`
  - 检查 gripper 动作是否真的能闭合（可以在 viewer 里观察）
  - 考虑 curriculum learning：先固定 cube 位置，学会夹取后再随机位置

**场景 C：drop 惩罚频繁出现**

- 原因：智能体把 cube 推下桌子
- 修复：
  - 增加 `drop.penalty` 到 -100 或 -200
  - 给“cube 在桌面上”一个小的额外奖励
  - 检查 PD 控制器是否太激进，导致碰撞不稳定

**场景 D：奖励曲线往下掉**

- 原因：训练发散或 reward shaping 误导
- 修复：
  - 降低学习率
  - 检查是否有 reward hacking（比如奖励项可被循环刷取）
  - 改用更 sparse 的奖励

---

## 5. 调参原则

### 5.1 成功奖励必须主导

一个 episode 里 dense reward 最多能拿到多少？简单估算：

```
max_dense_per_step = approach.scale + transport.scale + lift.scale + grasp.bonus
max_dense_per_episode = max_dense_per_step * max_episode_steps
```

默认配置：

```
max_dense_per_step = 2.0 + 2.0 + 5.0 + 2.0 = 11.0
max_dense_per_episode = 11.0 * 1000 = 11000
```

这种情况下 `success.bonus` 只有 500，**完全不足以主导**。所以通常需要：

- 要么把 `success.bonus` 加到 5000-10000
- 要么把 dense reward 系数降到 0.5-1.0 量级
- 要么缩短 episode 长度到 200-300 步

### 5.2 一次只改一个变量

奖励函数调优最怕一次改多个参数。建议每次只改一个：

1. 先固定 `success.bonus`，看成功率
2. 再调 `approach.scale`，看是否能接近 cube
3. 再调 `grasp.bonus` 和 `lift.scale`，看是否能夹取抬起
4. 最后调 `transport.scale` 和 `time_penalty`，看运输速度

### 5.3 用短实验快速迭代

不要每次跑 200k 步。50k 步（约 2-3 分钟）足以判断方向：

```bash
python examples/train_ppo.py --total-timesteps 50000 --no-viewer --device cpu \
  --reward-config config/reward_my_v1.yaml
```

方向对了再跑长训练。

---

## 6. 常见奖励规则设计

### 6.1 默认规则（dense + sparse）

适合当前项目：有明确的抓取、抬升、放置阶段，dense reward 引导学习，sparse success 给最终目标。

### 6.2 稀疏规则（sparse + minimal shaping）

如果想让策略更 robust：

```yaml
rewards:
  time_penalty: -0.1
  approach:
    scale: 0.1
    distance: 0.1
  grasp:
    bonus: 0.0
  lift:
    scale: 0.0
  transport:
    scale: 0.1
    distance: 0.1
  success:
    bonus: 1000.0
```

学习更慢，但不容易 reward hacking。

### 6.3 高成功奖励规则

如果想快速解决 laziness：

```yaml
rewards:
  success:
    bonus: 5000.0
  approach:
    scale: 0.5
  transport:
    scale: 0.5
  lift:
    scale: 1.0
```

成功奖励主导，智能体有强烈动机完成任务。

---

## 7. 进阶：Potential-Based Reward Shaping（PBRS）

如果想保证不改变最优策略，可以使用 PBRS：

```python
phi = -distance_ee_to_cube - distance_cube_to_tray + 5.0 * cube_height
reward = gamma * phi_next - phi
```

这种 shaping 在理论上不会引入 reward hacking，但设计好的势能函数需要更多领域知识。当前 YAML 配置未使用 PBRS，如需实现可以手动修改 `panda_pick_env.py` 的奖励计算部分。

---

## 8. 快速检查清单

跑实验前确认：

- [ ] YAML 文件格式正确（可以用 `python -c "import yaml; yaml.safe_load(open('config/xxx.yaml'))"` 检查）
- [ ] `success.bonus` 显著大于每 episode 能拿到的最大 dense reward
- [ ] 每次只改一个参数
- [ ] 先用 50k 步短实验验证方向
- [ ] 同时看 `rollout/success_rate` 和 `rollout/ep_rew_mean`
- [ ] 看 `reward/xxx_mean` 判断卡在哪个阶段

---

## 9. 命令速查

```bash
# 默认配置训练
cd /mnt/d/work/jaka_zu35_mujoco_rl
python examples/train_ppo.py --total-timesteps 200000 --no-viewer --device cpu

# 自定义配置训练
python examples/train_ppo.py --total-timesteps 200000 --no-viewer --device cpu \
  --reward-config config/reward_my_v1.yaml

# 短实验快速迭代
python examples/train_ppo.py --total-timesteps 50000 --no-viewer --device cpu \
  --reward-config config/reward_my_v1.yaml

# 查看 TensorBoard
tensorboard --logdir logs/
```
