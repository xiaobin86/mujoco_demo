# ACT 训练 GPU 优化记录 + LeRobot ACT 源码分析

> 日期：2026-08-08
> 范围：`scripts/train_act.py` 训练管线优化，以及与 LeRobot（/mnt/d/work/lerobot，v0.4.1）ACT 实现的对比分析。

---

## 0. ACT 模型基础架构（背景）

ACT（Action Chunking with Transformers，Zhao et al. 2023，ALOHA 论文）是面向精细操作的任务级模仿学习策略。本节介绍**标准版**（= LeRobot 实现 = 原版论文）的架构，作为后文所有优化方案的对照基线。

### 0.1 两个核心思想

**① Action chunking（动作分块）**：不逐步预测单个动作，而是一次预测未来 k 步动作序列（chunk）。ALOHA 默认 chunk_size=100（50Hz 下 2 秒）。两个收益：

- **缓解复合误差**：模型对一段时间的动作"做出承诺"，不会因为单步预测抖动而左右摇摆；
- **容忍多模态动作分布**：同一观测下演示数据可能有多种合理走法，逐步回归会把它们平均成"四不像"，成块预测则每次只需选中一种连贯走法。

**② CVAE 潜变量**：训练时用一个 VAE encoder 把**真值动作序列**压缩成潜变量 z（捕捉这条演示的"风格/意图"），decoder 以 z + 观测为条件重建动作；KL 损失把 z 拉向标准正态。推理时没有真值动作，z 取 0（或从先验采样）。z 的存在让 decoder 学会"给定一个意图，生成连贯轨迹"，而不是对所有演示做平均。

### 0.2 网络结构

```
                 训练时存在，推理时丢弃
                ┌──────────────┐
  真值动作序列 ─►│  VAE encoder  │──► (μ, log σ²) ──► 采样 z（重参数化）
  机器人状态   ─►│ (transformer)│
                └──────────────┘
                                       ┌─────────────────────────┐
   图像 ─► ResNet-18 ─► 特征图 token ─►│                         │
   状态 ─► 线性投影 ──────────────────►│  Transformer encoder    │
   z    ─► 线性投影 ──────────────────►│         │               │
                                       │         ▼               │
                                       │  Transformer decoder    │◄─ k 个可学习
                                       │  （cross-attention）    │   action queries
                                       │         │               │   （DETR 风格）
                                       └─────────┼───────────────┘
                                                 ▼
                                    action head（Linear）
                                                 │
                                                 ▼
                                   (chunk_size, action_dim) 动作块
```

### 0.3 关键环节拆解

| 环节 | 作用 | 原版/LeRobot 配置 |
|---|---|---|
| **视觉骨干** | 图像 → layer4 特征图，**每个空间位置展开为一个 token**（保留空间信息），配 2D 正弦位置编码 | ResNet-18，ImageNet 预训练，FrozenBatchNorm |
| **VAE encoder**（仅训练） | 输入 [cls, 状态, 动作序列]，cls token 输出 (μ, log σ²)，采样 z | latent_dim=32，4 层 transformer |
| **Transformer encoder** | 融合 [z, 状态 token, 图像 token 序列] 为 memory | dim=512，heads=8，ffn=3200，4 层 |
| **Transformer decoder** | k 个 action queries 对 memory 做 cross-attention，每个 query 对应 chunk 中一个时间步 | **1 层**（原版代码 bug 致 7 层只有 1 层生效，LeRobot 沿用） |
| **action head** | Linear 回归出 action_dim 维动作 | — |
| **损失** | **L1** 重建损失（pad 帧被 mask 掉）+ kl_weight × KL 散度 | kl_weight=10 |
| **优化器** | AdamW，backbone 与其余参数分组 | lr=1e-5，wd=1e-4（Aloha 任务默认） |
| **推理** | z=0 预测 chunk；**temporal ensembling**：每步都预测，对重叠预测按 w_i=exp(-0.01·i) 加权融合（老预测权重更高） | coeff=0.01 |
| **数据侧** | chunk 超出 episode 末尾时 pad，并用 `action_is_pad` 掩码排除出损失 | chunk 永不跨 episode |

### 0.4 我们的简化版 vs 标准版（优化前基线）

本项目的 `ACTPolicy`（`jaka_zu35_mujoco_rl/act/act_policy.py`）是轻量简化版，与标准版的差距正是后文优化项的来源：

| 环节 | 标准版 ACT（LeRobot） | 我们的简化版（优化前） |
|---|---|---|
| CVAE 潜变量 | ✓ latent 32 + KL | ✗ 纯行为克隆 |
| 视觉特征 | 特征图按空间位置展开成 token 序列 | 全局平均池化 → 1 个 token |
| Transformer | dim 512 / enc 4 层 / dec 1 层 / dropout 0.1 | dim 128 / enc 2 + dec 2 / 无 dropout |
| 损失 | L1 + pad mask | MSE |
| 优化器 | AdamW（wd=1e-4） | Adam（无权重衰减） |
| 推理 | temporal ensembling | chunk 整段开环执行完再预测 |
| episode 边界 | pad + mask，chunk 不跨边界 | chunk 可跨边界 |

> 注：简化版没有 VAE 是刻意取舍——潜变量主要解决"演示多模态"问题，对单臂抓取+数据量小的场景，先把骨干、损失、正则、推理融合这些环节做对，收益更直接。CVAE 列为长期项（见第 5 节第三梯队）。

---

## 1. 背景：训练慢，GPU 没吃饱

**现象**：60 个 episode（17762 帧）训练 ACT，140 epoch 用了 30 分钟（~12.9 s/epoch），1500 epoch 预计 5.4 小时。

**诊断证据**（`nvidia-smi` 实测）：

| 指标 | 读数 | 含义 |
|---|---|---|
| GPU-Util | 85% | 有迷惑性：只统计"有 kernel 在跑的时间占比" |
| 功耗 | ~65W | 5070 Ti Laptop 满载 100W+，远未吃满 |
| 显存 | 530MB / 12GB | 模型+激活很小 |
| 显存带宽利用 | 18% | 很低 |

**结论**：高利用率 + 低功耗 = 典型"一堆小 kernel"模式。瓶颈不在算力，而在每步开销：

1. **DataLoader 单进程（num_workers=0）**：每个 batch 主线程逐样本 `numpy→tensor` 转换再 collate，GPU 等 CPU；
2. **batch=16 太小**：ResNet-18 在 64×64 图上 kernel 极小，launch 开销占比高；
3. **Adam 每步更新 12.17M 参数**：elementwise 操作，GPU"忙着"但功耗上不去。

每步 ~11.6ms，其中真正的 GPU 计算只有几毫秒，其余是 CPU/launch 开销。

---

## 2. 已实施的优化（scripts/train_act.py）

### 2.1 数据管线重写为 GPU 常驻（最大收益）

原来：DataLoader 单进程，每样本 `torch.from_numpy`，每 batch CPU→GPU 拷贝。

现在（`TeleopDataset` 类）：

- 所有帧**一次性**上传显存：图像保持 **uint8**（约为 float32 的 1/4 显存，~218MB），编码器内部再转 float；proprio/action 转 float32；
- action chunk 用 `act.unfold(0, chunk_size, 1)` **预构建**为 `(n, chunk_size, action_dim)` 张量，避免每样本切片；
- 每个 batch 就是**显存内的索引切片**（`gather(idx)`），零 CPU 开销、零逐 batch H2D 传输；
- train/val 切分用设备上的 `torch.randperm` 索引数组表示。

### 2.2 torch.compile（CUDA 默认开启，`--no-compile` 可关）

融合 kernel，消除小 kernel 的 launch 开销。注意两点实现细节：

- 首次调用有一次性编译开销（~30-60s），摊到前几十个 epoch；
- **保存 checkpoint 必须用原始模块**（`raw_model.state_dict()`）：`torch.compile` 包装后的 state_dict 键会带 `_orig_mod.` 前缀，导致 `act_rollout.py` 加载失败。

### 2.3 bfloat16 autocast（CUDA 默认开启，`--no-amp` 可关）

`torch.autocast(device_type, dtype=torch.bfloat16)`，参数仍为 fp32 主权重（autocast 只降计算精度），无需 GradScaler。RTX 50 系 bf16 算力高，实测无收敛问题。

### 2.4 默认 batch-size 32 → 64

大 batch 是喂饱 GPU 的前提。batch 增大 N 倍时 lr 建议同步上调（本次 16→64 配 3e-4→6e-4），`ReduceLROnPlateau` 会兜底。

### 2.5 附带改进

- 每 10 epoch 打印 `s/epoch`（累计平均），方便监控速度；
- TensorBoard 记录 `train/loss`、`val/loss`、`train/learning_rate`（写入 `logs/ACT_N`，与 SB3 的 `PPO_N` 并列）。

---

## 3. 优化结果

| 指标 | 优化前 | 优化后 |
|---|---|---|
| s/epoch（稳态） | 12.9 | **~2.3** |
| 提速 | — | **~5.6×** |
| 1500 epoch 总时长 | ~5.4 小时 | **~1 小时** |
| GPU 显存占用 | 530MB | ~1.7GB（数据常驻） |
| GPU 利用率 | 85%（假性） | 95%（真实负载） |

验证记录：

- CPU/CUDA 冒烟训练通过；torch.compile + bf16 正常；
- checkpoint 无 `_orig_mod.` 前缀，`act_rollout.py` 加载兼容；
- `pytest tests/` 30 passed。

---

## 4. LeRobot ACT 源码分析（v0.4.1）

关键文件：

- `src/lerobot/scripts/lerobot_train.py` — 训练循环（基于 HF Accelerate）
- `src/lerobot/policies/act/modeling_act.py` — ACT 策略实现
- `src/lerobot/policies/act/configuration_act.py` — 默认超参
- `src/lerobot/datasets/sampler.py` — `EpisodeAwareSampler`

### 4.1 LeRobot 的做法 vs 我们

| 机制 | LeRobot | 我们 | 评价 |
|---|---|---|---|
| 训练范式 | step-based（默认 100k steps）+ `cycle(dataloader)` | epoch-based | 两者皆可 |
| 混合精度 | Accelerate autocast | bf16 autocast ✓ | 等价 |
| torch.compile | 未用 | ✓ | 我们更激进 |
| 数据加载 | DataLoader(num_workers=4, pin_memory, prefetch_factor=2) | GPU 常驻数据集 | 我们更激进 |
| cudnn.benchmark / TF32 | `cudnn.benchmark=True` + `allow_tf32=True` | **未设置** | **可借鉴（零成本）** |
| Scheduler | ACT preset 不用 scheduler，按 batch step | ReduceLROnPlateau 按 epoch | 皆可 |
| Resume | 完整训练状态（optimizer+scheduler+step） | 仅权重 | 可借鉴 |
| 指标 | update_s / dataloading_s 分离、grad_norm | s/epoch、loss/lr | 可加 grad_norm |
| chunk 跨 episode | 数据集 pad + `action_is_pad` 掩码，loss 忽略 pad 帧 | **未处理（chunk 会跨边界）** | **可借鉴** |
| Loss | **L1** + pad mask | MSE | 可尝试 |
| 优化器 | **AdamW**（wd=1e-4），backbone 独立 lr 参数组 | Adam 无 wd | 可借鉴 |
| 视觉特征 | ResNet-18 取 **layer4 feature map**，每个空间位置一个 token + 学习位置编码 | ResNet-18 全局平均池化 → 1 个 token | 可借鉴（保留空间信息） |
| Transformer 规模 | dim 512 / heads 8 / ffn 3200 / enc 4 层 / dec **1** 层 / dropout 0.1 | dim 128 / heads 4 / enc 2 + dec 2 / 无 dropout | dropout 可直接加 |
| CVAE | use_vae=True，latent 32，kl_weight=10 | 无 | 大改动，长期项 |
| 推理 | **temporal ensembling**（coeff=0.01，每步预测+指数加权融合） | chunk 开环执行完再预测 | **高优先级可移植** |

### 4.2 值得注意的细节

1. **decoder 只有 1 层**：原版 ACT 代码有个 bug 导致 7 层 decoder 只有第 1 层生效，LeRobot 沿用了这个行为（`n_decoder_layers=1`，见 configuration_act.py 注释）。
2. **temporal ensembling 的权重方向**：`w_i = exp(-coeff * i)`，**越老的预测权重越高**（coeff=0.01）。LeRobot PR #319 的实验表明激进地偏重新预测会削弱 action chunking 的收益。实现是在线加权平均（`ACTTemporalEnsembler`，~90 行），每步都重新预测一个 chunk 并融合。
3. **episode 边界处理**：LeRobot 数据集对超出 episode 末尾的 chunk 做 pad，并给出 `action_is_pad` 掩码；loss 计算为 `F.l1_loss(...) * ~action_is_pad`——不浪费数据也不跨边界。训练脚本里还有 `EpisodeAwareSampler` 可选直接丢弃每个 episode 末尾 N 帧。
4. **归一化**：LeRobot 对 state/action 用 mean_std 归一化；我们的 env 动作空间已是 [-1,1] + tanh 输出，等价覆盖。
5. **lr 配置**：LeRobot ACT preset 用 lr=1e-5（Aloha 双臂任务默认），比我们的 6e-4 小一个量级——超参高度依赖任务/数据规模，不照搬。

---

## 5. 后续优化路线（按性价比排序）

> **更新（2026-08-08 第二轮）**：第一梯队 1-4 项与第二梯队第 5 项（temporal ensembling）已全部实施，见第 7 节。

### 第一梯队：零/低成本，直接做

1. **性能开关**（train_act.py 加两行）：
   ```python
   torch.backends.cudnn.benchmark = True      # 输入形状固定，自动选最优 conv 算法
   torch.backends.cuda.matmul.allow_tf32 = True
   ```
2. **修 chunk 跨 episode 边界**：利用 npz 里的 `episode_starts`，丢弃每个 episode 末尾 `chunk_size` 帧的采样索引（参考 `EpisodeAwareSampler`），或进一步做 pad+mask。
3. **L1 loss**（替换 MSE，对离群更鲁棒）+ **dropout 0.1**（transformer 层）。
4. **AdamW + weight_decay 1e-4**（train/val 剪刀差出现时有助于抑制过拟合）。

### 第二梯队：中等成本，收益明确

5. **temporal ensembling 推理**：把 `ACTTemporalEnsembler`（modeling_act.py:164-252）移植到 `act_rollout.py`，每步预测 + 指数加权融合（coeff=0.01）。原版 ACT 效果的重要来源，通常对抓取平滑度/成功率有直接帮助。
6. **完整 resume**：保存/恢复 optimizer 状态 + epoch，配合现有 `--resume`。
7. **grad_norm 记录**到 TensorBoard，监控训练稳定性。

### 第三梯队：大改动，长期考虑

8. **feature-map 骨干**：ResNet layer4 特征图按空间位置展开成 token 序列 + 学习位置编码（保留空间信息，优于全局池化）。
9. **CVAE**（latent 32 + kl_weight 10）：处理多模态动作分布，原版 ACT 核心组件。
10. **终极选项**：lerobot 已 editable 安装，可直接用 `lerobot.policies.act.ACTPolicy`（完整原版 ACT + 归一化/前后处理管线），代价是要适配它的 batch 格式与 preprocessor。

---

## 6. 复现命令

```bash
# 优化后的训练（当前推荐配置）
python scripts/train_act.py --input data/teleop_demos_*.npz \
  --epochs 1500 --lr 6e-4 --batch-size 64 --chunk-size 16 \
  --output checkpoints/act_so101.pt

# 逃生开关
#   --no-compile   关闭 torch.compile
#   --no-amp       关闭 bf16 autocast

# 监控
tensorboard --logdir logs    # ACT_N 与 PPO_N 并列
```

---

## 7. 第二轮改动（2026-08-08，训练质量向）

第一轮解决"速度"，第二轮解决"结果质量"。全部已实施并验证。

### 7.1 train_act.py

| 改动 | 说明 |
|---|---|
| `cudnn.benchmark=True` + `allow_tf32=True` | LeRobot 同款性能开关，输入形状固定时自动选最优 conv 算法 |
| **episode 边界感知采样** | 利用 npz 的 `episode_starts`，剔除跨 episode 的 chunk 采样位置（同 LeRobot `EpisodeAwareSampler` 思路）。实测 350 帧/chunk 8：342 个位置中剔除 42 个跨界位置，剩 300 个边界安全样本 |
| **episode 级 train/val 切分** | 按 episode 而非按帧切分（`--seed` 可复现），消除验证集泄漏，best checkpoint 选择更可信 |
| **MSE → L1 loss** | LeRobot/原版 ACT 用 L1，对离群更鲁棒。注意 loss 数值尺度与 MSE 不可直接比较 |
| **Adam → AdamW**（`--weight-decay` 默认 1e-4） | 权重衰减抑制过拟合（针对已观察到的 train/val 剪刀差） |

### 7.2 act_policy.py

- ACTPolicy 新增 `dropout=0.1`（transformer encoder/decoder，同 LeRobot）。不新增参数，旧 checkpoint 兼容。

### 7.3 act_rollout.py

- 移植 **temporal ensembling**（`TemporalEnsembler`，忠实还原 LeRobot `ACTTemporalEnsembler` 的在线加权算法，数值验证 online == offline 加权平均，误差 <1e-5）。
- `--temporal-ensemble-coeff` 默认 **0.01**（原版 ACT 值）：每步都预测一个 chunk 并与历史预测指数加权融合（老预测权重更高），替代原来"整段开环执行完再预测"。`<=0` 关闭回到旧行为。
- 代价：推理频率从每 chunk 一次变为每步一次；100 episode rollout 约多几分钟。

### 7.4 验证记录

- 三文件 py_compile 通过；`pytest tests/` 30 passed；
- 合成数据冒烟：边界采样数与手算一致（300/342），episode 切分 6/1 正确，L1+AdamW 正常；
- TemporalEnsembler 数值正确性验证通过。

### 7.5 第二轮后的推荐命令

```bash
python scripts/train_act.py --input data/teleop_demos_*.npz \
  --epochs 1500 --lr 6e-4 --batch-size 64 --chunk-size 16 --seed 42 \
  --output checkpoints/act_so101.pt

python scripts/act_rollout.py --model checkpoints/act_so101.pt \
  --chunk-size 16 --episodes 100    # temporal ensembling 默认开启
```
