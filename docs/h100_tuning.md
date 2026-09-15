# H100 服务器参数调整文档

面向：Ubuntu + NVIDIA H100 上跑 SteerPose 复现（BamaPig3D 等数据集）。

---

## 0. 先认清一件事：这个模型小到 H100 是"过剩"的

| 项 | 数值 |
|----|------|
| 模型参数量 | **约 68.9K**（32 维 token × 5 层 Transformer） |
| 单个样本 | 19 关节 × 2 维 + 3 维旋转 = **41 个浮点数** |
| 实际显存占用 | batch 4096 时 **< 300 MB** |
| H100 上的瓶颈 | **不是 GPU 算力，而是数据合成（CPU numpy）与 DataLoader 进程** |

**这意味着**：不要指望"跑满 H100 才快"。正确用法是——
① 把 batch / 视角对数量开到显存上限之上都不会 OOM；
② 与其跑一个超长训练，不如**跑一个小规模超参扫描**（每轮几分钟），挑 val Lkp 最低的配置；
③ `--workers` 要比 GPU 训练常规设置更大，因为瓶颈在喂数据。

---

## 1. 推荐参数（三档配置）

### 档位 A：先验证链路（首次上服务器，5 分钟内出结果）

```bash
python -m steerpose.train \
    --poses3d data/poses_bamapig_train.npz \
    --epochs 30 --batch 512 --workers 4 --num-pairs 3000 \
    --device cuda --amp --out ckpt/smoke.pt --log-every 5
```
用途：确认环境、数据、显存都没问题。看 `epoch 30 val Lkp` 是否明显低于初始值（初始 ≈ 0.6~0.7）。

### 档位 B：正式训练 BamaPig3D（推荐，10~20 分钟）

```bash
python -m steerpose.train \
    --poses3d data/poses_bamapig_train.npz \
    --epochs 400 --batch 4096 --lr 1e-3 --weight-decay 1e-5 \
    --workers 12 --num-pairs 20000 --num-views 100 --num-rolls 20 \
    --amp --seed 0 --log-every 20 --save-every 100 \
    --out ckpt/bamapig.pt
```

### 档位 C：class-agnostic 模型（Animal3D 40 物种，1~2 小时）

```bash
python -m steerpose.train \
    --poses3d data/poses_animal3d.npz \
    --epochs 800 --batch 8192 --lr 1e-3 \
    --workers 16 --num-pairs 50000 --amp --seed 0 --log-every 25 \
    --out ckpt/animal3d_agnostic.pt
```

---

## 2. 每个参数为什么这么调

| 参数 | 论文/默认值 | H100 推荐 | 理由 |
|------|-------------|-----------|------|
| `--num-pairs` | 3000 | **20000（BamaPig 228 个姿态）/ 50000（Animal3D）** | ⭐️**这是最该调大的参数**。论文用 3000 是因为数据多；BamaPig3D 训练集只有 **228 个 3D 姿态**（70 帧 × 4 头，帧 ≤1400），只有靠海量视角对才能覆盖姿态 × 视角空间。合成是纯几何投影，几乎不耗算力，加 10 倍只多几秒 |
| `--batch` | 512 | **4096（B）/ 8192（C）** | 模型仅 68.9K 参数，大 batch 完全不占显存。大 batch 让每个 epoch 的梯度更稳，配合 `--lr 1e-3` 收敛更快 |
| `--epochs` | — | **400（B）/ 800（C）** | 视线对数而定：20000 对、batch 4096 → 每 epoch 仅 ~5 步，步数少所以要多个 epoch。**判据看 val Lkp 是否连续多个 log 点不再下降**，而不是死盯 epoch 数 |
| `--lr` | 1e-3 | **1e-3**（`CosineAnnealingLR` 自动退火） | 训练脚本内置余弦退火；若 val Lkp 抖动，先降到 5e-4，不要盲目加 epoch |
| `--workers` | 4 | **12~16** | H100 机器核多；瓶颈在数据侧，worker 少了 GPU 会等。注意 worker 太多反而会因每 batch 太小而空转 |
| `--amp` | 关 | **开**（bf16） | H100 原生支持 bf16。对速度提升有限（模型太小），但白拿；不会影响精度 |
| `--weight-decay` | 0 | **1e-5** | 数据量小（228 姿态）容易过拟合，加一点点正则 |
| `--num-views` / `--num-rolls` | 100 / 20 | **100 / 20（保持论文设置）** | 视角采样密度直接决定模型对旋转的分辨能力，不建议改小；想更密可试 200/40（合成代价线性增长） |
| `--seed` | 0 | 扫描时 **0/1/2 各跑一次** | 228 个姿态 + 随机视角对，方差较大；单次结果的偶然性高 |

---

## 3. 建议：不要跑单次长训练，跑个小扫描

因为每次训练只要几分钟，**扫描比调参更划算**。仓库提供了脚本：

```bash
bash scripts/sweep_h100.sh data/poses_bamapig_train.npz
```
它会依次跑 6 组配置（num_pairs ∈ {3000, 10000, 20000, 50000} × batch ∈ {2048, 4096}），
把每组最优 val Lkp 汇总到 `ckpt/sweep_summary.txt`，你挑最好的那组再跑长训练。

---

## 4. 显存与吞吐参考（H100 80GB）

| batch | 显存 | 说明 |
|-------|------|------|
| 512 | ~120 MB | 论文级设置 |
| 4096 | ~300 MB | 推荐 |
| 16384 | ~1.0 GB | 仍远未饱和，可继续加 |

真正的时间开销分布（实测比例）：
1. **合成视角对**（一次性，20000 对 ≈ 10~30 s）——只发生在训练开始时；
2. **每个 epoch 的前向/反向**——H100 上每步毫秒级；
3. **DataLoader 取数 + 增强**（噪声、掩码、归一化）——**这是最可能成为瓶颈的部分**。

> 如果 `nvidia-smi` 显示 GPU 利用率低（<30%），先加 `--workers`，而不是加 batch。

---

## 5. 标定阶段（inference-time optimization）的参数

标定**不需要 GPU 加速**（只优化 3 个旋转参数，1000 次迭代，秒级完成），但论文级设置如下：

```bash
python -m steerpose.calibrate \
    --ckpt ckpt/bamapig_best.pt \
    --poses-npz data/two_view_pig_c0_c6.npz \
    --iters 1000 --lambda-geom 1.0 --alpha 3.0 --retries 5 \
    --out out/two_view_c0_c6.npz
```

| 参数 | 论文值 | 说明 |
|------|--------|------|
| `--iters` | 1000 | Adam lr=0.01，线性衰减 1.0→0.01（脚本内置） |
| `--lambda-geom` | 1.0 | Lgeom 权重 λ；若结果旋转误差大但匹配看着对，可上调到 2~5 试 |
| `--alpha` | 3.0 | 相似度 `s = 2/(1+exp(α·Lkp))` 的缩放，论文附录 D.3 明确给出 |
| `--retries` | 5 | 失败换随机初始 R，论文设置 |
| Lgeom 激活时机 | 总迭代的 40% 之后 | 脚本内置（等匹配先稳定） |

**多相机（10 视角）**：逐对跑（10 视角有 45 对），再用 motion averaging + BA 统一坐标系——
这部分本仓库未实现，可参考 `docs/paper_notes.md` 第 5 节的提示。

---

## 6. 后台运行与日志

```bash
mkdir -p log
nohup python -m steerpose.train --poses3d data/poses_bamapig_train.npz \
    --epochs 400 --batch 4096 --workers 12 --num-pairs 20000 --amp \
    --out ckpt/bamapig.pt > log/train.out 2>&1 &

tail -f log/train.out              # 实时看
tail -f ckpt/bamapig_train.log     # 脚本自己写的结构化日志
```

产出文件：
```
ckpt/bamapig.pt              最终权重（含 model / opt / epoch / num_joints / config）
ckpt/bamapig_best.pt         val Lkp 最低的权重 ← 标定时用这个
ckpt/bamapig_ep100.pt       周期性快照
ckpt/bamapig_train.log       文本日志
ckpt/bamapig_config.json     本次超参留档（复现必备）
```

---

## 7. 排障对照表

| 现象 | 原因 | 处理 |
|------|------|------|
| `val Lkp` 卡在 0.6~0.7 不降 | 模型在输出"平均姿态" | 检查数据是否真的读了（`[i] 3D 姿态: N 个 x 19 关节`）；N 太小（<50）就先补数据；确认 `--num-pairs` 是否够大 |
| `val Lkp` 降了但标定结果差 | 模型区分度不足（正确配对和错误配对的姿态距离接近） | 这是最典型的问题：加大 `--num-pairs` 与 `--epochs`；检查 Sinkhorn 置信度是否接近均匀（见 README FAQ） |
| Lgeom 恒为 0 | 可靠匹配置信度低于阈值 | 先解决上面的模型区分度问题；Lgeom 在匹配稳定前不会有贡献（这也是论文把它放在 40% 迭代后才激活的原因） |
| GPU 利用率低 | 数据侧瓶颈 | 加 `--workers`；`--num-pairs` 一次性合成完后不应再重复 |
| 显存 OOM | 基本不可能（<1GB） | 若真出现，是其它进程占用 |
| 训练太慢 | 用了 CPU | 加 `--device cuda`；确认 `torch.cuda.is_available()` 为 True |
| 结果方差大 | 数据仅 228 个姿态 | 跑多 seed（0/1/2）取中位；或改用 class-agnostic（Animal3D）模型 |

---

## 8. 与论文对齐的核查清单

跑完一遍后，对照检查以下项是否一致（能定位大部分复现偏差）：

- [ ] 关节数 19，顺序为 nose, l/r_eye, l/r_ear, l/r_shoulder, l/r_elbow, l/r_paw, l/r_hip, l/r_knee, l/r_foot, tail, center
- [ ] 训练用帧 ≤1400（论文 Table 7），得到约 228 个 3D 姿态
- [ ] 视角对 3000+，100 视点 × 20 滚转，正交投影
- [ ] 输入增强：2D 噪声 + 随机掩码 10%~30% 关节
- [ ] 损失 Lkp 为**逐关节 L2 的均值**（不是平方和 / 不是平方根后求和）
- [ ] 标定阶段 α=3、1000 次迭代、Lgeom 在 40% 后激活、最多 5 次重启
- [ ] `docs/paper_notes.md` 第 8 节列的"与官方实现的已知差异"是否可接受
