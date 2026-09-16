# H100 服务器参数调整文档

面向：Ubuntu + NVIDIA H100 上跑 SteerPose 复现（BamaPig3D 等数据集）。

---

## ⚠️ 上手前必读：先跑数据体检 + 推理链自检（都是秒级）

```bash
# 1) 数据体检：3D GT 用官方外参+内参投影回图像 vs 官方 2D 标注
#    重投影误差小 → 骨架顺序 / 外参约定 / 内参口径 / 2D 坐标系四项一次全对
python scripts/check_mammal_data.py --root /data/BamaPig3D_pure_pickle

# 2) 推断链自检：不依赖训练、不依赖数据集，用合成真值场景验证 calibrate 的三处要害
#    （匹配矩阵形状/归一化、几何一致性项方向与平移求解、内参 K 通路）
python scripts/selftest_infer.py
```

**训练损失 `Lkp` 不经过匹配与几何这两条路径**，所以 `Lkp` 下降完全不能说明它们是对的
——本仓库就曾因此藏过三个隐蔽 bug（详见 README "实现修正记录"）。

另外，**档位 A（quick）容易让人误判**：它的定位是"跑通链路"，但步数给少时
val Lkp 会一直贴在平凡基线上（≈0.65），打乱 R 也不影响损失——**看起来像数据或实现坏了，
其实只是没训够**。所以现在 quick 档也给了约 1.2 万步，且在日志里显式打印总步数与基线；
拿 quick 的权重去 `calibrate` 若仍看到 R 误差接近随机、`几何项 Lgeom：使用 0 对匹配`，
优先加步数，不要改数据。详见第 0 节末尾。

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

### ⚠️ 但是：epoch 便宜 ≠ 步数够。唯一要看的是"总优化步数"

```
总步数 = ceil(训练对数 / batch) × epochs
```

`--batch` 一开大，每轮步数就掉下来，`--epochs` 看着不小、总步数其实很少。
本仓库踩过这个坑：`epochs=50 / batch=512 / pairs=3000` → 每轮只有 5 步，**总计 250 步**，
模型停在"输出平均姿态"上完全没动，从日志上看像数据或实现坏了。实测对照（同一份数据、同一实现）：

| 总步数 | val Lkp | 打乱 R 后的 ΔLkp | 模型状态 |
|--------|---------|------------------|----------|
| 250 | 0.6356 | +0.0195 | **完全忽略 R**（Δ 几乎为 0 = 没在看旋转输入） |
| 2000 | 0.1325 | — | 开始真正使用 R |
| 16500 | **0.0506** | +0.7741 | 正常 |

参照：平凡基线（无论输入都输出平均姿态）= **0.6481**。

`train.py` 现在会把这两行写进日志，**先看它们再决定要不要调参**：

```
[i] 优化规模：每轮 9 步 × 1400 epoch = 共 12600 步
[i] 平凡基线（输出平均姿态）val Lkp = 0.6489
```

val Lkp 与基线相近时，日志末尾会显式警告并给出处理顺序（优先加 `--epochs`、
其次调小 `--batch`）。`scripts/run_h100.sh` 已改为**按目标总步数反推 epoch**。

复现这套对照：`python scripts/diag_lkp_plateau.py`（基线 + 过拟合 + 长训练），
`python scripts/diag_r_ablation.py`（R 消融，判定模型是否真的在用 R）。

---

## 1. 推荐参数（三档配置）

> 三档都由 `scripts/run_h100.sh` 按目标总步数自动折算 epoch，下面列出的 epoch 数
> 是按 `--num-pairs` 与 `--batch` 换算后的**结果**，换数据量时会自动变。

### 档位 A：先验证链路 + 拿到一个"真的学到了东西"的权重

```bash
python -m steerpose.train \
    --poses3d data/poses_bamapig_train.npz \
    --epochs 4000 --batch 256 --workers 4 --num-pairs 3000 \
    --device cuda --amp --out ckpt/smoke.pt --log-every 200
```
用途：确认环境、数据、显存都没问题。**判据不是"跑完了"，而是日志里
`val Lkp` 明显低于同一行上下方的平凡基线**（一般要降到 0.2 以下才算开始有用）。

### 档位 B：正式训练 BamaPig3D（推荐，10~20 分钟）

```bash
python -m steerpose.train \
    --poses3d data/poses_bamapig_train.npz \
    --epochs 3000 --batch 1024 --lr 1e-3 --weight-decay 1e-5 \
    --workers 12 --num-pairs 20000 --num-views 100 --num-rolls 20 \
    --amp --seed 0 --log-every 150 --save-every 750 \
    --out ckpt/bamapig.pt
```
（= 每轮 14 步 × 3000 = 约 4.2 万步，是实测 1.65 万步的 2.5 倍余量。）

### 档位 C：class-agnostic 模型（Animal3D 40 物种，1~2 小时）

```bash
python -m steerpose.train \
    --poses3d data/poses_animal3d.npz \
    --epochs 1000 --batch 2048 --lr 1e-3 \
    --workers 16 --num-pairs 50000 --amp --seed 0 --log-every 50 \
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
| `--cameras` | sphere | **sphere（不要改）** | ⭐️ 视点必须在**完整单位球面**上取。早期实现只取上半球（z ≥ 0.05），等价于只覆盖一半旋转群，训练集里不存在"镜像视图"：训练曲线完全正常，但推理时遇到这类视角（例如 `calibrate --demo` 的相机 A = 单位矩阵）网络输出全错 → 跨视角匹配 0/12 → `Lgeom` 静默为 0。`hemisphere` 只为 A/B 对照保留（`python scripts/diag_sphere_fix.py`，服务器上跑） |
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
| `--focal` | — | 几何项把 2D 坐标当归一化图像坐标（视线 = `(u,v,1)`），所以坐标必须按焦距归一化。**优先用内参**：`--K-file <file>` 或直接用 `prepare_mammal_2d.py` 产出的 npz（内含 `K`，即 BamaPig3D 去畸变后的 `newcameramtx`，fx≈1340）。没有内参时才传 `--focal`（估不准就用 `0.9 × 图像长边`）。**不要**传按包围盒缩放的坐标——缩放会改变射线方向，使正确的 R 不再对应零奇异值 |
| Lgeom 激活时机 | 总迭代的 40% 之后 | 脚本内置（等匹配先稳定） |

标定结束会打印 **匹配置信度**（平均最大指派概率 + 可信匹配数）与 **几何项用了几对匹配**。
若置信度 < 0.5，或几何项标注"不足 3 对，Lgeom 静默失效"，说明 Sinkhorn 接近均匀指派、
R 与 t 都不可信 —— 此时应先解决模型区分度问题，而不是调标定参数。

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
| `val Lkp` 卡在 0.6~0.7 不降 | **总优化步数不够** —— 模型还没开始学，只是在输出"平均姿态" | 看日志的"优化规模"与"平凡基线"两行；先加 `--epochs`，再考虑调小 `--batch`。用 `scripts/diag_lkp_plateau.py` 复核：同数据 250 步 → 0.64（贴基线），2000 步 → 0.13 |
| `val Lkp` 降了但标定结果差 | 模型区分度不足（正确配对和错误配对的姿态距离接近） | 先跑 `python scripts/selftest_infer.py` 排除实现问题；再加大总步数（不是只加 `--num-pairs`）；标定输出里的"匹配置信度" < 0.5 就是这个原因 |
| 拿 quick 档位的权重去标定，R 误差 ≈ 随机 | quick 档的步数不足以让模型学会使用 R | 换 `full` 档位重训；用 `scripts/diag_r_ablation.py` 可以直接验证"打乱 R 是否影响损失" |
| Lgeom 恒为 0 | 可靠匹配置信度低于阈值（自适应，至少 0.3），不足 3 对时按约定返回 0 | **按顺序查三件事**：① `--cameras` 是 `sphere` 吗（半球会让镜像视角落在分布外，`calibrate --demo` 直接崩）；② 权重步数够不够；③ `scripts/selftest_infer.py` 是否全通过。看日志是否标了"不足 3 对，Lgeom 静默失效"：是则是匹配区分度问题（先解决上一行），不是几何项实现问题；Lgeom 在匹配稳定前不会有贡献（这也是论文把它放在 40% 迭代后才激活的原因） |
| `calibrate --demo` 匹配全错、ER ≈ 80°、Lmatch 停在 ~17 | 训练相机集合只覆盖上半球，而 `--demo` 场景的相机 A 是单位矩阵（正交下等价于"从下方看"的镜像视角）→ 网络输入落在分布外 | 用 `--cameras sphere` 重训（**默认值即为此**，除非你手动改过）。根因与证据链见 README『实现修正记录 · 第三批』 |
| `t` 解出来是 `[0,0,0]` | 可信匹配不足 3 个，脚本按约定返回零向量 | 同上，属于区分度问题而非求解器问题 |
| 几何项报 `W 形状与目标数不一致` | 匹配矩阵维度不对（`pose_distance` 被写成 cdist 的典型症状） | 跑 `scripts/selftest_infer.py` 的 M1 组；M1 会直接指出形状错误 |
| 标定日志里焦距/主点看着不对（如 fx=1728） | 没有走内参通路，退化成 `0.9 × 图像长边` 的猜测值 | 用 `--poses-npz`（内含 `K`）或显式 `--K-file`；BamaPig3D 应为 `newcameramtx`，fx≈1340 |
| `_best.pt` 没生成 / 日志末尾 `best val Lkp inf` | 旧版把最佳权重的保存条件写在 `% log_every` 里，`--epochs` 不是 `log_every` 整数倍时永不触发 | 已修（末轮强制评估）；若仍出现，检查 `--log-every` 是否大于 `--epochs` |
| GPU 利用率低 | 数据侧瓶颈 | 加 `--workers`；`--num-pairs` 一次性合成完后不应再重复 |
| 显存 OOM | 基本不可能（<1GB） | 若真出现，是其它进程占用 |
| 训练太慢 | 用了 CPU | 加 `--device cuda`；确认 `torch.cuda.is_available()` 为 True |
| 结果方差大 | 数据仅 228 个姿态 | 跑多 seed（0/1/2）取中位；或改用 class-agnostic（Animal3D）模型 |

---

## 8. 与论文对齐的核查清单

跑完一遍后，对照检查以下项是否一致（能定位大部分复现偏差）：

- [ ] `python scripts/selftest_infer.py` 全通过（17 项），且**放在任何改动之后重跑**
- [ ] `python scripts/check_mammal_data.py --root <pure_pickle>` 重投影误差中位 < 5 px
- [ ] 标定用的是内参 `newcameramtx`（fx≈1340），不是原始 K（1625）也不是猜测的 0.9×长边
- [ ] 关节数 19，顺序为 nose, l/r_eye, l/r_ear, l/r_shoulder, l/r_elbow, l/r_paw, l/r_hip, l/r_knee, l/r_foot, tail, center
- [ ] 训练用帧 ≤1400（论文 Table 7），得到约 228 个 3D 姿态
- [ ] 视角对 3000+，100 视点 × 20 滚转，**正交投影**（论文附录 D.1）
- [ ] 输入增强：2D 噪声 + 随机掩码 10%~30% 关节
- [ ] 损失 Lkp 为**逐关节 L2 的均值**（不是平方和 / 不是平方根后求和）
- [ ] 标定阶段 α=3、1000 次迭代、Lgeom 在 40% 后激活、最多 5 次重启
- [ ] 标定时传了正确的 `--focal`（几何项是**透视**共线约束，与训练用的正交投影不同，这是论文本身就有的设定差异）
- [ ] 标定输出里的"匹配置信度" ≥ 0.5，否则结果不可采信
- [ ] `docs/paper_notes.md` 第 8 节列的"与官方实现的已知差异"是否可接受
