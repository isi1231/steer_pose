# SteerPose 论文要点（复现对照用）

论文：**SteerPose: Simultaneous Extrinsic Camera Calibration and Matching from Articulation**
- 作者：Sang-Eun Lee、Ko Nishino（京都大学）、Shohei Nobuhara（京都工艺纤维大学）
- 发表：BMVC 2025（arXiv:2506.01691 v2, 2025-08-07）
- 项目页：https://kcvl-public.github.io/steerpose/
- 官方代码：https://github.com/kcvl-public/steerpose —— **截至 2026-09-15 仍为 404（未公开）**
- 联系方式（索取代码）：Sang-Eun Lee `slee@vision.ist.i.kyoto-u.ac.jp`

## 1. 问题与核心思想

无需标定板，直接用画面中自由运动的人/动物的**关节结构**当标定靶，在一次优化里同时完成：
① 相机外参标定；② 跨视角实例/关节匹配。灵感来自认知心理学的**心理旋转**（mental rotation）：
人可以通过在脑中旋转一个视角下的形状来对齐另一个视角，从而感知两视角间的相对位姿。

数学表述：SteerPose 实现**旋转协变**变换 `f(g(·)) = g'(f(·))`，
其中 `f` 是 3D→2D 投影，`g` 是 3D 旋转 R，`g'` 即 SteerPose 对 2D 姿态做的"心理旋转"。

## 2. 网络结构（Fig.2 + 附录 D.2）

| 环节 | 细节 |
|------|------|
| 输入 | N 个关节的 2D 坐标（每个关节 2 维）+ 相对旋转 R（Rodrigues 向量，3 维） |
| 编码 | 各自经 MLP 投到 **32 维** token；旋转向量也是一个 token → 共 N+1 个 token |
| 位置编码 | **可学习查找表**，N+1 个 token（编码关节层级结构），逐元素相加 |
| 主干 | **Transformer 编码器 5 层**（多头自注意力 + FFN） |
| 输出 | 输出 token **均值池化** → 32 维 → MLP 回归出 N 个关节的 2D 坐标 `q(R)` |

论文明确给出的超参：token 维 32、encoder 5 层、注意力头数未给、FFN 维度未给、MLP 层数未给
（本复现取 nhead=4、dim_feedforward=128、MLP 两层）。

## 3. 训练（3.1 + 附录 D.1/D.2）

**数据合成**（关键：不需要真实多相机数据）
1. 取目标类目的 3D 姿态数据集；
2. **Fibonacci 球面法**在上半单位球面均匀放 **100** 个视点，朝向原点；
3. 每个视点绕光轴均匀采 **20** 个滚转角（模拟相机 roll）；
4. 随机抽 **3000** 对相机，对 3D 姿态做**正交投影** → 2D 姿态对 (P, P′)，相对旋转 R 是真值；
5. 划分 train:val:test = **7:2:1**。

**增强**：输入 2D 关节加随机噪声（模拟检测误差）；随机掩码 **10%~30%** 关节（模拟遮挡）。

**损失**（式 1）：`Lkp(R) = (1/M) Σ D(q_i(R), p'_i)`，D 为逐关节 L2 距离。

**训练粒度**：按类目分别训练（人类 / 四足 / 鸟类…）；另有 **class-agnostic** 版本，
仅在 **Animal3D**（40 个四足物种）上训练一次即可泛化到未见过物种。

## 4. 推理：标定与匹配联合优化（3.2 + 附录 D.3）

固定 SteerPose 参数，只优化相对旋转 R（Rodrigues 3 参数）：

1. SteerPose 把视角 A 的 2D 姿态集 P 变换为 `Q(R)`；
2. **Sinkhorn**（可微最优传输）在 `Q(R)` 与视角 B 观测的 P′ 之间做软指派（集合大小不等时加 dummy）；
3. **匹配损失**（式 2）：相似度 `s = 2/(1+exp(α·Lkp))`，`α = 3`；
   `Lmatch = Σ_j w_j (1 - s_j)`，**双向**计算保证视角间循环一致；
4. **几何一致性损失** `Lgeom`：对当前 R 下匹配上的 N 对姿态，用**共线 + 共面约束**堆线性系统
   `A [x_1 … x_N t]ᵀ = 0`（x_i 为沿射线的深度、t 为相对平移，尺度/符号二义，符号由 chirality check 定），
   取 **σ₁/σ₂**（最小与次小奇异值之比）作为损失，迫使 R 落在"平移有解"的流形上；
5. 总损失 `Lmatch + λ·Lgeom` 反向传播更新 R。

**超参（附录 D.3）**：Adam，lr **0.01**，**1000** 次迭代；lr 线性调度 start 1.0 → end 0.01；
`Lgeom` 在总迭代 **40%** 之后激活（等匹配先稳定）；α = 3；
每个场景抽 100–120 张图像对；**重投影误差 < 10px** 视为标定成功，否则换随机初始 R 重试，**最多 5 次**。

## 5. 多视角（第 4 节）

逐对做两视角标定 → **motion averaging** + 基于循环一致的匹配 → 统一坐标系 →
非线性优化（bundle adjustment）最小化重投影误差进一步精化。
（本仓库暂未实现该部分；前作 RA-L 2022 的 `ba.py` 可参考。）

## 6. 实验结果要点

**对比几何方法（五点法 + RANSAC，RRA/RTA/AUC@30）**：全面胜出，尤其共面退化与宽基线场景。
例：Beagle Dog RRA 0.47 → 0.96；Bama Pig 0.64 → 0.92；Cheetah 0.50 → 0.81。

**对比学习法（RRA/RTA/AUC@20）**：
- vs **LightGlue**（SuperPoint + RANSAC + 本质矩阵 + BA）：全面优于，Pigeon 上 0.02 → 0.98；
- vs **MASt3R**（单目点图 + 3D 重建）：精度相当（猪 AUC 0.85 vs 0.88），
  但在**低纹理背景 / 宽基线**（猎豹、鸽子）场景更稳，因为关节是结构化语义对应，不依赖背景特征。

**消融（Table 3，Lgeom 的作用）**：
| 场景 | 指标 | w/o Lgeom | w/ Lgeom |
|------|------|-----------|----------|
| Bama Pig | ER↓ / Et↓ / P↑ | 11.35 / 7.78 / 0.83 | **8.30 / 4.71 / 0.90** |
| Pigeon | ER↓ / Et↓ / P↑ | 20.70 / 12.02 / 0.99 | **9.20 / 5.70 / 0.99** |

**泛化（Table 4，平均关节误差）**：class-specific vs class-agnostic（Animal3D 训练）
在 Cheetah 0.098 vs 0.202、Bama Pig 0.121 vs 0.173、Beagle Dog 0.166 vs 0.160。

## 7. 数据集（含下载入口）

| 数据集 | 物种 | 下载 | 规模 / 关节 |
|--------|------|------|-------------|
| MAMMAL / BamaPig3D | 巴马猪 | https://github.com/anl13/MAMMAL_datasets | 10 视角 × 1750 帧 @25fps，70 帧完整标注 3D；`pig_{i}frame{k}.txt` 为 23×3，第 18/20/22/23 行恒 0 → **19 个有效关节**；精简版 `BamaPig3D_pure_pickle` 481MB |
| MAMMAL / Beagle dog | 比格犬 | https://github.com/anl13/Beagle_dog_dataset | 10 视角，113 帧带 3D 标注，29 关节 |
| Animal3D | 40 种四足哺乳动物 | https://xujiacong.github.io/Animal3D | 3379 图、26 关节 + **SMAL 姿态/形状参数**（需 SMAL 模型转成关节坐标） |
| AcinoSet | 猎豹 | https://github.com/African-Robotics-Unit/AcinoSet | 6 相机、119,490 帧、7,588 帧标注、含 3D GT 与标定 |
| 3D-PoP | 鸽子 | https://github.com/alexhang212/Dataset-3DPOP （数据：https://tinyurl.com/4ckbjcpx ） | 4 相机 4K、~30 万帧、1/2/5/10 只、2D+3D 关键点 |
| CMU Panoptic | 人类 | http://domedb.perception.cs.cmu.edu/ （需注册，用 `getData.sh`） | 多视角；幼儿序列 `160906_ian1` |
| EgoHumans | 人类（排球） | https://github.com/rawalkhirodkar/egohumans | 多视角 |

**论文实际用的序列与帧区间（附录 Table 7）**：

| 目标 | 训练 | 标定（合成） | 标定（真实 2D 估计器） |
|------|------|--------------|------------------------|
| Cheetah | Jules flick1 (20190309) | Romeo flick (20190227) | Romeo flick (HRNet?) |
| Bama Pig | 帧 0–1400 | 帧 1400–1750 | 帧 0–1750（HRNet） |
| Beagle Dog | 帧 0–80 | 帧 80–112 | 帧 0–112（SuperAnimal） |
| Pigeon | Sequence8_n01_01072022 | Sequence5_n05_01072022 | Sequence5_n05_01072022 |
| Toddler | 171204_pose1–6 | 170915_toddler5 | 170915_toddler5（RTMO） |
| Volleyball | 171204_pose1–6 | 001_volleyball | 001_volleyball（RTMO） |

2D 姿态估计器：每类动物用其专用模型（猪 HRNet、狗 SuperAnimal、人类 RTMO 等）。


## 8. 本复现与论文的差异（已知）

**论文原文对 Lgeom 的表述（已核实，用于对齐）**：
> "Since the partial derivatives of the singular values **σ₂ ≥ σ₁** can be computed in closed-form
> [36, 29], we include the **ratio σ₁/σ₂** as a loss Lgeom in the optimization of R, to guarantee
> that the estimated R makes the **smallest singular value as small as possible**, and hence it has
> a solution of t. Consequently, we optimize **Lmatch + λ·Lgeom**."
>
> 以及："For all **N corresponding 2D pose pairs** under current rotation R, we transform their
> positions into a **normalized coordinate system**. By stacking linear equations derived from
> collinearity and coplanarity constraints [19]..."

据此确定的三点实现口径：

1. **σ₁ = 最小、σ₂ = 次小**，Lgeom 取二者之比 → 0 表示当前 R 下存在合法平移解。
   注意 `torch.linalg.svdvals` 返回**降序**，取最小两个是 `s[-1]`、`s[-2]`。
2. **N = 匹配的"姿态对"数**，每个姿态对一个 3D 点 —— 本复现取该姿态的**关节质心**
   作为观测点（`reduce="centroid"`）。也可用 `reduce="joints"` 逐关节取点（约束更多、
   矩阵更大），两者都通过自检，质心版更贴论文口径且更快。
3. **"normalized coordinate system" = 归一化图像坐标**（按焦距归一化），视线方向为
   `(u, v, 1)`。**不能**用按包围盒缩放的数据自适应坐标 —— 缩放等价于偷偷改焦距，
   会改变每条射线的方向，使正确的 R 不再对应零奇异值（实测对比度从 ~700× 掉到 ~0.2×）。

其余仍在的差异：

- 论文未给 MLP 层数 / 头数 / FFN 维度 / SteerPose 训练超参（lr、epoch、batch）→ 本复现取常规默认值；
- 共面（coplanarity）约束未用于 Lgeom：论文把共线 + 共面一起堆进线性系统，但共面项在
  **绝对外参**表述下才有"只含 t"的行（文献[19]用它做线性初值求解）；SteerPose 的相对两视角
  场景下共线项已足以让 σ_min 在 R 正确时塌到 0（实测对比度 ~700×），故本复现只用共线项。
  若要严格对齐，需按 `reference/calib_from_moving_person/calib_linear.py` 的
  `collinearity_w2c` / `coplanarity_w2c` 一并构造；
- 训练数据合成用**正交投影**（论文附录 D.1），而 Lgeom 是**透视**共线约束（文献[19]）——
  这是论文本身就存在的设定混用，本复现保持一致，但意味着若场景有强透视（大 FOV / 近距目标），
  几何项与网络学到的成像模型会失配；
- 多视角整合（motion averaging / cycle-consistent matching / BA）未实现；
- **相机集合：默认用完整球面，而非论文的上半球**（*有意偏离*）。论文附录 D.1 原文是
  "we placed 100 cameras uniformly over the unit hemisphere"，本复现实现为
  `train.py --cameras sphere|hemisphere`，默认 `sphere`。
  理由：网络输入 P 的 **2D 形状手性**取决于相机相对姿态规范系的朝向，半球训练只覆盖一侧；
  真实相机的朝向是任意的，落在另一侧时模型会**静默**失效（匹配 0/N → Sinkhorn 近均匀 →
  可靠匹配不足 3 对 → `Lgeom` 恒为 0，而训练曲线完全正常）。
  要严格对齐论文请显式 `--cameras hemisphere`（此时 `calibrate --demo` 仍可用，因为演示
  场景的相机 A 已改为"训练同源"取法）。A/B 脚本：`scripts/diag_sphere_fix.py`；
- 官方实现用的 Sinkhorn 细节（迭代数、温度 τ、dummy 处理）论文未给，本复现取 τ=0.05、50 次迭代；
- σ 的梯度用 PyTorch 自动微分（论文用闭式导数）。

## 9. 实现修正记录

### 9.1 推断链三个隐蔽 bug（2026-09-16）

一次代码审查中发现三个隐蔽 bug，**训练曲线全看不出来**（训练损失 `Lkp` 不经过匹配与几何路径）。
发现手段是补上 `scripts/selftest_infer.py`（合成真值场景、秒级）。

| # | 位置 | 问题 | 后果 | 修正 |
|---|------|------|------|------|
| 1 | `losses.pose_distance` | `torch.cdist(q, p).mean(-1)`：cdist 把 `(B,J,2)` 当**批量矩阵**沿 J 维广播，返回 `(B,J)` | 匹配矩阵 `W` 形状与语义全错（Sinkhorn 归一化维度也错了），几何项随后越界崩溃 | 广播作差 + 逐关节范数 → `(B1,B2)`；加 `pose_distance_ref` 交叉验证 |
| 2 | `losses.geometric_loss` | 返回 `σ_2ndmin/σ_min`（方向反了） | 会把 R 推向**远离**真值的方向 | `s[-1]/(s[-2]+ε)` |
| 3 | `losses.build_linear_system` | 行宽 `torch.zeros(3, 3+3)` 与 `torch.cat(rows, dim=1)` 自相矛盾；`Q[a]` 是整姿态 `(J,2)` 而非单点 | Lgeom 一旦激活（置信度 > 0.3）即 `RuntimeError`；训练早期 Lgeom 恒 0，**问题被掩盖** | 重写为 `A (3N, N+3)`，行方向 `cat(dim=0)` |

附带改进：`calibrate` 现在输出**匹配置信度**诊断（平均最大指派概率、可信匹配数），
置信度 < 0.5 时明确警告"R 与 t 不可信"；`calibrate --demo` 改用透视投影的合成场景
（原先用正交投影，与 Lgeom 的模型不自洽）；`make_demo_dataset` 对非法关节数改为显式报错。

### 9.2 相机坐标系 / 分布外输入（2026-09-16 第二批）

`calibrate --demo` 的 `Lgeom` 恒为 0、匹配全错。排查（脚本见 `scripts/diag_*.py`）依次排除了
"演示数据太扁"、"正交 vs 透视不匹配"、"训练不足所以没用 R"、"批内是否共用相机对"，
最终定位到**相机 A 的绝对朝向**：相机 A=I 时 Lkp 0.6029（匹配 0/12），
换成训练相机集合里的相机后 0.1674（全对）。

关键认知（写代码时最容易想反的一条）：
- 相对位姿 `(R, t)` 对世界系的选取是**规范不变**的 → 相机 A 放哪里不影响标定问题的解；
- 但网络输入 `P` 的 2D 形状**不是**规范不变的，它取决于相机相对姿态规范系的朝向。
  所以"换个世界坐标系"和"换一台物理相机"是两件事，只有后者会改变网络输入。

修正：`make_two_view_scene` 的相机 A 默认取自训练同源相机集合（`camera_a="training"`，
`"identity"` 保留用于复现该现象）；相机 B 的位置改为按"物体在 B 前方"反解，
避免旧代码随机生成 t 时出现深度变号（`selftest_infer.py` 新增"场景物理合法性"检查项）。

### 9.2 内参口径 / 静默失效 / 最佳权重（2026-09-16 第二批）

| # | 位置 | 问题 | 后果 | 修正 |
|---|------|------|------|------|
| 4 | `calibrate` + `prepare_mammal_2d` | 内参没贯通：前者只认 `--focal`，后者算出的 `newcameramtx` 没被使用 | BamaPig3D 的 `label_images` 已去畸变，用原始 K（1625）或猜测的 0.9×长边（1728）都会改变射线方向，`Lgeom` 判别力下降、R 变差，**且不报错** | `to_ray_coords` 支持 `K`；`prepare_mammal_2d` 把 `newcameramtx` 写进 npz 的 `K`；`calibrate` 按 `--K-file` > npz `K` > `--focal` 取用并打印实际内参 |
| 5 | `losses.geometric_loss` | 匹配不足 3 对时静默返回 0 | `Lgeom 0.000` 分不清"正常工作"与"从未参与"，模型没训好时 R 实际只由 `Lmatch` 决定 | 新增 `confident_pairs()` 与 `stats` 诊断；日志打印实际使用的匹配对数并标注失效；多重启选优时优先选几何项生效的那次 |
| 6 | `train.py` | 最佳权重保存条件写在独立的 `% log_every` 分支里，且每个记录点评估两次 | `--epochs` 不是 `log_every` 整数倍时 `_best.pt` 永不生成（如 `--epochs 5`），末尾打印 `best val Lkp inf`；验证开销翻倍 | 合并为一次评估、末轮强制评估并更新 `best`、跑完兜底；结束语同时给出最后权重与最佳权重路径 |

**内参细节**（`undistortion.py` 与 `visualize_BamaPig3D.py` 推出）：原始
`K = [[1625.31,0,963.89],[0,1625.35,523.46],[0,0,1]]`，`coeff = [-0.35582,0.14595,-0.00031,-0.00004,0]`；
官方 `label_images` 是去畸变图，故标签对应
`cv2.getOptimalNewCameraMatrix(K, coeff, (1920,1080), alpha=1)` = `fx≈1340.43, fy≈1342.80,
cx≈964.89, cy≈521.54`。这与官方重投影示例
`points2d = (points3d @ R.T + T) @ K.T` 用 `distortion_info.pkl` 里的 `newcameramtx` 完全一致。

