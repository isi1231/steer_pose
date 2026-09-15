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

## 7. 数据集

| 数据集 | 物种 | 备注 |
|--------|------|------|
| MAMMAL (Nature Comm. 2023) | 巴马猪、比格犬 | 多实例，与猪场场景最相关 |
| Animal3D | 40 种四足 | 训练 class-agnostic 模型 |
| AcinoSet | 猎豹 | 野外、单实例 |
| 3D-PoP | 鸽子 | 多实例、鸟类 |
| CMU Panoptic | 人类幼儿 | 室内多视角 |
| EgoHumans | 排球场景 | 人类 |

## 8. 本复现与论文的差异（已知）

- 论文未给 MLP 层数 / 头数 / FFN 维度 / SteerPose 训练超参（lr、epoch、batch）→ 本复现取常规默认值；
- `Lgeom` 的线性系统按论文 3.2 节文字描述实现（射线共线 + 两视图共面 → SVD 最小奇异值），
  与官方实现（可能沿用文献[19]的稀疏矩阵构造）可能存在细节差异；
  **奇异值梯度用 PyTorch 自动微分**（论文用闭式导数）；
- 多视角整合（motion averaging / cycle-consistent matching / BA）未实现；
- 官方实现用的 Sinkhorn 细节（迭代数、温度 τ、dummy 处理）论文未给，本复现取 τ=0.05、50 次迭代。
