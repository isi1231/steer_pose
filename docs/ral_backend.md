# RA-L2022 几何标定后端（`steerpose.calib_ral` + `steerpose.lifting`）

> 目的：把同作者前作 **`Extrinsic Camera Calibration From a Moving Person`
> (IROS/RA-L 2022)** 的标定链路移植进本仓库，补上 SteerPose 缺失的
> **2D → 单目 3D → 外参** 这一环，使其能直接用在 BamaPig3D（猪）这类
> 多物种 / 畜牧场景上。参考实现见 `reference/calib_from_moving_person/`。

---

## 1. 为什么需要它：SteerPose 与 RA-L2022 的分工

| | SteerPose（BMVC 2025） | RA-L2022 |
|---|---|---|
| 网络做什么 | **2D → 2D**（跨视角"心理旋转"匹配） | **2D → 相机系 3D**（单目提升器） |
| 标定怎么做 | 可微的 2D 重投影 / 线性系统 | **骨方向（oriented points）** + 射线共线/共面 |
| 需要的先验 | 骨架、内参 | 骨架、内参、**一个自由移动的个体** |
| 强项 | 无需 3D 标注，端到端可微 | 对单目深度误差**鲁棒**（只用方向） |

两者观测的是同一件事——**articulation 的朝向**——所以共用一套骨架定义
（`steerpose.skeleton`）。本仓库把两条路都留着：

* `steerpose.model.SteerPose` + `steerpose.calibrate`：原论文路径（2D→2D）；
* `steerpose.lifting.PoseLifter` + `steerpose.calib_ral`：RA-L2022 路径（2D→3D→外参）。

---

## 2. 三段式链路（逐段放松对单目 3D 的精度要求）

```
[0] 单目 3D 提升     每台相机各自跑 PoseLifter：归一化 2D -> 相机系 3D（根相对）
        |             ↳ scripts/train_lifter.py 训练
        v
[1] 旋转（SVD）      每根骨的**方向单位向量** v_c = oriented point。
        |            世界系里同一根骨在所有相机方向相同，只差各自的 R_c：
        |                v_c = v_world · R_c^T
        |            把所有相机的 v_c 沿列拼成 (N, 3C)，前 3 个右奇异向量即 v_world 的基，
        |            R_all = sqrt(C)·Zt[:3,:]（列块 = 各相机的 R_c）。固定 R_0 = I（gauge）。
        v
[2] 平移（线性）     共线：每相机每点 [n_c]_x (R_c X + t_c) = 0
        |            共面：相机对 (n_a × n_b) ⟂ (t_a - t_b)
        |            拼成 C·x = 0，取 C^T C 最小的 4 个特征向量
        |            （零空间 4 维 = 3 平移 + 1 尺度，正是标定固有的 gauge）
        |            t_0 = 0 消平移 gauge；|t_1 - t_0| = 1 消尺度 gauge；
        v            z-test（三角化后点落在相机前方）定手性符号
[3] BA（可选）       least_squares 联合精化，三项残差：
        |            a. 重投影（按 2D 置信度加权）
        |            b. 跨相机骨方向一致性    ← 只用提升器的"朝向"
        |            c. 三角化 3D 的骨长方差  ← 刚体假设，约束各帧
        v
     输出 (R, t)
```

**关键设计（也是原论文的核心洞察）**：第 [1][2] 段只用到 **骨方向**，
不用单目 3D 的绝对位置或尺度。单目深度不可靠，但方向可靠——于是标定精度
对提升器的深度误差几乎免疫。这是第 [3] 段 BA 里 `_objfun_var3d` 也用
"跨相机方向一致性"而不是"3D 位置一致性"的原因。

---

## 3. Gauge：什么可辨识，什么不可辨识

外参只能恢复到**相差一个世界系相似变换**的程度。任何
`(R_c, t_c) → (R_c H, s·R_c... )` 形式的整体变换都不改变任何观测。
所以：

| 量 | 可否辨识 |
|---|---|
| **相对旋转** `R_b R_a^T` | ✅ 可辨识（整体 `H` 抵消） |
| **基线方向** `t_b - t_a` 的朝向 | ✅ 可辨识 |
| **基线长度比**（如 `|t₂-t₀| / |t₁-t₀|`） | ✅ 可辨识 |
| 绝对姿态 / 绝对位置 / 绝对尺度 | ❌ 不可辨识 |

→ 因此评估**必须**用 gauge-free 指标，否则测的是 gauge 而不是精度：

* `relative_rotation_error_deg(R_est, R_gt)` —— 比较相对旋转，`H` 自动抵消；
* `translation_direction_error_deg(...)` —— 把基线表达到参考相机坐标系里；
  **相机 0 是 gauge 参考（`t_0 = 0`），它的基线方向天然无定义 → 返回 `nan`，
  统计时要排除**；
* `similarity_align`（Umeyama）—— 相似变换对齐后比较光心，用于跨数据集比较
  （`alignment_error()` 把它们打成一包）。

> 实测（`scripts/selftest_calib_ral.py`）：`alignment_error()['scale']` 不是误差，
> 而是**以 `|t₁-t₀|` 为单位恢复出的真实基线长度**。用假数据验证过：
> 2 相机（0,6）时 `scale = 7.6084`，正好等于 `√(7.236² + 2.351²)` m；
> 10 相机时 `scale = 2.4721 = 2·4·sin18°` m。两者都精确到小数点后 4 位。

---

## 4. 提升器（`steerpose.lifting.PoseLifter`）的三个有意选择

### 4.1 损失必须是**骨方向余弦**，不能是普通 MPJPE

单目 3D 的**绝对尺度不可观测**。用普通 MPJPE 当主损失时，代码不报错、loss 也下降，
但网络会把容量浪费在"猜绝对深度"上，而标定根本不需要这个量。所以：

```
主损失 = bone_direction_loss      1 - cos(pred, gt)，对尺度/平移完全免疫
辅助项 = scale_invariant_mpjpe    先各自归一到"平均骨长=1"再算逐关节 L2
```

`scale_invariant_mpjpe` 用**同一个**归一化规则处理 pred 与 gt（见其 docstring）——
否则会引入系统性偏差。

### 4.2 输入用**按内参归一化**的坐标，不是按图像尺寸归一化

```
(u, v) = ((x - cx)/fx, (y - cy)/fy)
```

好处有三：① 换相机 / 换分辨率不用重训；② 与几何后端的射线口径
`n = (u, v, 1)` 完全一致（`normalized_rays` 就是这么算的），两边不会打架；
③ 弱透视下 `(u,v)` 直接就是"朝向"的近似，网络更容易学。

### 4.3 单帧 Transformer，不做时序卷积

BamaPig3D 只有 **70 个标注帧、帧间隔 25 帧**，时序上下文本来就稀。
沿用本仓库 SteerPose 的 Transformer 骨架（单帧），逐关节回归 3D。
唯一的结构差别：SteerPose 对 token 做均值池化后回归整个 2D 姿态；
这里每个关节要单独的 3D，所以**逐 token** 回归。

---

## 5. 数据口径（踩过的坑，务必对齐）

| 项 | 正确口径 | 错了会怎样 |
|---|---|---|
| **内参** | `newcameramtx`（fx≈1340） | 用 `undistortion.py` 的原始 K（fx≈1625）→ 重投影误差从 ~0 涨到几十 px |
| 为什么 | `label_images` **已经去畸变**，标签对应的是去畸变后的相机矩阵 | — |
| **外参约定** | `x_cam = R @ X_world + t` | 反了 → 标定结果完全错误但不会报错 |
| **3D 标注顺序** | 23 → `g_all_parts = [0..16,18,20]` → 19 | 错位 → 骨方向全乱 |
| **3D 缺失关节** | 官方用 `0` 填充 | 必须靠掩码剔除（见 §6） |
| **帧号** | 第 k 个标注帧 = `k*25` | 训练/标定按帧划分会泄漏 |
| **`label_mix` vs `label_3d`** | 用 `label_mix`（用 mesh 补齐了缺失关节） | `label_3d` 约 40% 关节为 0，监督信号稀 |

自检命令（秒级，纯 numpy）：

```bash
python scripts/check_mammal_data.py --root /data/BamaPig3D_pure_pickle --plot out/reproj.png
# 期望：重投影误差中位 < 5 px，判定"优秀"
```

---

## 6. 两个必须知道的陷阱

### 6.1 无效关节被填 0 → 会产生「伪骨方向」（已在损失层修掉）

预处理把未标注关节的 3D 目标填成 `0`。于是**只要一根骨有一端缺失**，
它的"目标方向"就变成 `Y3[有效端] - 0` —— 一个**完全错误但数值很大**的向量。
如果损失不看掩码，网络会被主动推向这个虚假方向。

修复：`bone_direction_loss` / `bone_direction_error_deg` / `mpjpe` /
`procrustes_mpjpe` 都接受可选 `mask`，只在"**两端都可见**"的骨上求平均
（`_bone_valid_weight`）。

`scripts/selftest_lifting.py` 的 C 节是这件事的**回归测试**，它会同时验证：

* 伪骨方向确实非零（不是 no-op，是真错误信号）；
* `mask=None` 时梯度**确实会**流到无效关节（证明修复必要）；
* 带掩码后梯度在该关节上**精确为 0**。

### 6.2 多视角存活率：**必须用相机对**，不要一次上 10 台

`oriented_points` 与 `normalized_rays` 都要求"同一根骨在**所有**相机都可见"
（实现上是 `~isnan(nrm).any(axis=0)`）。所以存活率约为

```
存活率 ≈ (单相机可见率) ^ 相机数
```

BamaPig3D 的关节可见率约 **60%~70%**（猪会被栏杆、同伴遮挡）：

| 相机数 | 估算存活率 | 可用性 |
|---|---|---|
| **2** | 0.65² ≈ **42%** | ✅ 推荐 |
| 3 | ~27% | 勉强 |
| 10 | 0.65¹⁰ ≈ **1.3%** | ❌ 基本没有样本 |

这正是参考实现（RA-L2022）以**相机对**为单位标定的原因。
所以 `prepare_mammal_lifter.py` 提供 `--calib-cams 0,6`，
**默认用全部相机时会主动警告**，并打印实测存活率与骨观测数。

> 实测（假数据，单相机可见率 0.971）：
> `--calib-cams 0,6` → 标定骨存活 **91.5%**（717 个骨观测）；
> 全 10 台 → **54.7%**（429 个）。假数据可见率高，差距看不明显；
> 真实数据 0.65 可见率下 10 台会直接崩到 ~1%。

---

## 7. 完整跑法

```bash
# ---------- ⓪ 数据体检（纯 numpy，秒级，不需要权重）----------
python scripts/check_mammal_data.py --root /data/BamaPig3D_pure_pickle
#   期望：重投影中位 < 5 px

# ---------- ① 提升器训练数据 ----------
#   论文 Table 7：训练用帧 0-1400
python scripts/prepare_mammal_lifter.py --root /data/BamaPig3D_pure_pickle \
    --max-frame 1400 --out data/lifter_bamapig_train.npz
#   只想先看统计（不写文件）：
python scripts/prepare_mammal_lifter.py --root /data/BamaPig3D_pure_pickle \
    --max-frame 1400 --dry-run

# ---------- ② 训练（★ 在服务器 H100 上跑）----------
python scripts/train_lifter.py --npz data/lifter_bamapig_train.npz \
    --out ckpt/lifter_bamapig.pt --device cuda --workers 8 --steps 40000 --amp
#   先体检数据与平凡基线（CPU，不建模型）：
python scripts/train_lifter.py --npz data/lifter_bamapig_train.npz --dry-run

# ---------- ③ 标定输入包（★ 相机对！论文 Table 7：帧 1400-1750）----------
python scripts/prepare_mammal_lifter.py --root /data/BamaPig3D_pure_pickle \
    --min-frame 1400 --calib-cams 0,6 --no-train-out \
    --dump-calib data/lifter_calib_c0_c6.npz

# ---------- ④ 推理 + 标定 ----------
python - <<'PY'
import numpy as np, torch
from steerpose.lifting import PoseLifter, normalize_2d_intrinsics
from steerpose.calib_ral import calibrate_from_lifter, alignment_error

d = np.load("data/lifter_calib_c0_c6.npz")
p2d, w2d, K = d["p2d"], d["w2d"], d["K"]          # (C,N,J,2) 像素

ck = torch.load("ckpt/lifter_bamapig.pt", map_location="cpu", weights_only=False)
model = PoseLifter(num_joints=ck["num_joints"]).eval()
model.load_state_dict(ck["model"])

# ★ 提升器的输入必须与训练时同一口径：按内参归一化
C, N, J, _ = p2d.shape
xn = normalize_2d_intrinsics(p2d.reshape(-1, J, 2), K).reshape(C, N, J, 2)
with torch.no_grad():
    p3d = model(torch.as_tensor(xn, dtype=torch.float32),
                torch.as_tensor(w2d, dtype=torch.float32)).numpy()   # (C,N,J,3)

out = calibrate_from_lifter(p3d, p2d, K, w2d, refine=True)   # refine=是否跑 BA
print("R", out["R"].shape, "t", out["t"].shape)
if "R_gt" in d:   # 假数据/有真值时可以直接评估（gauge-free）
    print(alignment_error(out["R"], out["t"], d["R_gt"], d["t_gt"]))
PY
```

---

## 8. 自检（上服务器前先跑，全部纯 numpy/scipy）

```bash
python scripts/selftest_calib_ral.py    # 几何求解器：48 项断言
python scripts/selftest_lifting.py      # 损失/指标/数据集：53 项断言（CPU 张量，不含前向）
#   连前向一起测（会实例化一个 4 样本的小模型）：
python scripts/selftest_lifting.py --allow-model
```

假数据端到端闭环（不训练、不推理，纯几何）：

```bash
python scripts/make_mock_mammal.py --out /tmp/mock_bamapig
python scripts/prepare_mammal_lifter.py --root /tmp/mock_bamapig \
    --min-frame 1400 --calib-cams 0,6 --no-train-out --dump-calib /tmp/cal.npz
# 用**精确 GT** 当"提升器输出"喂进 calib_ral：线性解应精确恢复真值外参
```

---

## 9. 已知限制 / 未实现

* **未实现 RANSAC / 按相机对打分**：参考实现里的 `calib_ransac.py` 与
  `2d_joint_mask` 走的是"至少 K 台相机可见 + 相机对打分"，本模块暂不提供。
  在当前"必须全相机可见"的口径下，**多相机要用相机对分别标定再拼**。
* **未做时序建模**：提升器是单帧的。BamaPig3D 标注稀疏，时序收益有限；
  若换到连续标注的数据集（如 MAMMAL 的连续子集），可考虑加时序卷积。
* **手性靠 z-test 单点决定**：`z_test_sign` 只比较 `+t` / `-t` 哪边"点在相机前方"
  更多。样本很少时可能判错，建议 BA 前先打印该判定结果。
* **未做跨物种验证**：骨架靠 `steerpose.skeleton` 的 dict 抽象，换物种只需
  加一条 `SKELETONS` 记录（关节名/骨连接/根关节），但精度未测。
* **BA 很慢**：`scipy.optimize.least_squares` 每步要求全量重投影，
  相机数 × 样本大时会明显变慢。线性解通常已够用，`refine=False` 是默认。
