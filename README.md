# steer_pose

SteerPose 的**独立复现**实现 —— 无需标定板的多相机外参标定与跨视角匹配。

> 论文：**SteerPose: Simultaneous Extrinsic Camera Calibration and Matching from Articulation**
> （Sang-Eun Lee, Ko Nishino, Shohei Nobuhara；BMVC 2025；arXiv:[2506.01691](https://arxiv.org/abs/2506.01691)）
>
> **关于代码来源**：论文与项目页给出的官方仓库
> [`kcvl-public/steerpose`](https://github.com/kcvl-public/steerpose) 截至 2026-09 仍是 404
> （该 GitHub 组织存在但无任何公开仓库）。本仓库是按论文正文 + 附录 D
> （Training Details and Implementation）独立实现的复现版本，**不是官方代码**。
> 论文细节整理见 [`docs/paper_notes.md`](docs/paper_notes.md)；
> 官方代码放出后建议对照校准。

## 方法一句话

拿画面里自由走动的人/动物当标定靶：SteerPose 网络在给定相对旋转 R 时把视角 A 的 2D 姿态
"心理旋转"到视角 B，再用可微 Sinkhorn 匹配与视角 B 的观测姿态配对；通过反向传播**只优化 R**，
同时得到相机相对位姿与跨视角对应关系。两个损失分别是匹配损失（式 2）与
几何一致性损失（σ₁/σ₂，保证 R 存在合法平移解）。

## 文件结构

```
steer_pose/
├── steerpose/
│   ├── model.py        SteerPose 网络：N 关节 2D + Rodrigues 旋转 → 32 维 token×（N+1）
│   │                   → 可学习位置编码 → 5 层 Transformer → 均值池化 → MLP → N 个 2D 关节
│   ├── geometry.py     视角合成（Fibonacci 半球 100 视点 × 20 滚转）、Rodrigues、正交投影、误差指标
│   ├── data.py         3D 姿态加载、2D 归一化、训练对构建、噪声 + 10%~30% 掩码增强
│   ├── losses.py       Lkp（式 1）、相似度（α=3）、Sinkhorn、Lmatch（式 2，双向）、Lgeom
│   ├── train.py        训练入口（CUDA/AMP/DataLoader/日志/断点）
│   └── calibrate.py    两视角标定+匹配的推理时优化（含最多 5 次随机重启、平移求解）
├── scripts/
│   ├── setup_env.sh      服务器一键配环境（conda/venv、镜像、CUDA 验证）
│   ├── run_h100.sh       H100 推荐配置一键训练（quick / full / agnostic 三档）
│   ├── sweep_h100.sh     小规模超参扫描（模型小，扫描比调参划算）
│   ├── make_mock_mammal.py  生成"官方格式"的假数据，没下载数据集也能跑通全链路
│   ├── check_mammal_data.py  ⭐️ 数据体检 + 投影一致性验证（3D GT 重投影 vs 2D 标注）
│   ├── selftest_infer.py   ⭐️ 推理链自检（匹配矩阵 + 几何一致性 + 内参 K 通路），秒级、不依赖训练
│   ├── diag_lkp_plateau.py   ⭐️ 训练没进展时先跑这个：平凡基线 + 过拟合 + 长训练三对照
│   ├── diag_sphere_fix.py    ⭐️ A/B：训练相机集合「上半球 vs 完整球面」（会训两个小模型，服务器上跑）
│   ├── diag_r_ablation.py    判定网络是否真的在用 R（打乱 R 看损失变化）
│   ├── diag_camera_frame.py / diag_shared_camera.py / diag_scene_factor.py
│   │                          定位"--demo 标定必崩"的相机坐标系系列诊断
│   ├── diag_data_quality.py  自编数据 vs 真实数据的 3D 结构质量（s3/s1 等）
│   ├── prepare_data.py   任意 3D 姿态 -> poses.npz（通用）
│   ├── prepare_mammal.py BamaPig3D 3D 标注 -> poses.npz（支持 txt 目录与 pure_pickle）
│   └── prepare_mammal_2d.py  BamaPig3D 2D 标注 + 真值外参 -> 两视角标定输入（含 newcameramtx）
├── docs/
│   ├── paper_notes.md    论文关键细节、超参、结果、数据集、与官方实现的差异
│   └── h100_tuning.md    ⭐️ H100 参数调整文档（每个参数为什么这么调 + 排障表）
└── requirements.txt
```

## 环境要求

| 项 | 要求 |
|----|------|
| 系统 | Ubuntu 20.04 / 22.04（H100 服务器） |
| Python | 3.10 ~ 3.12 |
| GPU | NVIDIA H100（sm_90）—— 需 PyTorch ≥ 2.1，推荐 **2.4~2.6 + cu124** |
| 显存 | 本模型仅约 100K 参数，**2 GB 显存都够**；H100 属于严重过剩，好处是能一次把 batch / 视角对数量开大 |
| 依赖 | `torch`、`numpy`、`scipy`（见 `requirements.txt`） |

一键配置：

```bash
bash scripts/setup_env.sh            # conda 建环境 steerpose（默认走清华镜像）
bash scripts/setup_env.sh --venv     # 或使用 venv
bash scripts/setup_env.sh --cpu      # 无 GPU 机器（只跑演示流程）
```

手动配置（推荐先装 CUDA 版 torch，再装其余）：

```bash
conda create -n steerpose python=3.10 -y && conda activate steerpose
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 数据准备

训练需要**目标类目的 3D 姿态数据**，统一整理为 `poses.npz`（内含 `poses3d: (M, J, 3)`）。

### 0. 只想先跑通：只下载一个包

`anl13/MAMMAL_datasets` 里有 4 个下载项，**只需要 `BamaPig3D_pure_pickle`（481 MB）**。
下表是**实测解包核对**过的，不是照抄仓库 README：

| 下载项 | 实测大小 / 内容 | 要不要 |
|--------|----------------|--------|
| `BamaPig2D` | 8.02 GB，全帧原始图像 | ❌ 训练与标定都不需要原始图 |
| `BamaPig2D_sleap` | 32.9 MB，**只有 3 个 COCO/SLEAP 格式 JSON**（train / eval / full），19 关节 2D 关键点；无 3D、无外参、无内参，文件名里也没有相机号 | ❌ **不能用于本流水线**（见下方"Why not"） |
| `BamaPig3D` | 8.86 GB zip，标注与 pure_pickle **完全相同**，多出来的是全量图像 | ⚠️ 冗余，除非要未去畸变的原始图 |
| **`BamaPig3D_pure_pickle`** | **481 MB** | ✅ **就下这个** |

下载地址（与 [anl13/MAMMAL_datasets](https://github.com/anl13/MAMMAL_datasets) README 一致）：

- 百度网盘：`https://pan.baidu.com/s/1ZrAdLHwDDm1ZWqUpz94P7Q`，提取码 `jams`
- Google Drive：[17-jiZh4D8cYUNkzsZMSfjLL_5gyr-3i_](https://drive.google.com/file/d/17-jiZh4D8cYUNkzsZMSfjLL_5gyr-3i_/view?usp=sharing)

解压后目录里这几样就是全链路要用的全部内容（实测大小）：

```
BamaPig3D_pure_pickle/
├── label_mix.pkl                (70, 4, 23, 3)  151 KB  ★ 论文实验用，19 个关节全部有效
├── label_3d.pkl                 (70, 4, 23, 3)  151 KB  原始标注，60% 非零（缺关节）
├── label_mesh.pkl               (70, 4, 23, 3)  151 KB  与 label_mix 同（mesh 补齐版）
├── label_pose_params.pkl        SMAL 姿态参数（global/joint rotations 等）
├── label_keypoints2d.pkl        (70, 10, 4, 19, 3) 1.2 MB  [帧,相机,个体,关节,(x,y,valid)]
├── label_silhouettes2d.pkl      轮廓标注
├── extrinsic_camera_params/     00,01,02,05,06,07,08,09,10,11.txt（正好 10 台）
│                                每文件 6 个数：axis-angle 旋转(3) + 平移(3)，单位米
├── intrinsic_camera_params/
│   ├── distortion_info.pkl      32 MB，含 K / coeff / newcameramtx / mapx,mapy,inv_map*
│   └── inverse_map_dict.pkl     157 MB，去畸变查找表
└── label_images/cam{0,1,2,5,6,7,8,9,10,11}/   共 710 张（10 相机 × 70 帧），383 MB
```

> ⚠️ `label_3d.pkl` 有 40% 的关节以 0 填充（实测非零率 60.2%，每姿态中位只有 14/19 个有效关节）；
> `label_mix.pkl` 用 mesh 补齐后**每个姿态都是 19/19 有效**。论文实验用的是 `label_mix`，
> 所以默认用它。

**Why not `BamaPig2D_sleap`**（如果你想省带宽所以想只用它）：
它里面只有 2D 关键点，SteerPose 的三件事它一件都做不了 ——
① 训练需要 **3D 姿态**；② 合成标定实验需要 **3D GT + 真值外参**来算 ER/Et；
③ 真实标定实验需要 **相机内参**把像素换算成视线方向。
它连"哪张图是哪台相机、哪一帧"都不写在文件里（实测：图片编号 0~3339，按 idx 取模分组的
平均成像尺寸没有任何系统性差异，也没有相邻帧的自相关），所以连当"2D 估计器输出"用都缺映射。
唯一可能的用途是**以后**做论文 Table 7 的"真实标定（HRNet 那一路）"时当 2D 输入，
但那还需要另外下载 `BamaPig2D` 的图并自己恢复相机/帧编号 —— 与"先跑通"无关。

### 1. 跑通顺序（先体检，再训练）

**第一步必须做数据体检**——它把 3D GT 用官方外参+内参投影回图像，与官方 2D 标注比对。
重投影误差小（几个像素）就一次性证明了骨架顺序、外参约定、内参口径、2D 坐标系
四件事全都对；任何一件错了都不会报错，只会让后面标定的结果失去意义。

```bash
# ⓪ 数据体检（秒级，不需要训练、不需要权重；先跑通就靠它）
python scripts/check_mammal_data.py --root /data/BamaPig3D_pure_pickle
#   加 --save-json out/report.json 存报告，加 --plot out/reproj.png 出验证图
#   预期输出：重投影误差中位 < 5 px，判定"优秀"

# ① 3D 标注 -> 训练数据（论文 Table 7：Bama Pig 训练用帧 0-1400）
python scripts/prepare_mammal.py --src /data/BamaPig3D_pure_pickle/label_mix.pkl \
    --max-frame 1400 --out data/poses_bamapig_train.npz

# ② 训练（先在服务器上跑 quick 档位确认链路）
bash scripts/run_h100.sh data/poses_bamapig_train.npz quick

# ③ 两视角标定数据（论文 Table 7：帧 1400-1750，相机 0 <-> 6）
python scripts/prepare_mammal_2d.py --root /data/BamaPig3D_pure_pickle \
    --cam-a 0 --cam-b 6 --min-frame 1400 --out data/two_view_pig_c0_c6.npz

# ④ 标定（R 与 t 的旋转/平移误差会直接打出来）
python -m steerpose.calibrate --ckpt ckpt/bamapig_best.pt \
    --poses-npz data/two_view_pig_c0_c6.npz --out out/two_view_pig_c0_c6.npz
```

> **内参口径（最容易踩的坑）**：`label_images` 是**已经去畸变**的图，所以 2D 标注对应的
> 相机矩阵是 `newcameramtx`（fx≈1340，由 `getOptimalNewCameraMatrix(K, coeff, (1920,1080), alpha=1)`
> 得到），**不是** `undistortion.py` 里那个带畸变的原始 K（fx≈1625），更不是 0.9×图像尺寸（≈1728）。
> `prepare_mammal_2d.py` 会自动从 `intrinsic_camera_params/distortion_info.pkl` 读出来写进 npz 的
> `K`，`steerpose.calibrate` 会优先用它。焦距填错不会报错，只是 Lgeom 的判别力下降、R 的解变差。

**A. 没有数据也能先跑通**（程序化四足骨架演示数据）：

```bash
python scripts/prepare_data.py --demo --out data/poses_demo.npz
```

**B. 用公开数据集**（推荐，与论文一致）—— 论文用的就是这 6 个：

| 数据集 | 物种 | 下载地址 | 规模 / 关键点 |
|--------|------|----------|----------------|
| **MAMMAL** (BamaPig) | 巴马猪 | [anl13/MAMMAL_datasets](https://github.com/anl13/MAMMAL_datasets) | BamaPig3D：10 视角 × 1750 帧@25fps，70 帧标注 3D；精简版 `BamaPig3D_pure_pickle` 481MB（Baidu 提取码 `jams`） |
| **MAMMAL** (Beagle Dog) | 比格犬 | [anl13/Beagle_dog_dataset](https://github.com/anl13/Beagle_dog_dataset) | 10 视角，113 帧带 3D 标注，29 关节（Google Drive / Baidu 提取码 `13tw`） |
| **Animal3D** | 40 种四足哺乳动物 | [xujiacong.github.io/Animal3D](https://xujiacong.github.io/Animal3D/) | 3379 张图，26 关节 + SMAL 姿态形状参数（⚠️ 是 SMAL 参数不是关节坐标，见下） |
| **AcinoSet** | 猎豹 | [African-Robotics-Unit/AcinoSet](https://github.com/African-Robotics-Unit/AcinoSet) | 6 相机，119,490 帧，7,588 帧人工标注，含 3D GT 与标定文件 |
| **3D-PoP** | 鸽子 | [alexhang212/Dataset-3DPOP](https://github.com/alexhang212/Dataset-3DPOP)（数据在 [这里](https://tinyurl.com/4ckbjcpx)） | 4 相机 4K，~30 万帧，1/2/5/10 只，2D + 3D 关键点 |
| **CMU Panoptic** | 人类 | [domedb.perception.cs.cmu.edu](http://domedb.perception.cs.cmu.edu/)（需注册） | 用 `getData.sh` 下载，toddler 序列 `160906_ian1` |
| **EgoHumans** | 人类（排球） | [rawalkhirodkar/egohumans](https://github.com/rawalkhirodkar/egohumans) | 多视角第一/第三人称 |

**论文实际用的序列与帧区间**（论文附录 Table 7，想严格对齐就照这个切）：

| 目标 | 训练集 | 标定-合成 | 标定-真实（2D 估计器） |
|------|--------|-----------|------------------------|
| Cheetah | `Jules flick1 (20190309)` | `Romeo flick (20190227)` | 同左 |
| Bama Pig | 帧 0–1400 | 帧 1400–1750 | 帧 0–1750（HRNet） |
| Beagle Dog | 帧 0–80 | 帧 80–112 | 帧 0–112（SuperAnimal） |
| Pigeon | `Sequence8_n01_01072022` | `Sequence5_n05_01072022` | 同左 |
| Toddler | `171204_pose1–6` | `170915_toddler5` | `170915_toddler5`（RTMO） |
| Volleyball | `171204_pose1–6` | `001_volleyball` | `001_volleyball`（RTMO） |

整理格式（通用）：

```bash
python scripts/prepare_data.py --input raw/mammal_pig.npz --out data/poses_quadruped.npz
# 指定键名 / 挑选重排关节
python scripts/prepare_data.py --input raw/animal3d.npz --key joints3d \
    --joints 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --out data/poses_animal3d.npz
```

**MAMMAL 专用转换**（BamaPig3D 的 `label_3d/pig_{i}_frame_{k:06d}.txt` 是 23×3 矩阵，
第 18/20/22/23 行恒为 0 → 官方有效索引 `[0..16, 18, 20]`，即 **19 个有效关节**；
遮挡未标注的关节以 0 填充）：

```bash
# A. 原始 txt 目录（推荐用 label_mix，它用 label_mesh 补齐了缺失关节）
python scripts/prepare_mammal.py --src /data/BamaPig3D/label_mix --dry-run
python scripts/prepare_mammal.py --src /data/BamaPig3D/label_mix \
    --max-frame 1400 --out data/poses_bamapig_train.npz     # 论文训练划分：帧 <= 1400

# B. 精简版 pkl（481MB，最省事）
python scripts/prepare_mammal.py --src /data/BamaPig3D_pure_pickle/label_mix.pkl \
    --max-frame 1400 --out data/poses_bamapig_train.npz
```

**两视角标定实验的数据准备**（用官方 2D 标注 + 真值外参，直接能算旋转误差）：

```bash
python scripts/prepare_mammal_2d.py \
    --kp2d /data/BamaPig3D_pure_pickle/label_keypoints2d.pkl \
    --extrinsics /data/BamaPig3D/extrinsic_camera_params \
    --cam-a 0 --cam-b 6 --min-frame 1400 \           # 论文标定划分：帧 1400-1750
    --out data/two_view_pig_c0_c6.npz
```
输出 `P / P2 / R(真值) / t(真值)`，配合 `steerpose.calibrate` 即可算出论文的 ER/Et。
加 `--noise-px 3` 可模拟论文"GT 2D + 高斯噪声 σ=3px"的设置。

**没有数据集也能先跑通全链路**（生成官方格式的假数据，2D 标注由 3D GT 透视投影生成，
所以体检时重投影误差应接近 0，可用来确认脚本本身没问题）：

```bash
python scripts/make_mock_mammal.py --out data/mock_bamapig
python scripts/check_mammal_data.py --root data/mock_bamapig          # 先体检，应判"优秀"
python scripts/prepare_mammal.py --src data/mock_bamapig/label_mix.pkl \
    --max-frame 1400 --out data/poses_mock.npz
python -m steerpose.train --poses3d data/poses_mock.npz --epochs 30 --batch 256 \
    --workers 0 --device cpu --out ckpt/mock.pt
python scripts/prepare_mammal_2d.py --root data/mock_bamapig \
    --cam-a 0 --cam-b 6 --min-frame 1400 --out data/two_view_mock.npz
python -m steerpose.calibrate --ckpt ckpt/mock_best.pt --poses-npz data/two_view_mock.npz
```
⚠️ 该假数据骨架为程序化生成，**只用于验证代码链路，不能用于任何精度结论**。

⚠️ **Animal3D 的坑**：它给的是 SMAL 模型的姿态/形状参数而非 3D 关节坐标，
要用于 SteerPose 训练需先用 SMAL 模型回归出关节位置（论文也是这么做的）。
嫌麻烦的话，用 BamaPig3D + Beagle Dog（直接就是 3D 关节）训四足模型最快。

脚本会自动：展平多余维度、剔除含 NaN/Inf 的姿态、逐姿态居中、打印数据规模与坐标范围。



## 训练

> ⚠️ **先读这一段，否则很容易误判"数据/实现有问题"**：
> 决定成败的是**总优化步数** = `ceil(训练对数 / batch) × epochs`，**不是 epoch 数**。
> 模型只有 ~69K 参数，epoch 极其便宜；但 `--batch` 一开大，每轮就只剩几步。
> 本仓库实测（同一份数据、同一个实现）：
>
> | 总步数 | Lkp | 打乱 R 后的 ΔLkp | 模型状态 |
> |--------|-----|------------------|----------|
> | 250 | 0.6356 | +0.0195 | **完全忽略 R**，只是在输出"平均姿态" |
> | 2000 | 0.1325 | — | 开始真正使用 R |
> | 16500 | **0.0506** | +0.7741 | 正常 |
>
> 作为参照，"无论输入什么都输出平均姿态"的平凡基线是 **0.6481**。
> 训练日志现在会主动打印总步数与这个基线；若 val Lkp 与基线相近会显式警告。
> 复现这套对照实验：`python scripts/diag_lkp_plateau.py`（或 `scripts/diag_r_ablation.py`）。

> ⚠️ **还有一个不体现在损失曲线上的坑：网络输入 P 的手性取决于相机 A 的绝对朝向。**
> 论文附录 D.1 把相机放在**单位半球面**上（原文："100 cameras uniformly over the unit
> hemisphere"），网络只见过"从一侧看"的 2D 形状。一旦输入来自另一侧（正交投影下是镜像
> 形状），网络输出全错——症状全在推理端：匹配 0/12 → Sinkhorn 近均匀 → 可靠匹配不足 3 对
> → **`Lgeom` 恒为 0**。实测同一权重 Lkp 0.23 → 0.60。
>
> 因此两点约定：
> 1. `make_two_view_scene` 默认 `camera_a="training"`，**相机 A 必须取自训练同源相机集合**
>    （`camera_a="identity"` 保留，专门用来复现这个现象，别当默认值用）；
> 2. 训练侧默认用 `--cameras sphere`（完整球面）而不是论文的半球，这样**任意**相机朝向
>    都能工作——真实相机的朝向本来就是任意的，而半球训练下的失效是**静默**的。
>    要严格对齐论文：`--cameras hemisphere`。
>
> 复核这个 A/B（**在服务器上跑**，本机不要跑模型）：
>
> ```bash
> python scripts/diag_sphere_fix.py   # 同一数据/超参/种子，只改球面覆盖，三种口径打分
> ```

```bash
# 真实数据：先给足步数，batch 不要一上来就开到几千
python -m steerpose.train --poses3d data/poses_quadruped.npz \
    --epochs 3000 --batch 256 --workers 8 --device cuda --amp \
    --num-pairs 3000 --out ckpt/steerpose_quad.pt

# 演示数据快速跑通（同样要够步数）
python -m steerpose.train --demo --epochs 1500 --batch 256 \
    --num-pairs 3000 --out ckpt/demo.pt
```

**相机集合**（`--cameras`，默认 `sphere`）：合成训练对时在完整单位球面上取 100 个视点
× 20 个滚转 = 2000 台相机。`--cameras hemisphere` 是旧的"只取上半球"行为，
**保留仅用于 A/B 对照**——它会让训练集缺失一半朝向（镜像视角），推理时静默失效。
换真实物种/数据集时不要动这个默认值；真实相机的朝向本来就是任意的。

**H100 上直接用推荐配置（三档：quick / full / agnostic）** —— 脚本按"目标总步数"自动折算 epoch：

```bash
bash scripts/run_h100.sh data/poses_bamapig_train.npz quick   # 先跑通，约 1.2 万步
bash scripts/run_h100.sh data/poses_bamapig_train.npz         # 正式训练，约 4 万步
bash scripts/sweep_h100.sh data/poses_bamapig_train.npz       # 超参扫描（推荐）
```
> ⭐️ **参数怎么调、为什么这么调、遇到问题怎么办，全部写在
> [`docs/h100_tuning.md`](docs/h100_tuning.md)** —— 上服务器前请先读它。
> 一句话结论：模型只有 ~69K 参数，H100 是过剩的；真正该调大的是
> `--num-pairs`（3000 → 20000+）和 `--batch`（512 → 4096），瓶颈在 DataLoader 而不在 GPU。

要点（对应论文附录 D.1/D.3）：

- 训练对由 3D 姿态 + 随机视角对**在线合成**，默认 3000 对，按 **7:2:1** 划分；
- 输入加 2D 噪声，随机掩码 **10%~30%** 关节；
- 损失 `Lkp` = 预测与目标 2D 姿态的平均逐关节 L2 距离；
- 输出：`ckpt/*.pt`（最后一轮，含 `model`/`opt`/`epoch`/`num_joints`）、
  `*_best.pt`（**验证损失最低**，标定时请用这个）、`*_epN.pt`（每 `--save-every` 轮）、
  `*_train.log`（文本日志，改进的轮次标 `*best`）、`*_config.json`（超参留档）；
- 断点续训：`--resume ckpt/steerpose_quad_best.pt`。

> 若要**复现论文的 class-agnostic 模型**：用 Animal3D（40 种四足）导出的 `poses.npz` 训练一次即可，
> 之后可直接用于没见过的四足物种（论文 Table 1/4/6）。

## 推理链自检（上服务器前先跑，5 秒）

推断阶段（`calibrate`）的正确性几乎全在"匹配矩阵"和"几何一致性项"这两处，而**训练损失
`Lkp` 不经过这两条路径**——所以 `Lkp` 下降完全不代表它们是对的。本仓库就曾因此藏过三个
隐蔽 bug（见文末"实现修正记录"），因此把自检固化下来：

```bash
python scripts/selftest_infer.py              # 17 项检查，CPU 秒级
python scripts/selftest_infer.py --reduce joints --noise 0.002
```

它构造一个已知真值 R、t 的合成两视角场景（透视投影），检查：

| 组 | 检查内容 |
|----|----------|
| **M1 匹配** | `pose_distance` 形状为 `(B1,B2)` 且与朴素参考实现一致；`W` 形状/行列归一化；加大 `n_iter` 行和误差下降；同视角自匹配对角线命中率 = 1.0 |
| **M2 几何** | `Lgeom(R_gt)` 显著小于 `Lgeom(R_wrong)`（实测对比度 ~700×）；用 R_gt 解出的 t 方向误差 < 5°；错误 R 下 t 误差明显更大；梯度无 NaN；真值附近起步不发散 |
| **M3 内参 K** | 像素坐标 →`to_ray_coords(K=...)`→ 归一化坐标往返闭合（<1e-9）；只用 K 做几何时 t 方向误差仍 <5°、`Lgeom(R_gt)` 仍 <1e-3；`geometric_loss` 能报告匹配对数与是否生效 |

**任何改动 `losses.py` / `calibrate.py` 后都应重跑这个自检。**

## 标定（推理时联合优化）

```bash
# 自测：程序化场景 + 真值对比（输出旋转误差 ER）
python -m steerpose.calibrate --ckpt ckpt/demo_best.pt --demo

# 真实两视角：npz 内含 'P' (B1,J,2)、'P2' (B2,J,2)，可选 'R'/'t' 作真值、'K' 作内参
python -m steerpose.calibrate --ckpt ckpt/steerpose_quad.pt \
    --poses-npz data/two_view_poses.npz --out out/two_view.npz
```

**内参优先级**：`--K-file <file>` > npz 内的 `K` > `--focal`/自动估计。
`prepare_mammal_2d.py` 产出的 npz 自带 `K`（= `newcameramtx`），因此 BamaPig3D 的实验
**不需要**手动指定 `--focal`，直接跑即可（日志会打印 `几何约束使用内参 K：fx=...`）。

> **`Lgeom` 静默失效的坑**：几何项只会用"行最大置信度 > 阈值（自适应，至少 0.3）"的匹配，
> 不足 3 对时**直接返回 0**（不报错）。模型没训练好时 Sinkhorn 接近均匀指派，`Lgeom`
> 就恒为 0，而此时 R 完全由 `Lmatch` 决定，结果不可信。现在日志会显式打印
> `几何项 Lgeom：使用 N 对匹配`，不足 3 对时标注 `<-- 不足 3 对，Lgeom 静默失效`；
> 多次重启选最优时也会优先选几何项真正生效的那次。
>
> ⚠️ 所以看到 `Lgeom 0.000` **不要**先去调几何项——按这个顺序排查：
> ① 训练相机集合是 `sphere`（完整球面）吗？② `--demo` 时权重训够了吗（val Lkp 远低于基线）？
> ③ `scripts/selftest_infer.py` 通过吗？三个都正常，`Lgeom` 自然会参与。

⚠️ **焦距（`--focal`）**：几何一致性项把 2D 坐标当作**归一化图像坐标** `(u, v)`，
视线方向是 `(u, v, 1)`。所以坐标必须是"按焦距归一化"的，不能用按包围盒缩放的坐标——
缩放会改变每条射线的方向，正确的 R 就不再对应零奇异值。

- 输入是**像素坐标**时：优先给内参 `K`；否则传 `--focal <像素焦距>`（不知道就估计 `f ≈ 0.9 × 图像长边`）；
- 不传则按数据量级自动估（像素量级自动用 `0.9×尺寸`，已归一化的坐标直接用 `1.0`）；
- 主点默认取所有点的均值（无内参时的代理）；有内参时请用 `K`，别用 `--focal`。

**如何准备两视角 2D 姿态**：用现成 2D 姿态估计器（DeepLabCut / RTMPose / animal-ap10 等）
分别在两个视角上检测同一时刻的 2D 姿态，按同一骨架顺序堆成 `(B, J, 2)` 数组。
两个视角数量不同（遮挡导致漏检）也可以——论文用 dummy 目标处理。

复现的默认超参（论文附录 D.3）：Adam lr=0.01，1000 次迭代，lr 线性衰减 1.0→0.01，
`Lgeom` 在 40% 迭代后激活，α=3，失败最多 5 次随机重启。

标定结束会打印**匹配置信度**诊断。若平均置信度 < 0.5，说明 Sinkhorn 接近均匀指派、
R 与 t 都不可信（通常是 SteerPose 训练不足），脚本会给出明确提示。
```

也支持代码内调用：

```python
from steerpose import SteerPose, calibrate_with_retries
res = calibrate_with_retries(P, P2, model, iters=1000, lam=1.0)
res["R"]         # 相对旋转 (3,3)
res["t"]         # 相对平移方向（尺度不可观测）
res["matches"]   # 跨视角匹配结果
res["W"]         # Sinkhorn 软指派矩阵
```

**如何准备两视角 2D 姿态**：用现成 2D 姿态估计器（DeepLabCut / RTMPose / animal-ap10 等）
分别在两个视角上检测同一时刻的 2D 姿态，按同一骨架顺序堆成 `(B, J, 2)` 数组。
两个视角数量不同（遮挡导致漏检）也可以——论文用 dummy 目标处理。

复现的默认超参（论文附录 D.3）：Adam lr=0.01，1000 次迭代，lr 线性衰减 1.0→0.01，
`Lgeom` 在 40% 迭代后激活，α=3，失败最多 5 次随机重启。

## 实现修正记录

### 第一批（2026-09-16）：推断链三个隐蔽 bug

训练曲线看不出这三个问题，因为**训练损失 `Lkp` 完全不经过匹配与几何这两条路径**。
发现方式是为 `calibrate` 写了几何自检（现为 `scripts/selftest_infer.py`）。

| # | 位置 | 问题 | 后果 | 修正 |
|---|------|------|------|------|
| 1 | `losses.pose_distance` | 用 `torch.cdist(q, p).mean(-1)` 处理 `(B,J,2)`。cdist 把 `(B,J,2)` 当**批量矩阵**沿 J 维广播，返回 `(B,J)` 而非 `(B1,B2)` | 匹配矩阵维度与语义**全错**（Sinkhorn 归一化作用错维度），几何项随后越界崩溃；而 `Lkp` 下降看起来完全正常 | 改为广播作差 + 逐关节求范数：`(q[:,None]-p[None]).norm(-1).mean(-1)` → `(B1,B2)`；并加 `pose_distance_ref` 朴素参考实现做交叉验证 |
| 2 | `losses.geometric_loss` | 返回 `σ_2ndmin/σ_min`（论文要求 `σ_min/σ_2ndmin`，且 `torch.linalg.svdvals` 返回**降序**） | 损失方向反了，会把 R 往**远离**真值的方向推 | 改为 `s[-1]/(s[-2]+ε)`，并加"仅用 Lgeom 优化应能收敛"的验证 |
| 3 | `losses.build_linear_system` | 行的列宽 `torch.zeros(3, 3+3)` 与 `torch.cat(rows, dim=1)` 自相矛盾，且 `Q[a]` 是整姿态 `(J,2)` 而非单点 | Lgeom 一旦真正激活（匹配置信度 > 0.3）就抛 `RuntimeError`；训练早期置信度低、Lgeom 恒为 0，**问题被掩盖** | 重写为 `A (3N, N+3)`：每个匹配姿态对取质心作一个 3D 点（论文 "N corresponding 2D pose pairs" 口径），未知量为深度 `x_i` 与 `t`；`torch.cat(..., dim=0)` |

同时修正的建模问题：几何约束原先用"按包围盒缩放"的归一化坐标，缩放会改变射线方向，
正确的 R 不再对应零奇异值 —— 现改用按**焦距**归一化的图像坐标（见上文 `--focal` 说明）。

### 第三批（2026-09-16）：训练相机集合只覆盖一半旋转群 → `--demo` 标定必崩

**症状**（用户报的"损失看着就不对"）：`calibrate --demo` 的 `Lgeom` **一直是 0.000**，
`Lmatch` 停在 ~17，12 个目标的匹配全错、`ER` 83°、匹配置信度 0.177。

**排查路径**（每一步都排除一个假设，脚本都在 `scripts/diag_*.py`）：

| # | 假设 | 实测 | 结论 |
|---|------|------|------|
| 1 | 演示数据太"平"（3D 结构不足） | 演示数据 s3/s1 中位 **0.351** vs 真实 BamaPig3D 0.225 | ❌ 演示数据反而更"立体" |
| 2 | 正交投影 vs 透视投影不匹配 | 两种投影下 Lkp 都 ≈ 0.68 | ❌ 不是主因 |
| 3 | 训练不足导致 R 没被用起来 | 10 万步权重下 R 消融 ΔLkp = +0.77（确实在用 R） | ❌ 不是主因 |
| 4 | 批内是否共用相机对 | 每样本随机对 0.230 / 共用一对 0.167 / 固定 A 换 B 0.207 | ❌ 都不是 |
| 5 | **相机 A 的绝对朝向** | 相机 A=I → **0.6029（匹配 0/12）**；相机 A 取自训练集合 → 0.1674（全对） | ✅ **根因** |

**根因**：`make_two_view_scene`（`calibrate --demo` 用的场景，是**我们自己写的**，
不属于论文实现）把相机 A 放在 `np.eye(3)`（沿世界 +z 看）。对**标定问题本身**这没问题——
相机 A 的朝向只是世界系的规范选择，整体旋转同时作用于姿态与两台相机时 P、P′、(R, t) 全都不变。
但 **P 是网络的输入**，它的 2D 形状取决于相机 A 相对"姿态规范系"的朝向；而论文附录 D.1 的相机是
**放在单位半球面上**的（原文："100 cameras uniformly over the unit hemisphere"），即网络只见过
"从一侧看"的 2D 形状，`np.eye(3)` 恰好是另一侧的**镜像**视角 → 分布外输入 → 输出全错。
`Lgeom` 之所以恒为 0，是因为它只接受"行最大置信度 > 阈值"的匹配，匹配全崩 → 可靠匹配不足 3 对
→ **静默返回 0**（不报错）。所以"Lgeom 一直是 0"不是几何项坏了，而是匹配没起来的下游现象。

| # | 位置 | 问题 | 后果 | 修正 |
|---|------|------|------|------|
| 7 | `geometry.make_two_view_scene` | 相机 A 固定为 `np.eye(3)`，落在训练相机集合的**另一侧**（分布外输入） | 训练/验证损失完全正常，但 `calibrate --demo` 必崩：Lkp 0.23→0.60、匹配 **0/12**、Sinkhorn 近均匀指派、可靠匹配不足 3 对 → **Lgeom 恒为 0**。症状全在推理端，离病因很远 | 相机 A 改为默认取自**训练同源相机**（`camera_a="training"`，用上半球那批——它是 sphere/hemisphere 两种训练配置的公共区域，对两者都在分布内）；`camera_a="identity"` 保留，专门用于复现"分布外输入"这一现象。另外相机 B 的位置改为按"物体在 B 前方"反解（旧代码随机生成 t，可能让深度变号、场景物理上不成立） |
| 8 | `geometry.ViewSynthesizer` | 论文的相机集合只有**上半球**。真实相机相对"姿态规范系"若落在另一侧，半球训练的模型会**静默**失效（同 #7 的症状） | 同上 | 新增 `fibonacci_sphere()` 覆盖完整球面；`train.py` 暴露 `--cameras sphere\|hemisphere`，**默认 `sphere`**。这是**有意偏离论文**、换来对任意相机朝向的鲁棒性；要严格对齐论文请显式用 `--cameras hemisphere`（此时 `calibrate --demo` 也能正常工作，因为 #7 已修） |

**证据链的复现方式**（都在服务器上跑，本机不跑模型）：
```bash
python scripts/selftest_infer.py        # 几何链路（含场景物理合法性），秒级、不需要权重
python scripts/diag_sphere_fix.py       # A/B：上半球 vs 完整球面，三种测量口径
python -m steerpose.calibrate --ckpt <权重> --demo   # 看 Lgeom 是否真正生效
```
预期：修好之后 `calibrate --demo` 的日志里 `几何项 Lgeom：使用 N 对匹配` 中 N ≥ 3
（不再是"不足 3 对，Lgeom 静默失效"），匹配置信度 > 0.5，`argmax` 匹配与序号一致的比例接近 1。

### 第二批（2026-09-16）：内参口径 + 静默失效 + 最佳权重

| # | 位置 | 问题 | 后果 | 修正 |
|---|------|------|------|------|
| 4 | `calibrate` / `prepare_mammal_2d` | 内参没有贯通：`prepare_mammal_2d` 已经算出 `newcameramtx`，但 `calibrate` 只认 `--focal`，npz 里的 `K` 被忽略 | `label_images` 已去畸变，用错的焦距（原始 K 的 1625 或猜测的 1728）会改变射线方向，`Lgeom` 判别力下降、R 变差，**且不报错** | `to_ray_coords` 新增 `K` 参数；`prepare_mammal_2d` 把 `newcameramtx` 写进 npz 的 `K`；`calibrate` 按 `--K-file` > npz `K` > `--focal` 的优先级取用，并在日志打印实际使用的内参 |
| 5 | `losses.geometric_loss` | 匹配不足 3 对时静默返回 0，日志里的 `Lgeom 0.000` 无法区分"几何项正常"与"几何项从未参与" | 模型没训练好时 R 完全由 `Lmatch` 决定，却看起来像几何约束在工作 | 新增 `confident_pairs()` 与 `stats` 诊断字段；日志打印 `几何项 Lgeom：使用 N 对匹配`，不足 3 对时显式标注失效；`calibrate_with_retries` 选最优时优先选几何项真正生效的那次 |
| 6 | `train.py` | 最佳权重与日志写在两处独立的 `% log_every` 判断里：`--epochs` 不是 `log_every` 整数倍时 `_best.pt` 永远不生成；且每个记录点重复评估两次 | 例如 `--epochs 5`（默认 `log_every=10`）只得到最后一轮权重，最后打印 `best val Lkp inf` 误导用户；验证开销翻倍 | 合并为一次评估，末轮强制评估并更新 `best`，跑完兜底一次；日志标注 `*best`，结束语同时给出"最后权重"与"最佳权重"路径 |

## 已验证 / 未实现

- ✅ 已验证：训练与标定全流程可在小规模数据上端到端跑通（含真值误差评估）；
- ✅ 已验证：数据体检 `check_mammal_data.py` 在官方格式假数据上闭环（3D GT 重投影误差
  中位 0.00 px，10 台相机全通过），证明骨架顺序 / 外参约定 / 内参口径 / 2D 坐标系四项自洽；
- ✅ 已验证：推理链（匹配矩阵 + 几何一致性 + 平移求解 + 内参 K 通路）由 `scripts/selftest_infer.py`
  以合成真值场景逐项验证（17 项检查全通过，`Lgeom` 真值/随机对比度 ~700×，t 方向误差 0.02°，
  像素↔归一化坐标往返偏差 1.1e-16）；
- ⬜ 未实现：多视角整合（motion averaging + 循环一致匹配 + bundle adjustment，论文第 4 节）——
  可参考同作者前作 [RA-L 2022 代码](https://github.com/kyotovision-public/extrinsic-camera-calibration-from-a-moving-person)（MIT），
  其中的共线/共面约束、chirality check 与 BA 是 SteerPose 几何部分的技术来源；
- ⬜ 未做：与 MAMMAL / 3D-PoP / AcinoSet 的定量对比复现（需要数据集与逐类目 2D 姿态估计器）；
- ⚠️ 与官方实现的可能差异见 `docs/paper_notes.md` 第 8 节。

## 常见问题

**Q: 没有真实数据集能不能先验证代码正确性？**
可以。`--demo` 会用程序化四足骨架生成数据，跑通"训练 → 两视角标定 → 旋转误差"全链路。
注意演示数据规模小，损失不会降到论文水平（论文 Table 4 的关节误差在 0.1~0.2 量级）。

**Q: 标定结果很差 / Sinkhorn 匹配置信度几乎均匀，是不是代码有 bug？**
**先跑 `python scripts/selftest_infer.py`** —— 它 5 秒内就能告诉你实现本身有没有问题
（不依赖训练、不依赖数据）。如果自检全通过，那就是模型区分度不够：标定质量完全取决于
SteerPose 能否准确预测"同一姿态在另一视角下的样子"。正确配对的姿态距离一旦和错误配对
差不多，Sinkhorn 就会输出接近均匀的指派、`Lgeom` 也会因为筛不出可靠匹配而恒为 0。

**先看"是不是训够了"**，别急着怀疑数据：训练日志里的 val Lkp 必须明显低于同一行上方的
平凡基线（演示数据上约 0.648）。实测门槛：val Lkp ≥ 0.35 时标定基本没有意义；
降到 0.13（约 2000 步）开始能用；0.05（约 16500 步）时匹配才真正可信。
`python scripts/diag_lkp_plateau.py` 会在本地几分钟内复现这套对照，直接告诉你
"实现有没有 bug"和"要多少步"。

```bash
python scripts/selftest_infer.py                     # 先确认实现没问题
python scripts/diag_lkp_plateau.py                   # 再确认训练规模够不够
python -m steerpose.train --demo --epochs 1500 --batch 256 --out ckpt/demo.pt
python -m steerpose.calibrate --ckpt ckpt/demo_best.pt --demo --iters 1000
```

**Q: `calibrate --demo` 的 `Lgeom` 一直是 0.000，是不是几何项写错了？**
基本不是。`Lgeom` 只吃"可靠的匹配对"，匹配没起来它就静默为 0。按顺序查三件事：
① 训练时相机集合是不是完整球面（默认 `--cameras sphere`，别用 `hemisphere`）；
② 权重训够了没（`--demo` 场景下 val Lkp 要远低于平凡基线 0.648）；
③ `python scripts/selftest_infer.py` 是否 17 项全通过。这三项都正常时 `Lgeom` 必然参与，
日志会打印"使用 N 对匹配"。详细根因见文末"实现修正记录 · 第三批"。

**Q: 服务器上怎么后台跑训练？**
```bash
nohup python -m steerpose.train --poses3d data/poses_quadruped.npz \
    --epochs 3000 --batch 256 --workers 8 --device cuda \
    --out ckpt/steerpose_quad.pt > log/train.out 2>&1 &
```

**Q: 训练日志里 val Lkp 一直贴着某个数不动，是不是数据不对？**
大概率是**步数不够**，不是数据。日志里会打印"平凡基线（输出平均姿态）val Lkp"和
"共 N 步"。若 val Lkp ≈ 基线，说明模型还没开始学：把 `--epochs` 加到几千、
或把 `--batch` 调小（每轮步数会等比变多）。判据与对照实验见上文"训练"一节。

**Q: 为什么两个视角的实例数量不一样也能匹配？**
网络逐姿态工作，Sinkhorn 在集合层面做软指派；漏检会让集合变小，论文通过加 dummy 目标处理，
本实现用软指派 + 双向损失实现同样的容错效果。

**Q: 平移 t 为什么只有方向？**
单目/双目尺度不可观测，论文中 t 也是"解到尺度与符号二义"后由 chirality check 定符号。

## 引用

```bibtex
@inproceedings{lee2025steerpose,
  title     = {SteerPose: Simultaneous Extrinsic Camera Calibration and Matching from Articulation},
  author    = {Lee, Sang-Eun and Nishino, Ko and Nobuhara, Shohei},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year      = {2025},
  note      = {arXiv:2506.01691}
}
```
