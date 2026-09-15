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
│   ├── prepare_data.py   任意 3D 姿态 -> poses.npz（通用）
│   ├── prepare_mammal.py BamaPig3D 3D 标注 -> poses.npz（支持 txt 目录与 pure_pickle）
│   └── prepare_mammal_2d.py  BamaPig3D 2D 标注 + 真值外参 -> 两视角标定输入
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

**没有数据集也能先跑通全链路**（生成官方格式的假数据）：

```bash
python scripts/make_mock_mammal.py --out data/mock_bamapig
# 之后依次跑 prepare_mammal.py / steerpose.train / prepare_mammal_2d.py / steerpose.calibrate
```
⚠️ 该假数据骨架为程序化生成，**只用于验证代码链路，不能用于任何精度结论**。

⚠️ **Animal3D 的坑**：它给的是 SMAL 模型的姿态/形状参数而非 3D 关节坐标，
要用于 SteerPose 训练需先用 SMAL 模型回归出关节位置（论文也是这么做的）。
嫌麻烦的话，用 BamaPig3D + Beagle Dog（直接就是 3D 关节）训四足模型最快。

脚本会自动：展平多余维度、剔除含 NaN/Inf 的姿态、逐姿态居中、打印数据规模与坐标范围。



## 训练

```bash
# 真实数据（H100：直接开大 batch 和 worker）
python -m steerpose.train --poses3d data/poses_quadruped.npz \
    --epochs 300 --batch 512 --workers 8 --device cuda --amp \
    --num-pairs 3000 --out ckpt/steerpose_quad.pt

# 演示数据快速跑通
python -m steerpose.train --demo --epochs 50 --num-pairs 2000 --out ckpt/demo.pt
```

**H100 上直接用推荐配置（三档：quick / full / agnostic）**：

```bash
bash scripts/run_h100.sh data/poses_bamapig_train.npz quick   # 先验证链路
bash scripts/run_h100.sh data/poses_bamapig_train.npz         # 正式训练
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
- 输出：`ckpt/*.pt`（含 `model`/`opt`/`epoch`/`num_joints`）、`*_best.pt`、`*_epN.pt`、
  `*_train.log`（文本日志）、`*_config.json`（超参留档）；
- 断点续训：`--resume ckpt/steerpose_quad_best.pt`。

> 若要**复现论文的 class-agnostic 模型**：用 Animal3D（40 种四足）导出的 `poses.npz` 训练一次即可，
> 之后可直接用于没见过的四足物种（论文 Table 1/4/6）。

## 标定（推理时联合优化）

```bash
# 自测：程序化场景 + 真值对比（输出旋转误差 ER）
python -m steerpose.calibrate --ckpt ckpt/demo_best.pt --demo

# 真实两视角：npz 内含 'P' (B1,J,2)、'P2' (B2,J,2)，可选 'R' 作为真值
python -m steerpose.calibrate --ckpt ckpt/steerpose_quad.pt \
    --poses-npz data/two_view_poses.npz --out out/two_view.npz
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

## 已验证 / 未实现

- ✅ 已验证：训练与标定全流程可在小规模数据上端到端跑通（含真值误差评估）；
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
先检查 SteerPose 是否训练充分。标定质量完全取决于模型的区分度：如果模型对"同一姿态在不同
视角下的样子"预测不准，那么正确配对的姿态距离会和错误配对差不多，Sinkhorn 输出接近均匀分布、
`Lgeom` 也会因为筛不出足够可靠匹配（置信度低于阈值）而恒为 0。经验门槛：演示数据上
val Lkp 至少降到 0.35 以下，标定才开始给出有意义的旋转估计。建议先跑：
```bash
python -m steerpose.train --demo --epochs 200 --num-pairs 4000 --out ckpt/demo.pt
python -m steerpose.calibrate --ckpt ckpt/demo_best.pt --demo --iters 1000
```

**Q: 服务器上怎么后台跑训练？**
```bash
nohup python -m steerpose.train --poses3d data/poses_quadruped.npz \
    --epochs 300 --batch 512 --workers 8 --device cuda \
    --out ckpt/steerpose_quad.pt > log/train.out 2>&1 &
```

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
