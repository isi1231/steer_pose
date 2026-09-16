#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提升器「损失 / 指标 / 数据集」自检（纯张量运算，CPU，秒级）。

为什么单独一个自检
------------------
RA-L2022 的整条链路建立在一个**很容易悄悄搞错**的性质上：

    标定只用骨方向 -> 损失必须对"尺度、平移"完全免疫。

如果损失不小心带进了尺度或平移（比如用了普通 MPJPE 当主损失），代码不会报错、
loss 照样下降，但网络会把容量浪费在单目根本无法观测的绝对深度上，
最后标定精度莫名其妙地差。所以这里把"不变性"写成断言。

★ 本自检**不实例化任何神经网络、不做任何前向**（遵守"不在本机跑模型"的约定）。
  PoseLifter 的前向由服务器上的 train_lifter.py 覆盖；若你想连前向一起测，
  加 `--allow-model`（会在 CPU 上跑 4 个样本的一次前向）。

用法:
    python scripts/selftest_lifting.py
    python scripts/selftest_lifting.py --allow-model
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.lifting import (                                        # noqa: E402
    LifterDataset, bone_direction_error_deg, bone_direction_loss,
    bone_length_mean, bone_vectors, build_frame_split, build_split_indices,
    mpjpe, normalize_2d_intrinsics, normalize_3d, pose_scale, procrustes_mpjpe,
    root_center, scale_invariant_mpjpe,
)
from steerpose.skeleton import bones_to_index_array, get_skeleton         # noqa: E402

SK = get_skeleton("pig19")
BONES, ROOT, CALIB = SK["bones"], SK["root"], SK["calib_bones"]

_ap = argparse.ArgumentParser(description="提升器 损失/指标/数据集 自检")
_ap.add_argument("--allow-model", action="store_true",
                 help="额外跑一次 PoseLifter 前向（CPU，4 个样本；默认关闭）")
_args, _ = _ap.parse_known_args()
args_allow_model = _args.allow_model

N_PASS = N_FAIL = 0
NOTES = []


def check(cond, label, detail=""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))
    else:
        N_FAIL += 1
        print(f"  FAIL  {label}" + (f"   [{detail}]" if detail else ""))


def note(msg):
    NOTES.append(msg)
    print(f"  note  {msg}")


def rand_pose(B=4, J=19, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, J, 3, generator=g)


def rand_rot(B=1, seed=1):
    """随机旋转矩阵 (B,3,3)（float32，与 rand_pose 同 dtype）。"""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(B, 3, 3, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    return (Q * torch.where(torch.linalg.det(Q).unsqueeze(-1).unsqueeze(-1) < 0, -1.0, 1.0))


# =====================================================================
print("=" * 72)
print("A. 坐标归一化 / 骨架工具")
print("=" * 72)

K = np.array([[1200.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
P = np.array([[1560.0, 1040.0], [960.0, 540.0]])
n = normalize_2d_intrinsics(P, K)
check(np.allclose(n[0], [(1560 - 960) / 1200, (1040 - 540) / 1000]),
      "normalize_2d_intrinsics：主点映射到 (0,0)", f"{np.round(n[0], 6)}")
check(np.allclose(n[1], [0.0, 0.0]), "normalize_2d_intrinsics：主点 -> 原点")
try:
    normalize_2d_intrinsics(P, np.eye(2))
    check(False, "normalize_2d_intrinsics：非法 K 应当报错")
except ValueError:
    check(True, "normalize_2d_intrinsics：非法 K 报错")

b = bones_to_index_array(BONES)
check(b.shape == (19, 2), "bones_to_index_array -> (19,2)", str(b.shape))
check(bones_to_index_array(CALIB).shape == (14, 2), "calib_bones -> (14,2)")
# 所有骨索引必须在 [0,19)
check(b.min() >= 0 and b.max() < 19, "骨索引范围合法")
# root 必须被骨头连上（否则根相对后根关节对损失毫无贡献）
check(ROOT in b.ravel().tolist(), f"root={ROOT} 出现在骨端点里")

y = rand_pose()
check(torch.allclose(root_center(y, ROOT)[:, ROOT], torch.zeros(4, 3)),
      "root_center：根关节归零")

# pose_scale / normalize_3d 的契约是 **numpy 输入**（数据处理阶段用）
ynp = y.numpy().astype(np.float64)
s = pose_scale(ynp, BONES, ROOT)
check(np.shape(s) == (4,) and (s > 0).all(),
      "pose_scale -> (B,) 且为正", str(np.round(s, 4)))
yn_np, s2 = normalize_3d(ynp, BONES, ROOT)
check(np.allclose(yn_np[:, ROOT], 0.0), "normalize_3d：根关节归零")
check(np.allclose(s, s2), "normalize_3d：返回的 scale 与 pose_scale 一致")
# 归一化后平均骨长应为 1（这正是"骨长归一化"的定义）
bed = np.linalg.norm(yn_np[:, b[:, 0], :] - yn_np[:, b[:, 1], :], axis=-1)
check(np.allclose(bed.mean(axis=-1), 1.0, atol=1e-5),
      "normalize_3d 之后平均骨长 = 1", str(np.round(bed.mean(axis=-1), 6)))

# 张量版（可微）：根相对 + 加权平均骨长
yt = y.clone().requires_grad_(True)
bl = bone_length_mean(root_center(yt, ROOT), BONES)
check(tuple(bl.shape) == (4,) and bool((bl > 0).all()),
      "bone_length_mean（可微）-> (B,) 且为正", str(bl.detach().numpy().round(4)))
check(bl.requires_grad, "bone_length_mean 保持可微（能回传梯度）")

# =====================================================================
print("\n" + "=" * 72)
print("B. 骨方向损失：不变性（RA-L2022 依赖的核心性质）")
print("=" * 72)

gt = rand_pose()
pred = rand_pose(seed=21)
check(abs(float(bone_direction_loss(gt, gt, BONES))) < 1e-7,
      "完全相同 -> 损失 0")

# ---- 正确的不变性：**同时对 pred 与 gt** 施加相似变换 (R, s, t)，损失不变 ----
print("  (不变性 = 对 pred 和 gt 施加**同一个**相似变换；只变换一边是应该变的)")
for s_val in (0.3, 1.0, 7.5):
    R = rand_rot(1, seed=int(s_val * 10))[0]
    t = torch.tensor([3.0, -2.0, 5.0])
    f = lambda z: s_val * (z @ R.T) + t
    l0 = float(bone_direction_loss(pred, gt, BONES))
    l1 = float(bone_direction_loss(f(pred), f(gt), BONES))
    check(abs(l0 - l1) < 1e-5,
          f"同时施加 旋转/尺度({s_val})/平移 -> 损失不变",
          f"{l0:.6f} -> {l1:.6f}")

# 平移同时也是"自动"免疫的：骨向量是差，绝对位置根本不进入计算
l2 = float(bone_direction_loss(pred + 100.0, gt - 55.0, BONES))
check(abs(l2 - float(bone_direction_loss(pred, gt, BONES))) < 1e-6,
      "只平移（方向不受影响）-> 损失不变", f"{l2:.6f}")

# ★ 反向断言（很重要）：只旋转 gt，损失**必须**变大。
#   这正是 RA-L2022 能用骨方向反解相机旋转的原因——如果这一项也为 0，
#   说明损失对姿态朝向不敏感，整条标定链路就没有信息量了。
R = rand_rot(1, seed=99)[0]
l_rot_gt = float(bone_direction_loss(pred, gt @ R.T, BONES))
l_rot_pd = float(bone_direction_loss(pred @ R.T, gt, BONES))
check(l_rot_gt > 1e-3 and l_rot_pd > 1e-3,
      "【关键】只旋转一边 -> 损失显著变大（朝向敏感，标定才有信息量）",
      f"只转 gt {l_rot_gt:.4f} / 只转 pred {l_rot_pd:.4f}")

# 镜像（负缩放）：两边都镜像 -> 不变；只镜像一边 -> 损失变成 2.0
check(abs(float(bone_direction_loss(-pred, -gt, BONES)) -
          float(bone_direction_loss(pred, gt, BONES))) < 1e-6,
      "两边同时镜像 -> 损失不变")
# 只用一边镜像时 cos 变号：loss_mirror = 1 + cos_orig，故两者之和恒为 2
l_o = float(bone_direction_loss(pred, gt, BONES))
l_m = float(bone_direction_loss(pred, -gt, BONES))
check(abs(l_o + l_m - 2.0) < 1e-5,
      "只镜像一边 -> cos 变号（loss + loss_mirror = 2）",
      f"{l_o:.4f} + {l_m:.4f} = {l_o + l_m:.6f}")
l_id = float(bone_direction_loss(gt, -gt, BONES))
check(abs(l_id - 2.0) < 1e-6,
      "pred==gt 时只镜像一边 -> 精确 2.0（方向全反）", f"{l_id:.6f}")
note("骨方向损失**不含手性信息**（v 与 -v 的角只在同时镜像时不变）。"
     "所以标定里 (R, t) 的 chirality 必须单独定 —— 这就是 calib_ral "
     "要用 z-test（三角化后点数在相机前方）挑符号的原因。")

# mask=None 与 mask=全1 应完全等价
ones = torch.ones(4, 19)
lc = float(bone_direction_loss(pred, gt + 0.7, BONES))
lm = float(bone_direction_loss(pred, gt + 0.7, BONES, mask=ones))
check(abs(lc - lm) < 1e-9, "mask=None 等价于 mask=全1", f"{lc:.6f} vs {lm:.6f}")

# 误差指标：相同姿态必须是**精确** 0（atan2 形式；arccos 会有 0.006° 底噪）
e0 = bone_direction_error_deg(gt, gt, BONES)
check(e0 == 0.0, "bone_direction_error_deg：相同 -> 精确 0 度（atan2 无底噪）",
      f"{e0:.3e}")
# 有扰动时应明显大于 0，且与"一半的骨被扰动"的直觉量级一致
g = torch.Generator().manual_seed(3)
noisy = gt + torch.randn(4, 19, 3, generator=g) * 0.5
e1 = bone_direction_error_deg(noisy, gt, BONES)
check(e1 > 1.0, "bone_direction_error_deg：有扰动 -> 明显 > 0", f"{e1:.2f}°")

# =====================================================================
print("\n" + "=" * 72)
print("C. 回归测试：无效关节被填 0 后产生的『伪骨方向』")
print("=" * 72)
print("  背景：prepare_mammal_lifter.py 把未标注关节的 3D 目标填成 0。")
print("        若一根骨有一端缺失，它的目标方向会变成 Y3[有效端] - 0 —— 完全错的向量。")
print("        下面的断言确保：掩码为 0 的骨既不产生损失，也不产生梯度。")

B, J = 4, 19
pred = rand_pose(B, J, seed=7).clone().requires_grad_(True)
gt_clean = rand_pose(B, J, seed=8)

# 挑一根骨，把它的一端置为"未标注 -> 0"，并在掩码里标 0
k = 3
j0, j1 = int(b[3, 0]), int(b[3, 1])
gt_bad = gt_clean.clone()
gt_bad[:, j1] = 0.0
mask = torch.ones(B, J)
mask[:, j1] = 0.0

# (1) 伪目标确实是一个"大"向量，不是 0（所以不是 no-op，而是真错误信号）
spur = (gt_bad[:, j0] - gt_bad[:, j1]).norm(dim=-1)
real = (gt_clean[:, j0] - gt_clean[:, j1]).norm(dim=-1)
check(float(spur.min()) > 0.5, "伪骨方向确实非零（是真实存在的错误信号）",
      f"|v_spur| {spur.numpy().round(3)} vs |v_real| {real.numpy().round(3)}")

# (2) 不看掩码时，梯度会流到那个"本应无效"的关节上 —— 这就是坑
l_unmasked = bone_direction_loss(pred, gt_bad, BONES, mask=None)
g_unm = torch.autograd.grad(l_unmasked, pred, retain_graph=True)[0]
check(float(g_unm[:, j1].abs().max()) > 1e-8,
      "【坑】mask=None 时梯度确实流到无效关节（说明修复是必要的）",
      f"max|g(j{j1})| = {float(g_unm[:, j1].abs().max()):.3e}")

# (3) 带上掩码后，梯度在该关节上必须**精确为 0**
l_masked = bone_direction_loss(pred, gt_bad, BONES, mask=mask)
g_msk = torch.autograd.grad(l_masked, pred, retain_graph=True)[0]
check(float(g_msk[:, j1].abs().max()) == 0.0,
      "【修复】带掩码后梯度在无效关节上精确为 0",
      f"max|g(j{j1})| = {float(g_msk[:, j1].abs().max()):.3e}")

# (4) 掩码只剔除受影响的骨，不影响其它骨：与"直接用干净 gt"比，值应一致
mask_only_j1 = mask
l_alt = bone_direction_loss(pred, gt_bad, BONES, mask=mask_only_j1)
gt_alt = gt_clean.clone()
check(abs(float(l_alt) - float(bone_direction_loss(pred, gt_alt, BONES,
                                                  mask=mask_only_j1))) < 1e-6,
      "被掩码的骨不影响其它骨的损失值")

# (5) 全部关节都无效时不应崩（分母 clamp）
mask0 = torch.zeros(B, J)
l_zero = bone_direction_loss(pred, gt_bad, BONES, mask=mask0)
check(torch.isfinite(l_zero), "全掩码时损失为有限值（不产生 NaN）",
      f"{float(l_zero):.3e}")
g_zero = torch.autograd.grad(l_zero, pred, retain_graph=True,
                             allow_unused=True)[0]
check(g_zero is None or float(g_zero.abs().max()) == 0.0,
      "全掩码时梯度为 0（allow_unused 或精确 0）")

# (6) 单样本 / 单骨退化
one = pred[:1]
check(torch.isfinite(bone_direction_loss(one, gt_bad[:1], BONES, mask=mask[:1])),
      "batch=1 不崩")

# =====================================================================
print("\n" + "=" * 72)
print("D. 尺度不变 MPJPE（辅助损失）")
print("=" * 72)

check(abs(float(scale_invariant_mpjpe(gt, gt, BONES, root=ROOT))) < 1e-6,
      "相同姿态 -> 0")
# pred 整体缩放不应改变这一项（这正是"尺度不变"的含义）
l1 = float(scale_invariant_mpjpe(pred, gt, BONES, root=ROOT))
l2 = float(scale_invariant_mpjpe(pred * 3.7, gt, BONES, root=ROOT))
check(abs(l1 - l2) < 1e-5, "pred 整体缩放 -> 不变", f"{l1:.6f} vs {l2:.6f}")
# 平移也不应改变
l3 = float(scale_invariant_mpjpe(pred + 12.0, gt, BONES, root=ROOT))
check(abs(l1 - l3) < 1e-5, "pred 平移 -> 不变", f"{l1:.6f} vs {l3:.6f}")
# 但旋转应当改变（因为 MPJPE 是逐关节对应的，不做 Procrustes 对齐）
R = rand_rot(1, seed=5)[0]
l4 = float(scale_invariant_mpjpe(pred @ R.T, gt, BONES, root=ROOT))
check(l4 > l1, "pred 旋转 -> 变大（未做对齐，属预期）", f"{l1:.4f} -> {l4:.4f}")
check(float(mpjpe(pred, pred)) == 0.0, "mpjpe(相同) = 0")
check(abs(float(mpjpe(pred, pred, mask=torch.ones(pred.shape[:2]))) -
          float(mpjpe(pred, pred))) < 1e-9, "mpjpe：mask=全1 等价于 mask=None")
check(float(mpjpe(pred, pred, mask=torch.zeros(pred.shape[:2]))) == 0.0,
      "mpjpe：全掩码 -> 0（分母 clamp 生效）")

# Procrustes：对齐后旋转+缩放+平移都不应产生误差
gt_t = 2.5 * (gt @ R.T) + torch.tensor([1.0, 2.0, 3.0])
check(procrustes_mpjpe(gt, gt_t) < 1e-4,
      "procrustes_mpjpe：相似变换后 -> ~0", f"{procrustes_mpjpe(gt, gt_t):.2e}")
check(procrustes_mpjpe(gt, gt) < 1e-9, "procrustes_mpjpe：相同 -> ~0")

# =====================================================================
print("\n" + "=" * 72)
print("E. 数据集 / 划分")
print("=" * 72)

N, J = 40, 19
rng = np.random.default_rng(0)
X2 = rng.normal(0, 0.3, (N, J, 2)).astype(np.float32)
Mask = (rng.random((N, J)) > 0.1).astype(np.float32)
Y3 = rng.normal(0, 1, (N, J, 3)).astype(np.float32)

ds = LifterDataset(X2, Mask, Y3)
x, m, y_ = ds[0]
check(x.shape == (J, 2) and m.shape == (J,) and y_.shape == (J, 3),
      "LifterDataset 单样本形状", f"{tuple(x.shape)} {tuple(m.shape)} {tuple(y_.shape)}")
check(len(ds) == N, "LifterDataset 长度")
check(ds.num_joints == J, "LifterDataset.num_joints")

ds_aug = LifterDataset(X2, Mask, Y3, aug_mask=True, mask_ratio=(0.3, 0.3), seed=1)
frac = np.mean([float(ds_aug[i][1].sum()) for i in range(N)])
check(frac < float(Mask.sum(axis=1).mean()),
      "aug_mask 确实减少了可见关节数",
      f"{frac:.2f} < {float(Mask.sum(axis=1).mean()):.2f}")
# 增强只应"减少"可见性，不能把原本不可见的变可见
bad = 0
for i in range(N):
    if bool((ds_aug[i][1].numpy() > Mask[i]).any()):
        bad += 1
check(bad == 0, "aug_mask 不会把不可见关节变可见")

ds_n = LifterDataset(X2, Mask, Y3, noise_std=0.05, seed=3)
check(not np.allclose(ds_n[0][0].numpy(), X2[0]), "noise_std 确实改动了输入")
check(np.allclose(ds[0][0].numpy(), X2[0]), "noise_std=0 时输入原样")

# 按帧划分：不能有帧同时出现在 train 与 val
FrameId = np.repeat(np.arange(N // 4) * 25, 4)
sp = build_frame_split(FrameId, val_ratio=0.25, seed=0)
tr_f = set(FrameId[sp["train"]].tolist())
va_f = set(FrameId[sp["val"]].tolist())
check(len(tr_f & va_f) == 0, "build_frame_split：train/val 无共享帧",
      f"train {len(tr_f)} 帧, val {len(va_f)} 帧")
check(len(sp["train"]) + len(sp["val"]) == N, "build_frame_split 覆盖全部样本")
check(2 <= len(va_f) <= 3,
      "build_frame_split：val 帧数 ≈ 25% × 10（round(2.5)=2，银行家舍入）",
      f"{len(va_f)}")

sp2 = build_split_indices(N, val_ratio=0.25, seed=0)
check(len(sp2["val"]) == 10 and len(sp2["train"]) == 30,
      "build_split_indices：80/20-ish 划分",
      f"{len(sp2['train'])}/{len(sp2['val'])}")

# =====================================================================
print("\n" + "=" * 72)
print("F. 前向（可选；默认跳过，遵守『不在本机跑模型』）")
print("=" * 72)
if args_allow_model:
    from steerpose.lifting import PoseLifter
    mdl = PoseLifter(num_joints=J, d_model=16, nhead=4, num_layers=2,
                     dim_feedforward=32)
    with torch.no_grad():
        o = mdl(torch.randn(4, J, 2), torch.ones(4, J))
        check(tuple(o.shape) == (4, J, 3), "PoseLifter 前向形状 (4,19,3)",
              str(tuple(o.shape)))
        o2 = mdl(torch.randn(4, J, 2), torch.zeros(4, J))
        check(o2.shape == o.shape, "全掩码输入不崩")
        o3 = mdl(torch.randn(4, J, 2))
        check(o3.shape == o.shape, "joint_mask=None 不崩")
        check(bool(torch.isfinite(o).all()), "输出全为有限值")
else:
    print("  跳过。要一起测前向：python scripts/selftest_lifting.py --allow-model")

# =====================================================================
print("\n" + "=" * 72)
print(f"结果：{N_PASS} PASS / {N_FAIL} FAIL" + (f"  ({len(NOTES)} 条说明)" if NOTES else ""))
print("=" * 72)
for m_ in NOTES:
    print(f"  · {m_}")
if N_FAIL:
    print("\n[!] 有断言失败 —— 提升器的损失/指标与几何后端的假设不一致，不要上服务器训练。")
raise SystemExit(1 if N_FAIL else 0)
