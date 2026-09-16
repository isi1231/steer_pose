# -*- coding: utf-8 -*-
"""单目 3D 提升器：单视角 2D 关节 → 相机系 3D 姿态（根相对）。

它在整条链路里的位置
--------------------
RA-L2022（`Extrinsic camera calibration from a moving person`, 同作者前作）用的是
**每台相机各自**跑一个单目 3D 提升器（VideoPose3D），然后：

  * 用提升出来的 3D 的**骨方向**（oriented points）解相机旋转——因为单目 3D 的绝对位置
    与深度不可靠，但**朝向**是可靠的；
  * 用 2D 射线的共线 + 共面约束解平移；
  * 最后 BA 联合优化（重投影 + 跨相机骨方向一致性 + 骨长一致性）。

SteerPose 原本没有提升器（它只做 2D→2D 的"心理旋转"，用于匹配），所以这里补上这一环。
**关键约定**：本网络的输出只需要"朝向正确"，不需要"绝对位置/尺度正确"——
因此训练目标以**骨方向余弦损失为主**，MPJPE 只作辅助；3D 目标统一做
根相对 + 骨长归一化（单目尺度本来就不可观测）。

与 VideoPose3D 的差异（有意为之，写在 docs/ral_backend.md）：
  * 输入用**按内参归一化**的坐标 ((x-cx)/fx, (y-cy)/fy)，而不是按图像尺寸归一化——
    这样换相机/换分辨率不用重训，和几何后端的射线口径完全一致；
  * 结构复用本仓库 SteerPose 的 Transformer 骨架（单帧），不做时序卷积：
    BamaPig3D 只有 70 个标注帧、标注帧间隔 25 帧，时序上下文本来就稀。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .skeleton import bones_to_index_array, get_skeleton


# ============================ 坐标与尺度 ============================

def normalize_2d_intrinsics(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    """像素坐标 -> **按内参归一化**的图像坐标 ((x-cx)/fx, (y-cy)/fy)。

    这与 `losses.to_ray_coords(K=...)` 是同一套口径：归一化坐标下视线方向就是 (u, v, 1)。
    提升器用它当输入，几何约束也用它，两边不会打架。
    """
    P = np.asarray(P, np.float64)
    K = np.asarray(K, np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K 形状应为 (3,3)，实际 {K.shape}")
    return (P - K[:2, 2]) / np.array([K[0, 0], K[1, 1]], np.float64)


def root_center(y3: np.ndarray, root: int) -> np.ndarray:
    """根相对化：减去根关节坐标。"""
    return y3 - y3[..., root: root + 1, :]


def pose_scale(y3: np.ndarray, bones, root: int, eps: float = 1e-9) -> np.ndarray:
    """单目尺度基准：根相对后的**平均骨长**。"""
    yc = root_center(y3, root)
    b = bones_to_index_array(bones)
    d = np.linalg.norm(yc[..., b[:, 0], :] - yc[..., b[:, 1], :], axis=-1)
    return d.mean(axis=-1)


def normalize_3d(y3: np.ndarray, bones, root: int) -> tuple:
    """根相对 + 骨长归一化。返回 (y_norm, scale)，便于还原到原始量纲。"""
    s = pose_scale(y3, bones, root)
    return root_center(y3, root) / s[..., None, None], s


# ============================ 指标与损失 ============================

def bone_vectors(y3: torch.Tensor, bones) -> torch.Tensor:
    """(B,J,3) -> (B,Bn,3) 骨向量（未归一化）。"""
    b = bones_to_index_array(bones)
    idx0 = torch.as_tensor(b[:, 0], device=y3.device)
    idx1 = torch.as_tensor(b[:, 1], device=y3.device)
    return y3[:, idx0, :] - y3[:, idx1, :]


def _bone_valid_weight(mask: torch.Tensor, bones) -> torch.Tensor:
    """关节掩码 (B,J) -> 骨掩码 (B,Bn)：**两端都可见**的骨才计入损失。

    ★ 为什么必须做这件事（一个真实的坑）
    预处理把"未标注关节"的 3D 目标填成了 0（见 `prepare_mammal_lifter.py`），
    于是**只要一根骨有一端缺失**，它的"目标方向"就变成
    `Y3[有效端] - 0`——一个完全错误的向量。如果损失不看掩码，网络会被
    主动推向这个虚假方向。所以骨级加权不是锦上添花，是正确性要求。
    """
    b = bones_to_index_array(bones)
    i0 = torch.as_tensor(b[:, 0], device=mask.device)
    i1 = torch.as_tensor(b[:, 1], device=mask.device)
    return mask[:, i0] * mask[:, i1]


def bone_direction_loss(pred3: torch.Tensor, gt3: torch.Tensor, bones,
                        mask: torch.Tensor = None,
                        eps: float = 1e-8) -> torch.Tensor:
    """骨方向余弦损失 `1 - cos(pred, gt)`（对尺度、平移都不敏感）。

    **这是本网络的主损失**：RA-L2022 的标定只用到骨方向（oriented points），
    所以"方向对"比"位置对"重要得多。

    mask : (B,J) 关节可见性（1=可见）。给定时只在"两端都可见"的骨上求平均；
           不给定则对全部骨求平均（旧行为）。
    """
    vp = torch.nn.functional.normalize(bone_vectors(pred3, bones) + eps, dim=-1)
    vg = torch.nn.functional.normalize(bone_vectors(gt3, bones) + eps, dim=-1)
    cos = (vp * vg).sum(dim=-1)                     # (B,Bn)
    if mask is None:
        return (1.0 - cos).mean()
    w = _bone_valid_weight(mask.to(cos.dtype), bones)
    return ((1.0 - cos) * w).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def bone_direction_error_deg(pred3: torch.Tensor, gt3: torch.Tensor, bones,
                             mask: torch.Tensor = None) -> float:
    """骨方向角度误差（度）—— 标定质量最相关的指标。

    ★ 用 `atan2(|v×g|, v·g)` 而不是 `arccos(v·g)`：
      `arccos` 在 1 附近是**平方根型病态**的——float32 下 `cos` 只要因舍入掉到
      1-6e-8，角度就会跳到 ~0.006°，于是"完全相同"的两个姿态也报出非零误差。
      `atan2` 形式在 v == g 处给出**精确的 0**（叉积严格为 0），误差指标才干净。
      （同样的处理见 `steerpose/calib_ral.py` 的 `_so3_angle` / `_vec_angle`。）
    """
    vp = torch.nn.functional.normalize(bone_vectors(pred3, bones), dim=-1)
    vg = torch.nn.functional.normalize(bone_vectors(gt3, bones), dim=-1)
    cross = torch.linalg.cross(vp, vg, dim=-1)
    sin = torch.linalg.norm(cross, dim=-1)
    cos = (vp * vg).sum(dim=-1)
    ang = torch.rad2deg(torch.atan2(sin, cos))       # (B,Bn)
    if mask is None:
        return float(ang.mean())
    w = _bone_valid_weight(mask.to(ang.dtype), bones)
    return float((ang * w).sum() / w.sum().clamp_min(1.0))


def mpjpe(pred3: torch.Tensor, gt3: torch.Tensor,
          mask: torch.Tensor = None) -> torch.Tensor:
    """根相对 3D 的逐关节 L2（尺度已归一）。mask 给定时只统计可见关节。"""
    d = torch.linalg.norm(pred3 - gt3, dim=-1)      # (B,J)
    if mask is None:
        return d.mean()
    w = mask.to(d.dtype)
    return (d * w).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def procrustes_mpjpe(pred3: torch.Tensor, gt3: torch.Tensor,
                     mask: torch.Tensor = None) -> float:
    """P-MPJPE：先对每帧做相似变换对齐（旋转+尺度+平移）再算 MPJPE。

    单目 3D 的绝对朝向/尺度不可观测，报告"对齐后"的误差才是公平的。
    mask 给定时只统计该帧可见的关节（至少 3 个可见才计入）。
    """
    p = pred3.detach().cpu().numpy().astype(np.float64)
    g = gt3.detach().cpu().numpy().astype(np.float64)
    m = None if mask is None else (mask.detach().cpu().numpy() > 0.5)
    errs = []
    for i, (a, b) in enumerate(zip(p, g)):
        if m is not None:
            ok = m[i]
            if ok.sum() < 3:
                continue
            a, b = a[ok], b[ok]
        mu_a, mu_b = a.mean(0), b.mean(0)
        A, B = a - mu_a, b - mu_b
        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        scale = (S * np.array([1.0, 1.0, d])).sum() / max((A ** 2).sum(), 1e-12)
        errs.append(np.linalg.norm((R @ A.T).T * scale - B, axis=-1).mean())
    return float(np.mean(errs)) if errs else float("nan")



# ============================ 网络 ============================

class PoseLifter(nn.Module):
    """单帧单目 3D 提升器：`(B,J,2) [+掩码] -> (B,J,3)`（根相对、骨长归一化）。

    结构：关节 MLP → J 个 token（+ 可学习位置编码）→ Transformer 编码器 ×5 →
    **逐关节** 头 → 3 维。
    与 `model.SteerPose` 的唯一结构差别：SteerPose 对 token 做均值池化后回归整个
    2D 姿态（因为它的输入是"集合"，输出也是"集合"）；这里每个关节要单独的 3D，
    所以是逐 token 回归。
    """

    def __init__(self, num_joints: int = 19, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 5, dim_feedforward: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model

        self.joint_mlp = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.pos_embedding = nn.Parameter(torch.zeros(num_joints, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            activation="gelu", batch_first=True, norm_first=True, dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
        )

    def forward(self, joints2d: torch.Tensor,
                joint_mask: torch.Tensor = None) -> torch.Tensor:
        B = joints2d.shape[0]
        if joint_mask is None:
            joint_mask = torch.ones(B, self.num_joints,
                                    device=joints2d.device, dtype=joints2d.dtype)
        m = joint_mask.unsqueeze(-1)
        tok = (self.joint_mlp(joints2d * m)
               + (1.0 - m) * self.mask_token
               + self.pos_embedding[: self.num_joints])
        return self.head(self.encoder(tok))          # (B,J,3)


# ============================ 数据集 ============================

class LifterDataset(Dataset):
    """(2D, 掩码, 3D) 三件套。`aug_mask` 会随机把关节置为不可见（模拟检测漏检）。

    约定：
      X2   : (N,J,2) float32  **按内参归一化**的 2D
      Mask : (N,J)   float32  1=可见
      Y3   : (N,J,3) float32  根相对 + 骨长归一化的相机系 3D
    """

    def __init__(self, X2, Mask, Y3, indices=None,
                 aug_mask: bool = False, mask_ratio=(0.1, 0.3),
                 noise_std: float = 0.0, seed: int = 0):
        self.X2 = np.asarray(X2, np.float32)
        self.Mask = np.asarray(Mask, np.float32)
        self.Y3 = np.asarray(Y3, np.float32)
        self.indices = np.arange(len(self.X2)) if indices is None else np.asarray(indices)
        self.aug_mask = aug_mask
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.num_joints = self.X2.shape[1]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        x = self.X2[j].copy()
        m = self.Mask[j].copy()
        if self.noise_std > 0:
            x = x + self.rng.normal(0, self.noise_std, x.shape).astype(np.float32)
        if self.aug_mask:
            lo, hi = self.mask_ratio
            # 额外随机遮挡（只在与原掩码都可见的关节里挑），模拟"训练时也可能漏检"
            extra = self.rng.random(self.num_joints) < self.rng.uniform(lo, hi)
            m = m * (~extra)
        return (torch.from_numpy(x), torch.from_numpy(m),
                torch.from_numpy(self.Y3[j]))


def build_split_indices(n: int, val_ratio: float = 0.2, seed: int = 0) -> dict:
    """按 (帧) 随机划分：默认 8:2。注意：**同一帧的不同相机样本应落在同一侧**，
    否则会高估验证集表现；所以调用方应先按帧聚合（见 scripts/prepare_mammal_lifter.py
    写出的 `frame_id` 字段），本函数只做保底随机划分。
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_ratio))
    return {"val": np.sort(perm[:n_val]), "train": np.sort(perm[n_val:])}


def build_frame_split(frame_ids: np.ndarray, val_ratio: float = 0.2,
                      seed: int = 0) -> dict:
    """按**帧**划分 train/val（同帧的 10 个相机样本不会被拆开）。"""
    uni = np.unique(np.asarray(frame_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uni))
    n_val = max(1, int(round(len(uni) * val_ratio)))
    val_f = set(uni[perm[:n_val]].tolist())
    is_val = np.array([f in val_f for f in np.asarray(frame_ids)])
    return {"val": np.where(is_val)[0], "train": np.where(~is_val)[0]}


def bone_length_mean(y3: torch.Tensor, bones, mask: torch.Tensor = None,
                     root: int = None, eps: float = 1e-9) -> torch.Tensor:
    """有效骨的**加权平均骨长** (B,) —— 可微，用于把预测缩放到与目标同量纲。

    数据准备里用的是"有效骨骨长的中位数"，但中位数不可微；这里用加权均值，
    只要 pred 与 gt 用**同一个**函数归一化，两者就仍然可比（见 scale_invariant_mpjpe）。
    """
    if root is not None:
        y3 = y3 - y3[:, root:root + 1]
    d = torch.linalg.norm(bone_vectors(y3, bones), dim=-1)      # (B,Bn)
    if mask is None:
        return d.mean(dim=-1)
    w = _bone_valid_weight(mask, bones)
    return (d * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)


def scale_invariant_mpjpe(pred3: torch.Tensor, gt3: torch.Tensor, bones,
                          mask: torch.Tensor = None, root: int = None,
                          eps: float = 1e-9) -> torch.Tensor:
    """**尺度不变**的 MPJPE（可微，可当辅助损失）。

    单目 3D 的绝对尺度不可观测，直接回归绝对坐标是学不出来的。这里先用
    `bone_length_mean` 把 pred / gt 各自归一到"平均骨长 = 1"，再算逐关节 L2。
    于是这一项只约束**姿态的相对深度结构**，不惩罚尺度 —— 与骨方向主损失互补。

    ★ 关键是 pred 与 gt 用**同一个**归一化规则，否则会引入系统性偏差。
    """
    p = pred3 - pred3[:, root:root + 1] if root is not None else pred3
    g = gt3 - gt3[:, root:root + 1] if root is not None else gt3
    sp = bone_length_mean(p, bones, mask, root=None, eps=eps).clamp_min(eps)
    sg = bone_length_mean(g, bones, mask, root=None, eps=eps).clamp_min(eps)
    p = p / sp[:, None, None]
    g = g / sg[:, None, None]
    d = torch.linalg.norm(p - g, dim=-1)
    if mask is None:
        return d.mean()
    w = mask.to(d.dtype)
    return (d * w).sum() / w.sum().clamp_min(1.0)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "PoseLifter", "LifterDataset", "normalize_2d_intrinsics", "normalize_3d",
    "root_center", "pose_scale", "bone_vectors", "bone_direction_loss",
    "bone_direction_error_deg", "mpjpe", "procrustes_mpjpe",
    "bone_length_mean", "scale_invariant_mpjpe",
    "build_split_indices", "build_frame_split", "count_params", "get_skeleton",
]
