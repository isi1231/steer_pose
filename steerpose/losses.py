# -*- coding: utf-8 -*-
"""损失与优化组件（论文 3.1 式(1)、3.2 式(2) 与 Lgeom）.

- Lkp    : 训练损失，预测 2D 姿态与目标 2D 姿态的平均逐关节 L2 距离；
- alpha=3: 匹配相似度 s = 2 / (1 + exp(alpha * Lkp))（论文附录 D.3 给出 alpha=3）；
- Sinkhorn: 可微最优传输，把代价矩阵转成双随机的软指派 W；
- Lmatch : Σ_j w_j (1 - s_j)，双向计算保证视角间循环一致；
- Lgeom  : 对当前 R 下的匹配对，由共线/共面约束构造线性系统
           A [x_1 ... x_N t]^T = 0（x_i 为沿射线的深度，t 为相对平移），
           取最小与次小奇异值之比 σ1/σ2，迫使 R 使 t 有解。
"""
import torch


def torch_skew(v: torch.Tensor) -> torch.Tensor:
    """向量 (3,) -> 反对称矩阵 (3,3)，满足 skew(v) @ u = v × u。"""
    K = torch.zeros(3, 3, device=v.device, dtype=v.dtype)
    K[0, 1], K[0, 2] = -v[2], v[1]
    K[1, 0], K[1, 2] = v[2], -v[0]
    K[2, 0], K[2, 1] = -v[1], v[0]
    return K


def lkp_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """式(1)：平均逐关节 L2 距离（可加权掩码）。"""
    return torch.linalg.norm(pred - target, dim=-1).mean()


def pose_distance(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """两个姿态集合间的成对平均逐关节 L2：q (B1,J,2), p (B2,J,2) -> (B1,B2)。"""
    return torch.cdist(q, p).mean(dim=-1)


def similarity(d: torch.Tensor, alpha: float = 3.0) -> torch.Tensor:
    """s(q, p') = 2 / (1 + exp(alpha * Lkp))，取值 (0, 2)。"""
    return 2.0 / (1.0 + torch.exp(alpha * d))


def sinkhorn(cost: torch.Tensor, n_iter: int = 50, tau: float = 0.05) -> torch.Tensor:
    """log 域 Sinkhorn 行列归一化：代价矩阵 -> 软指派矩阵 W（可微）。"""
    logK = -cost / tau
    for _ in range(n_iter):
        logK = logK - torch.logsumexp(logK, dim=1, keepdim=True)
        logK = logK - torch.logsumexp(logK, dim=0, keepdim=True)
    return torch.exp(logK)


def matching_loss(Q: torch.Tensor, P2: torch.Tensor, alpha: float = 3.0,
                  n_iter: int = 50, tau: float = 0.05):
    """式(2) 双向匹配损失。返回 (Lmatch, W, S)。"""
    d = pose_distance(Q, P2)
    S = similarity(d, alpha)
    W = sinkhorn(d, n_iter=n_iter, tau=tau)
    loss = (W * (1.0 - S)).sum() + (W.t() * (1.0 - S.t())).sum()
    return loss, W, S


def build_linear_system(Q: torch.Tensor, P2: torch.Tensor,
                        pairs: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """共线 + 共面约束构成 A z = 0，z = [x_1..x_N, t]（论文 3.2 / 文献[19]）.

    相机 1 系下 3D 点 X_i = x_i * d_i（d_i 为 q_i 的单位射线，x_i 未知深度），
    在相机 2 系中应落在 p'_i 的射线上：
        cross(d'_i, R @ (x_i d_i) + t) = 0
    展开为：x_i * [cross(d'_i, R d_i)] + skew(d'_i) @ t = 0 —— 对 (x_i, t) 线性。
    """
    Np = pairs.shape[0]
    if Np == 0:
        return None
    rows = []
    for a, b in pairs.tolist():
        da = torch.cat([Q[a], torch.zeros(1, device=Q.device)])
        db = torch.cat([P2[b], torch.zeros(1, device=P2.device)])
        da = da / (torch.norm(da) + 1e-12)
        db = db / (torch.norm(db) + 1e-12)
        A = torch.zeros(3, 3 + 3, device=Q.device)
        A[:, 0] = torch.cross(db, R @ da)     # x_i 的系数
        A[:, 1:] = torch_skew(db)             # t 的系数
        rows.append(A)
    return torch.cat(rows, dim=1)              # (3*Np, Np+3)


def to_cam_frame(P: torch.Tensor) -> torch.Tensor:
    """2D 姿态 -> 近似归一化相机系坐标（加 z=0 平面并去中心/尺度）。"""
    hi = P.max(dim=1, keepdim=True).values
    lo = P.min(dim=1, keepdim=True).values
    c = (hi + lo) / 2.0
    scale = (hi - lo).max(-1, keepdim=True).values / 2.0 + 1e-8
    return (P - c) / scale


def geometric_loss(Q: torch.Tensor, P2: torch.Tensor, W: torch.Tensor,
                   R: torch.Tensor, conf_thr: float = None) -> torch.Tensor:
    """Lgeom = σ_min / σ_next（越接近 0 说明当前 R 存在合法平移解）。

    用软指派 W 的行最大置信度筛选可靠匹配；阈值随集合大小自适应。
    """
    conf, bidx = W.max(dim=1)
    aidx = torch.arange(W.shape[0], device=W.device)
    if conf_thr is None:
        conf_thr = max(0.3, 2.0 / max(W.shape[0], W.shape[1]))
    sel = conf > conf_thr
    if sel.sum() < 3:                      # 匹配太少，几何项无意义
        return torch.zeros((), device=W.device)
    pairs = torch.stack([aidx[sel], bidx[sel]], dim=1)
    A = build_linear_system(to_cam_frame(Q), to_cam_frame(P2), pairs, R)
    s = torch.linalg.svdvals(A)
    return s[0] / (s[1] + 1e-12)
