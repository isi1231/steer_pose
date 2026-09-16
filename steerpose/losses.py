# -*- coding: utf-8 -*-
"""损失与优化组件（论文 3.1 式(1)、3.2 式(2) 与 Lgeom）.

- Lkp    : 训练损失，预测 2D 姿态与目标 2D 姿态的平均逐关节 L2 距离；
- alpha=3: 匹配相似度 s = 2 / (1 + exp(alpha * Lkp))（论文附录 D.3 给出 alpha=3）；
- Sinkhorn: 可微最优传输，把代价矩阵转成双随机的软指派 W；
- Lmatch : Σ_j w_j (1 - s_j)，双向计算保证视角间循环一致；
- Lgeom  : 对当前 R 下的匹配对，由共线约束构造线性系统
           A [x_1 ... x_N t]^T = 0（x_i 为 3D 点沿相机 A 视线的深度，t 为相对平移），
           取最小与次小奇异值之比 σ1/σ2，迫使 R 使 t 有解。
           注意：Lgeom 必须用**整场景归一化**坐标（normalize_scene），
           逐姿态归一化会抹掉位置信息导致 t 不可辨识。
"""
import math
import numpy as np
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
    """两个姿态集合间的**成对**平均逐关节 L2：q (B1,J,2), p (B2,J,2) -> (B1,B2)。

    d[i,j] = (1/J) Σ_k ‖q[i,k] - p[j,k]‖₂ —— 即式(2)里 q_j(R) 与 p′_j 之间的 Lkp。

    **不要**写成 torch.cdist(q, p)：cdist 把 (B,J,2) 当成"批量矩阵"，会沿 J 维做
    广播，返回 (B,J,J) 而不是 (B1,B2)——匹配矩阵维度就错了，Sinkhorn 与
    Lgeom 全部失效，而训练损失 Lkp 不经过这里，所以画面上看不出问题。
    """
    if q.shape[-2:] != p.shape[-2:]:
        raise ValueError(
            f"两个姿态集合的 (J,2) 必须一致，实际 {tuple(q.shape[-2:])} vs {tuple(p.shape[-2:])}")
    diff = q[:, None, :, :] - p[None, :, :, :]     # (B1, B2, J, 2)
    return diff.norm(dim=-1).mean(dim=-1)          # (B1, B2)


def pose_distance_ref(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """pose_distance 的朴素参考实现（逐对循环），仅用于自检。"""
    q, p = np.asarray(q, np.float64), np.asarray(p, np.float64)
    out = np.zeros((q.shape[0], p.shape[0]))
    for i in range(q.shape[0]):
        for j in range(p.shape[0]):
            out[i, j] = np.linalg.norm(q[i] - p[j], axis=-1).mean()
    return out


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
    """式(2) 双向匹配损失。返回 (Lmatch, W, S)。

    W 的形状是 (B1, B2)：把 Q 中的第 i 个姿态指派到 P2 中的第 j 个。
    """
    d = pose_distance(Q, P2)
    S = similarity(d, alpha)
    W = sinkhorn(d, n_iter=n_iter, tau=tau)
    loss = (W * (1.0 - S)).sum() + (W.t() * (1.0 - S.t())).sum()
    return loss, W, S


def batch_skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3)，满足 batch_skew(v) @ u = v × u。"""
    o = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    o[..., 0, 1] = -v[..., 2]; o[..., 0, 2] = v[..., 1]
    o[..., 1, 0] = v[..., 2]; o[..., 1, 2] = -v[..., 0]
    o[..., 2, 0] = -v[..., 1]; o[..., 2, 1] = v[..., 0]
    return o


def guess_focal(P: np.ndarray, focal: float = None) -> float:
    """未显式给焦距时，按数据量级猜一个：像素坐标 -> 约 0.9×图像尺寸；否则认为已归一化。"""
    if focal is not None and focal > 0:
        return float(focal)
    flat = np.asarray(P, np.float64).reshape(-1, 2)
    extent = float((flat.max(0) - flat.min(0)).max())
    return 1.0 if extent < 10.0 else 0.9 * extent


def to_ray_coords(P: np.ndarray, focal: float = 1.0,
                  principal_point=None) -> np.ndarray:
    """2D 坐标 -> 归一化图像坐标 (x-cx)/f, (y-cy)/f —— 保留真实射线几何。

    **关键**：几何约束必须用"按焦距归一化"的坐标，因为归一化图像坐标下的视线方向
    就是 (u, v, 1)。若改用数据自适应的包围盒尺度（如 data.normalize_pose 那样除以
    最大边），相当于把焦距悄悄改成了那个尺度，每条射线的方向都被改变，正确的 R 就
    不再对应零奇异值，Lgeom 会失去判别力。

    principal_point 默认取所有点的均值（无内参时的常用代理）；有标定内参时请显式传入。

    与网络输入用的逐姿态归一化（data.normalize_pose）是两回事：
    网络只看姿态**形状**，几何约束还需要姿态在图像中的**位置**。
    """
    P = np.asarray(P, np.float64)
    if principal_point is None:
        principal_point = P.reshape(-1, 2).mean(0)
    return (P - np.asarray(principal_point, np.float64)) / float(focal)


def rays_from_2d(P: torch.Tensor) -> torch.Tensor:
    """(..., 2) 归一化图像坐标 -> (..., 3) 单位视线方向 (u, v, 1)/‖·‖。

    对应文献[19]的 normalized camera coordinates：焦距归一化为 1。
    """
    ones = torch.ones(*P.shape[:-1], 1, device=P.device, dtype=P.dtype)
    d = torch.cat([P, ones], dim=-1)
    return d / (d.norm(dim=-1, keepdim=True) + 1e-12)


def pose_centroid(P: torch.Tensor) -> torch.Tensor:
    """(N, J, 2) -> (N, 2)：每个姿态的关节质心，作为共线约束里的 3D 点观测。

    论文对 Lgeom 的表述是 "for all N corresponding 2D pose pairs"，
    unknowns x_i 是"由共线约束恢复出的 3D [点]"，即**每对匹配姿态一个 3D 点**，
    所以这里用姿态质心而不是逐个关节（文献[19]的 3D 点则来自逐关节匹配）。
    """
    return P.mean(dim=1)


def build_linear_system(Qg: torch.Tensor, P2g: torch.Tensor,
                        pairs: torch.Tensor, R: torch.Tensor,
                        reduce: str = "centroid", ridge: float = 1e-8):
    """共线约束堆叠为 A z = 0，z = [x_1 ... x_N, t]（论文 3.2 节，文献[19] §III）。

    相机 A 系下的 3D 点记为 X_i = x_i d_i（d_i 为该点在 A 中的单位视线，x_i 未知），
    它在相机 B 系中必须落在对应观测 d'_i 的视线上：

        cross(d'_i, R (x_i d_i) + t) = 0
        =>  x_i · [skew(d'_i) R d_i]  +  skew(d'_i) · t  = 0

    对 (x_i, t) 完全线性。R 与匹配都正确时该方程组存在非零解（t 只差一个尺度），
    即 A 的最小奇异值 → 0。

    参数
      Qg, P2g : 视角 A / B 的 2D 坐标，**整场景归一化**（normalize_scene），保留位置
      pairs   : (N, 2) 匹配索引对
      reduce  : 'centroid' 每对匹配姿态取质心作一个 3D 点（论文口径，默认，矩阵小）
                'joints'   逐关节作 3D 点（约束更多，矩阵大 N×J 倍）
      ridge   : 附加 sqrt(ridge)·I 行，保证奇异值有下界、梯度不因重根产生 NaN

    返回 A (3N(+3)，N+3)；N 为匹配数（reduce='joints' 时为 匹配数×J）。
    """
    N = pairs.shape[0] if pairs is not None else 0
    if N == 0:
        return None
    ia, ib = pairs[:, 0], pairs[:, 1]
    if reduce == "joints":
        da = rays_from_2d(Qg[ia]).reshape(-1, 3)
        db = rays_from_2d(P2g[ib]).reshape(-1, 3)
    else:
        da = rays_from_2d(pose_centroid(Qg)[ia])
        db = rays_from_2d(pose_centroid(P2g)[ib])
    M = da.shape[0]

    Rd = (R @ da.unsqueeze(-1)).squeeze(-1)          # R d_i           (M, 3)
    C = torch.cross(db, Rd, dim=-1)                  # x_i 的系数       (M, 3)
    S = batch_skew(db)                               # t 的系数         (M, 3, 3)

    Cmat = torch.zeros(3 * M, M, device=da.device, dtype=da.dtype)
    Cmat[torch.arange(3 * M, device=da.device),
         torch.arange(M, device=da.device).repeat_interleave(3)] = C.reshape(-1)
    Smat = S.reshape(3 * M, 3)
    A = torch.cat([Cmat, Smat], dim=1)               # (3M, M+3)
    if ridge and ridge > 0:
        eye = torch.eye(M + 3, device=da.device, dtype=da.dtype) * math.sqrt(ridge)
        A = torch.cat([A, eye], dim=0)
    return A


def geometric_loss(Qg: torch.Tensor, P2g: torch.Tensor, W: torch.Tensor,
                   R: torch.Tensor, conf_thr: float = None,
                   reduce: str = "centroid", max_pts: int = 512) -> torch.Tensor:
    """Lgeom = σ1/σ2（论文 3.2 节）——最小奇异值 / 次小奇异值。

    论文原文："the singular values σ2 ≥ σ1 ... we include the ratio σ1/σ2 as a loss
    Lgeom in the optimization of R, to guarantee that the estimated R makes the
    smallest singular value as small as possible, and hence it has a solution of t."
    即 σ1 为**最小**、σ2 为**次小**；比值 → 0 表示当前 R 下存在合法的平移解。

    注意：这里必须传**整场景归一化**的 2D 坐标（normalize_scene），不是逐姿态归一化。

    用软指派 W 的行最大置信度筛选可靠匹配；阈值随集合大小自适应。
    """
    if W is None or W.numel() == 0:
        return torch.zeros((), device=R.device)
    if W.dim() != 2 or W.shape[0] != Qg.shape[0] or W.shape[1] != P2g.shape[0]:
        raise ValueError(
            f"匹配矩阵 W 形状 {tuple(W.shape)} 与两个视角的目标数 "
            f"({Qg.shape[0]}, {P2g.shape[0]}) 不一致。W[i, j] 表示 Q 的第 i 个姿态"
            f"指派给 P2 的第 j 个姿态，必须是二维 (B1, B2)。")
    conf, bidx = W.max(dim=1)
    aidx = torch.arange(W.shape[0], device=W.device)
    if conf_thr is None:
        conf_thr = max(0.3, 2.0 / max(W.shape[0], W.shape[1]))
    sel = conf > conf_thr
    if sel.sum().item() < 3:               # 匹配太少，几何项无意义
        return torch.zeros((), device=W.device)
    pairs = torch.stack([aidx[sel], bidx[sel]], dim=1)
    if pairs.shape[0] > max_pts:           # 控制 SVD 规模
        step = max(1, pairs.shape[0] // max_pts)
        pairs = pairs[::step][:max_pts]
    A = build_linear_system(Qg, P2g, pairs, R, reduce=reduce)
    if A is None:
        return torch.zeros((), device=W.device)
    s = torch.linalg.svdvals(A)          # torch 返回**降序**
    return s[-1] / (s[-2] + 1e-12)       # σ_min / σ_2ndmin -> 0 表示存在合法 t


def solve_translation(Qg: torch.Tensor, P2g: torch.Tensor, pairs: torch.Tensor,
                      R: torch.Tensor, reduce: str = "centroid") -> np.ndarray:
    """给定 R 与匹配，解出相对平移**方向** t（尺度不可观测）。

    取 A 的最小右奇异向量 z = [x_1 ... x_N, t]，返回归一化后的 t。
    """
    A = build_linear_system(Qg, P2g, pairs, R, reduce=reduce, ridge=0.0)
    if A is None:
        return np.zeros(3)
    _, _, Vt = np.linalg.svd(A.detach().cpu().numpy())
    z = Vt[-1]
    M = A.shape[1] - 3
    t = z[M:]
    n = float(np.linalg.norm(t))
    return t / n if n > 1e-9 else np.zeros(3)
