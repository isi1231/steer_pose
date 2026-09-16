# -*- coding: utf-8 -*-
"""RA-L2022 风格的几何标定后端：从"移动中的个体的骨方向"解多相机外参。

来源
----
本模块是 IROS/RA-L 2022 "Extrinsic Camera Calibration From a Moving Person"
（与 SteerPose 同一作者组，见 `reference/calib_from_moving_person/`）的核心几何
求解器的**忠实移植**，只做三处工程化改写（都写在 `docs/ral_backend.md`）：

1. 参考实现的稀疏矩阵用 `lil_matrix` 逐块赋值再 `vstack`，本文改为等价的
   **COO 三元组一次性构造**（无行为差异，只是把 O(块数) 的 Python 循环搬到 numpy 里）；
   `scripts/selftest_calib_ral.py` 会用参考写法做对照，保证两者数值一致。
2. 旋转矩阵的 Rodrigues 换算用 `scipy.spatial.transform.Rotation`，
   不引入 OpenCV（与仓库其余脚本一致）。
3. 增加了 `orthonormalize` 开关与退化诊断返回值；默认行为与参考实现**完全一致**。

它解决的问题
------------
只有 2D 关节、相机之间没有任何共同已知结构时，怎么恢复外参？
参考实现的答案是：**用一个在场景中自由移动的人（这里是猪）当标定物**。

整条链路（三段，逐段放松对单目 3D 的精度要求）
------------------------------------------------

    [1] 旋转：SVD 求"共同世界方向"
        每台相机用单目 3D 提升器（`steerpose.lifting.PoseLifter`）得到相机系 3D，
        取**骨方向单位向量** v_c ∈ R^3 当 oriented point。
        对同一姿势，世界系里同一根骨头在所有相机看起来方向相同，只是各自差一个 R_c：
            v_c = R_c · v_world        （行向量写法：v_c = v_world @ R_c^T）
        把所有相机的 v_c 沿列拼成 (N, 3C)，其前 3 个右奇异向量就是 v_world 的基，
        于是 R_all = sqrt(C)·Zt[:3,:]（列块即各相机的 R_c）。
        **gauge**：整体左乘一个世界旋转不影响任何观测，所以固定 R_0 = I。
        → 输出只有**相对**旋转有意义（`relative_rotation_error_deg` 就是这么算的）。

    [2] 平移：共线 + 共面的线性系统，零空间 4 维
        观测量 n_c 是归一化图像坐标下的射线方向 ((x-cx)/fx,(y-cy)/fy,1)。
          * 共线（collinearity）：每台相机每个 3D 点，射线 ⟂ 该点在相机下的成像方向
                [n_c]_x · (R_c X + t_c) = 0
          * 共面（coplanarity）：两台相机的两条射线与基线共面
                (n_a × n_b) ⟂ (t_a - t_b) 的相机系形式
        拼成 C·x = 0，取 C^T C 的**最小的 4 个特征向量**：
        零空间维数 = 3（整体平移）+ 1（整体尺度）——这正是标定的固有 gauge。
        用 `t_0 = 0` 消掉平移 gauge、用 |t_1 - t_0| = 1 消掉尺度 gauge，
        再用 z-test（射线正负一致性）确定手性（chiral）符号。

    [3] BA（可选）：`scipy.optimize.least_squares` 联合精化
        三项残差（权重 lambda1 / lambda2）：
          a. 重投影残差（用 2D 置信度加权）
          b. **跨相机骨方向一致性**：同一根骨头在各相机→世界后应指向同一方向
             —— 这一项只用提升器的"朝向"，不用它的绝对位置/尺度
          c. 三角化 3D 的**骨长方差** —— 用刚体假设约束各帧
        注意 b 项的输入是**每台相机各自的** 3D（`p3d_CxNxJx3`），
        而不是融合后的 3D；所以单目提升器的深度误差不会直接污染标定。

与 SteerPose 的分工
-------------------
* SteerPose 的网络（`steerpose.model.SteerPose`）做 **2D→2D**：跨视角匹配；
* 本模块 + `steerpose.lifting.PoseLifter` 做 **2D→3D→外参**：标定。
两者观测的是同一件事（articulation 的朝向），所以用同一套骨架定义
（`steerpose.skeleton`）。参考实现的论文结论是：一旦标定出来，
三角化出的 3D 又可以反过来当"伪标签"训练单目 3D 估计器（自监督闭环，
见 `docs/ral_backend.md` 的"自监督闭环"一节）。

依赖：**只用 numpy / scipy**，不导入 torch —— 本模块可以在任何地方、任何时间跑。
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .skeleton import bones_to_index_array

__all__ = [
    # 观测构造
    "visible_from_all", "oriented_points", "normalized_rays",
    # 线性解
    "solve_rotation", "solve_translation", "z_test_sign", "triangulate",
    "dlt_batch", "calib_ral",
    # BA
    "ba_refine", "reprojection_errors",
    # 评估
    "camera_centers", "relative_rotation_error_deg", "so3_angle_deg",
    "translation_direction_error_deg", "similarity_align", "alignment_error",
]


# =====================================================================
# 0. 小工具
# =====================================================================
def _skew(n: np.ndarray) -> np.ndarray:
    """(...,3) -> (...,3,3) 反对称矩阵 [n]_x，满足 [n]_x @ x = n × x。"""
    n = np.asarray(n, np.float64)
    z = np.zeros_like(n[..., 0])
    r0 = np.stack([z, -n[..., 2], n[..., 1]], axis=-1)
    r1 = np.stack([n[..., 2], z, -n[..., 0]], axis=-1)
    r2 = np.stack([-n[..., 1], n[..., 0], z], axis=-1)
    return np.stack([r0, r1, r2], axis=-2)


def _as_K(C: int, K) -> np.ndarray:
    """内参统一成 (C,3,3)。允许传 (3,3) 或 (C,3,3)。"""
    K = np.asarray(K, np.float64)
    if K.ndim == 2:
        K = np.broadcast_to(K, (C, 3, 3)).copy()
    if K.shape != (C, 3, 3):
        raise ValueError(f"内参形状应为 (3,3) 或 ({C},3,3)，实际 {K.shape}")
    return K


def _rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rv, np.float64).reshape(-1, 3)).as_matrix()


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, np.float64).reshape(-1, 3, 3)).as_rotvec()


# =====================================================================
# 1. 观测构造：oriented points & 归一化射线
# =====================================================================
def visible_from_all(mask_CxNxJ: np.ndarray) -> np.ndarray:
    """(C,N,J) bool -> (N,J) bool：在所有相机中都可见的关节。

    参考实现 `util.visible_from_all_cam` 就是 `np.min(mask, axis=0)`。

    **为什么要求"全相机可见"**：旋转解要求所有相机的 oriented point 一一对应；
    平移解的共面约束也是按"同一个 3D 点在各相机都有射线"来配对的。
    代价是样本量会明显减少——BamaPig3D 里猪经常被栏杆/同伴遮挡。
    想放宽可以改成"至少 K 台相机可见"再按相机对打分（参考实现的
    `2d_joint_mask` / `calib_ransac.py` 走的是这条路，本模块暂不提供）。
    """
    mask = np.asarray(mask_CxNxJ)
    if mask.ndim != 3:
        raise ValueError(f"mask 应为 (C,N,J)，实际 {mask.shape}")
    return np.min(mask, axis=0).astype(bool)


def oriented_points(p3d_CxNxJx3: np.ndarray, mask_CxNxJ: np.ndarray, bones,
                    normalize: bool = True, eps: float = 1e-8) -> np.ndarray:
    """每台相机的 3D 关节 -> **骨方向单位向量**（oriented points），返回 (C, M, 3)。

    这是 RA-L2022 最关键的一步设计：**丢弃骨头的长度、只留方向**。
    原因是单目提升器的深度不可靠（尺度不可观测），但"这根骨头的朝向"相对可靠；
    而方向本身就是共轭量——世界系里同一根骨头在所有相机眼中方向相同，
    只差各自的 R_c，于是可以直接 SVD。

    参数
    ----
    p3d_CxNxJx3 : 每台相机各自的相机系 3D（根相对/尺度任意都行，只用到方向）
    mask_CxNxJ  : True = 该相机该帧该关节可信
    bones       : [(i,j), ...] 骨连接；建议用 `skeleton['calib_bones']`
                  （猪的 14 根"大骨"，剔除了眼/耳这类只有几像素、方向噪声极大的小骨）

    返回
    ----
    v : (C, M, 3)，M = 候选骨数中**在所有相机都有效**的那些；
        每行是单位向量（normalize=True 时）。
    """
    p3d = np.asarray(p3d_CxNxJx3, np.float64)
    mask = np.asarray(mask_CxNxJ, bool)
    if p3d.ndim != 4 or p3d.shape[3] != 3:
        raise ValueError(f"p3d 应为 (C,N,J,3)，实际 {p3d.shape}")
    if mask.shape != p3d.shape[:3]:
        raise ValueError(f"mask 应为 {p3d.shape[:3]}，实际 {mask.shape}")

    C, N, J, _ = p3d.shape
    b = bones_to_index_array(bones)

    # 无效关节置 NaN，NaN 会顺着减法污染整根骨头
    p = np.where(mask[..., None], p3d, np.nan)
    pairs = p[:, :, b, :]                              # (C,N,B,2,3)
    dirs = pairs[:, :, :, 1, :] - pairs[:, :, :, 0, :]  # (C,N,B,3)  e1 - e0
    dirs = dirs.reshape(C, N * b.shape[0], 3)

    nrm = np.linalg.norm(dirs, axis=2)                 # (C, M0)
    # 所有相机都有效 + 骨长非零（零长骨头方向无定义）
    ok = ~np.isnan(nrm).any(axis=0) & (nrm.min(axis=0) > eps)
    dirs = dirs[:, ok, :]

    if normalize:
        nn = np.linalg.norm(dirs, axis=2, keepdims=True)
        dirs = dirs / np.maximum(nn, eps)
    return dirs


def normalized_rays(p2d_CxNxJx2: np.ndarray, K, mask_CxNxJ: np.ndarray,
                    joints: Optional[Sequence[int]] = None
                    ) -> np.ndarray:
    """像素 2D -> **按内参归一化**的齐次射线 (C, M, 3)，第三分量恒为 1。

    等价于参考实现里 `n = (u,v,1) @ inv(K).T`（见 `calib_linear.main_linear`）。
    归一化后视线方向就是 (u,v,1) 本身，于是位置矩阵可以当成 K = I 用——
    这也是为什么线性解里出现的是 `pycalib.calib.triangulate(n, [R|t])`。

    参数
    ----
    joints : 只取这些关节（默认全部）。参考实现虽然把 `OP_KEY_SUB` 传了进来但函数体
             里没用，实际用了全部关节；本函数默认与参考一致（全部），
             需要复现"只用骨端点"的行为时显式传 `joints`。

    注意
    ----
    `mask_CxNxJ` 建议传 `visible_from_all(...)` 的结果，保证 M 个点在**所有**相机
    都有射线，否则共面约束的配对会错位。
    """
    p2d = np.asarray(p2d_CxNxJx2, np.float64)
    mask = np.asarray(mask_CxNxJ, bool)
    if p2d.ndim != 4 or p2d.shape[3] != 2:
        raise ValueError(f"p2d 应为 (C,N,J,2)，实际 {p2d.shape}")
    if mask.shape != p2d.shape[:3]:
        raise ValueError(f"mask 应为 {p2d.shape[:3]}，实际 {mask.shape}")

    C = p2d.shape[0]
    K = _as_K(C, K)
    if joints is not None:
        p2d = p2d[:, :, list(joints), :]
        mask = mask[:, :, list(joints)]

    p = np.where(mask[..., None], p2d, np.nan)
    p = p.reshape(C, -1, 2)
    ok = ~np.isnan(p).any(axis=(0, 2))
    p = p[:, ok, :]                                     # (C, M, 2)

    Kinv = np.linalg.inv(K)                             # (C,3,3)
    # 先升到齐次 (u,v,1)，再用 K^-1 作用（行向量右乘 Kinv.T）；
    # 标准内参下结果就是 ((u-cx)/fx, (v-cy)/fy, 1)
    ph = np.concatenate([p, np.ones((C, p.shape[1], 1))], axis=2)
    return ph @ np.transpose(Kinv, (0, 2, 1))           # (C,M,3)


# =====================================================================
# 2. 旋转：SVD 找共同世界方向
# =====================================================================
def solve_rotation(v_CxMx3: np.ndarray, orthonormalize: bool = False) -> np.ndarray:
    """oriented points -> 各相机的 world→camera 旋转，返回 (C,3,3)（gauge: R_0 = I）。

    参考实现 `calib_linear.calib_linear` 的旋转部分：

        v_Nx3C = np.hstack(v_CxNx3)          # (N, 3C)
        Y, D, Zt = np.linalg.svd(v_Nx3C)
        R_all = np.sqrt(C) * Zt[:3, :]       # (3, 3C)，列块 = 各相机的 R_c
        Rx = np.linalg.inv(R_all[:, :3]); R_all = Rx @ R_all
        R_w2c_list = R_all.T.reshape((-1, 3, 3))

    原理：世界系里所有相机的同一根骨头方向相同，于是 v_c = R_c v_world。
    把 v_world（3 个未知方向）当成主成分，各相机的 v_c 张成的 3 维子空间
    正是 v_world 的旋转副本；SVD 的前 3 个右奇异向量给出它们的基。
    `sqrt(C)` 是行数的归一化（两个尺度因子互相抵消，改为 I 后无影响）。

    gauge：整体世界旋转不可观测，所以强制 R_0 = I —— 之后**只有相对旋转**有意义。

    参数
    ----
    orthonormalize : True 时把每个 R 投影到最近的 SO(3)（SVD 正交化，保手性）。
                     参考实现不投影；数据干净时 R 本来就正交，两者完全一致。
                     有噪声时投影能防止 det<0 / 非正交带来的数值漂移，
                     但会轻微破坏"R_0 恰好是 I"。默认 False = 与参考一致。
    """
    v = np.asarray(v_CxMx3, np.float64)
    if v.ndim != 3 or v.shape[2] != 3:
        raise ValueError(f"v 应为 (C,M,3)，实际 {v.shape}")
    C, M, _ = v.shape
    if M < 3:
        raise ValueError(f"至少需要 3 个共同骨方向才能定旋转，实际 {M}")
    if C < 2:
        raise ValueError(f"至少需要 2 台相机，实际 {C}")

    v_Nx3C = np.hstack(v)                                # (M, 3C)
    _, D, Zt = np.linalg.svd(v_Nx3C, full_matrices=False)
    if D.shape[0] < 3 or D[2] < 1e-10:
        raise ValueError(f"骨方向退化（第 3 奇异值 {D[2] if D.shape[0] > 2 else 0:.2e}），"
                         f"请检查是否只喂了一根骨头/一个方向")

    R_all = np.sqrt(C) * Zt[:3, :]                       # (3, 3C)
    # gauge: 让第 0 台相机是 I
    Rx = np.linalg.inv(R_all[:, :3])
    R_all = Rx @ R_all
    R = R_all.T.reshape((C, 3, 3))

    if orthonormalize:
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        flip = np.linalg.det(Rn) < 0
        if np.any(flip):
            U[flip, :, -1] *= -1.0
            Rn = U @ Vt
        R = Rn
    return R


# =====================================================================
# 3. 平移：共线 + 共面线性系统
# =====================================================================
def _build_C(R_Cx3x3: np.ndarray, n_CxMx3: np.ndarray):
    """构造线性约束矩阵 C（稀疏 CSR），未知量 = [X_0..X_{M-1}, t_0..t_{C-1}]（各 3 维）。

    与参考实现逐块 `lil_matrix` 赋值**数值等价**，只是改用 COO 三元组一次性装配。

    共线块（每台相机每点 3 行）：A[3(iM+j)+a, 3j+b] = ([n_ij]_x R_i)[a,b]
                                  A[3(iM+j)+a, 3M+3i+b] = [n_ij]_x[a,b]
    共面块（每对相机每点 1 行）：m = n_ia × n_ib（世界系）
                                  B[..., 3a+b] = (m R_a^T)[b]，B[..., 3b+b] = -(m R_b^T)[b]
    """
    C, M, _ = n_CxMx3.shape
    nm = _skew(n_CxMx3.reshape(-1, 3))                   # (C*M, 3, 3)
    ncols = (M + C) * 3

    rows, cols, vals = [], [], []
    j = np.arange(M)
    a3 = np.arange(3)

    # --- 共线 ---
    for i in range(C):
        blk = nm[i * M:(i + 1) * M]                      # (M,3,3) = [n_ij]_x
        B1 = np.einsum("jab,bc->jac", blk, R_Cx3x3[i])   # ([n]_x R_i)[j,a,b]
        r = (3 * (i * M + j)[:, None] + a3[None, :])     # (M,3)
        rr = np.broadcast_to(r[:, :, None], (M, 3, 3)).ravel()
        # 第 1 块：对未知量 X_j 的系数 = [n_ij]_x R_i
        rows.append(rr)
        cols.append(np.broadcast_to((3 * j[:, None] + a3[None, :])[:, None, :],
                                    (M, 3, 3)).ravel())
        vals.append(B1.ravel())
        # 第 2 块：对未知量 t_i 的系数 = [n_ij]_x（同一批行）
        rows.append(rr)
        cols.append(np.broadcast_to((3 * M + 3 * i + a3[None, :])[:, None, :],
                                    (M, 3, 3)).ravel())
        vals.append(np.broadcast_to(blk, (M, 3, 3)).ravel())

    nA = 3 * C * M

    # --- 共面 ---
    off = nA
    nB = 0
    for p, (ia, ib) in enumerate(itertools.combinations(range(C), 2)):
        m = np.cross(n_CxMx3[ia] @ R_Cx3x3[ia], n_CxMx3[ib] @ R_Cx3x3[ib])   # (M,3)
        V1 = m @ R_Cx3x3[ia].T
        V2 = -(m @ R_Cx3x3[ib].T)
        r = off + p * M + np.broadcast_to(j[:, None], (M, 3))        # (M,3)
        # 注意：共面块只作用在平移未知量上，所以列要加 3M 的偏移
        #（参考实现里 B 的局部宽度是 C*3，被放到最后 C*3 列）
        rows.append(r.ravel()); cols.append(
            np.broadcast_to((3 * M + 3 * ia + a3[None, :]), (M, 3)).ravel())
        vals.append(V1.ravel())
        rows.append(r.ravel()); cols.append(
            np.broadcast_to((3 * M + 3 * ib + a3[None, :]), (M, 3)).ravel())
        vals.append(V2.ravel())
        nB += M

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    Cmat = sp.coo_matrix((vals, (rows, cols)), shape=(nA + nB, ncols)).tocsr()
    return Cmat, nA, nB


def solve_translation(R_Cx3x3: np.ndarray, n_CxMx3: np.ndarray,
                      verbose: bool = False) -> dict:
    """共线 + 共面线性系统 -> 各相机平移（gauge: t_0 = 0, |t_1 - t_0| = 1）。

    返回 dict：
        t      : (C,3,1) 平移（t_0 恒为 0，整体尺度被归一化到 1）
        X      : (M,3)   由线性解三角化出的世界系 3D（同一个 gauge）
        sign   : ±1，z-test 定出的手性符号（已经乘进 t 和 X）
        w      : 前 6 个最小特征值（w[3]/w[4] 应当 << 1；否则退化）
        degenerate : bool，w[3]/w[4] > 1e-4

    为什么零空间是 4 维
    ------------------
    C·x = 0 的解空间含 3 个"整体平移"（所有相机一起平移，任何 3D 点不动）
    + 1 个"整体尺度"（t 和 X 一起缩放，射线几何不变）。所以只有 4 维是**真正的**解，
    第 5 个特征值应当显著大于 0。参考实现用 `w[3]/w[4] > 1e-4` 判退化。

    尺度归一化为什么是 |t_1 - t_0|
    ------------------------------
    基线长度是唯一的尺度参考；把它定为 1 后，恢复出的 3D 有真实的比例关系
    （猪的骨长 / 相机基线），所以 BA 里的骨长方差项才有意义。
    """
    R = np.asarray(R_Cx3x3, np.float64)
    n = np.asarray(n_CxMx3, np.float64)
    C, M, _ = n.shape

    Cmat, nA, nB = _build_C(R, n)
    # 参考实现：svd(A.T@A) 的后 4 列  ==  eigh(B.T@B) 的前 4 列（按升序）
    w, v = sla.eigh((Cmat.T @ Cmat).toarray(), subset_by_index=(0, 5),
                    overwrite_a=True, overwrite_b=True)
    degenerate = bool(w[3] / max(w[4], 1e-300) > 1e-4)
    if degenerate or verbose:
        print(f"[calib_ral] 线性解特征值(前6小的) = {np.array2string(w, precision=3)} "
              f"→ w[3]/w[4] = {w[3] / max(w[4], 1e-300):.3e}"
              f"{'  ⚠️ 退化：零空间不是 4 维' if degenerate else '  ✓ 零空间 4 维'}")

    k = v[:, :4]                                          # (ncols, 4)

    # --- 用 t_0 = 0 消平移 gauge ---
    # 未知量排布是 [X_0..X_{M-1}, t_0..t_{C-1}]，各 3 维。取 t_0 那 3 行组成的
    # (3,4) 子块做 SVD，其最后一个右奇异向量 vt[3] 就是"让 t_0 归零"的系数。
    t0_rows = k[M * 3: M * 3 + 3, :]                     # t_0 在未知量里的那 3 行
    _, _, vt = np.linalg.svd(t0_rows)
    t = k @ vt[3, :]
    X = t[: M * 3].reshape((-1, 3))
    t = t[M * 3:]
    # --- 用 |t_1 - t_0| = 1 消尺度 gauge ---
    s = np.linalg.norm(t[3:6])
    if s < 1e-12:
        raise ValueError("平移解退化：|t_1 - t_0| ≈ 0（相机 0 与 1 的基线不可解）")
    t = (t / s).reshape((-1, 3))
    X = X / s
    t = t.reshape((C, 3, 1))

    # --- z-test 定手性 ---
    sign, zp, zn = z_test_sign(R[0], t[0, :, 0], R[1], t[1, :, 0], n[0], n[1])
    t = sign * t
    X = sign * X

    # --- 用 R,t 重新三角化（参考实现的做法：线性 X 只用于定 gauge） ---
    P = np.concatenate([R, t], axis=2)                    # (C,3,4)
    X2 = np.stack([triangulate(n[:, i, :], P)[:3] for i in range(M)], axis=0)

    return {"t": t, "X": X2, "X_linear": X, "sign": sign,
            "w": w, "degenerate": degenerate, "z_pos": zp, "z_neg": zn}


def z_test_sign(R1, t1, R2, t2, n1, n2):
    """手性判定：分别用 +t 与 -t 三角化，看哪个解更常落在两台相机的**前方**。

    射线几何对手性是完全对称的（把所有相机搬到原点对面，重投影一模一样），
    唯一的区别就是 3D 点会跑到相机背后。所以数"z>0 的点更多"的那一侧。

    返回 (sign, z_pos_count, z_neg_count)；sign ∈ {+1, -1}。
    """
    def tri(R1, t1, R2, t2):
        # 用归一化射线（K=I）做 DLT，与 util.triangulate 同一套
        P1 = np.hstack([R1, np.asarray(t1).reshape(3, 1)])
        P2 = np.hstack([R2, np.asarray(t2).reshape(3, 1)])
        return np.stack([triangulate(np.stack([n1[i], n2[i]]), np.stack([P1, P2]))[:3]
                         for i in range(n1.shape[0])], axis=0)

    def z_count(R, t, Xw):
        Xc = Xw @ R.T + np.asarray(t).reshape(1, 3)
        return int(np.sum(Xc[:, 2] > 0))

    t1 = np.asarray(t1, np.float64).reshape(3)
    t2 = np.asarray(t2, np.float64).reshape(3)
    Xp = tri(R1, t1, R2, t2)
    Xn = tri(R1, -t1, R2, -t2)
    zp = z_count(R1, t1, Xp) + z_count(R2, t2, Xp)
    zn = z_count(R1, t1, Xn) + z_count(R2, t2, Xn)
    return (1 if zp > zn else -1), zp, zn


def triangulate(rays_Cx3: np.ndarray, P_Cx3x4: np.ndarray) -> np.ndarray:
    """多视图 DLT 三角化：归一化射线 (C,3) + 投影矩阵 (C,3,4) -> 齐次 (4,)。

    与参考实现 `util.triangulate` 一致：堆 2C×4 约束取最小奇异向量
    （用 4×4 的 `AtA` 做 `eigh` 更省）。射线第三分量为 1，故只用前两个分量。
    """
    rays = np.asarray(rays_Cx3, np.float64)
    P = np.asarray(P_Cx3x4, np.float64)
    if rays.shape[0] != P.shape[0]:
        raise ValueError(f"射线/投影矩阵数量不一致：{rays.shape[0]} vs {P.shape[0]}")

    AtA = np.zeros((4, 4))
    for i in range(rays.shape[0]):
        x = np.empty((2, 4))
        x[0] = P[i, 0, :] - rays[i, 0] * P[i, 2, :]
        x[1] = P[i, 1, :] - rays[i, 1] * P[i, 2, :]
        AtA += x.T @ x
    _, v = np.linalg.eigh(AtA)
    Xh = v[:, 0]
    if np.isclose(Xh[3], 0.0):
        return Xh
    return Xh / Xh[3]


def calib_ral(v_CxMx3: np.ndarray, n_CxMx3: np.ndarray,
              max_views: Optional[int] = None, seed: int = 0,
              orthonormalize: bool = False, verbose: bool = False) -> dict:
    """线性标定全流程：oriented points + 归一化射线 -> (R, t)。

    返回 dict：
        R, t, X, sign, w, degenerate, n_views, K_gauge（是否被归一化尺度）

    参数
    ----
    max_views : 限制参与线性解的 3D 点数（内存/时间 ∝ M²，因为要对
                ((M+C)·3)² 的稠密矩阵做 `eigh`）。M=1330（70帧×19关节）时
                约 130 MB / 十几秒；想快速试跑可以先给 400。None = 全部用。
    """
    v = np.asarray(v_CxMx3, np.float64)
    n = np.asarray(n_CxMx3, np.float64)
    if v.shape[0] != n.shape[0]:
        raise ValueError(f"相机数不一致：v {v.shape[0]} vs n {n.shape[0]}")

    if max_views is not None:
        rng = np.random.default_rng(seed)
        if v.shape[1] > max_views:
            v = v[:, rng.choice(v.shape[1], max_views, replace=False), :]
        if n.shape[1] > max_views:
            n = n[:, rng.choice(n.shape[1], max_views, replace=False), :]

    R = solve_rotation(v, orthonormalize=orthonormalize)
    out = solve_translation(R, n, verbose=verbose)
    out.update({"R": R, "n_views": int(n.shape[1]), "n_bones": int(v.shape[1]),
                "orthonormalize": bool(orthonormalize)})
    return out


def dlt_batch(nn_CxNxJx3: np.ndarray, P_Cx3x4: np.ndarray,
              vis_CxNxJ: np.ndarray) -> np.ndarray:
    """批量多视图 DLT：齐次归一化射线 (C,N,J,3) + 投影矩阵 (C,3,4) -> 世界系 (N,J,3)。

    与逐点调用 `triangulate` 完全等价，只是把每个点的 4×4 约束矩阵一次性算完。
    **关键**：约束必须对**所有可见相机求和**，否则每个视图单独解会退化
    （单视图的 DLT 没有唯一解）。

    无效点（可见视图 < 2 或齐次尺度 ≈ 0）返回 NaN，由调用方兜底。
    """
    nn = np.asarray(nn_CxNxJx3, np.float64)
    P = np.asarray(P_Cx3x4, np.float64)
    vis = np.asarray(vis_CxNxJ, bool)
    C, N, J = vis.shape
    Pb = P[:, None, None, :, :]                                       # (C,1,1,3,4)
    wm = vis[..., None].astype(np.float64)                            # (C,N,J,1)
    r0 = (Pb[..., 0, :] - nn[..., 0:1] * Pb[..., 2, :]) * wm
    r1 = (Pb[..., 1, :] - nn[..., 1:2] * Pb[..., 2, :]) * wm
    AtA = (np.einsum("...a,...b->...ab", r0, r0)
           + np.einsum("...a,...b->...ab", r1, r1)).sum(axis=0)       # (N,J,4,4)
    _, vec = np.linalg.eigh(AtA)                                      # (N,J,4,4) 升序
    Xh = vec[..., :, 0]                                               # 最小特征向量
    X = Xh[..., :3] / np.where(np.abs(Xh[..., 3:4]) < 1e-12, np.nan, Xh[..., 3:4])
    X[vis.sum(axis=0) < 2] = np.nan
    return X


# =====================================================================
# 4. 评估（全部 gauge-free：只比相对量）
# =====================================================================
def camera_centers(R_Cx3x3: np.ndarray, t_Cx3x1: np.ndarray) -> np.ndarray:
    """相机光心在世界系的坐标 C_c = -R_c^T t_c，返回 (C,3)。"""
    R = np.asarray(R_Cx3x3, np.float64)
    C = len(R)
    t = np.asarray(t_Cx3x1, np.float64).reshape(C, -1)
    if t.shape != (C, 3):
        raise ValueError(f"t 应为 (C,3) 或 (C,3,1)，实际 {np.asarray(t_Cx3x1).shape}")
    return -np.einsum("cij,ci->cj", R, t)


def _so3_angle(R: np.ndarray) -> float:
    """SO(3) 旋转角（弧度），**数值稳定**版本。

    直接用 `arccos((tr R - 1)/2)` 在 R ≈ I 时有 ~1e-8 rad 的浮点底噪
    （cos 被算成 1-2e-16，arccos 的导数为无穷，放大成 1e-8 rad ≈ 6e-7°）。
    `atan2(||skew 部分||, tr)` 形式在 R = I 时给出**精确的 0**，
    对小角度也保持 ~1e-16 的相对精度，所以评估指标不会被数值噪声污染。
    """
    R = np.asarray(R, np.float64)
    s = 0.5 * np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    c = 0.5 * (np.trace(R) - 1.0)
    return float(np.arctan2(s, c))


def _vec_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两批向量之间的夹角（弧度），同样用 atan2 以避免 0 附近的 arccos 底噪。"""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(axis=-1))


def so3_angle_deg(R) -> object:
    """SO(3) 旋转的角度（度）。`R` 可为 (3,3) 或 (C,3,3)（逐台返回数组）。

    用它而不是 `arccos((tr R-1)/2)`：后者在 R≈I 时有 ~6e-7° 的浮点底噪。
    """
    R = np.asarray(R, np.float64)
    if R.ndim == 2:
        return float(np.degrees(_so3_angle(R)))
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R 应为 (3,3) 或 (...,3,3)，实际 {R.shape}")
    return np.array([np.degrees(_so3_angle(x)) for x in R.reshape(-1, 3, 3)])


def relative_rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray,
                                ref: int = 0) -> np.ndarray:
    """逐相机旋转误差（度），**gauge-free**。

    标定出的 R 只在"相差一个**整体世界旋转**"的意义下唯一，而且这个 gauge 是
    **右乘**形式（推导见下）：
        R_est_i = R_gt_i · H            （H = R_gt_ref^{-1}，使 R_est_ref = I）
    推导：模型是 v_c = v_w R_c^T，算法恢复的世界基 v_w' 与真值 v_w 只差一个旋转
    （v_w' = v_w H，因为两者张成同一个 3 维子空间），代回去得 H R_est_c^T = R_gt_c^T，
    即 R_est_c = R_gt_c H。
    直接比 R_est 与 R_gt 会得到任意大的误差；改成比**相对旋转**（都以第 ref 台为参考）：
        ΔR_i = (R_est_i R_est_ref^T) · (R_gt_i R_gt_ref^T)^T     ← H 被消掉
    角度 = arccos((tr ΔR - 1) / 2)。参考实现 `util.eval_R` 报的是同一量的
    θ/√2 形式；本函数报真实角度（度），几何意义更直观。
    """
    R_est = np.asarray(R_est, np.float64)
    R_gt = np.asarray(R_gt, np.float64)
    errs = []
    for i in range(len(R_est)):
        dR = (R_est[i] @ R_est[ref].T) @ (R_gt[i] @ R_gt[ref].T).T
        errs.append(np.degrees(_so3_angle(dR)))
    return np.array(errs)


def translation_direction_error_deg(R_est, t_est, R_gt, t_gt,
                                    ref: int = 0) -> np.ndarray:
    """逐相机的**基线方向**误差（度），gauge-free。

    难点：估计和真值的世界系不同（差一个 gauge），所以不能在各自的世界系里直接比
    (C_i - C_ref) 的方向。做法是把基线方向**表达在第 ref 台相机的坐标系里**：

        b_i = R_ref · (C_i - C_ref)

    这个量对 gauge 完全免疫：世界旋转 H̃ 下 R → R H̃^T、C → H̃ C，于是
    b_i → R_ref H̃^T H̃ (C_i - C_ref) = b_i（H̃ 被消掉）。
    同时也对整体平移（不改变差值）与整体尺度（不改变方向）免疫。

    真正有意义的是基线方向的**朝向**：它直接决定三角化的形状；
    基线长度的绝对值不可解（尺度 gauge），所以这里只比方向。
    """
    Ce = camera_centers(R_est, t_est)
    Cg = camera_centers(R_gt, t_gt)
    Re = np.asarray(R_est, np.float64)
    Rg = np.asarray(R_gt, np.float64)
    ve = (Ce - Ce[ref]) @ Re[ref].T      # 列形式 R_ref (C_i - C_ref)
    vg = (Cg - Cg[ref]) @ Rg[ref].T
    errs = np.degrees(_vec_angle(ve, vg))
    errs[np.linalg.norm(ve, axis=1) < 1e-9] = np.nan   # 参考相机自身方向无定义
    return errs


def similarity_align(A: np.ndarray, B: np.ndarray):
    """Umeyama：求 (s, R, t) 使 s·R·A + t 最接近 B（B ≈ sRA + t）。

    返回 (s, R, t, rms)。用于把标定结果（尺度任意）对齐到真值后比较。
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError(f"A/B 形状应相同且为 (N,d)，实际 {A.shape} vs {B.shape}")

    n, d = A.shape
    muA, muB = A.mean(0), B.mean(0)
    Ac, Bc = A - muA, B - muB
    # 协方差
    Sigma = Bc.T @ Ac / n
    U, S, Vt = np.linalg.svd(Sigma)
    d_sign = np.sign(np.linalg.det(U @ Vt))
    D = np.eye(d)
    D[-1, -1] = d_sign
    R = U @ D @ Vt
    varA = (Ac ** 2).sum() / n
    s = float((S * np.diag(D)).sum() / max(varA, 1e-300))
    t = muB - s * (R @ muA)
    rms = float(np.sqrt((((s * (R @ A.T).T + t) - B) ** 2).sum(axis=1).mean()))
    return s, R, t, rms


def alignment_error(R_est, t_est, R_gt, t_gt) -> dict:
    """一套完整的 gauge-free 标定评估指标。

    * `rot_deg`        : (C,) 相对旋转误差（度）
    * `trans_dir_deg`  : (C,) 基线方向误差（度）
    * `scale`          : 估计尺度 / 真值尺度（≈1 表示基线长度恢复正确）
    * `center_rms_m`   : 相似变换对齐后的光心 RMS 误差（与真值同量纲，单位取决于真值）
    * `center_rel_err` : 对齐后的相对光心误差（除以真值基线长度，无量纲，跨数据集可比）
    """
    Ce = camera_centers(R_est, t_est)
    Cg = camera_centers(R_gt, t_gt)
    s, R_al, t_al, rms = similarity_align(Ce, Cg)
    base_gt = np.linalg.norm(Cg - Cg[0], axis=1).mean()
    return {
        "rot_deg": relative_rotation_error_deg(R_est, R_gt),
        "trans_dir_deg": translation_direction_error_deg(R_est, t_est, R_gt, t_gt),
        "scale": s,
        "center_rms_m": rms,
        "center_rel_err": rms / max(base_gt, 1e-12),
    }


# =====================================================================
# 5. BA：联合精化（重投影 + 跨相机骨方向一致 + 骨长一致）
# =====================================================================
def _to_theta(R, t, x):
    rv = _matrix_to_rotvec(R).ravel()
    return np.concatenate([rv, np.asarray(t).ravel(), np.asarray(x).ravel()])


def _from_theta(theta, C):
    rv = theta[:3 * C].reshape(C, 3)
    t = theta[3 * C:6 * C].reshape(C, 3, 1)
    x = theta[6 * C:].reshape(-1, 3)
    return _rotvec_to_matrix(rv), t, x


def _objfun_nll(K, R, t, x, y, y_mask, w):
    """重投影残差（按 2D 置信度加权）。逐相机：e = √2·w·(y − π(K(R X + t)))。

    ★ 必须把 K 带上：`y` 是**像素**坐标，而 (R X + t) 是相机系 3D。
      参考实现写的是 `y_hat = K @ (R @ x.T + t)`。漏掉 K 会让残差量纲错位
      （归一化坐标 vs 像素，差 fx≈1340 倍），BA 直接发散。
    """
    e_all = []
    for Ki, Ri, ti, mask, yi, wi in zip(K, R, t, y_mask, y, w):
        xx = x[mask]
        yy = yi[mask]
        ww = (np.sqrt(2.0) * wi[mask]).reshape(-1, 1)
        P = Ki @ (Ri @ xx.T + np.asarray(ti).reshape(3, 1))
        z = P[2:3, :]
        # 参考实现直接相除；这里把 |z|≈0 的退化射线置 0 残差，避免 inf 拖垮 least_squares。
        # （注意不能用 np.maximum 夹住负 z——那会把"点在相机背后"变成 1e12 级别的假残差）
        z = np.where(np.abs(z) < 1e-12, np.nan, z)
        yhat = P[:2, :] / z
        e_all.append(np.nan_to_num((yy - yhat.T) * ww, nan=0.0,
                                   posinf=0.0, neginf=0.0).ravel())
    return np.concatenate(e_all)


def _objfun_var3d(R, p3d_CxNxJx3, mask_CxNxJ, bone_idx):
    """跨相机骨方向一致性：把每台相机的骨方向转到世界系，要求它们方差最小。

    残差 = 1 - ||mean_c(unit(v_c R_c))||，逐骨一个数。
    这一项**只用提升器的朝向**：即使单目深度（尺度/位置）全错，方向仍可约束旋转。
    """
    C = len(R)
    vw = np.stack([p3d @ Ri for p3d, Ri in zip(p3d_CxNxJx3, R)], axis=0)   # (C,N,J,3)
    vw = np.where(mask_CxNxJ[..., None], vw, np.nan)

    bones = vw[:, :, bone_idx, :]                       # (C,N,B,2,3)
    dirs = bones[:, :, :, 0, :] - bones[:, :, :, 1, :]  # (C,N,B,3)
    dirs = dirs.reshape(C, -1, 3)
    nrm = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs / np.maximum(nrm, 1e-12)

    invalid = np.isnan(dirs).any(axis=(0, 2))           # 有些骨在所有相机都无效
    if invalid.all():
        return np.zeros(1)
    mean_dir = np.nanmean(dirs[:, ~invalid], axis=0)
    return 1.0 - np.linalg.norm(mean_dir, axis=1)


def _objfun_varbone(x_NxJx3, bone_idx):
    """骨长一致性：同一根骨头在所有帧的长度方差（刚体假设）。"""
    bone = x_NxJx3[:, bone_idx, :]                      # (N,B,2,3)
    length = np.linalg.norm(bone[:, :, 0, :] - bone[:, :, 1, :], axis=2)   # (N,B)
    return np.var(length, axis=0)


def _objfun(theta, K, sp2d, ss2d, sp3d, ss3d, bone_idx, C, N, J, lam1, lam2):
    R, t, x = _from_theta(theta, C)
    e1 = _objfun_nll(K, R, t, x, sp2d.reshape(C, N * J, 2),
                     (ss2d > 0).reshape(C, N * J), ss2d.reshape(C, N * J))
    e2 = _objfun_var3d(R, sp3d, ss3d > 0, bone_idx) * lam1
    e3 = _objfun_varbone(x.reshape(N, J, 3), bone_idx) * lam2
    return np.concatenate([e1, e2, e3])


def reprojection_errors(K, R, t, X_world, p2d_CxNxJx2, mask_CxNxJ):
    """逐相机的重投影残差（像素），返回 (C, N, J, 2)（无效处为 NaN）。

    `X_world` 为世界系 3D，可传 (N,J,3) 或 (N*J,3)。
    """
    C = len(K)
    X = np.asarray(X_world, np.float64).reshape(-1, 3)
    out = []
    for i in range(C):
        P = K[i] @ np.concatenate([R[i], np.asarray(t[i]).reshape(3, 1)], axis=1)
        q = X @ P[:3, :3].T + P[:3, 3]
        z = q[:, 2:3]
        z = np.where(np.abs(z) < 1e-12, np.nan, z)
        q = q[:, :2] / z
        e = q - p2d_CxNxJx2[i].reshape(-1, 2)
        e = e.reshape(p2d_CxNxJx2.shape[1], p2d_CxNxJx2.shape[2], 2)
        out.append(np.where(mask_CxNxJ[i][..., None], e, np.nan))
    return np.stack(out, axis=0)


def ba_refine(R_Cx3x3, t_Cx3x1, p2d_CxNxJx2, w2d_CxNxJ, p3d_CxNxJx3, w3d_CxNxJ,
              K, bones, lambda1=0.01, lambda2=0.01,
              max_nfev=200, verbose=False) -> dict:
    """BA 联合精化：返回 {"R", "t", "X", "cost", "nfev", "status"}。

    参数
    ----
    p2d / w2d : 像素 2D 与置信度（0/1 或 float，作权重）
    p3d / w3d : **每台相机各自的** 3D（提升器输出；只用到骨方向）+ 有效性
    lambda1   : 跨相机骨方向一致性权重
    lambda2   : 三角化 3D 骨长方差权重
    K         : (3,3) 或 (C,3,3)

    注意事项
    --------
    * `X`（世界系 3D）也在优化变量里，初值由 2D 三角化给出；
      自由度很大（N·J·3 维），所以 `max_nfev` 别开太小，或减少参与帧数。
    * 三项残差的量纲完全不同（像素 / 无量纲 / 长度²），所以 lambda 必须调。
      参考实现默认 1.0；我们默认 0.01，因为 BamaPig3D 的基线尺度（米）下
      骨长方差残差数值偏大。**这两个数应当在服务器上做一次小网格搜索**
      （见 `docs/ral_backend.md` 的调参表）。
    * gauge：BA 会保持 R_0 ≈ I 与 |t_1 - t_0| ≈ 1 的近似（因为这三项残差
      对整体平移/旋转/尺度不变，初始值定了就不会漂远），但会略微漂移；
      评估仍然只用 gauge-free 指标。
    """
    R = np.asarray(R_Cx3x3, np.float64)
    t = np.asarray(t_Cx3x1, np.float64).reshape(len(R), 3)
    C = len(R)
    N, J = p2d_CxNxJx2.shape[1], p2d_CxNxJx2.shape[2]
    K = _as_K(C, K)
    bone_idx = bones_to_index_array(bones)

    # 初值：用线性解的 R,t 把 2D 三角化到世界系（批量 DLT，允许"只被部分相机看到"）
    P = np.concatenate([R, t[:, :, None]], axis=2)                    # (C,3,4)
    Kinv = np.linalg.inv(K)                                           # (C,3,3)
    ph = np.concatenate([p2d_CxNxJx2, np.ones(p2d_CxNxJx2.shape[:3] + (1,))], axis=3)
    nn = np.stack([ph[c] @ Kinv[c].T for c in range(C)], axis=0)      # (C,N,J,3) 归一化
    vis = (np.asarray(w2d_CxNxJ, np.float64) > 0)
    Xi = dlt_batch(nn, P, vis)                                       # (N,J,3)
    X = Xi.reshape(-1, 3)
    # 不可解/数值异常的点：用有效点的均值兜底（比置 0 安全——0 可能正好是退化位置）
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        good = ~bad
        X[bad] = X[good].mean(axis=0) if good.any() else 0.0

    theta0 = _to_theta(R, t, X)
    res = least_squares(
        _objfun, theta0, method="trf", ftol=1e-4, max_nfev=max_nfev,
        verbose=2 if verbose else 0,
        args=(K, p2d_CxNxJx2, w2d_CxNxJ, p3d_CxNxJx3, w3d_CxNxJ, bone_idx, C, N, J,
              lambda1, lambda2),
    )
    R_opt, t_opt, X_opt = _from_theta(res.x, C)
    return {"R": R_opt, "t": t_opt, "X": X_opt.reshape(N, J, 3),
            "cost": float(res.cost), "nfev": int(res.nfev),
            "status": int(res.status), "message": str(res.message)}


# =====================================================================
# 6. 一键：从提升器输出直接标定
# =====================================================================
def calibrate_from_lifter(p3d_CxNxJx3: np.ndarray, p2d_CxNxJx2: np.ndarray, K,
                          w2d_CxNxJ: np.ndarray, bones=None, root: int = 18,
                          refine: bool = False, lambda1: float = 0.01,
                          lambda2: float = 0.01, max_views: Optional[int] = None,
                          visible_all: bool = True, verbose: bool = False) -> dict:
    """端到端：单目 3D（每相机各一份）+ 2D + 内参 -> 外参 (R, t)。

    参数
    ----
    p3d_CxNxJx3 : 提升器给出的**相机系** 3D（根相对即可；尺度任意）
    p2d_CxNxJx2 : 像素 2D（用于平移解的射线）
    K           : newcameramtx（BamaPig3D 的 label_images 已去畸变！不要用原始 K）
    w2d_CxNxJ   : 2D 有效性/置信度
    bones       : 默认取 `skeleton['calib_bones']`
    refine      : 是否跑 BA（线性解足够好时可以先不开）

    返回线性解 + （可选）BA 解：{"R","t","X_lin","R_ba","t_ba","lin","ba","n_views",...}
    """
    from .skeleton import get_skeleton
    sk = get_skeleton("pig19")
    if bones is None:
        bones = sk["calib_bones"]
    if root is None:
        root = sk["root"]

    w2d = np.asarray(w2d_CxNxJ, bool)
    vis = visible_from_all(w2d) if visible_all else np.ones(w2d.shape[1:], bool)
    vis_3d = np.broadcast_to(vis, w2d.shape)             # 全相机可见 -> 各相机一致
    v = oriented_points(p3d_CxNxJx3, vis_3d, bones)
    n = normalized_rays(p2d_CxNxJx2, K, vis_3d)
    lin = calib_ral(v, n, max_views=max_views, verbose=verbose)

    out = {"R": lin["R"], "t": lin["t"], "X_lin": lin["X"],
           "lin": lin, "n_views": lin["n_views"], "n_bones": lin["n_bones"],
           "bones": list(bones), "root": int(root)}

    if refine:
        # BA 的 3D 项需要"每台相机各自的 3D"，所以这里用提升器输出本体
        ba = ba_refine(lin["R"], lin["t"], p2d_CxNxJx2, np.asarray(w2d, np.float64),
                       p3d_CxNxJx3, np.asarray(w2d, np.float64), K, bones,
                       lambda1=lambda1, lambda2=lambda2, verbose=verbose)
        out.update({"R_ba": ba["R"], "t_ba": ba["t"], "X_ba": ba["X"], "ba": ba})
    return out
