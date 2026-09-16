#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RA-L2022 标定后端自检（**纯 numpy/scipy，不含任何网络/训练**，秒级完成）。

它做什么
--------
用合成真值（已知 R_gt/t_gt/3D/2D）打通 `steerpose.calib_ral` 的每一环，
把"数值正确性"和"接口正确性"分开查：

  A. 基础工具        _skew / visible_from_all / oriented_points / normalized_rays
  B. 与参考实现对照  自研 COO 装配的约束矩阵 vs `reference/` 里 lil 逐块赋值的写法
  C. 线性解精度      4 相机 / 2 相机 / 加噪 3D / 加噪 2D 四种设置下的旋转+平移误差
  D. gauge 不变性    给真值整体左乘世界旋转 G，相对误差必须一模一样
  E. 手性            把 t 取负后 z-test 必须翻号
  F. 三角化         干净数据下 DLT 应精确复现真值 3D
  G. BA             开 refine 后误差不劣化（残差应当下降）
  H. 端到端          calibrate_from_lifter 的完整返回结构

为什么可以在这里跑
------------------
本脚本只做矩阵运算，不加载权重、不训练、不推理。它验证的是**几何求解器**，
与"H100 上跑模型"完全无关。

用法（在服务器或本地任意环境）
------------------------------
    python scripts/selftest_calib_ral.py
    python scripts/selftest_calib_ral.py --noise-3d 0.02 --noise-2d 2.0 --seed 1
退出码 0 = 全部通过；非 0 = 有失败项。
"""
import argparse
import itertools
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.calib_ral import (                                          # noqa: E402
    _build_C, _skew, alignment_error, calib_ral, calibrate_from_lifter,
    camera_centers, dlt_batch, normalized_rays, oriented_points,
    relative_rotation_error_deg, similarity_align, solve_rotation,
    solve_translation, triangulate, translation_direction_error_deg,
    visible_from_all, z_test_sign, ba_refine,
)
from steerpose.skeleton import get_skeleton, bones_to_index_array            # noqa: E402

# ---------------------------------------------------------------------------
# 结果收集
# ---------------------------------------------------------------------------
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def note(name, detail=""):
    """信息项：只打印，不计入通过/失败（用于"加了噪声后本来就该偏离"的检查）。"""
    print(f"  [INFO] {name}" + (f"   {detail}" if detail else ""))


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """构造 world->camera 的 R, t，满足 x_cam = R @ X_world + t，相机朝 +Z 看。

    R 的三行就是相机 x/y/z 轴在世界系里的方向（因为 z_cam = R[2,:]·X + t[2]）。
    """
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f = f / np.linalg.norm(f)
    up = np.asarray(up, float)
    x = np.cross(up, f)
    if np.linalg.norm(x) < 1e-8:            # 视线与 up 平行，换个辅助轴
        x = np.cross(np.array([0.0, 1.0, 0.0]), f)
    x = x / np.linalg.norm(x)
    y = np.cross(f, x)
    R = np.stack([x, y, f], axis=0)
    t = -R @ eye
    return R, t


def make_scene(J=19, N=24, C=4, radius=2.5, scale=0.5, seed=0,
               noise_3d=0.0, noise_2d=0.0, image=(1920, 1200)):
    """合成一个"猪在相机阵列中心扭动"的场景。

    返回 dict：X_gt (N,J,3) 世界系真值；R_gt/t_gt；p2d (C,N,J,2) 像素；
    K；per-camera 3D（模拟提升器输出）。
    """
    rng = np.random.default_rng(seed)
    W, H = image
    K = np.array([[1340.0, 0, W / 2], [0, 1340.0, H / 2], [0, 0, 1.0]])

    # --- 真值 3D：让方向充分变化（旋转 SVD 需要秩 3）---
    base = rng.normal(0, 1, (J, 3))
    base -= base.mean(0)
    base = base / np.linalg.norm(base, axis=1, keepdims=True) * 0.15   # 关节相对根的位置
    X = np.empty((N, J, 3))
    axis = rng.normal(size=(N, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ang = rng.uniform(-0.6, 0.6, N)
    for i in range(N):
        a = axis[i]
        Kmat = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        Rr = np.eye(3) + np.sin(ang[i]) * Kmat + (1 - np.cos(ang[i])) * (Kmat @ Kmat)
        X[i] = base @ Rr.T * scale / 0.15
    X += rng.normal(0, 0.05, (N, 1, 3))                                # 每帧整体轻微移动

    # --- 相机 ---
    dirs = rng.normal(size=(C, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    R_gt = np.empty((C, 3, 3))
    t_gt = np.empty((C, 3, 1))
    for c in range(C):
        R_gt[c], tc = look_at(dirs[c] * radius, np.zeros(3))
        t_gt[c] = tc.reshape(3, 1)

    # --- 投影 ---
    p2d = np.empty((C, N, J, 2))
    p3d_cam = np.empty((C, N, J, 3))
    for c in range(C):
        Xc = X @ R_gt[c].T + t_gt[c].reshape(1, 1, 3)
        p3d_cam[c] = Xc
        q = Xc @ K.T
        z = q[..., 2:3]
        p2d[c] = q[..., :2] / np.maximum(np.abs(z), 1e-9) * np.sign(z)

    if noise_2d > 0:
        p2d = p2d + rng.normal(0, noise_2d, p2d.shape)
    if noise_3d > 0:
        # 模拟单目提升器：位置有噪声 + 每台相机各自的深度尺度误差
        p3d_cam = p3d_cam + rng.normal(0, noise_3d, p3d_cam.shape)
        p3d_cam = p3d_cam * rng.uniform(0.9, 1.1, (C, 1, 1, 1))

    z_min = p3d_cam[..., 2].min()
    return {"X_gt": X, "R_gt": R_gt, "t_gt": t_gt, "p2d": p2d, "p3d": p3d_cam,
            "K": K, "in_front": z_min > 0, "z_min": float(z_min)}


# ---------------------------------------------------------------------------
# 参考实现（lil 逐块赋值），用于对拍
# ---------------------------------------------------------------------------
def _build_C_reference(R, n):
    """`reference/calib_from_moving_person/calib_linear.py` 的原写法。"""
    C, M, _ = n.shape
    A = []
    for idx_t in range(C):
        for idx_v in range(M):
            nv = n[idx_t, idx_v]
            nmat = np.array([[0, -nv[2], nv[1]],
                             [nv[2], 0, -nv[0]],
                             [-nv[1], nv[0], 0]], float)
            t0 = M * 3
            a = sp.lil_matrix((3, (M + C) * 3))
            a[:, idx_v * 3: idx_v * 3 + 3] = nmat @ R[idx_t]
            a[:, t0 + idx_t * 3: t0 + idx_t * 3 + 3] = nmat
            A.append(a)
    A = sp.vstack(A)

    B = []
    for (ia, Ra, na), (ib, Rb, nb) in itertools.combinations(
            list(zip(range(C), R, n)), 2):
        m = np.cross(na @ Ra, nb @ Rb)
        blk = sp.lil_matrix((M, C * 3))
        blk[:, ia * 3: ia * 3 + 3] = m @ Ra.T
        blk[:, ib * 3: ib * 3 + 3] = -m @ Rb.T
        B.append(blk)
    B = sp.vstack(B)

    Cm = sp.lil_matrix((A.shape[0] + B.shape[0], A.shape[1]), dtype=np.float64)
    Cm[:A.shape[0]] = A
    Cm[A.shape[0]:, -B.shape[1]:] = B
    return Cm.tocsr(), A.shape[0], B.shape[0]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RA-L2022 标定后端自检（无模型）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=24, help="合成场景的帧数")
    ap.add_argument("--noise-3d", type=float, default=0.0, help="提升器 3D 噪声（米）")
    ap.add_argument("--noise-2d", type=float, default=0.0, help="2D 噪声（像素）")
    ap.add_argument("--rot-tol", type=float, default=0.5, help="旋转误差容差（度）")
    ap.add_argument("--trans-tol", type=float, default=1.5, help="基线方向误差容差（度）")
    ap.add_argument("--run-ba", action="store_true", default=True,
                    help="是否测 BA（默认测；--no-ba 关闭）")
    ap.add_argument("--no-ba", dest="run_ba", action="store_false")
    args = ap.parse_args()

    sk = get_skeleton("pig19")
    bones = sk["calib_bones"]
    root = sk["root"]
    clean = (args.noise_3d == 0.0 and args.noise_2d == 0.0)
    # 有噪时把精度容差放宽（噪声本来就会带来误差；这里只检查"没跑飞"）
    rot_tol = args.rot_tol if clean else max(args.rot_tol, 15.0)
    trans_tol = args.trans_tol if clean else max(args.trans_tol, 15.0)

    print("=" * 72)
    print("RA-L2022 标定后端自检    (纯 numpy/scipy，无模型、无训练)")
    print(f"骨架: pig19 ({sk['num_joints']} 关节 / {len(bones)} 标定骨 / root={root})")
    print(f"合成设置: {args.frames} 帧, 3D噪声={args.noise_3d}m, 2D噪声={args.noise_2d}px, "
          f"seed={args.seed}")
    print("=" * 72)

    # ---------------- A. 基础工具 ----------------
    section("A. 基础工具")
    n_rand = np.array([[0.3, -0.4, 0.5], [1.0, 0.0, 0.0]])
    S = _skew(n_rand)
    x = np.array([0.7, 0.2, -0.3])
    check("_skew: [n]_x @ x == n × x",
          np.allclose(S[0] @ x, np.cross(n_rand[0], x)) and
          np.allclose(S[1] @ x, np.cross(n_rand[1], x)),
          f"max err={np.abs(S[0] @ x - np.cross(n_rand[0], x)).max():.2e}")
    check("_skew: 反对称",
          np.allclose(S, -np.transpose(S, (0, 2, 1))))

    m = np.array([[[True, True], [False, True]],
                  [[True, False], [True, True]]])
    check("visible_from_all: 取逐关节最小",
          np.array_equal(visible_from_all(m), [[True, False], [False, True]]))

    rng = np.random.default_rng(0)
    p3d = rng.normal(0, 0.3, (3, 5, 19, 3))
    mask = np.ones((3, 5, 19), bool)
    mask[0, 1, 5] = False              # 关节 5（在 calib_bones 里）在相机 0 的第 1 帧不可见
    v = oriented_points(p3d, mask, bones)
    b = bones_to_index_array(bones)
    check("oriented_points: 全部单位长度",
          np.allclose(np.linalg.norm(v, axis=2), 1.0),
          f"shape={v.shape}")
    ref_dir = p3d[0, 0, b[0, 1]] - p3d[0, 0, b[0, 0]]
    check("oriented_points: 与手算方向一致",
          np.allclose(v[0, 0], ref_dir / np.linalg.norm(ref_dir)),
          f"骨 {b[0]} 第 0 帧")
    # 5 帧 × 14 骨 = 70；关节 5 只出现在骨 (18,5) 与 (5,7) 中 -> 恰好少 2 根
    n_expected = 5 * len(bones) - 2
    check("oriented_points: 被遮挡关节波及的骨被剔除（且只剔除这些）",
          v.shape[1] == n_expected,
          f"M={v.shape[1]}，期望 {n_expected}（5×{len(bones)} − 2）")

    K = np.array([[1340.0, 0, 960.0], [0, 1340.0, 600.0], [0, 0, 1.0]])
    px = np.array([[[1300.0, 500.0], [1000.0, 400.0]]], float)         # (1,2)
    p2d = np.broadcast_to(px, (2, 1, 2, 2)).copy()
    mm_all = np.ones((2, 1, 2), bool)
    rays = normalized_rays(p2d, K, mm_all)
    check("normalized_rays: 齐次坐标 + 手算归一化",
          rays.shape == (2, 2, 3) and
          np.allclose(rays[0, 0], [(1300 - 960) / 1340, (500 - 600) / 1340, 1.0]) and
          np.allclose(rays[1, 1], [(1000 - 960) / 1340, (400 - 600) / 1340, 1.0]) and
          np.allclose(rays[..., 2], 1.0),
          f"shape={rays.shape}")
    mm_drop = mm_all.copy()
    mm_drop[0, 0, 1] = False
    check("normalized_rays: 不在所有相机可见的点被剔除（M: 2 → 1）",
          normalized_rays(p2d, K, mm_drop).shape == (2, 1, 3))

    # ---------------- B. 与参考实现对拍 ----------------
    section("B. 约束矩阵 vs 参考实现（lil 逐块赋值）")
    rng = np.random.default_rng(1)
    Cs, Ms = 4, 30
    Rr = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(Cs)])
    for i in range(Cs):
        if np.linalg.det(Rr[i]) < 0:
            Rr[i][:, 0] *= -1
    nn = np.concatenate([rng.normal(size=(Cs, Ms, 2)), np.ones((Cs, Ms, 1))], axis=2)
    C1, nA1, nB1 = _build_C(Rr, nn)
    C2, nA2, nB2 = _build_C_reference(Rr, nn)
    check("行/列数一致",
          C1.shape == C2.shape and nA1 == nA2 and nB1 == nB2,
          f"{C1.shape} vs {C2.shape}, nA={nA1}/{nA2}, nB={nB1}/{nB2}")
    check("矩阵元素逐一相等（COO 装配 == lil 装配）",
          np.allclose(C1.toarray(), C2.toarray(), atol=1e-12),
          f"max|Δ|={np.abs(C1.toarray() - C2.toarray()).max():.3e}")

    # ---------------- C. 线性解精度 ----------------
    section("C. 线性解精度")
    scene = make_scene(J=19, N=args.frames, C=4, seed=args.seed,
                       noise_3d=args.noise_3d, noise_2d=args.noise_2d)
    check("合成场景合法（所有点都在相机前方）", scene["in_front"],
          f"min z = {scene['z_min']:.3f} m")
    K = scene["K"]
    C = scene["R_gt"].shape[0]
    w2d = np.ones((C, args.frames, 19))
    v = oriented_points(scene["p3d"], np.broadcast_to(w2d > 0, scene["p3d"].shape[:3]), bones)
    n = normalized_rays(scene["p2d"], K, np.broadcast_to(w2d > 0, scene["p3d"].shape[:3]))
    K4 = np.broadcast_to(K, (C, 3, 3)).copy()
    Cg_all = camera_centers(scene["R_gt"], scene["t_gt"])
    # 尺度 gauge 由 |t_1 - t_0| = 1 固定，而 |t_1 - t_0| = |C_1 - C_0|（见 calib_ral 文档），
    # 所以对齐尺度 s 应当 ≈ 相机 0 与 1 的**真值基线长**，而不是所有基线的均值。
    base01_gt = float(np.linalg.norm(Cg_all[1] - Cg_all[0]))

    out4 = calib_ral(v, n, verbose=True)
    err = alignment_error(out4["R"], out4["t"], scene["R_gt"], scene["t_gt"])
    check("4 相机：旋转误差 < 容差",
          np.nanmax(err["rot_deg"][1:]) < rot_tol,
          f"max={np.nanmax(err['rot_deg'][1:]):.4f}° (除参考相机) 逐台="
          f"{np.array2string(err['rot_deg'], precision=3)}")
    check("4 相机：基线方向误差 < 容差",
          np.nanmax(err["trans_dir_deg"]) < trans_tol,
          f"max={np.nanmax(err['trans_dir_deg']):.4f}°")
    check("4 相机：尺度恢复 ≈ 真值基线 |C1-C0|（t 被归一化为 1）",
          abs(err["scale"] - base01_gt) / base01_gt < 0.02,
          f"s={err['scale']:.5f} vs |C1−C0|={base01_gt:.5f} m")
    tol_center = 0.01 if clean else 0.05
    check("4 相机：对齐后光心相对误差", err["center_rel_err"] < tol_center,
          f"{err['center_rel_err'] * 100:.4f}%（容差 {tol_center * 100:.0f}%）")
    # 零空间 4 维 / 相似变换 这两条只在**无噪**时有意义：
    # 加了 3D 噪声后线性解本来就是有偏的，退化告警（w[3]/w[4] > 1e-4）出现是预期行为，
    # 这正是参考实现要再加 BA / RANSAC 的原因。
    if clean:
        check("4 相机：零空间 4 维（未退化）", not out4["degenerate"],
              f"w[3]/w[4]={out4['w'][3] / out4['w'][4]:.2e}")
    else:
        note("4 相机：零空间退化告警", 
             f"w[3]/w[4]={out4['w'][3] / out4['w'][4]:.2e}"
             f"（有噪时为预期；参考实现同款阈值 1e-4）")

    # ★ 最有信息量的一个检查：三角化出的 3D 必须是真值 3D 的**相似变换**
    #   （旋转 + 尺度 + 平移）。这一条同时覆盖了旋转、平移、三角化三环。
    X_est = out4["X"].reshape(-1, 3)
    Xgt_flat = scene["X_gt"].reshape(-1, 3)
    s_x, R_x, t_x, rms_x = similarity_align(X_est, Xgt_flat)
    span_gt = np.linalg.norm(Xgt_flat - Xgt_flat.mean(0), axis=1).mean()
    tol_rel = 1e-9 if clean else 0.15
    check("4 相机：三角化 3D 是 GT 3D 的相似变换",
          rms_x / span_gt < tol_rel,
          f"相对 rms={rms_x / span_gt:.2e}（容差 {tol_rel:g}）")
    check("4 相机：λ（骨长尺度）与 1/|C1-C0| 一致",
          abs(s_x - base01_gt) / base01_gt < 0.05,
          f"1/λ={s_x:.5f} vs |C1−C0|={base01_gt:.5f}")

    # 2 相机
    sc2 = make_scene(J=19, N=args.frames, C=2, seed=args.seed + 7)
    m2 = np.ones(sc2["p3d"].shape[:3], bool)
    v2 = oriented_points(sc2["p3d"], m2, bones)
    n2 = normalized_rays(sc2["p2d"], K, m2)
    out2 = calib_ral(v2, n2)
    e2 = alignment_error(out2["R"], out2["t"], sc2["R_gt"], sc2["t_gt"])
    check("2 相机：旋转误差 < 容差", e2["rot_deg"][1] < rot_tol,
          f"{e2['rot_deg'][1]:.4f}°")
    check("2 相机：基线方向误差 < 容差", e2["trans_dir_deg"][1] < trans_tol,
          f"{e2['trans_dir_deg'][1]:.4f}°")
    check("2 相机：零空间 4 维", not out2["degenerate"],
          f"w[3]/w[4]={out2['w'][3] / out2['w'][4]:.2e}")

    # ---------------- D. gauge 不变性 ----------------
    section("D. gauge 不变性（整体世界旋转不应改变任何相对量）")
    G = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]
    if np.linalg.det(G) < 0:
        G[:, 0] *= -1
    err_rot_G = relative_rotation_error_deg(out4["R"] @ G, scene["R_gt"] @ G)
    same_G = np.allclose(err_rot_G, relative_rotation_error_deg(out4["R"], scene["R_gt"]),
                         atol=1e-4)
    check("相对旋转误差对整体世界旋转不变", same_G,
          f"max Δ={np.abs(err_rot_G - relative_rotation_error_deg(out4['R'], scene['R_gt'])).max():.2e}")
    err_t_G = translation_direction_error_deg(out4["R"] @ G, out4["t"], scene["R_gt"] @ G,
                                              scene["t_gt"])
    check("基线方向误差对整体世界旋转不变",
          np.allclose(err_t_G, err["trans_dir_deg"], atol=1e-3, equal_nan=True),
          f"max Δ={np.nanmax(np.abs(err_t_G - err['trans_dir_deg'])):.2e}°")
    # 整体尺度不变（乘 3.7 后 t 与 X 一起缩放）
    check("光心对整体缩放不变（方向量）",
          np.allclose(translation_direction_error_deg(out4["R"], 3.7 * out4["t"],
                                                      scene["R_gt"], scene["t_gt"]),
                      err["trans_dir_deg"], atol=1e-6, equal_nan=True))
    check("alignment_error：t 整体乘 3.7 不改变任何方向读数",
          np.allclose(alignment_error(out4["R"], 3.7 * out4["t"], scene["R_gt"],
                                      scene["t_gt"])["trans_dir_deg"],
                      err["trans_dir_deg"], atol=1e-6, equal_nan=True))
    rng_s = np.random.default_rng(3)
    A_s = rng_s.normal(size=(8, 3))
    R_true = np.linalg.qr(rng_s.normal(size=(3, 3)))[0]
    if np.linalg.det(R_true) < 0:
        R_true[:, 0] *= -1
    s_true, t_true = 2.5, np.array([1.0, -2.0, 0.5])
    B_s = s_true * (A_s @ R_true.T) + t_true
    s, R_al, t_al, rms = similarity_align(A_s, B_s)
    check("similarity_align: 已知 (s,R,t) 精确恢复",
          abs(s - s_true) < 1e-9 and np.allclose(R_al, R_true, atol=1e-9)
          and np.allclose(t_al, t_true, atol=1e-9) and rms < 1e-9,
          f"s={s:.9f} (真值 {s_true}), rms={rms:.2e}")

    # ---------------- D2. 评估指标自身的正确性（用 GT 做参照：与噪声无关的精确测试）----
    section("D2. 评估指标自检（注入已知误差）")
    rng_e = np.random.default_rng(11)

    def _rot(axis, deg):
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        t = np.radians(deg)
        return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * (Kx @ Kx)

    Rg, tg = scene["R_gt"], scene["t_gt"]
    Cg = camera_centers(Rg, tg)
    check("旋转指标：自己比自己 = 0（精确 0，无 arccos 底噪）",
          np.allclose(relative_rotation_error_deg(Rg, Rg), 0, atol=1e-12),
          f"max={relative_rotation_error_deg(Rg, Rg).max():.1e}°")
    H = np.linalg.qr(rng_e.normal(size=(3, 3)))[0]
    if np.linalg.det(H) < 0:
        H[:, 0] *= -1
    check("旋转指标：整体右乘 H（纯 gauge）读数仍为 0",
          np.allclose(relative_rotation_error_deg(Rg @ H, Rg), 0, atol=1e-12),
          f"max={relative_rotation_error_deg(Rg @ H, Rg).max():.2e}°")

    for i_inj, deg in [(1, 1.234), (2, 7.5)]:
        dR = _rot([0.3, -0.5, 0.8], deg)
        R_bad = Rg.copy()
        R_bad[i_inj] = dR @ Rg[i_inj]          # 左乘 = 直接拧第 i 台的相对旋转
        e_bad = relative_rotation_error_deg(R_bad, Rg)
        others = [e_bad[j] for j in range(1, len(R_bad)) if j != i_inj]
        check(f"旋转指标：注入 {deg}° 到相机 {i_inj} 能**精确**读出",
              abs(e_bad[i_inj] - deg) < 1e-9 and max(others) < 1e-9,
              f"读出 {e_bad[i_inj]:.9f}°（期望 {deg}），其余最大 {max(others):.2e}°")

    # 平移指标：在**相机 0 坐标系**里把相机 2 的基线方向拧 5°（必须绕 ⊥ 轴，否则读数≠5°）
    vg0 = (Cg[2] - Cg[0]) @ Rg[0].T                 # 相机 0 系里的基线方向
    ax = np.cross(vg0, [0.0, 1.0, 0.0])
    ax = ax / np.linalg.norm(ax)
    vg0_rot = _rot(ax, 5.0) @ vg0
    Cg_bad = Cg.copy()
    Cg_bad[2] = Cg[0] + vg0_rot @ Rg[0]            # 反推光心
    t_bad = np.stack([-Rg[i] @ Cg_bad[i] for i in range(C)])[:, :, None]
    e_t_bad = translation_direction_error_deg(Rg, t_bad, Rg, tg)
    check("平移指标：相机 2 的基线方向拧 5°，读数**精确** ≈ 5°",
          abs(e_t_bad[2] - 5.0) < 1e-9,
          f"读出 {e_t_bad[2]:.9f}°；其余相机 = "
          f"{np.array2string(np.delete(e_t_bad, 2), precision=2)}")
    # 平移 gauge 是 t_i -> t_i - R_i τ（世界整体平移 τ）；缩放是 t -> λ t。
    # 注意：给所有 t 加同一个常数**不是** gauge（各相机 R 不同，光心不会整体平移）。
    tau = np.array([0.7, -1.3, 2.1])
    t_gauged = np.stack([t_bad[i, :, 0] - Rg[i] @ tau for i in range(C)])[:, :, None]
    check("平移指标：对平移 gauge（t_i - R_i τ）与整体缩放不变",
          np.allclose(translation_direction_error_deg(Rg, t_gauged, Rg, tg), e_t_bad,
                      atol=1e-9, equal_nan=True) and
          np.allclose(translation_direction_error_deg(Rg, 5.0 * t_bad, Rg, tg), e_t_bad,
                      atol=1e-9, equal_nan=True))

    # ---------------- E. 手性 ----------------
    section("E. 手性 / z-test")
    sign, zp, zn = z_test_sign(out4["R"][0], out4["t"][0, :, 0], out4["R"][1],
                               out4["t"][1, :, 0], n[0], n[1])
    check("z-test 选中前方点数更多的一侧", sign == 1 and zp > zn,
          f"sign={sign}, z+={zp}, z-={zn}")
    sign_n, zp_n, zn_n = z_test_sign(out4["R"][0], -out4["t"][0, :, 0], out4["R"][1],
                                     -out4["t"][1, :, 0], n[0], n[1])
    check("把 t 整体取负 -> z-test 翻号", sign_n == -sign and zn_n > zp_n,
          f"sign={sign_n}, z+={zp_n}, z-={zn_n}")

    # ---------------- F. 三角化 ----------------
    section("F. 三角化（DLT）")
    P = np.concatenate([scene["R_gt"], scene["t_gt"]], axis=2)
    X_tri = np.stack([triangulate(n[:, i, :], P)[:3] for i in range(n.shape[1])])
    X_gt_flat = scene["X_gt"].reshape(-1, 3)
    # n 的顺序是 (帧, 关节) 展平，与 X_gt.reshape(-1,3) 同序
    d = np.linalg.norm(X_tri - X_gt_flat, axis=1)
    if clean:
        check("干净 2D 下 DLT 精确复现真值 3D", d.max() < 1e-6, f"max err={d.max():.3e} m")
    else:
        note("2D 有噪，DLT 复现真值的偏差（预期非 0）", f"median={np.median(d):.3e} m")

    # 批量版 DLT 必须与逐点版**逐点**一致（★ 曾经漏了对相机轴求和 -> 退化成单视图）
    nnb = n.reshape(C, args.frames, 19, 3)
    visb = np.ones((C, args.frames, 19), bool)
    Xb = dlt_batch(nnb, P, visb).reshape(-1, 3)
    db = np.linalg.norm(Xb - X_tri, axis=1)
    check("dlt_batch 与逐点 triangulate 逐点数值一致", db.max() < 1e-9,
          f"max|Δ|={db.max():.3e} m（全 {Xb.shape[0]} 个点）")
    check("dlt_batch: 可见视图 < 2 的点返回 NaN",
          np.isnan(dlt_batch(nnb, P, np.zeros_like(visb))).all())
    visb2 = visb.copy()
    visb2[1:] = False          # 只剩 1 个视图 -> 不可解
    check("dlt_batch: 单视图不可解（返回 NaN，而不是给个假解）",
          np.isnan(dlt_batch(nnb, P, visb2)).all())

    # ---------------- G. BA ----------------
    if args.run_ba:
        section("G. BA 精化（用小规模子问题：BA 的自由度 = N·J·3，全量请放服务器）")
        nb = min(4, args.frames)
        w_b = np.ones((C, nb, 19))
        print(f"  [i] BA 子问题：{nb} 帧 × 19 关节 → 自由参数 {nb * 19 * 3 + C * 6}"
              f"（全量 {args.frames} 帧请放到服务器上跑）")
        print("  [i] 注意 least_squares(trf) 用数值 Jacobian，一次迭代要约 (参数数+1) 次函数求值")
        t0 = __import__("time").time()
        ba = ba_refine(out4["R"], out4["t"], scene["p2d"][:, :nb], w_b,
                       scene["p3d"][:, :nb], w_b, K4, bones,
                       lambda1=0.01, lambda2=0.01, max_nfev=2000)
        print(f"  [i] BA 用时 {__import__('time').time() - t0:.1f}s")
        e_ba = alignment_error(ba["R"], ba["t"], scene["R_gt"], scene["t_gt"])
        check("BA 跑通（least_squares 正常收敛、残差有限）",
              ba["status"] in (1, 2, 3, 4) and np.isfinite(ba["cost"]),
              f"nfev={ba['nfev']}, cost={ba['cost']:.4g}, msg={ba['message'][:48]}")
        if clean:
            check("BA 初始残差已很小（干净数据下线性解就是最优）",
                  ba["cost"] < 1e-6, f"cost={ba['cost']:.3e}")
        else:
            note("BA 残差量级（有噪时非 0；反映 2D/3D 噪声）", f"cost={ba['cost']:.4g}")
        check("BA 后旋转误差不劣化（+0.05° 容差）",
              np.nanmax(e_ba["rot_deg"][1:]) <= np.nanmax(err["rot_deg"][1:]) + 0.05,
              f"{np.nanmax(err['rot_deg'][1:]):.4f}° -> {np.nanmax(e_ba['rot_deg'][1:]):.4f}°")
        check("BA 后尺度仍 ≈ 真值基线 |C1-C0|", abs(e_ba["scale"] - base01_gt) / base01_gt < 0.1,
              f"s={e_ba['scale']:.5f} vs {base01_gt:.5f}")

    # ---------------- H. 端到端 ----------------
    section("H. 端到端 calibrate_from_lifter")
    full = calibrate_from_lifter(scene["p3d"], scene["p2d"], K4, w2d, bones=bones,
                                 refine=False)
    need = {"R", "t", "X_lin", "lin", "n_views", "n_bones", "bones", "root"}
    check("返回结构完整（含 gauge/退化诊断）", need.issubset(full.keys()),
          f"keys={sorted(need - set(full))} 缺失" if not need.issubset(full.keys())
          else f"n_views={full['n_views']}, n_bones={full['n_bones']}")
    check("端到端 R 形状 (C,3,3)", np.asarray(full["R"]).shape == (C, 3, 3))
    check("端到端 t 形状 (C,3,1)", np.asarray(full["t"]).shape == (C, 3, 1))
    check("t[0] 恒为 0（平移 gauge）", np.allclose(full["t"][0], 0.0),
          f"|t0|={np.abs(full['t'][0]).max():.2e}")
    base = np.linalg.norm(camera_centers(full["R"], full["t"])[1]
                          - camera_centers(full["R"], full["t"])[0])
    tol_g = 1e-9 if clean else 0.05     # 有噪时 t0 只能"近似"归零（投影残差）
    check("基线长度归一化为 1（尺度 gauge）", abs(base - 1.0) < tol_g,
          f"|t1-t0|={base:.12f}（容差 {tol_g:g}）")
    e_full = alignment_error(full["R"], full["t"], scene["R_gt"], scene["t_gt"])
    check("端到端精度与直接调用一致",
          np.allclose(np.sort(e_full["rot_deg"]), np.sort(err["rot_deg"]), atol=1e-9))

    # ---------------- 汇总 ----------------
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 72)
    print(f"汇总: {len(_RESULTS) - n_fail}/{len(_RESULTS)} 通过"
          + (f"，{n_fail} 项失败" if n_fail else "  ✅ 全部通过"))
    if n_fail:
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"  FAIL  {name}   {detail}")
    print("=" * 72)
    print("说明：本自检只验证几何求解器（numpy/scipy），不含任何网络前向/训练。")
    print("     真正的提升器精度要在服务器上用 scripts/train_lifter.py 训练后再评。")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
