#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理链自检：匹配矩阵 M1 + 几何一致性 Lgeom M2（不依赖训练，秒级完成）。

为什么需要这个自检
------------------
SteerPose 的**推断阶段**（calibrate）几乎全部的正确性都藏在两个地方：
匹配矩阵 W 的形状/语义，以及几何一致性项 Lgeom 的方向。而这两处的错误
**都无法从训练曲线看出来** —— 训练损失 Lkp 走的是另一条路径，Lkp 下降
不代表 Lmatch / Lgeom 是对的。唯一的验证方式是构造已知真值 R, t 的合成
两视角场景，然后检查：

  M1 匹配：pose_distance 必须是 (B1,B2) 的成对距离，与朴素参考实现一致
  M2 几何：Lgeom(R_gt) 必须显著小于 Lgeom(R_wrong)
           用 R_gt + 正确匹配解出的 t 方向必须接近 t_gt
           对 R 求导不能出现 NaN

历史教训（本仓库曾踩过，故留此自检）
------------------------------------
  * pose_distance 曾写成 torch.cdist(q, p).mean(-1)。cdist 把 (B,J,2) 当成
    批量矩阵沿 J 维广播，返回 (B,J) 而非 (B1,B2) —— 匹配矩阵维度和语义
    全错（连 Sinkhorn 的归一化都作用错了维度），几何项跟着越界崩溃。
    训练损失不经过这里，所以**训练看起来完全正常**。
  * build_linear_system 的行/列拼接维度不一致，Lgeom 一旦真正被激活
    （匹配置信度 > 0.3）就抛 RuntimeError；训练早期置信度低、Lgeom 恒为 0，
    问题被掩盖。
  * geometric_loss 曾返回 σ_2ndmin/σ_min（论文要求 σ_min/σ_2ndmin），
    方向正好相反，会把 R 往错误方向推。
  * 几何约束曾使用"按包围盒缩放"的归一化坐标。缩放会改变每条射线的方向，
    正确的 R 不再对应零奇异值 —— 必须用按**焦距**归一化的图像坐标。

用法:
    python scripts/selftest_infer.py
    python scripts/selftest_infer.py --n-obj 20 --seed 3 --noise 0.002
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.geometry import (make_two_view_scene, matrix_to_rodrigues,  # noqa: E402
                                rodrigues_to_matrix, rotation_error_deg)
from steerpose.losses import (geometric_loss, matching_loss, pose_distance,  # noqa: E402
                              pose_distance_ref, solve_translation, to_ray_coords)


def rand_rotation(rng, deg: float) -> np.ndarray:
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    return Rotation.from_rotvec(ax * np.deg2rad(deg)).as_matrix()


def check_matching(n_obj: int, n_joints: int, rng, results: list):
    """M1：pose_distance / matching_loss 的形状与数值。"""
    q = rng.normal(size=(n_obj, n_joints, 2))
    p = rng.normal(size=(n_obj, n_joints, 2))
    d = pose_distance(torch.tensor(q, dtype=torch.float32),
                      torch.tensor(p, dtype=torch.float32)).numpy()
    d_ref = pose_distance_ref(q, p)
    results.append(("pose_distance 形状为 (B1,B2)", d.shape == (n_obj, n_obj)))
    results.append(("pose_distance 与朴素参考实现一致（1e-5）",
                    float(np.abs(d - d_ref).max()) < 1e-5))

    lm, W, S = matching_loss(torch.tensor(q, dtype=torch.float32),
                             torch.tensor(p, dtype=torch.float32))
    results.append((f"W 形状为 (B1,B2)，实际 {tuple(W.shape)}", tuple(W.shape) == (n_obj, n_obj)))
    # sinkhorn 是"行归一化 → 列归一化"交替投影，循环以列归一化收尾，
    # 所以列和精确、行和会残留迭代不足的误差（tau=0.05、50 次迭代时量级 1e-3~1e-2）。
    col_err = float((W.sum(dim=0) - 1).abs().max())
    row_err = float((W.sum(dim=1) - 1).abs().max())
    _, W_more, _ = matching_loss(torch.tensor(q, dtype=torch.float32),
                                 torch.tensor(p, dtype=torch.float32), n_iter=500)
    row_err_more = float((W_more.sum(dim=1) - 1).abs().max())
    results.append((f"W 列和 ≈ 1（1e-5），实际 {col_err:.1e}", col_err < 1e-5))
    results.append((f"W 行和 ≈ 1（2e-2，允许迭代不足），实际 {row_err:.1e}", row_err < 2e-2))
    results.append((f"加大 n_iter 到 500 后行和误差下降（{row_err:.1e} -> {row_err_more:.1e}）",
                    row_err_more <= row_err + 1e-9))
    results.append(("Lmatch 为标量且有限", lm.dim() == 0 and bool(torch.isfinite(lm))))
    print(f"[M1] pose_distance {d.shape}  最大偏差 vs 参考 {np.abs(d - d_ref).max():.2e}")
    print(f"[M1] W {tuple(W.shape)}  列和误差 {col_err:.2e}  行和误差 {row_err:.2e} "
          f"(n_iter=500 -> {row_err_more:.2e})  Lmatch {float(lm):.4f}")


def main():
    ap = argparse.ArgumentParser(description="推理链自检（匹配 + 几何）")
    ap.add_argument("--n-obj", type=int, default=12, help="场景中的目标数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="归一化图像坐标下的观测噪声 sigma")
    ap.add_argument("--reduce", type=str, default="centroid", choices=["centroid", "joints"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    checks = []

    # ================= M1：跨视角匹配 =================
    print("========== M1 跨视角匹配 ==========")
    check_matching(args.n_obj, 19, rng, checks)

    # 真实两视角场景上再验一次：把 Q 直接设为视角 A 的观测并做全局归一化，
    # 此时"正确匹配"就是恒等映射，Sinkhorn 的 argmax 应当基本落在对角线上。
    sc0 = make_two_view_scene(n_obj=args.n_obj, seed=args.seed, rot_deg=0.0,
                              baseline=0.0, xy=0.6, depth=(3.0, 6.0))
    Qa = to_ray_coords(sc0["P"], focal=1.0, principal_point=sc0["principal_point"])
    Qb = to_ray_coords(sc0["P2"], focal=1.0, principal_point=sc0["principal_point"])
    _, W0, _ = matching_loss(torch.tensor(Qa, dtype=torch.float32),
                             torch.tensor(Qb, dtype=torch.float32))
    diag = float((W0.argmax(dim=1) == torch.arange(args.n_obj)).float().mean())
    checks.append((f"同视角自匹配的对角线命中率 = 1.0（实际 {diag:.2f}）", diag >= 0.99))
    print(f"[M1] 同视角自匹配对角线命中率 {diag:.2f}（W {tuple(W0.shape)}）")
    print()

    # ================= M2：几何一致性 =================
    print("========== M2 几何一致性 Lgeom / 平移求解 ==========")
    sc = make_two_view_scene(n_obj=args.n_obj, seed=args.seed, rot_deg=64.6, baseline=2.04)
    P, P2, R_gt, t_gt = sc["P"], sc["P2"], sc["R"], sc["t"]
    if args.noise > 0:
        P = P + rng.normal(0, args.noise, P.shape)
        P2 = P2 + rng.normal(0, args.noise, P2.shape)

    # 合成场景已知内参（f=1, 主点=0），直接给它 -> 几何是精确的
    rc = dict(focal=sc["focal"], principal_point=sc["principal_point"])
    Pg = torch.from_numpy(to_ray_coords(P, **rc)).float()
    P2g = torch.from_numpy(to_ray_coords(P2, **rc)).float()
    K = P.shape[0]
    pairs = torch.stack([torch.arange(K), torch.arange(K)], dim=1)   # 真值匹配
    W = torch.zeros(K, K)
    W[torch.arange(K), torch.arange(K)] = 1.0                        # 完美软指派
    R_t = torch.tensor(R_gt, dtype=torch.float32)

    print(f"[场景] 目标 {K} 个 x {P.shape[1]} 关节 | 真值旋转 "
          f"{np.rad2deg(np.linalg.norm(matrix_to_rodrigues(R_gt))):.1f}° | "
          f"基线 |t| = {np.linalg.norm(t_gt):.2f} | reduce={args.reduce}")

    # ---- 1) Lgeom 判别力 ----
    lg_true = float(geometric_loss(Pg, P2g, W, R_t, reduce=args.reduce))
    lg_rand = []
    for s in range(8):
        Rw_np = rand_rotation(np.random.default_rng(1000 + s), rng.uniform(5, 120))
        lg_rand.append(float(geometric_loss(Pg, P2g, W,
                                            torch.tensor(Rw_np, dtype=torch.float32),
                                            reduce=args.reduce)))
    lg_rand = np.array(lg_rand)
    lg_med = float(np.median(lg_rand))
    contrast = lg_med / max(lg_true, 1e-12)

    print(f"[Lgeom] R = R_gt    : smin/s2nd = {lg_true:.6f}")
    print(f"[Lgeom] R = 随机 x8 : min {lg_rand.min():.6f}  中位 {lg_med:.6f}  最大 {lg_rand.max():.6f}")
    print(f"[Lgeom] 对比度（随机中位 / 真值）= {contrast:.0f}x")

    # ---- 2) 平移方向恢复 ----
    t_est = solve_translation(Pg, P2g, pairs, R_t, reduce=args.reduce)
    u_gt = t_gt / np.linalg.norm(t_gt)
    t_err = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(t_est, u_gt))), -1, 1))))
    R_wrong = torch.tensor(rand_rotation(np.random.default_rng(77), 60), dtype=torch.float32)
    t_wrong = solve_translation(Pg, P2g, pairs, R_wrong, reduce=args.reduce)
    t_err_wrong = float(np.degrees(np.arccos(
        np.clip(abs(float(np.dot(t_wrong, u_gt))), -1, 1))))
    print(f"[t] R=R_gt  方向误差 {t_err:6.2f}°   t_est = {np.round(t_est, 3)}   t_gt = {np.round(u_gt, 3)}")
    print(f"[t] R=错误  方向误差 {t_err_wrong:6.2f}°")

    # ---- 3) 梯度健全性 ----
    rv = torch.tensor(matrix_to_rodrigues(R_gt), dtype=torch.float32).requires_grad_(True)
    lg = geometric_loss(Pg, P2g, W, rodrigues_to_matrix(rv), reduce=args.reduce)
    lg.backward()
    g = rv.grad.detach().numpy()
    ok_grad = bool(np.isfinite(g).all())

    # ---- 4) 预期用法检验：Lmatch 已把 R 拉到真值附近后，Lgeom 负责"收尾" ----
    # 论文里 Lgeom 从不是单独使用的：它只在 40% 迭代后作为正则项加到 Lmatch 上
    # （Lmatch + λ·Lgeom），并且最多 5 次随机重启。所以这里检验的是
    # "从真值附近出发能否被 Lgeom 拉得更准"，而不是"冷启动能否全局收敛"。
    def grad_norm(R_np: np.ndarray) -> float:
        r = torch.tensor(matrix_to_rodrigues(R_np), dtype=torch.float32).requires_grad_(True)
        geometric_loss(Pg, P2g, W, rodrigues_to_matrix(r), reduce=args.reduce).backward()
        return float(r.grad.detach().norm())

    gn_true = grad_norm(R_gt)
    gn_rand = grad_norm(rand_rotation(np.random.default_rng(31), 60))
    print(f"[grad] |dLgeom/drotvec| @R_gt = {gn_true:.5f}   @R_错误 = {gn_rand:.5f}")

    # 从 45° 冷启动（仅作参考，非判定项）
    rv_cold = torch.tensor(matrix_to_rodrigues(rand_rotation(np.random.default_rng(5), 45)),
                           dtype=torch.float32).requires_grad_(True)
    oc = torch.optim.Adam([rv_cold], lr=0.02)
    for _ in range(300):
        oc.zero_grad()
        geometric_loss(Pg, P2g, W, rodrigues_to_matrix(rv_cold), reduce=args.reduce).backward()
        oc.step()
    err_cold = rotation_error_deg(rodrigues_to_matrix(rv_cold).detach().numpy(), R_gt)

    # 从真值 +5° 扰动起步（预期的收尾场景）
    R_pert = R_gt @ Rotation.from_rotvec(
        np.deg2rad(5.0) * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])).as_matrix()
    rv_warm = torch.tensor(matrix_to_rodrigues(R_pert), dtype=torch.float32).requires_grad_(True)
    ow = torch.optim.Adam([rv_warm], lr=0.01)
    for _ in range(300):
        ow.zero_grad()
        geometric_loss(Pg, P2g, W, rodrigues_to_matrix(rv_warm), reduce=args.reduce).backward()
        ow.step()
    err_warm = rotation_error_deg(rodrigues_to_matrix(rv_warm).detach().numpy(), R_gt)
    print(f"[opt] 仅用 Lgeom：45° 冷启动 -> ER = {err_cold:6.2f}°（非判定项，非凸目标存在局部极小）")
    print(f"[opt] 仅用 Lgeom：真值 +5° 起步 -> ER = {err_warm:6.2f}°（预期的收尾场景）")

    checks += [
        ("Lgeom(R_gt) < 0.2 x Lgeom(随机) —— 损失方向正确、有判别力",
         lg_true < 0.2 * lg_med),
        ("平移方向误差 < 5° —— 线性系统与射线构造正确", t_err < 5.0),
        ("错误 R 下平移误差明显更大 —— 线性系统对 R 敏感", t_err_wrong > t_err + 5.0),
        ("梯度无 NaN/Inf —— 反向传播可用", ok_grad),
        ("真值附近起步不发散（ER < 15°）", err_warm < 15.0),
    ]

    # 说明：不做"从 45° 冷启动必须收敛"的判定 —— σ_min/σ_2ndmin 是非凸目标，
    # 且在 σ_min→0 处有尖点（梯度不趋于 0），所以单独用 Lgeom 冷启动会掉局部极小。
    # 论文的用法是把 Lgeom 作为 Lmatch 收敛后的正则项（40% 迭代后激活 + 最多 5 次重启），
    # 上面 err_cold 一栏仅作参考。
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in checks):
        print("\n自检未通过 —— Lgeom / 线性系统构造有问题")
        return 1
    print("\n自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
