# -*- coding: utf-8 -*-
"""两视角外参标定 + 跨视角匹配：推理时联合优化 R（论文 3.2 节）.

流程:
  1. SteerPose 冻结，只把相对旋转 R（Rodrigues 向量）当作可学习参数；
  2. 给定 R，SteerPose 把视角 A 的 2D 姿态 P 变换到视角 B 得到 Q(R)；
  3. Sinkhorn 可微匹配在 Q(R) 与观测到的 P' 之间做软指派；
  4. 损失 = Lmatch + λ·Lgeom，反向传播只更新 R；
  5. Lgeom 在 40% 迭代后激活（附录 D.3）；Adam lr=0.01，1000 次迭代，线性衰减 1.0→0.01；
  6. 失败（论文判据：重投影误差 ≥ 10px）换随机初始 R 重试，最多 5 次。

命令行:
    # 自测（无数据，程序化场景，含真值对比）
    python -m steerpose.calibrate --ckpt ckpt/demo_best.pt --demo

    # 真实两视角 2D 姿态：npz 内含 'P' (B1,J,2) 与 'P2' (B2,J,2)
    python -m steerpose.calibrate --ckpt ckpt/steerpose_quad.pt \\
        --poses-npz data/two_view_poses.npz --out out/two_view.npz
"""
import argparse
import math
import numpy as np
import torch

from .data import normalize_pose
from .geometry import (make_two_view_scene, matrix_to_rodrigues,
                       rodrigues_to_matrix, rotation_error_deg)
from .losses import (geometric_loss, guess_focal, matching_loss,
                     solve_translation, to_ray_coords)
from .model import SteerPose


def estimate_translation(Qg: torch.Tensor, P2g: torch.Tensor, W: torch.Tensor,
                         R: torch.Tensor) -> np.ndarray:
    """由匹配结果解出相对平移方向 t（单位向量，尺度不可观测）。

    Qg / P2g 必须是**按内参归一化**的 2D 坐标（calibrate 里由 to_ray_coords 生成），
    不能是逐姿态包围盒归一化（normalize_pose）的结果。
    """
    conf, bidx = W.max(dim=1)
    aidx = torch.arange(W.shape[0], device=W.device)
    sel = conf > max(0.3, 2.0 / max(W.shape[0], W.shape[1]))
    if sel.sum().item() < 3:
        return np.zeros(3)
    pairs = torch.stack([aidx[sel], bidx[sel]], dim=1)
    return solve_translation(Qg, P2g, pairs, R)


def calibrate(P: np.ndarray, P2: np.ndarray, model: SteerPose, iters: int = 1000,
              lr: float = 0.01, lam: float = 1.0, alpha: float = 3.0,
              geom_start: float = 0.4, tau: float = 0.05, sinkhorn_iter: int = 50,
              R_init: np.ndarray = None, focal: float = None, K: np.ndarray = None,
              seed: int = 0, verbose: bool = True) -> dict:
    """两视角联合标定 + 匹配。P (B1,J,2) 与 P2 (B2,J,2) 为两视角的 2D 姿态集合。

    K    : (3,3) 内参矩阵。**首选**，fx/fy/cx/cy 全部取自它，等效于同时指定焦距与主点。
    focal: 像素焦距（没有 K 时的备选）。几何约束需要它把像素坐标换算成归一化图像坐标；
           不传则按数据量级自动估计（见 losses.guess_focal）。已归一化的坐标传 1.0。

    为什么内参必须给对：Lgeom 的共线约束要求"归一化图像坐标下的视线方向 = (u,v,1)"。
    像素坐标除以错误的焦距，等于把每条射线的方向都扭了一下，正确的 R 不再是零奇异值解，
    Lgeom 就失去判别力（且不会报错，只是结果变差）。BamaPig3D 的 label_images 已去畸变，
    因此必须传 newcameramtx（由 prepare_mammal_2d.py 写进 npz 的 'K'）。
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    # 网络输入：逐姿态归一化（与训练一致，只看姿态形状）
    Pn = torch.from_numpy(np.stack([normalize_pose(p) for p in P])).float().to(device)
    P2n = torch.from_numpy(np.stack([normalize_pose(p) for p in P2])).float().to(device)
    # 几何约束：归一化图像坐标（保留位置与真实射线几何）
    if K is not None:
        K = np.asarray(K, np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K 形状应为 (3,3)，实际 {K.shape}")
        Pg = torch.from_numpy(to_ray_coords(P, K=K)).float().to(device)
        P2g = torch.from_numpy(to_ray_coords(P2, K=K)).float().to(device)
        if verbose:
            print(f"[i] 几何约束使用内参 K：fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
                  f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    else:
        # 两视角共用同一 pp/f
        pp = np.asarray(P, np.float64).reshape(-1, 2).mean(0)
        f0 = guess_focal(P, focal)
        Pg = torch.from_numpy(to_ray_coords(P, focal=f0, principal_point=pp)).float().to(device)
        P2g = torch.from_numpy(to_ray_coords(P2, focal=f0, principal_point=pp)).float().to(device)
        if verbose:
            print(f"[i] 未给内参 K，几何约束用估计参数：focal={f0:.2f} pp={np.round(pp,2)}")

    if R_init is not None:
        rv0 = torch.tensor(matrix_to_rodrigues(R_init), device=device)
    else:
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(3, generator=g)
        rv0 = (v / torch.norm(v) * (math.pi * 0.5)).to(device)
    rotvec = rv0.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([rotvec], lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01,
                                              total_iters=iters)
    model.eval()
    curve = []
    geom_stats = {}
    geom_on = int(iters * geom_start)
    for it in range(iters):
        R = rodrigues_to_matrix(rotvec)
        Q = model(Pn, rotvec)
        lm, W, S = matching_loss(Q, P2n, alpha=alpha, n_iter=sinkhorn_iter, tau=tau)
        if it >= geom_on:
            geom_stats = {}
            lg = geometric_loss(Pg, P2g, W, R, stats=geom_stats)
        else:
            lg = torch.zeros((), device=device)
        loss = lm + lam * lg
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        curve.append((loss.item(), lm.item(), float(lg)))
        if verbose and it % 100 == 0:
            extra = ""
            if it >= geom_on:
                extra = (f"  几何匹配 {geom_stats.get('n_pairs', 0):3d} 对"
                         f"{'' if geom_stats.get('active') else '（不足 3 对，几何项未生效）'}")
            print(f"  iter {it:4d}  total {loss.item():8.4f}  "
                  f"Lmatch {lm.item():8.4f}  Lgeom {float(lg):6.3f}{extra}")

    with torch.no_grad():
        R = rodrigues_to_matrix(rotvec)
        Q = model(Pn, rotvec)
        _, W, S = matching_loss(Q, P2n, alpha=alpha, n_iter=sinkhorn_iter, tau=tau)
        t = estimate_translation(Pg, P2g, W, R)
        # 匹配置信度：区分度不足时 Sinkhorn 会接近均匀指派，此时标定结果不可信
        conf = W.max(dim=1).values
        match_conf = float(conf.mean())
        n_conf = int((conf > 0.3).sum())
        # 几何项最终是否真正参与（n<3 时会静默为 0，必须显式暴露）
        final_stats = {}
        geometric_loss(Pg, P2g, W, R, stats=final_stats)
    return {"R": R.detach().cpu().numpy(),
            "rotvec": rotvec.detach().cpu().numpy(),
            "t": t,
            "W": W.detach().cpu().numpy(),
            "matches": W.argmax(dim=1).detach().cpu().numpy(),
            "match_conf": match_conf,
            "n_confident": n_conf,
            "n_targets": int(Pn.shape[0]),
            "n_geom_pairs": int(final_stats.get("n_pairs", 0)),
            "geom_active": bool(final_stats.get("active", False)),
            "curve": curve}


def calibrate_with_retries(P: np.ndarray, P2: np.ndarray, model: SteerPose,
                          n_retries: int = 5, verbose: bool = False, **kwargs) -> dict:
    """论文附录 D.3：失败则换随机初始旋转重试，最多 n_retries 次。

    复现版以最终损失作为"是否成功"的判据（论文用的是重投影误差 < 10px）。
    若某次重启的几何项真正生效（n_geom_pairs ≥ 3），优先选它——否则可能选中一个
    Lmatch 数值更低、但几何约束从未参与、R 其实没有依据的退化解。
    """
    best, best_key = None, None
    for a in range(n_retries):
        res = calibrate(P, P2, model, seed=a, verbose=verbose, **kwargs)
        key = (bool(res["geom_active"]), -res["curve"][-1][0])
        if best_key is None or key > best_key:
            best, best_key = res, key
            best["retry"] = a
    return best


def parse_args():
    ap = argparse.ArgumentParser(description="SteerPose 两视角标定 + 匹配")
    ap.add_argument("--ckpt", type=str, required=True, help="训练好的 SteerPose 权重")
    ap.add_argument("--demo", action="store_true", help="程序化场景自测（带真值）")
    ap.add_argument("--poses-npz", type=str, default=None,
                    help="真实数据 npz，内含 'P' (B1,J,2) 与 'P2' (B2,J,2)")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lambda-geom", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--focal", type=float, default=None,
                    help="像素焦距（几何约束用）。不传则按数据量级自动估计；"
                         "已归一化的坐标传 1.0。若 npz 内含 'K' 或给了 --K-file，则以 K 为准")
    ap.add_argument("--K-file", type=str, default=None,
                    help="内参文件（.npy 的 3x3，或含 9 个数的文本）。"
                         "优先级：--K-file > npz 内的 'K' > --focal/自动估计。"
                         "BamaPig3D 请传去畸变后的 newcameramtx")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None, help="结果 npz 保存路径")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    ck = torch.load(args.ckpt, map_location=device)
    model = SteerPose(num_joints=ck["num_joints"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"[i] 载入权重 {args.ckpt}（{ck['num_joints']} 关节，epoch {ck.get('epoch', '?')}）")

    gt, t_gt, K = None, None, None
    if args.K_file:
        K = np.load(args.K_file) if args.K_file.endswith(".npy") \
            else np.loadtxt(args.K_file).reshape(3, 3)
        K = np.asarray(K, np.float64)
        print(f"[i] 内参来自 {args.K_file}：fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
              f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    if args.poses_npz:
        arr = np.load(args.poses_npz)
        P, P2 = arr["P"].astype(np.float64), arr["P2"].astype(np.float64)
        if "R" in arr:
            gt = arr["R"]
        if "t" in arr:
            t_gt = arr["t"]
        if K is None and "K" in arr:
            K = np.asarray(arr["K"], np.float64)
            print(f"[i] npz 内含内参 K：fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
                  f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}（去畸变后的相机矩阵）")
    else:
        if not args.demo:
            print("[!] 未指定 --poses-npz，转到 --demo 自测模式")
        sc = make_two_view_scene(n_obj=12, seed=99)
        P, P2, gt, t_gt = sc["P"], sc["P2"], sc["R"], sc["t"]
        print(f"[i] 演示场景：{P.shape[0]} 个目标，真值相对旋转与平移均已知（透视投影）；"
              "相机 A 取自训练同源相机集合（保证网络输入 P 落在训练分布内）")

    print(f"[i] 输入：视角 A {P.shape}，视角 B {P2.shape}")
    res = calibrate_with_retries(P, P2, model, n_retries=args.retries,
                                 iters=args.iters, lam=args.lambda_geom,
                                 alpha=args.alpha, focal=args.focal, K=K, verbose=True)
    print(f"[i] 最优重启次数: {res['retry']}")
    print(f"[i] 匹配置信度：平均 {res['match_conf']:.3f}，"
          f"可信匹配 {res['n_confident']}/{res['n_targets']}")
    print(f"[i] 几何项 Lgeom：使用 {res['n_geom_pairs']} 对匹配"
          f"{'' if res['geom_active'] else '  <-- 不足 3 对，Lgeom 静默失效，R 完全由 Lmatch 决定'}")
    if res["match_conf"] < 0.5 or res["n_confident"] < 3:
        print("[!] 匹配置信度偏低 —— Sinkhorn 接近均匀指派，R 与 t 都不可信。\n"
              "    常见原因：SteerPose 训练不足（quick 档位 30 epoch / 3000 pairs 只够跑通链路），\n"
              "    或 2D 姿态噪声/掩码比例过高。请用 full 档位（≥400 epoch、≥20000 pairs）重训，\n"
              "    或提高数据质量后重跑；先用 scripts/selftest_infer.py 确认实现本身没问题。")
    print(f"[result] R =\n{np.round(res['R'], 4)}")
    print(f"[result] t(单位方向) = {np.round(res['t'], 4)}")
    print(f"[result] 匹配: {res['matches'][:20]}{' ...' if len(res['matches']) > 20 else ''}")

    if gt is not None:
        err = rotation_error_deg(res["R"], gt)
        acc = float((res["matches"] == np.arange(len(res["matches"]))).mean()) \
            if len(res["matches"]) == len(P2) else float("nan")
        print(f"[result] 旋转误差 ER = {err:.2f}°   (匹配 argmax 与序号一致的比例 {acc:.2f})")
    if t_gt is not None:
        a = np.asarray(t_gt).ravel()
        b = np.asarray(res["t"]).ravel()
        if np.linalg.norm(a) > 1e-9 and np.linalg.norm(b) > 1e-9:
            c = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
            print(f"[result] 平移方向误差 Et = {np.degrees(np.arccos(np.clip(c, -1, 1))):.2f}°")

    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out, R=res["R"], t=res["t"], W=res["W"],
                 matches=res["matches"], curve=np.array(res["curve"]))
        print(f"[ok] 结果已保存 -> {args.out}")


if __name__ == "__main__":
    main()
