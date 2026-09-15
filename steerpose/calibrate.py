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
from .geometry import (ViewSynthesizer, make_demo_dataset, matrix_to_rodrigues,
                       ortho_project, rodrigues_to_matrix, rotation_error_deg)
from .losses import build_linear_system, geometric_loss, matching_loss, similarity, pose_distance, to_cam_frame
from .model import SteerPose


def estimate_translation(Q: torch.Tensor, P2: torch.Tensor, W: torch.Tensor,
                         R: torch.Tensor) -> np.ndarray:
    """由匹配结果解出相对平移方向 t（单位向量，尺度不可观测）。"""
    conf, bidx = W.max(dim=1)
    aidx = torch.arange(W.shape[0], device=W.device)
    sel = conf > max(0.3, 2.0 / max(W.shape[0], W.shape[1]))
    if sel.sum() < 3:
        return np.zeros(3)
    pairs = torch.stack([aidx[sel], bidx[sel]], dim=1)
    A = build_linear_system(to_cam_frame(Q), to_cam_frame(P2), pairs, R)
    A = A.detach().cpu().numpy()
    Np = pairs.shape[0]
    _, _, Vt = np.linalg.svd(A)
    z = Vt[-1]                       # 零空间向量 = [x_1..x_N, t]
    t = z[Np:]
    n = np.linalg.norm(t)
    return t / n if n > 1e-9 else t


def calibrate(P: np.ndarray, P2: np.ndarray, model: SteerPose, iters: int = 1000,
              lr: float = 0.01, lam: float = 1.0, alpha: float = 3.0,
              geom_start: float = 0.4, tau: float = 0.05, sinkhorn_iter: int = 50,
              R_init: np.ndarray = None, seed: int = 0, verbose: bool = True) -> dict:
    """两视角联合标定 + 匹配。P (B1,J,2) 与 P2 (B2,J,2) 为两视角的 2D 姿态集合。"""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    Pn = torch.from_numpy(np.stack([normalize_pose(p) for p in P])).float().to(device)
    P2n = torch.from_numpy(np.stack([normalize_pose(p) for p in P2])).float().to(device)

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
    for it in range(iters):
        R = rodrigues_to_matrix(rotvec)
        Q = model(Pn, rotvec)
        lm, W, S = matching_loss(Q, P2n, alpha=alpha, n_iter=sinkhorn_iter, tau=tau)
        lg = geometric_loss(Q, P2n, W, R) if it >= int(iters * geom_start) \
            else torch.zeros((), device=device)
        loss = lm + lam * lg
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        curve.append((loss.item(), lm.item(), float(lg)))
        if verbose and it % 100 == 0:
            print(f"  iter {it:4d}  total {loss.item():8.4f}  "
                  f"Lmatch {lm.item():8.4f}  Lgeom {float(lg):6.3f}")

    with torch.no_grad():
        R = rodrigues_to_matrix(rotvec)
        Q = model(Pn, rotvec)
        _, W, S = matching_loss(Q, P2n, alpha=alpha, n_iter=sinkhorn_iter, tau=tau)
        t = estimate_translation(Q, P2n, W, R)
    return {"R": R.detach().cpu().numpy(),
            "rotvec": rotvec.detach().cpu().numpy(),
            "t": t,
            "W": W.detach().cpu().numpy(),
            "matches": W.argmax(dim=1).detach().cpu().numpy(),
            "curve": curve}


def calibrate_with_retries(P: np.ndarray, P2: np.ndarray, model: SteerPose,
                          n_retries: int = 5, verbose: bool = False, **kwargs) -> dict:
    """论文附录 D.3：失败则换随机初始旋转重试，最多 n_retries 次。

    复现版以最终损失作为"是否成功"的判据（论文用的是重投影误差 < 10px）。
    """
    best = None
    for a in range(n_retries):
        res = calibrate(P, P2, model, seed=a, verbose=verbose, **kwargs)
        if best is None or res["curve"][-1][0] < best["curve"][-1][0]:
            best, best["retry"] = res, a
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

    gt = None
    if args.poses_npz:
        arr = np.load(args.poses_npz)
        P, P2 = arr["P"].astype(np.float64), arr["P2"].astype(np.float64)
        if "R" in arr:
            gt = arr["R"]
    else:
        if not args.demo:
            print("[!] 未指定 --poses-npz，转到 --demo 自测模式")
        poses3d = make_demo_dataset(num_anims=20, num_frames=1, seed=99)
        vs = ViewSynthesizer(num_pairs=1, seed=7)
        Ra, Rb, Rrel = vs.sample_pair()
        poses_c = poses3d - poses3d.mean(axis=1, keepdims=True)
        P = np.stack([normalize_pose(ortho_project(p, Ra)) for p in poses_c])
        P2 = np.stack([normalize_pose(ortho_project(p, Rb)) for p in poses_c])
        gt = Rrel
        print(f"[i] 演示场景：{P.shape[0]} 个目标，真值相对旋转已知")

    print(f"[i] 输入：视角 A {P.shape}，视角 B {P2.shape}")
    res = calibrate_with_retries(P, P2, model, n_retries=args.retries,
                                 iters=args.iters, lam=args.lambda_geom,
                                 alpha=args.alpha, verbose=True)
    print(f"[i] 最优重启次数: {res['retry']}")
    print(f"[result] R =\n{np.round(res['R'], 4)}")
    print(f"[result] t(单位方向) = {np.round(res['t'], 4)}")
    print(f"[result] 匹配: {res['matches'][:20]}{' ...' if len(res['matches']) > 20 else ''}")

    if gt is not None:
        err = rotation_error_deg(res["R"], gt)
        acc = float((res["matches"] == np.arange(len(res["matches"]))).mean()) \
            if len(res["matches"]) == len(P2) else float("nan")
        print(f"[result] 旋转误差 ER = {err:.2f}°   (匹配 argmax 与序号一致的比例 {acc:.2f})")

    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out, R=res["R"], t=res["t"], W=res["W"],
                 matches=res["matches"], curve=np.array(res["curve"]))
        print(f"[ok] 结果已保存 -> {args.out}")


if __name__ == "__main__":
    main()
