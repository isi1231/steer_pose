# -*- coding: utf-8 -*-
"""单因子隔离：make_two_view_scene 上标定失败的**唯一**原因是哪一个？

两个候选渲染路径的差异有三处：
  i.  投影：正交（训练用的） vs 透视（场景用的）
  ii. 旋转：每个样本随机 R（训练用的） vs 整个场景共用一个 R
  iii.平移：无基线（训练用的） vs 有基线 t
把 2^3 = 8 种组合都跑一遍，看 Lkp 在哪一步突变。
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.data import build_pairs, normalize_pose  # noqa: E402
from steerpose.geometry import (make_demo_dataset, make_two_view_scene,  # noqa: E402
                                matrix_to_rodrigues, ortho_project,
                                perspective_project)
from steerpose.losses import lkp_mean, similarity  # noqa: E402
from steerpose.model import SteerPose  # noqa: E402


def norm_t(P):
    return torch.from_numpy(
        np.stack([normalize_pose(p) for p in P]).astype(np.float32))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ck = torch.load("ckpt/demo_best.pt", map_location="cpu")
    model = SteerPose(num_joints=ck["num_joints"])
    model.load_state_dict(ck["model"]); model.eval()

    sc = make_two_view_scene(n_obj=12, seed=99)
    X = sc["poses3d"]                                   # (12,20,3) 场景里的 3D 姿态
    Xc = X - X.mean(axis=1, keepdims=True)
    R_fix, t_fix = sc["R"], sc["t"]
    print(f"场景共用的 R：旋转 {np.degrees(np.linalg.norm(matrix_to_rodrigues(R_fix))):.1f}°；"
          f"基线 |t| = {np.linalg.norm(t_fix):.2f}\n")

    # 随机 R（每个样本一个，来自训练用的相机集合）
    pr = build_pairs(X, num_pairs=64, num_views=100, num_rolls=20, seed=3)

    @torch.no_grad()
    def run(tag, P_, P2_, RV_):
        Pn, P2n = norm_t(P_), norm_t(P2_)
        Q = model(Pn, torch.from_numpy(RV_.astype(np.float32)))
        lkp = float(lkp_mean(Q, P2n))
        d = torch.linalg.norm(Q - P2n, dim=-1).mean(-1)
        s = similarity(d)
        print(f"  {tag:38s} Lkp {lkp:6.4f}   对角线 s 中位 {s.median():.3f}")
        return lkp

    print("8 种组合（投影 × 旋转 × 基线）：")
    rows = []
    for persp, rand_r, base in itertools.product((False, True), repeat=3):
        if persp:
            P_ = perspective_project(Xc, np.eye(3), np.zeros(3))
            P2_ = perspective_project(Xc, R_fix, t_fix if base else np.zeros(3))
        else:
            P_ = np.stack([ortho_project(Xc[i], np.eye(3)) for i in range(len(Xc))])
            P2_ = np.stack([ortho_project(Xc[i], R_fix) for i in range(len(Xc))])
            if base:   # 正交投影下"基线"没有意义，直接给 2D 加同一个常数位移
                P2_ = P2_ + np.asarray(t_fix)[None, None, :2] * 0.3
        RV = pr["rotvec"][:len(Xc)] if rand_r else \
            np.repeat(matrix_to_rodrigues(R_fix)[None], len(Xc), 0)
        tag = (f"{'透视' if persp else '正交'}-"
               f"{'随机R' if rand_r else '固定R'}-"
               f"{'有基线' if base else '无基线'}")
        rows.append((tag, run(tag, P_, P2_, RV)))

    print()
    lo = min(r[1] for r in rows); hi = max(r[1] for r in rows)
    print(f"最小 Lkp {lo:.4f} / 最大 {hi:.4f}")
    print("\n只保留 Lkp < 0.3 的组合（能工作的）：")
    for tag, v in rows:
        if v < 0.3:
            print(f"  {tag:38s} {v:.4f}")
    print("\n逐个因子看（在同一投影下比较）：")
    for persp in (False, True):
        pn = "透视" if persp else "正交"
        sub = {t.replace(pn + "-", ""): v for t, v in rows if t.startswith(pn)}
        print(f"  {pn}: { {k: round(v, 3) for k, v in sub.items()} }")


if __name__ == "__main__":
    main()
