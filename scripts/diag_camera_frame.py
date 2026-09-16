# -*- coding: utf-8 -*-
"""验证根因：make_two_view_scene 把相机 A 放在单位矩阵 = 从"下方"看，
而训练用的相机集合是 look_at_origin 的**上半球俯视**相机。

对正交投影，"从下方看"恰好是"从上方看"沿一条轴的**镜像**，因此是训练分布外的输入。
本脚本只改一件事：把相机 A 从 np.eye(3) 换成训练相机集合里的一个。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.data import normalize_pose  # noqa: E402
from steerpose.geometry import (ViewSynthesizer, make_two_view_scene,  # noqa: E402
                                matrix_to_rodrigues, ortho_project)
from steerpose.losses import lkp_mean, similarity  # noqa: E402
from steerpose.model import SteerPose  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ck = torch.load("ckpt/demo_best.pt", map_location="cpu")
    model = SteerPose(num_joints=ck["num_joints"])
    model.load_state_dict(ck["model"]); model.eval()

    sc = make_two_view_scene(n_obj=12, seed=99)
    X = sc["poses3d"]
    Xc = X - X.mean(axis=1, keepdims=True)

    vs = ViewSynthesizer(num_views=100, num_rolls=20, num_pairs=64, seed=7)
    Ra_set, Rb_set, Rrel = vs.sample_pair()
    # 让相机 B 相对相机 A 的旋转等于场景那台相机的旋转（保持"两视角"语义）
    Rb = sc["R"] @ Ra_set

    def run(tag, P_, P2_, rv):
        Pn = torch.from_numpy(np.stack([normalize_pose(p) for p in P_]).astype(np.float32))
        P2n = torch.from_numpy(np.stack([normalize_pose(p) for p in P2_]).astype(np.float32))
        with torch.no_grad():
            Q = model(Pn, torch.from_numpy(np.asarray(rv, np.float32)))
            lkp = float(lkp_mean(Q, P2n))
            d = torch.linalg.norm(Q - P2n, dim=-1).mean(-1)
            s = float(similarity(d).median())
        print(f"  {tag:44s} Lkp {lkp:6.4f}   s 中位 {s:.3f}")
        return lkp

    print("同一批姿态、同一个相对旋转 R，只改相机 A 的绝对朝向：")
    a = run("相机 A = I（make_two_view_scene 的做法）",
            np.stack([ortho_project(Xc[i], np.eye(3)) for i in range(len(Xc))]),
            np.stack([ortho_project(Xc[i], sc["R"]) for i in range(len(Xc))]),
            matrix_to_rodrigues(sc["R"])[None].repeat(len(Xc), 0))

    b = run("相机 A = 训练相机集合里的一个 look_at_origin 相机",
            np.stack([ortho_project(Xc[i], Ra_set) for i in range(len(Xc))]),
            np.stack([ortho_project(Xc[i], Rb) for i in range(len(Xc))]),
            matrix_to_rodrigues(sc["R"])[None].repeat(len(Xc), 0))

    print("\n" + "=" * 72)
    print(f"相机 A = I            : Lkp {a:.4f}")
    print(f"相机 A = 训练分布相机   : Lkp {b:.4f}")
    if b < 0.5 * a:
        print("→ 根因确认：**相机 A 放在单位矩阵**导致输入落在训练分布之外（镜像视角）。")
        print("  make_two_view_scene 必须用训练同源的 look_at_origin 相机，而不是 np.eye(3)。")
    else:
        print("→ 不是这个原因，需继续排查。")
    print("\n参照：训练分布 val Lkp ≈ 0.24；E(lkp) 0.5 左右时 Sinkhorn 才会挑出正确配对。")


if __name__ == "__main__":
    main()
