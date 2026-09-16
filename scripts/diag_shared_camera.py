# -*- coding: utf-8 -*-
"""干净对照：同一批 12 个姿态，只改"12 个样本是否共用同一个相机对"。

  A. 训练式：每个样本各自随机抽一对相机（= build_pairs 的做法）
  B. 场景式：12 个样本共用同一对相机（= make_two_view_scene / calibrate 的做法）
如果 A 好 B 差，说明模型只在一批样本里**旋转 R 有变化**时才工作 ——
那就不是"每样本独立函数"该有的行为，需要往模型/归一化上查。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.data import normalize_pose  # noqa: E402
from steerpose.geometry import (ViewSynthesizer, make_demo_dataset,  # noqa: E402
                                make_two_view_scene, matrix_to_rodrigues,
                                ortho_project)
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
    X = sc["poses3d"]
    Xc = X - X.mean(axis=1, keepdims=True)
    n = len(Xc)

    vs = ViewSynthesizer(num_views=100, num_rolls=20, num_pairs=64, seed=7)

    @torch.no_grad()
    def score(tag, pairs):
        Ps = np.stack([ortho_project(Xc[j], Ra) for j, Ra, _ in pairs])
        P2s = np.stack([ortho_project(Xc[j], Rb) for j, _, Rb in pairs])
        RV = np.stack([matrix_to_rodrigues(Rb @ Ra.T) for _, Ra, Rb in pairs])
        Pn, P2n = norm_t(Ps), norm_t(P2s)
        Q = model(Pn, torch.from_numpy(RV.astype(np.float32)))
        lkp = float(lkp_mean(Q, P2n))
        d = torch.linalg.norm(Q - P2n, dim=-1).mean(-1)
        s = similarity(d)
        print(f"  {tag:40s} Lkp {lkp:6.4f}   s 中位 {s.median():.3f}  "
              f"s>0.5 的比例 {(s > 0.5).float().mean():.2f}")
        return lkp

    print("12 个姿态，从训练相机集合里抽相机对：\n")

    # A. 每个样本各自随机一对相机
    pa = []
    for i in range(n):
        Ra, Rb, _ = vs.sample_pair()
        pa.append((i, Ra, Rb))
    la = score("A. 每样本各自随机一对相机（训练式）", pa)

    # A'. 同上但每个姿态重复 50 次（模拟 build_pairs 的 600 对）
    pa2 = []
    for _ in range(50):
        for i in range(n):
            Ra, Rb, _ = vs.sample_pair()
            pa2.append((i, Ra, Rb))
    la2 = score("A'. 同上，600 对（与前面 0.195 同口径）", pa2)

    # B. 12 个样本共用同一对相机
    Ra_f, Rb_f, _ = vs.sample_pair()
    lb = score("B. 共用同一对相机（= 场景/标定做法）", [(i, Ra_f, Rb_f) for i in range(n)])

    # C. 共用相机对，但只变"R"（固定 Ra，Rb 每个样本不同）
    pc = []
    for i in range(n):
        _, Rb, _ = vs.sample_pair()
        pc.append((i, Ra_f, Rb))
    lc = score("C. 固定相机 A、每样本换相机 B", pc)

    # D. 每样本固定 Ra= I（场景原始做法）作参照
    pd = []
    for i in range(n):
        _, Rb, _ = vs.sample_pair()
        pd.append((i, np.eye(3), Rb))
    ld = score("D. 相机 A 固定为单位矩阵", pd)

    print("\n" + "=" * 72)
    print(f"  A  每样本随机相机对        {la:.4f}")
    print(f"  A' 600 对（同 build_pairs） {la2:.4f}")
    print(f"  B  12 样本共用一对相机      {lb:.4f}")
    print(f"  C  固定 A、每样本换 B       {lc:.4f}")
    print(f"  D  相机 A = I              {ld:.4f}")
    print(f"\n  训练分布 val 参照 0.24")
    if la2 < 0.3 and lb > 0.5:
        print("  → 差别确实在'共用相机对 vs 每样本随机'。继续查模型是否依赖批内 R 的变化。")
    elif la2 > 0.3:
        print("  → A 也不好了？说明 12 个姿态本身在这个测量口径下就偏难，"
              "需要换更细的分组统计。")


if __name__ == "__main__":
    main()
