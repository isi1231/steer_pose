# -*- coding: utf-8 -*-
"""先校准"量尺"本身：同一段诊断代码，在训练分布上跑，能不能复现 val Lkp ≈ 0.24？

如果不能，说明是诊断脚本里的约定（rotvec 方向 / 关节顺序 / 归一化）用错了，
那么前面关于"投影不一致"的结论也不可信。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.data import build_pairs, normalize_pose  # noqa: E402
from steerpose.geometry import make_demo_dataset, make_two_view_scene  # noqa: E402
from steerpose.losses import lkp_mean, pose_distance, similarity  # noqa: E402
from steerpose.model import SteerPose  # noqa: E402


def norm_t(P):
    return torch.from_numpy(
        np.stack([normalize_pose(p) for p in P]).astype(np.float32))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ck = torch.load("ckpt/demo_best.pt", map_location="cpu")
    model = SteerPose(num_joints=ck["num_joints"])
    model.load_state_dict(ck["model"]); model.eval()
    print(f"模型 epoch {ck.get('epoch')}，{ck['num_joints']} 关节\n")

    def diag_d(P_, P2_, RV_):
        Pn, P2n = norm_t(P_), norm_t(P2_)
        rv = torch.from_numpy(RV_.astype(np.float32))
        with torch.no_grad():
            Q = model(Pn, rv)
            d = torch.diagonal(pose_distance(Q, P2n))
            s = similarity(d)
            # 直接照 train.evaluate 的口径算一次 Lkp（这才是和日志可比的量）
            lkp = float(lkp_mean(Q, P2n))
        return float(d.median()), float(s.median()), lkp

    # ---- 对照组 1：直接用训练分布的 val 划分（这就是日志里 0.2446 的来源）----
    poses = make_demo_dataset(num_anims=200, num_frames=20, seed=1)
    pairs = build_pairs(poses, num_pairs=3000, num_views=100, num_rolls=20, seed=0)
    va = slice(2100, 2700)
    d1, s1, lkp1 = diag_d(pairs["P"][va], pairs["P2"][va], pairs["rotvec"][va])
    print("对照 1：训练分布的 val（与日志同源）")
    print(f"   Lkp {lkp1:.4f}（日志为 0.2446）  对角线 d {d1:.4f}  s {s1:.3f}")

    # ---- 对照组 2：拿场景那 12 个姿态，套用训练的数据合成流程 ----
    scene_poses = make_demo_dataset(num_anims=12, num_frames=1, seed=99)
    pr2 = build_pairs(scene_poses, num_pairs=600, num_views=100, num_rolls=20, seed=0)
    d2, s2, lkp2 = diag_d(pr2["P"], pr2["P2"], pr2["rotvec"])
    print("\n对照 2：场景的 12 个姿态 + 训练的数据合成流程")
    print(f"   Lkp {lkp2:.4f}  对角线 d {d2:.4f}  s {s2:.3f}")

    # ---- 对照组 3：场景本身（前一步测出 d=0.68）----
    sc = make_two_view_scene(n_obj=12, seed=99)
    from steerpose.geometry import matrix_to_rodrigues
    rv_sc = matrix_to_rodrigues(sc["R"])[None, :].repeat(12, 0)
    d3, s3, lkp3 = diag_d(sc["P"], sc["P2"], rv_sc)
    print("\n对照 3：make_two_view_scene（透视 + 基线）")
    print(f"   Lkp {lkp3:.4f}  对角线 d {d3:.4f}  s {s3:.3f}")

    print("\n" + "=" * 72)
    if lkp1 < 0.35:
        print("量尺可信：同一段代码能在训练分布上复现日志里的低 Lkp。")
        print(f"→ 于是对照 2/3 的高 Lkp 是真实差异，不是诊断脚本的约定错误。")
        print(f"   对照2（12 个场景姿态，训练流程）= {lkp2:.4f}")
        print(f"   对照3（make_two_view_scene）    = {lkp3:.4f}")
    else:
        print(f"量尺不可信：训练分布上也算出 {lkp1:.4f}，与日志 0.2446 不符；"
              "先修诊断脚本再谈别的。")

    # 顺带比一下输入/目标的分布，找 OOD 线索
    from steerpose.geometry import ViewSynthesizer
    print("\n输入分布体检（归一化后）：")
    for tag, P_, P2_ in (("训练分布", pairs["P"][va], pairs["P2"][va]),
                         ("场景12姿态(训练流程)", pr2["P"], pr2["P2"]),
                         ("make_two_view_scene", sc["P"], sc["P2"])):
        Pn = np.stack([normalize_pose(p) for p in P_])
        P2n = np.stack([normalize_pose(p) for p in P2_])
        print(f"  {tag:22s} |P|均值 {np.abs(Pn).mean():.3f}  "
              f"|P2|均值 {np.abs(P2n).mean():.3f}  "
              f"corr(P,P2) 逐关节中位 "
              f"{np.median([np.corrcoef(Pn[i].ravel(), P2n[i].ravel())[0,1] for i in range(len(Pn))]):.3f}")


if __name__ == "__main__":
    main()
