# -*- coding: utf-8 -*-
"""为什么训练好了（val Lkp 0.24）标定还是失败？—— 查"投影模型不一致"。

训练数据的合成用的是**正交投影**（论文附录 D.1，geometry.ortho_project）；
而 calibrate 的场景（make_two_view_scene）用的是**透视投影**（为了让 Lgeom 的
透视共线约束自洽）。如果这个差异足够大，网络在标定场景上就"看不懂"输入，
匹配矩阵退化，Lgeom 也筛不出可靠匹配。

做法：把同一个场景分别按"正交"和"透视"两种投影渲染，喂给同一个模型，
比较在**真值 R**下模型输出与目标的逐关节距离 d。d 越小说明模型越能对上。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.data import normalize_pose  # noqa: E402
from steerpose.geometry import make_demo_dataset, make_two_view_scene, ortho_project  # noqa: E402
from steerpose.losses import pose_distance, similarity  # noqa: E402
from steerpose.model import SteerPose  # noqa: E402


def norm_all(P):
    return torch.from_numpy(
        np.stack([normalize_pose(p) for p in P]).astype(np.float32))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ck = torch.load("ckpt/demo_best.pt", map_location="cpu")
    model = SteerPose(num_joints=ck["num_joints"])
    model.load_state_dict(ck["model"]); model.eval()
    print(f"载入 ckpt/demo_best.pt（epoch {ck.get('epoch')}）")

    sc = make_two_view_scene(n_obj=12, seed=99)      # calibrate --demo 用的场景
    P, P2, R_gt = sc["P"], sc["P2"], sc["R"]
    print(f"场景：{P.shape[0]} 个目标，相机 A 在原点，B 有旋转+基线 "
          f"|t|={np.linalg.norm(sc['t']):.2f}")
    print(f"      目标深度 3~6 m，横向 ±1.2 m，单目标尺寸 ~0.35")
    # 单目标内部因透视造成的深度变化比例
    zs = sc["poses3d"][..., 2]
    print(f"      每个目标内部的深度极差：中位 "
          f"{np.median(zs.max(1)-zs.min(1)):.3f} m，相对其深度约 "
          f"{np.median((zs.max(1)-zs.min(1))/zs.mean(1))*100:.1f}%（透视强度）")

    rv = torch.from_numpy(np.asarray(sc["R"]).astype(np.float32).copy()).ravel()
    from steerpose.geometry import matrix_to_rodrigues
    rv = torch.from_numpy(matrix_to_rodrigues(sc["R"]).astype(np.float32))

    def report(tag, P_, P2_):
        Pn, P2n = norm_all(P_), norm_all(P2_)
        with torch.no_grad():
            Q = model(Pn, rv)
            d = pose_distance(Q, P2n)                    # (12,12)，对角线是真值配对
            diag = torch.diagonal(d)
            s_diag = similarity(diag)
            # 正确配对 vs 错误配对的区分度
            off = d[~torch.eye(d.shape[0], dtype=torch.bool)]
            print(f"  {tag}")
            print(f"    正确配对 d：中位 {diag.median():.4f}   "
                  f"错误配对 d：中位 {off.median():.4f}   "
                  f"区分度 {off.median()/diag.median():.2f}x")
            print(f"    正确配对 s=2/(1+exp(3d))：中位 {s_diag.median():.3f}"
                  f"   （>0.5 才有机会被 Sinkhorn 挑出来）")
        return float(diag.median()), float(off.median())

    print("\n同一个场景，两种投影：")
    d_per, o_per = report("透视投影（calibrate --demo 实际用的）", P, P2)
    # 正交版本：用同样的 3D 姿态与相机朝向，但正交投影
    Po = np.stack([ortho_project(sc["poses3d"][i] - sc["poses3d"][i].mean(0),
                                 np.eye(3)) for i in range(P.shape[0])])
    P2o = np.stack([ortho_project(sc["poses3d"][i] - sc["poses3d"][i].mean(0),
                                  sc["R"]) for i in range(P.shape[0])])
    d_ort, o_ort = report("正交投影（与训练分布一致）", Po, P2o)

    print("\n" + "=" * 72)
    print("结论")
    print(f"  正交投影下 正确配对 d = {d_ort:.4f}  区分度 {o_ort/d_ort:.2f}x")
    print(f"  透视投影下 正确配对 d = {d_per:.4f}  区分度 {o_per/d_per:.2f}x")
    if d_per > 2 * d_ort:
        print("  → 投影模型不一致是主因：网络在正交投影上训练，却被要求处理透视投影。")
        print("    calibrate 的演示场景应改成弱透视（目标放远、横向范围收窄），")
        print("    或者训练时也加透视投影的数据增强。")
    else:
        print("  → 投影差异不是主因，需要继续查别的原因。")
    print("\n参照：训练分布上的 val Lkp = 0.2446（由 demo_best.pt 的日志给出）")


if __name__ == "__main__":
    main()
