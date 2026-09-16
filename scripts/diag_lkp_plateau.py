# -*- coding: utf-8 -*-
"""诊断：训练损失 Lkp 卡在 ~0.64 到底是"数据/实现有问题"还是"优化步数不够"。

三个对照实验：
  A. 平凡基线 —— 直接输出"平均姿态"、直接输出输入视角的姿态（不旋转），看 Lkp 是多少。
     如果模型只做到基线水平，说明它什么都没学到；如果基线本身就 0.6 上下，
     说明 0.64 这个数其实"离基线不远"。
  B. 过拟合小样本 —— 只取 64 对训练，跑到几百 epoch。若能把 Lkp 压到接近 0，
     说明网络结构/损失/数据管线本身没问题，之前只是欠拟合。
  C. 长训练 —— 同样数据，把步数拉长 20 倍，看 Lkp 是否显著下降。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.data import normalize_pose, build_pairs  # noqa: E402
from steerpose.geometry import make_demo_dataset  # noqa: E402
from steerpose.losses import lkp_mean  # noqa: E402
from steerpose.model import SteerPose  # noqa: E402


def norm_all(P):
    return np.stack([normalize_pose(p) for p in P]).astype(np.float32)


def baselines(Pn, P2n):
    """三个平凡预测器的 Lkp（正样本对内部，逐对计算再求均值）。"""
    J = Pn.shape[1]
    mean_target = P2n.mean(axis=0)                      # (J,2) 全数据集平均姿态
    out = {}
    d = np.linalg.norm(np.broadcast_to(mean_target, P2n.shape) - P2n, axis=-1).mean()
    out["输出平均姿态"] = float(d)
    d = np.linalg.norm(np.zeros_like(P2n) - P2n, axis=-1).mean()
    out["输出零（归一化姿态的中心）"] = float(d)
    d = np.linalg.norm(Pn - P2n, axis=-1).mean()
    out["直接抄输入（不旋转）"] = float(d)
    return out


def train(Pn, P2n, RV, epochs, batch, lr=1e-3, seed=0, log_every=None, tag=""):
    torch.manual_seed(seed)
    model = SteerPose(num_joints=Pn.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    P = torch.from_numpy(Pn)
    P2 = torch.from_numpy(P2n)
    R = torch.from_numpy(RV)
    n = P.shape[0]
    steps = 0
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            q = model(P[idx], R[idx])
            loss = lkp_mean(q, P2[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            tot += loss.item() * idx.numel()
            cnt += idx.numel()
        sched.step()
        hist.append(tot / max(cnt, 1))
        if log_every and ((ep + 1) % log_every == 0 or ep == epochs - 1):
            print(f"      [{tag}] epoch {ep+1:4d}  steps {steps:6d}  Lkp {hist[-1]:.4f}")
    return hist, steps


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    poses = make_demo_dataset(num_anims=200, num_frames=20, seed=1)
    print(f"演示数据 {poses.shape}  坐标系跨度 "
          f"{np.round(poses.reshape(-1,3).max(0)-poses.reshape(-1,3).min(0),3)}")

    pairs = build_pairs(poses, num_pairs=3000, num_views=100, num_rolls=20, seed=0)
    Pn = norm_all(pairs["P"])
    P2n = norm_all(pairs["P2"])
    RV = pairs["rotvec"].astype(np.float32)
    print(f"训练对 {Pn.shape}  旋转角分布："
          f"中位 {np.median(np.degrees(np.linalg.norm(RV,axis=1))):.1f}°  "
          f"最大 {np.degrees(np.linalg.norm(RV,axis=1)).max():.1f}°")

    print("\n===== A. 平凡基线（正样本对）=====")
    for k, v in baselines(Pn, P2n).items():
        print(f"   {k:24s} Lkp = {v:.4f}")

    print("\n===== B. 过拟合小样本（64 对，看能否压到 ~0）=====")
    b_idx = np.arange(64)
    hist_b, steps_b = train(Pn[b_idx], P2n[b_idx], RV[b_idx], epochs=600,
                            batch=64, lr=1e-3, log_every=100, tag="64对")
    print(f"   最终 Lkp {hist_b[-1]:.4f}（{steps_b} 步）")

    print("\n===== C. 全量数据长训练（3000 对 → 用 2100 训练）=====")
    tr = np.arange(2100)
    for epochs, batch in ((50, 512), (400, 512)):
        hist_c, steps_c = train(Pn[tr], P2n[tr], RV[tr], epochs=epochs, batch=batch,
                               lr=1e-3, log_every=max(epochs // 5, 1),
                               tag=f"{epochs}ep/b{batch}")
        print(f"   → {epochs} epoch = {steps_c} 步，最终 Lkp {hist_c[-1]:.4f}")

    print("\n===== 结论判据 =====")
    mean_base = baselines(Pn, P2n)["输出平均姿态"]
    print(f"   基线（平均姿态）Lkp = {mean_base:.4f}")
    print(f"   小样本过拟合 Lkp   = {hist_b[-1]:.4f}  "
          f"({'能压下去 → 实现无 bug，只是欠拟合' if hist_b[-1] < 0.15 else '压不下去 → 怀疑实现有 bug'})")


if __name__ == "__main__":
    main()
