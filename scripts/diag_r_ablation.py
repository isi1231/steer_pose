# -*- coding: utf-8 -*-
"""R 消融实验：网络到底有没有在用相对旋转 R？

判据：训练完之后，把 batch 内的 R 打乱（模型看到错误的旋转）或直接置零，重算 Lkp。
  * Lkp 基本不变   -> 网络**忽略了 R**，它只是输出了"平均姿态"，这就是 Lkp 卡住的根因；
  * Lkp 显著变差   -> 网络确实在用 R，损失卡住只是欠拟合 / 步数不够。

对照组：
  - 模型 M_mem  ：只在 64 对上过拟合（Lkp 能到 ~0.07）——用它检验"记忆"能不能绕过 R；
  - 模型 M_full ：全量数据正常训练若干步。
另外给出"平均姿态基线"和一个**理想参考**：同一姿态在两个视角下、按真值 R 由
3D 重投影算出的目标，看看 Lkp 的理论可达范围。
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


def train(Pn, P2n, RV, epochs, batch, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = SteerPose(num_joints=Pn.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    P, P2, R = (torch.from_numpy(x) for x in (Pn, P2n, RV))
    n = P.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = lkp_mean(model(P[idx], R[idx]), P2[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def eval_under(model, Pn, P2n, RV, mode="true", seed=0):
    """mode: true / shuffle / zero / half。返回 Lkp。"""
    P, P2, R = (torch.from_numpy(x) for x in (Pn, P2n, RV))
    g = torch.Generator().manual_seed(seed)
    if mode == "shuffle":
        R = R[torch.randperm(R.shape[0], generator=g)]
    elif mode == "zero":
        R = torch.zeros_like(R)
    elif mode == "half":
        R = R * 0.5
    return float(lkp_mean(model(P, R), P2).item())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    poses = make_demo_dataset(num_anims=200, num_frames=20, seed=1)
    pairs = build_pairs(poses, num_pairs=3000, num_views=100, num_rolls=20, seed=0)
    Pn, P2n, RV = norm_all(pairs["P"]), norm_all(pairs["P2"]), pairs["rotvec"].astype(np.float32)

    ang = np.degrees(np.linalg.norm(RV, axis=1))
    print(f"R 的旋转角：中位 {np.median(ang):.1f}°  P90 {np.percentile(ang,90):.1f}°  "
          f"最大 {ang.max():.1f}°")
    print(f"平均姿态基线 Lkp = {np.linalg.norm(np.broadcast_to(P2n.mean(0), P2n.shape) - P2n, axis=-1).mean():.4f}")

    runs = []
    print("\n--- 训练 ---")
    print("  M_mem : 64 对过拟合 600 epoch")
    m_mem = train(Pn[:64], P2n[:64], RV[:64], epochs=600, batch=64)
    print("  M_full: 2100 对，50 epoch × batch512（= 用户那次训练）")
    m_full = train(Pn[:2100], P2n[:2100], RV[:2100], epochs=50, batch=512)
    print("  M_long: 2100 对，400 epoch × batch64（= 16500 步）")
    m_long = train(Pn[:2100], P2n[:2100], RV[:2100], epochs=400, batch=64)

    for name, model, sl in (("M_mem(64对)", m_mem, slice(0, 64)),
                            ("M_full(250步)", m_full, slice(0, 600)),
                            ("M_long(16500步)", m_long, slice(0, 600))):
        print(f"\n===== {name} =====")
        vals = {m: eval_under(model, Pn[sl], P2n[sl], RV[sl], mode=m)
                for m in ("true", "shuffle", "zero", "half")}
        print(f"   R=真值     Lkp {vals['true']:.4f}")
        print(f"   R=打乱     Lkp {vals['shuffle']:.4f}   "
              f"（相比真值变化 {vals['shuffle']-vals['true']:+.4f}）")
        print(f"   R=0        Lkp {vals['zero']:.4f}   （{vals['zero']-vals['true']:+.4f}）")
        print(f"   R=减半     Lkp {vals['half']:.4f}   （{vals['half']-vals['true']:+.4f}）")
        delta = vals['shuffle'] - vals['true']
        print(f"   判定：{'网络**忽略 R**（打乱 R 几乎不影响损失）' if delta < 0.05 else '网络在**使用 R**'}")
        runs.append((name, vals, delta))

    print("\n" + "=" * 74)
    print("结论")
    for name, vals, delta in runs:
        print(f"  {name:18s} Lkp(真值R) {vals['true']:.4f}   Δ(打乱R) {delta:+.4f}")


if __name__ == "__main__":
    main()
