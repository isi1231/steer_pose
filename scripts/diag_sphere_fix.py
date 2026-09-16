# -*- coding: utf-8 -*-
"""决定性对照：训练相机集合「上半球」vs「完整球面」，只改这一件事。

⚠️ 本脚本会**训练两个模型**（各 2500 步，CPU 上约 6 分钟、H100 上几十秒）。
   它是给**服务器/你自己终端**跑的 A/B 验证脚本，不要在本机随手跑。
   现在这个开关已经做进 `steerpose.train`（`--cameras sphere|hemisphere`），
   如果你已经有 quick 档权重，用下面两条命令做 A/B 通常更省事：

       bash scripts/run_h100.sh data/poses_bamapig_train.npz quick   # 默认 sphere
       python -m steerpose.train --poses3d ... --cameras hemisphere ...  # 旧行为对照

背景（上一轮排查的结论）
------------------------
`make_two_view_scene`（`calibrate --demo` 用的场景）把**相机 A 放在 np.eye(3)**，
即沿世界 +z 方向看；而 `ViewSynthesizer` 原来的 `fibonacci_hemisphere` 只取
z >= 0.05 的视点。对正交投影，"从下方看"恰好是"从上方看"沿一条轴的**镜像**，
训练集里根本不存在这种 2D 形状 → 网络输入落在分布外 → 输出崩掉
（实测 Lkp 0.23 → 0.60，12 个目标 0 个被正确匹配，Lmatch 一直降不下来，
因此 Sinkhorn 给不出可靠匹配，Lgeom 因为匹配数不足 3 而**恒等于 0**）。

⚠️ 关键认知：relative pose (R, t) 对"世界系怎么选"是**规范不变**的
（整体旋转同时作用于姿态与两台相机，P、P'、R、t 全都不变）。
所以相机 A 放哪里**不影响标定问题本身**，影响的只是网络输入 P 的分布。
把相机 A 放在单位矩阵 = 换了一台物理上不同的相机（镜像视角），
而不是换了个世界坐标系 —— 这就是那个 bug 的真正性质。

本脚本做什么
------------
同一份 3D 姿态、同一套超参、同一随机种子，训练两个模型，唯一差别是
`ViewSynthesizer(full_sphere=...)`。然后用三种测量口径打分：

  V1 正交投影 / 相机 A = I        —— 相机 A 落在训练集合的另一侧（分布外输入）
  V2 **透视投影 / 相机 A = I**    —— 就是 `calibrate --demo` 曾经的输入路径
  V3 正交投影 / 相机 A 取自训练集合 —— 修复后 `--demo` 用的口径（分布内参照）

⚠️ 与"修复"的关系（别把两件事混起来）：
`--demo` 崩掉的直接原因是**场景本身**选了分布外的相机 A，修法是把
`make_two_view_scene(camera_a=...)` 的默认值改成 `"training"`（已改）。
本脚本衡量的是**另一个**决定：训练侧要不要把相机集合从论文的"上半球"
（附录 D.1 原文：100 cameras uniformly over the unit hemisphere）扩到**完整球面**。
关系到实用鲁棒性——真实相机相对姿态规范系可能落在任意一侧，而半球训练下的失效是静默的。

期望结果：
  * 两个模型都能把 val Lkp 压到明显低于平凡基线 → "损失不收敛"不是数据问题（是步数）；
  * 上半球模型：V1/V2 崩（Lkp ≈ 0.6，匹配 0 个正确），V3 正常
    → 证明"输出全错"与训练好坏无关，纯粹是输入落在分布外；
  * 完整球面模型：V1/V2/V3 都正常 → 完整球面训练可以直接免疫这类失效。

用法（**在服务器/你自己的终端跑**）：python scripts/diag_sphere_fix.py
（脚本会顺带把两个权重写到 ckpt/diag_hemi.pt / ckpt/diag_full.pt，
  之后可以直接 `python -m steerpose.calibrate --ckpt ckpt/diag_full.pt --demo`）
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.data import normalize_pose  # noqa: E402
from steerpose.geometry import (ViewSynthesizer, make_two_view_scene,  # noqa: E402
                                matrix_to_rodrigues, ortho_project)
from steerpose.losses import lkp_mean, matching_loss  # noqa: E402
from steerpose.model import SteerPose  # noqa: E402

NUM_PAIRS = 3000
STEPS = 2500
BATCH = 256
LR = 1e-3
SEED = 0


def to_norm(a: np.ndarray) -> np.ndarray:
    return np.stack([normalize_pose(p) for p in a]).astype(np.float32)


def train(full_sphere: bool, poses3d: np.ndarray, num_joints: int, log=print):
    """和 steerpose.train 同样的数据/增强/损失，只把优化循环写成简单的定步数版本。"""
    vs = ViewSynthesizer(num_views=100, num_rolls=20, num_pairs=NUM_PAIRS,
                         seed=SEED, full_sphere=full_sphere)
    pairs = vs.make_training_set(poses3d)
    P, P2 = to_norm(pairs["P"]), to_norm(pairs["P2"])
    RV = pairs["rotvec"].astype(np.float32)

    n_tr, n_va = int(NUM_PAIRS * 0.7), int(NUM_PAIRS * 0.2)
    tr = np.arange(0, n_tr)
    va = np.arange(n_tr, n_tr + n_va)

    # 平凡基线：无论输入什么都输出训练集目标姿态的均值
    mu = P2[tr].mean(axis=0)
    baseline = float(np.linalg.norm(mu[None] - P2[va], axis=-1).mean())

    torch.manual_seed(SEED)
    model = SteerPose(num_joints=num_joints)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    g = torch.Generator().manual_seed(SEED)

    P_t, P2_t, RV_t = (torch.from_numpy(x) for x in (P, P2, RV))
    t0 = time.time()
    for step in range(STEPS):
        idx = torch.randint(0, n_tr, (BATCH,), generator=g)
        p, p2, rv = P_t[idx], P2_t[idx], RV_t[idx]
        # 与 PosePairDataset 一致：输入加 N(0, 0.02) 噪声 + 随机掩码 10%~30% 关节
        p = p + 0.02 * torch.randn(p.shape, generator=g)
        ratio = torch.empty(BATCH, 1).uniform_(0.1, 0.3, generator=g)
        mask = (torch.rand(BATCH, num_joints, generator=g) >= ratio).float()
        q = model(p, rv, mask)
        loss = lkp_mean(q, p2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % (STEPS // 5) == 0:
            log(f"    step {step+1:5d}/{STEPS}  train Lkp {loss.item():.4f}"
                f"  ({time.time()-t0:.0f}s)")

    model.eval()
    with torch.no_grad():
        val = float(lkp_mean(model(P_t[va], RV_t[va]), P2_t[va]))
    return model, val, baseline


@torch.no_grad()
def score(model, P0, P20, rv):
    """按 calibrate 的口径打分：Lkp / 匹配置信度 / argmax 正确率。"""
    Pn = torch.from_numpy(to_norm(P0))
    P2n = torch.from_numpy(to_norm(P20))
    rvt = torch.from_numpy(np.asarray(rv, np.float32))
    Q = model(Pn, rvt)
    lkp = float(lkp_mean(Q, P2n))
    _, W, _ = matching_loss(Q, P2n)
    conf = float(W.max(dim=1).values.mean())
    acc = float((W.argmax(dim=1).numpy() == np.arange(len(Pn))).mean())
    return lkp, conf, acc


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from steerpose.geometry import make_demo_dataset

    poses3d = make_demo_dataset(num_anims=200, seed=1)
    J = poses3d.shape[1]
    print(f"[i] 演示 3D 姿态 {poses3d.shape}，目标 {STEPS} 步 × batch {BATCH}\n")

    models = {}
    for tag, full in (("上半球（旧默认）", False), ("完整球面（新默认）", True)):
        print(f"[i] 训练 {tag} 模型  full_sphere={full}")
        m, val, baseline = train(full, poses3d, J)
        print(f"    → val Lkp {val:.4f}   平凡基线 {baseline:.4f}"
              f"   （{(1-val/baseline)*100:.1f}% 相对下降）\n")
        torch.save({"model": m.state_dict(), "num_joints": J, "epoch": STEPS},
                   f"ckpt/diag_{'full' if full else 'hemi'}.pt")
        models[tag] = (m, val, baseline)

    # ---------- 三个测量口径，用同一批 12 个姿态 ----------
    sc = make_two_view_scene(n_obj=12, seed=99)
    X = sc["poses3d"]
    Xc = X - X.mean(axis=1, keepdims=True)
    n = len(Xc)
    rv_scene = matrix_to_rodrigues(sc["R"])[None].repeat(n, 0)

    V = {}
    V["V1 正交 / 相机A=I（老口径）"] = (
        np.stack([ortho_project(Xc[i], np.eye(3)) for i in range(n)]),
        np.stack([ortho_project(Xc[i], sc["R"]) for i in range(n)]), rv_scene)
    V["V2 透视 / 相机A=I（--demo 实际路径）"] = (
        sc["P"], sc["P2"], rv_scene)

    print("=" * 88)
    print("按 calibrate 的口径打分（Lkp 越低越好；conf 是 Sinkhorn 行最大置信度；"
          "acc 是 argmax 匹配正确比例）")
    print("=" * 88)
    for tag, (m, val, baseline) in models.items():
        print(f"\n【{tag}】val Lkp {val:.4f}（基线 {baseline:.4f}）")
        for vname, (P0, P20, rv) in V.items():
            lkp, conf, acc = score(m, P0, P20, rv)
            print(f"  {vname:34s} Lkp {lkp:6.4f}   conf {conf:.3f}   acc {acc:.2f}")
        # V3 用与**该模型训练时同源**的相机集合，公平比较
        vs = ViewSynthesizer(num_views=100, num_rolls=20, num_pairs=64,
                             seed=7, full_sphere=(tag.startswith("完整")))
        Ra, Rb, _ = vs.sample_pair()
        lkp, conf, acc = score(
            m,
            np.stack([ortho_project(Xc[i], Ra) for i in range(n)]),
            np.stack([ortho_project(Xc[i], Rb) for i in range(n)]),
            matrix_to_rodrigues(Rb @ Ra.T)[None].repeat(n, 0))
        print(f"  {'V3 正交 / 相机A 取自训练集合':34s} Lkp {lkp:6.4f}   "
              f"conf {conf:.3f}   acc {acc:.2f}")

    print("\n" + "=" * 88)
    print("判读：")
    print("  · 两个模型的 val Lkp 都远低于平凡基线 → 收敛没问题，之前的'损失不收敛'")
    print("    是**优化步数**造成的（250 步时模型停在基线上）。")
    print("  · 上半球模型在 V1/V2 崩、在 V3 正常 → '输出全错'与训练好坏无关，")
    print("    是相机 A 落在训练集合另一侧（分布外输入）导致的。")
    print("  · 完整球面模型若 V1/V2/V3 都正常 → 训练侧用 sphere 可以免疫这类失效；")
    print("    论文严格复现用 --cameras hemisphere，此时 --demo 也已由 camera_a='training' 兜住。")


if __name__ == "__main__":
    main()
