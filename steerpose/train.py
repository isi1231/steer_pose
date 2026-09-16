# -*- coding: utf-8 -*-
"""SteerPose 训练入口（全监督，损失 Lkp，论文 3.1 式(1) 与附录 D.1/D.3）.

示例（服务器）:
    # 1) 用真实 3D 姿态训练（推荐：Animal3D / MAMMAL 导出为 poses.npz）
    python -m steerpose.train --poses3d data/poses_quadruped.npz \\
        --epochs 300 --batch 512 --lr 1e-3 --out ckpt/steerpose_quad.pt

    # 2) 无数据集时先跑通流程（程序化演示数据）
    python -m steerpose.train --demo --epochs 50 --num-pairs 2000 --out ckpt/demo.pt

训练细节（论文附录 D.3）：
    - 训练对由 3D 姿态 + 随机视角对合成（默认 3000 对，7:2:1 划分）
    - 输入加 2D 噪声、随机掩码 10%~30% 关节
    - 优化器 Adam，lr 1e-3（论文对 SteerPose 网络未给 lr，这里给常用默认值）
"""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PosePairDataset, build_pairs, split_indices, load_poses3d, normalize_pose
from .geometry import make_demo_dataset
from .losses import lkp_mean
from .model import SteerPose


def parse_args():
    ap = argparse.ArgumentParser(description="SteerPose 训练")
    src = ap.add_argument_group("数据")
    src.add_argument("--poses3d", type=str, default=None,
                     help="3D 姿态 npz（内含 poses3d: (M,J,3)）")
    src.add_argument("--demo", action="store_true", help="用程序化演示数据（无数据集时自测）")
    src.add_argument("--demo-anims", type=int, default=200)
    src.add_argument("--num-pairs", type=int, default=3000, help="合成相机对数量（默认 3000）")
    src.add_argument("--num-views", type=int, default=100, help="Fibonacci 视点数（默认 100）")
    src.add_argument("--num-rolls", type=int, default=20, help="每视点滚转数（默认 20）")
    src.add_argument("--cameras", choices=["sphere", "hemisphere"], default="sphere",
                     help="合成训练对所用的相机集合。sphere=完整球面（默认，覆盖全部 SO(3)）；"
                          "hemisphere=只取上半球（旧行为）。"
                          "只用上半球时训练集里不存在'镜像视角'（等价于从下半球看），"
                          "遇到这类输入网络会输出完全错误的结果（实测 Lkp 0.23→0.60，"
                          "跨视角匹配 0 个正确），而训练曲线完全看不出来。"
                          "改这个是 A/B 实验，请在服务器上跑，别在本机跑。")

    tr = ap.add_argument_group("训练")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch", type=int, default=512)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=0.0)
    tr.add_argument("--workers", type=int, default=4, help="DataLoader worker 数（Windows 设 0）")
    tr.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    tr.add_argument("--amp", action="store_true", help="混合精度（H100 上可用 bf16）")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log-every", type=int, default=10)

    out = ap.add_argument_group("输出")
    out.add_argument("--out", type=str, default="ckpt/steerpose.pt", help="权重保存路径")
    out.add_argument("--save-every", type=int, default=50, help="每 N 个 epoch 存一次（0=只存最后）")
    out.add_argument("--resume", type=str, default=None)
    return ap.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> float:
    model.eval()
    tot, n = 0.0, 0
    for p, p2, rv, m in loader:
        p, p2, rv, m = p.to(device), p2.to(device), rv.to(device), m.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            q = model(p, rv, m)
            loss = lkp_mean(q, p2)
        tot += loss.item() * p.shape[0]; n += p.shape[0]
    model.train()
    return tot / max(n, 1)


def mean_pose_baseline(pairs: dict, idx_tr, idx_va) -> float:
    """平凡预测器基线：无论输入什么都输出"训练集目标姿态的平均值"。

    这是判断模型到底有没有学到东西的**唯一可靠参照**。若 val Lkp 与它相差无几，
    说明网络什么都没学到 —— 常见原因是**优化步数不够**（epochs × 每轮批数），
    而不是数据/实现有问题。本仓库实测：同一份数据 250 步时 Lkp 0.636（≈基线 0.648），
    2000 步时降到 0.133。
    """
    tr = np.stack([normalize_pose(pairs["P2"][i]) for i in idx_tr])
    va = np.stack([normalize_pose(pairs["P2"][i]) for i in idx_va])
    mu = tr.mean(axis=0)
    return float(np.linalg.norm(mu[None] - va, axis=-1).mean())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- 数据 ----------
    if args.poses3d:
        poses3d = load_poses3d(args.poses3d)
    else:
        print("[i] 未提供 --poses3d，使用程序化演示数据（仅供跑通流程）")
        poses3d = make_demo_dataset(num_anims=args.demo_anims, seed=1)
    M, J, _ = poses3d.shape
    print(f"[i] 3D 姿态: {M} 个 x {J} 关节")

    t0 = time.time()
    pairs = build_pairs(poses3d, num_pairs=args.num_pairs,
                        num_views=args.num_views, num_rolls=args.num_rolls,
                        seed=args.seed, full_sphere=args.cameras == "sphere")
    idx = split_indices(args.num_pairs)
    print(f"[i] 合成训练对 {args.num_pairs} 对（耗时 {time.time()-t0:.1f}s），"
          f"train/val/test = {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}")
    print(f"[i] 相机集合: {args.cameras}"
          f"（{'完整球面，覆盖全部朝向' if args.cameras == 'sphere' else '仅上半球 —— 缺失一半朝向，推理时遇镜像视角会失效'}）")

    ds_tr = PosePairDataset(pairs, idx["train"], noise_std=0.02, seed=args.seed)
    ds_va = PosePairDataset(pairs, idx["val"], noise_std=0.01, seed=args.seed + 1)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=False,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers)

    # ---------- 模型 ----------
    device = pick_device(args.device)
    model = SteerPose(num_joints=J).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[i] 设备 {device} | 模型参数量 {n_param/1e3:.1f}K")
    if args.amp and device.type == "cuda":
        print("[i] 启用自动混合精度（bf16）")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start_ep = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_ep = ck.get("epoch", 0)
        print(f"[i] 从 {args.resume} 恢复（epoch {start_ep}）")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log_path = os.path.splitext(args.out)[0] + "_train.log"
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"[i] 开始训练: epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"pairs={args.num_pairs} joints={J}")

    # ---- 关键参照：平凡基线 + 总优化步数 ----
    # 训练是否"真的在学"，唯一可靠的判据是 val Lkp 相对这个基线下降了多少。
    # 只跑几十个 epoch（batch 又大）时总步数可能只有两三百步，模型会停在基线上不动，
    # 看起来就像"数据烂/实现错"，其实只是没训够。这里主动算出来并存进日志。
    steps_per_epoch = max(1, math.ceil(len(ds_tr) / max(args.batch, 1)))
    total_steps = steps_per_epoch * max(args.epochs - start_ep, 1)
    baseline = mean_pose_baseline(pairs, idx["train"], idx["val"])
    log(f"[i] 优化规模：每轮 {steps_per_epoch} 步 × {args.epochs} epoch = "
        f"共 {total_steps} 步")
    log(f"[i] 平凡基线（输出平均姿态）val Lkp = {baseline:.4f}")
    log("    ↳ 判定标准：val Lkp 必须明显低于这个基线才算学到东西。"
        "若两者相近，先加总步数（提高 --epochs 或调小 --batch），"
        "而不是怀疑数据或实现。")

    best = float("inf")
    for ep in range(start_ep, args.epochs):
        tot, n = 0.0, 0
        for p, p2, rv, m in dl_tr:
            p, p2, rv, m = p.to(device, non_blocking=True), p2.to(device, non_blocking=True), \
                rv.to(device, non_blocking=True), m.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                q = model(p, rv, m)
                loss = lkp_mean(q, p2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * p.shape[0]; n += p.shape[0]
        sched.step()

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1,
              "num_joints": J, "config": vars(args)}

        # 每个记录点只评估一次，并据此更新 best。末轮一定评估：
        # 否则当 epochs 不是 log_every 的整数倍（例如 --epochs 5 而 log_every=10）
        # 时 _best.pt 永远不会被写出，最终只能拿到最后一轮的权重。
        if (ep + 1) % args.log_every == 0 or ep == args.epochs - 1:
            val = evaluate(model, dl_va, device, args.amp and device.type == "cuda")
            improved = val < best
            if improved:
                best = val
                torch.save(ck, os.path.splitext(args.out)[0] + "_best.pt")
            log(f"  epoch {ep+1:4d}/{args.epochs}  train Lkp {tot/max(n,1):.4f}  "
                f"val Lkp {val:.4f}  lr {sched.get_last_lr()[0]:.2e}"
                f"{'  *best' if improved else ''}")

        if args.save_every and (ep + 1) % args.save_every == 0:
            torch.save(ck, os.path.splitext(args.out)[0] + f"_ep{ep+1}.pt")

    # 训练跑完但一次都没评估（epochs<=0 等极端情况）时兜底，避免 best 仍是 inf
    if not math.isfinite(best):
        best = evaluate(model, dl_va, device, args.amp and device.type == "cuda")
        torch.save(ck, os.path.splitext(args.out)[0] + "_best.pt")

    torch.save(ck, args.out)
    with open(os.path.splitext(args.out)[0] + "_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    log(f"[ok] 训练结束，最后权重 -> {args.out}；最佳权重 -> "
        f"{os.path.splitext(args.out)[0]}_best.pt（best val Lkp {best:.4f}）")
    log(f"     日志 -> {log_path}")
    if best > 0.8 * baseline:
        log("")
        log("[!] 警告：val Lkp 与'输出平均姿态'的平凡基线几乎相同"
            f"（{best:.4f} vs {baseline:.4f}），说明模型基本**没有学到东西**。")
        log("    最可能的原因是优化步数太少：本次共 "
            f"{total_steps} 步（每轮 {steps_per_epoch} 步）。")
        log("    处理办法（按优先级）：")
        log(f"      1) 提高 --epochs（当前 {args.epochs}）：小模型 + 小数据集时 epoch 很便宜，"
            "直接开到 1000~3000")
        log(f"      2) 调小 --batch（当前 {args.batch}）：每轮步数会按比例变多")
        log(f"      3) 加大 --num-pairs（当前 {args.num_pairs}）：每步见到的多样性更多")
        log("    注意：这一条**不是**数据或实现的问题。同样数据实测 250 步 → Lkp 0.64（=基线），"
            "2000 步 → Lkp 0.13。")
        log("    标定（calibrate）必须用真正降下来的权重，否则匹配置信度会接近均匀指派、"
            "Lgeom 静默失效。")
    log_f.close()


if __name__ == "__main__":
    main()
