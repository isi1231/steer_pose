#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练单目 3D 提升器 `steerpose.lifting.PoseLifter`（RA-L2022 链路的第一环）。

为什么损失要这么设计
--------------------
RA-L2022 的标定**只用骨方向**（oriented points）：单目 3D 的绝对深度/尺度不可观测，
但"这根骨头朝哪个方向"相对可靠，而方向恰好是标定所需的全部信息。所以：

  主损失 = 骨方向余弦 `1 - cos`        （对尺度、平移完全免疫）
  辅助项 = 尺度不变的 MPJPE            （只约束相对深度结构，不惩罚尺度）

**不能用普通 MPJPE 当主损失**：单目尺度本来就不存在，"把坐标回归准"是个无解的
目标，只会让网络在尺度上浪费容量。

★ 决定成败的是**总优化步数**（每轮批数 × epoch），不是 epoch 数。
  本模型只有 ~100K 参数，一个 epoch 极便宜；给不够步数时它会停在
  "输出训练集平均姿态"的平凡基线上，看起来就像数据/实现有问题。
  本脚本会打印**总步数**和**两条平凡基线的实测指标**，val 接近基线时显式警告。

两条平凡基线（用于判断"模型到底学到东西没有"）
----------------------------------------------
  * `triv_ortho`：正交提升 —— 直接把归一化 2D `(u,v)` 当成 `(u,v,0)`。
    这是单目 3D 最经典的下界；网络打不过它说明基本没学到深度。
  * `triv_mean` ：永远输出训练集平均姿态（根相对+归一化的均值）。
    这是"完全不看输入"的下界；打不过它说明网络还不如常量。

用法
----
    # ⓪ 只体检数据和划分（不建模型、不占显存）
    python scripts/train_lifter.py --npz data/lifter_bamapig_train.npz --dry-run

    # ① 正式训练（H100）
    python scripts/train_lifter.py --npz data/lifter_bamapig_train.npz \\
        --out ckpt/lifter_bamapig.pt --device cuda --workers 8 --steps 40000

    # ② 先小步数确认链路
    python scripts/train_lifter.py --npz data/lifter_bamapig_train.npz \\
        --out ckpt/lifter_quick.pt --steps 2000 --batch 256 --workers 4

    # ③ 断点续训
    python scripts/train_lifter.py --npz data/lifter_bamapig_train.npz \\
        --resume ckpt/lifter_bamapig.pt --device cuda

训练完成后：
    # 用提升器 + 几何后端做多视角标定（脚本见 docs/ral_backend.md）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steerpose.lifting import (                                        # noqa: E402
    LifterDataset, PoseLifter, bone_direction_error_deg, bone_direction_loss,
    build_frame_split, build_split_indices, count_params,
    procrustes_mpjpe, scale_invariant_mpjpe,
)
from steerpose.skeleton import get_skeleton, bones_to_index_array         # noqa: E402


# ============================ 平凡基线 ============================

def _scale_norm_np(y3: np.ndarray, bones, mask: np.ndarray, root: int):
    """numpy 版"根相对 + 平均骨长归一化"（与训练时的可微版本同规则）。"""
    b = bones_to_index_array(bones)
    y = y3 - y3[:, root: root + 1]
    d = np.linalg.norm(y[:, b[:, 0], :] - y[:, b[:, 1], :], axis=-1)     # (N,Bn)
    w = mask[:, b[:, 0]] * mask[:, b[:, 1]]
    s = (d * w).sum(-1) / np.maximum(w.sum(-1), 1.0)
    s = np.maximum(s, 1e-9)
    return y / s[:, None, None]


def baseline_ortho(X2: np.ndarray, bones, mask: np.ndarray, root: int) -> np.ndarray:
    """正交提升：归一化 2D (u,v) -> (u,v,0)，再做根相对+骨长归一化。"""
    z = np.zeros_like(X2[..., :1])
    return _scale_norm_np(np.concatenate([X2, z], axis=-1).astype(np.float64),
                          bones, mask, root)


def baseline_mean(Y3_train: np.ndarray, n: int) -> np.ndarray:
    """永远输出训练集平均姿态（完全不看输入）。"""
    return np.repeat(Y3_train.mean(axis=0, keepdims=True), n, axis=0)


@torch.no_grad()
def metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, bones,
            device: str) -> dict:
    """把 numpy 预测/真值算成一组指标（骨方向误差 + P-MPJPE + 尺度不变 MPJPE）。"""
    p = torch.as_tensor(pred, dtype=torch.float32, device=device)
    g = torch.as_tensor(gt, dtype=torch.float32, device=device)
    m = torch.as_tensor(mask, dtype=torch.float32, device=device)
    return {
        "bone_deg": bone_direction_error_deg(p, g, bones, mask=m),
        "p_mpjpe": procrustes_mpjpe(p, g, mask=m),
        "si_mpjpe": float(scale_invariant_mpjpe(p, g, bones, mask=m).item()),
    }


# ============================ 训练 / 验证 ============================

def validate(model, loader, device, sk, args):
    """验证：不更新参数，顺便把预测收集起来算指标。"""
    model.eval()
    bones, root = sk["bones"], sk["root"]
    tot_loss = tot_bone = tot_aux = 0.0
    n_batch = 0
    preds, gts, msks = [], [], []
    with torch.no_grad():
        for x2, msk, y3 in loader:
            x2 = x2.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            y3 = y3.to(device, non_blocking=True)
            pred = model(x2, msk)
            l_bone = bone_direction_loss(pred, y3, bones, mask=msk)
            l_aux = (scale_invariant_mpjpe(pred, y3, bones, mask=msk, root=root)
                     if args.w_mpjpe > 0 else torch.zeros((), device=device))
            tot_loss += float((l_bone + args.w_mpjpe * l_aux).item())
            tot_bone += float(l_bone.item())
            tot_aux += float(l_aux.item())
            n_batch += 1
            preds.append(pred.float().cpu().numpy())
            gts.append(y3.float().cpu().numpy())
            msks.append(msk.float().cpu().numpy())

    out = {"loss": tot_loss / max(n_batch, 1),
           "bone_loss": tot_bone / max(n_batch, 1),
           "aux_loss": tot_aux / max(n_batch, 1), "n_batch": n_batch}
    if preds:
        out.update(metrics(np.concatenate(preds, 0), np.concatenate(gts, 0),
                           np.concatenate(msks, 0), bones, "cpu"))
    return out



def main():
    ap = argparse.ArgumentParser(description="训练单目 3D 提升器（RA-L2022 链路第一环）")
    ap.add_argument("--npz", type=str, required=True,
                    help="prepare_mammal_lifter.py 产出的训练数据 npz")
    ap.add_argument("--out", type=str, default="ckpt/lifter_bamapig.pt")
    ap.add_argument("--skeleton", type=str, default="pig19")

    # --- 优化 ---
    ap.add_argument("--steps", type=int, default=0,
                    help="目标**总优化步数**（推荐 20000~40000）。给了就按它反推 epoch，"
                         "比 --epochs 可靠：决定成败的是总步数，不是 epoch 数")
    ap.add_argument("--epochs", type=int, default=200, help="--steps 为 0 时才用它")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW 权重衰减")
    ap.add_argument("--warmup-frac", type=float, default=0.05, help="warmup 占总步数比例")
    ap.add_argument("--w-mpjpe", type=float, default=0.1,
                    help="尺度不变 MPJPE 辅助损失权重（0=只用骨方向）")
    ap.add_argument("--clip", type=float, default=1.0, help="梯度裁剪（0=不裁）")
    ap.add_argument("--amp", action="store_true", help="混合精度（H100 上建议开）")

    # --- 数据 ---
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug-mask", action="store_true",
                    help="训练时随机再遮挡一部分关节（模拟检测漏检）")
    ap.add_argument("--aug-mask-lo", type=float, default=0.1)
    ap.add_argument("--aug-mask-hi", type=float, default=0.3)
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="训练时给归一化 2D 加的高斯噪声（0.005 ≈ 6px @fx=1200）")

    # --- 模型 ---
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)

    # --- 运行 ---
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50, help="每多少步打印一次训练损失")
    ap.add_argument("--resume", type=str, default=None, help="从 ckpt 续训")
    ap.add_argument("--dry-run", action="store_true",
                    help="只体检数据与划分，**不建模型**（纯 numpy，可随便跑）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    sk = get_skeleton(args.skeleton)
    bones, root = sk["bones"], sk["root"]

    # ---------------- 数据 ----------------
    if not os.path.isfile(args.npz):
        raise SystemExit(f"[!] 找不到训练数据 {args.npz}\n"
                         f"    先跑: python scripts/prepare_mammal_lifter.py "
                         f"--root <BamaPig3D_pure_pickle> --max-frame 1400 "
                         f"--out {args.npz}")
    d = np.load(args.npz, allow_pickle=True)
    need = ("X2", "Mask", "Y3", "FrameId")
    missing = [k for k in need if k not in d.files]
    if missing:
        raise SystemExit(f"[!] {args.npz} 缺少字段 {missing}；现有 {list(d.files)}\n"
                         f"    请用 scripts/prepare_mammal_lifter.py 重新生成。")
    X2, Mask, Y3, FrameId = d["X2"], d["Mask"], d["Y3"], d["FrameId"]
    J = X2.shape[1]
    if J != len(sk["names"]):
        raise SystemExit(f"[!] 数据关节数 {J} 与骨架 '{args.skeleton}' 的 "
                         f"{len(sk['names'])} 不一致")

    print("=" * 72)
    print("训练单目 3D 提升器（RA-L2022 链路第一环）")
    print("=" * 72)
    print(f"[i] 数据 {args.npz}")
    print(f"    X2 {X2.shape}  Mask {Mask.shape}  Y3 {Y3.shape}")
    print(f"    关节可见率 {Mask.mean():.3f}   样本数 {len(X2)}")
    n_frames = len(np.unique(FrameId))
    print(f"    帧数 {n_frames}（帧范围 {FrameId.min()}~{FrameId.max()}），"
          f"相机 {sorted(np.unique(d['CamId']).tolist()) if 'CamId' in d.files else '?'}")

    # 按帧划分：同一帧的不同相机样本必须落在同一侧，否则验证集会被"泄漏"抬高
    split = build_frame_split(FrameId, val_ratio=args.val_ratio, seed=args.seed)
    vals = set(np.array(FrameId)[split["val"]].tolist())
    trs = set(np.array(FrameId)[split["train"]].tolist())
    leak = vals & trs
    print(f"[i] 按帧划分：train {len(split['train'])} 条 / val {len(split['val'])} 条；"
          f"共享帧 {len(leak)}（应为 0）")
    if leak:
        raise SystemExit("[!] train/val 共享帧，划分失效，请检查 FrameId 字段")
    if len(split["val"]) < 20:
        print(f"[!] 验证集只有 {len(split['val'])} 条，指标噪声会很大")

    # ---- 平凡基线（关键：用来判断模型有没有学到东西）----
    va = split["val"]
    gt_val, mask_val = Y3[va], Mask[va]
    bl = {
        "triv_ortho": metrics(baseline_ortho(X2[va], bones, mask_val, root),
                              gt_val, mask_val, bones, "cpu"),
        "triv_mean": metrics(baseline_mean(Y3[split["train"]], len(va)),
                             gt_val, mask_val, bones, "cpu"),
    }
    print("\n[i] 平凡基线（验证集上，模型必须明显优于它们）")
    for k, v in bl.items():
        print(f"    {k:11s} 骨方向 {v['bone_deg']:6.2f}°   "
              f"P-MPJPE {v['p_mpjpe']:.4f}   尺度不变MPJPE {v['si_mpjpe']:.4f}")

    n_train, n_val = len(split["train"]), len(split["val"])
    steps_per_epoch = max(1, int(np.ceil(n_train / args.batch)))
    if args.steps > 0:
        epochs = max(1, int(np.ceil(args.steps / steps_per_epoch)))
    else:
        epochs = args.epochs
        args.steps = epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    print(f"\n[i] 每轮 {steps_per_epoch} 步 → {epochs} 轮 = **总 {total_steps} 步**"
          f"（batch {args.batch}，lr {args.lr}）")
    if total_steps < 3000:
        print(f"[!] 总步数 {total_steps} < 3000，模型很可能停在平凡基线上。"
              f"请提高 --steps（建议 20000~40000）。")

    if args.dry_run:
        print("\n[dry-run] 未建模型、未训练（纯 numpy 体检）。")
        return 0

    # ---------------- 模型 ----------------
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[!] 没有可用的 CUDA，回退到 CPU（会非常慢，建议上服务器跑）")
        device = "cpu"
    model = PoseLifter(num_joints=J, d_model=args.d_model, nhead=args.nhead,
                       num_layers=args.layers, dim_feedforward=args.ff,
                       dropout=args.dropout).to(device)
    print(f"\n[i] 模型 PoseLifter：{count_params(model):,} 参数  "
          f"d_model={args.d_model} layers={args.layers} nhead={args.nhead}  "
          f"device={device}")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[i] 已载入 {args.resume}"
              f"（其中记录的 step={ck.get('step', '?')}）")

    ds_tr = LifterDataset(X2, Mask, Y3, indices=split["train"], aug_mask=args.aug_mask,
                          mask_ratio=(args.aug_mask_lo, args.aug_mask_hi),
                          noise_std=args.noise_std, seed=args.seed)
    ds_va = LifterDataset(X2, Mask, Y3, indices=split["val"])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=False, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, int(total_steps * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warm) / max(1, total_steps - warm)))))
    use_amp = bool(args.amp) and device.startswith("cuda")
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):          # 老版本 torch 的写法
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    best = float("inf")
    hist, gstep, t0 = [], 0, time.time()

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = run_bone = run_aux = 0.0
        n_seen = 0
        for x2, msk, y3 in dl_tr:
            x2 = x2.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            y3 = y3.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(x2, msk)
                l_bone = bone_direction_loss(pred, y3, bones, mask=msk)
                l_aux = (scale_invariant_mpjpe(pred, y3, bones, mask=msk, root=root)
                         if args.w_mpjpe > 0 else torch.zeros((), device=device))
                loss = l_bone + args.w_mpjpe * l_aux
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
            sched.step()

            run_loss += float(loss.item()); run_bone += float(l_bone.item())
            run_aux += float(l_aux.item()); n_seen += 1; gstep += 1
            if args.log_every > 0 and gstep % args.log_every == 0:
                print(f"    ep{ep} step {gstep}/{total_steps}  "
                      f"loss {run_loss/n_seen:.4f} (骨 {run_bone/n_seen:.4f} "
                      f"辅助 {run_aux/n_seen:.4f})  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  "
                      f"{gstep/max(time.time()-t0,1e-9):.1f} step/s")

        va_m = validate(model, dl_va, device, sk, args)
        tr_loss = run_loss / max(n_seen, 1)
        line = (f"  [ep {ep:4d}/{epochs}] step {gstep:6d}  "
                f"train loss {tr_loss:.4f}   val loss {va_m['loss']:.4f}  "
                f"val 骨方向 {va_m['bone_deg']:6.2f}°  "
                f"P-MPJPE {va_m['p_mpjpe']:.4f}  "
                f"(基线 ortho {bl['triv_ortho']['bone_deg']:.2f}° / "
                f"mean {bl['triv_mean']['bone_deg']:.2f}°)")
        print(line)
        hist.append({"epoch": ep, "step": gstep, "train_loss": tr_loss, **va_m})

        if va_m["loss"] < best:
            best = va_m["loss"]
            torch.save({"model": model.state_dict(), "step": gstep, "epoch": ep,
                        "cfg": vars(args), "num_joints": J,
                        "joint_names": list(sk["names"]), "bones": list(sk["bones"]),
                        "calib_bones": list(sk["calib_bones"]), "root": root,
                        "val": va_m, "baselines": bl, "skeleton": args.skeleton},
                       args.out)
            print(f"        ✓ 保存 {args.out}（val loss {best:.4f}，"
                  f"骨方向 {va_m['bone_deg']:.2f}°）")

    # ---------------- 收尾 ----------------
    print("\n" + "=" * 72)
    ck = torch.load(args.out, map_location="cpu", weights_only=False)
    v, b = ck["val"], ck["baselines"]
    print(f"[ok] 最佳 ckpt -> {args.out}（step {ck['step']}）")
    print(f"     val 骨方向误差 {v['bone_deg']:.2f}°   "
          f"P-MPJPE {v['p_mpjpe']:.4f}   尺度不变MPJPE {v['si_mpjpe']:.4f}")
    print(f"     基线        ortho {b['triv_ortho']['bone_deg']:.2f}°  "
          f"mean {b['triv_mean']['bone_deg']:.2f}°")
    gain = b["triv_ortho"]["bone_deg"] - v["bone_deg"]
    if gain < 1.0:
        print(f"[!] 相对正交基线只提升 {gain:.2f}°，几乎没学到东西。"
              f"\n    排查：① 总步数是否够（建议 ≥20000）；② lr 是否合适"
              f"（1e-3 起）；③ 数据里 X2 是否为**按内参归一化**的坐标。")
    else:
        print(f"     ✓ 相对正交基线提升 {gain:.2f}°")
    print(f"     日志 -> {os.path.splitext(args.out)[0]}_history.json")
    with open(os.path.splitext(args.out)[0] + "_history.json", "w",
              encoding="utf-8") as f:
        json.dump({"baselines": bl, "history": hist, "total_steps": gstep,
                   "epochs": epochs, "steps_per_epoch": steps_per_epoch},
                  f, ensure_ascii=False, indent=2)

    print(f"\n下一步（多视角标定）：")
    print(f"  # ① 准备标定输入包（★ 用相机对，不要用全部 10 台）")
    print(f"  python scripts/prepare_mammal_lifter.py --root <pure_pickle> "
          f"--min-frame 1400 --calib-cams 0,6 --no-train-out "
          f"--dump-calib data/lifter_calib_c0_c6.npz")
    print(f"  # ② 见 docs/ral_backend.md 的 calibrate 示例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
