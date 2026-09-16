#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 MAMMAL / BamaPig3D 的 3D 关键点标注转成 SteerPose 训练用的 poses.npz。

支持两种来源（官方格式，见 https://github.com/anl13/MAMMAL_datasets ）：
  A. 原始 txt 目录（BamaPig3D/label_3d/ 或 label_mix/，推荐 label_mix）
       pig_{i}_frame_{k:06d}.txt, i=0..3, k=0,25,...,1725（共 70 帧 × 4 头）
       每个文件 23x3 矩阵；第 18/20/22/23 行（1-based）恒为 0，
       官方有效索引 g_all_parts = [0..16, 18, 20] -> 19 个关节
       未被标注（遮挡）的关节以 0 填充
  B. 精简版 pure_pickle 的 label_3d.pkl / label_mix.pkl（只要 481MB，推荐）
       形状 (70, 4, 19, 3)（旧版脚本可能导成 (70,4,23,3)，本脚本自动兼容）

19 个有效关节顺序（官方 g_jointnames 去掉 4 个 "none"）：
   0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder,
   7 l_elbow, 8 r_elbow, 9 l_paw, 10 r_paw, 11 l_hip, 12 r_hip, 13 l_knee,
   14 r_knee, 15 l_foot, 16 r_foot, 17 tail, 18 center

论文 Bama Pig 的划分（附录 Table 7）：训练用帧 0–1400，标定用帧 1400–1750。
即：用 --max-frame 1400 取训练姿态；标定那部分用 prepare_mammal_2d.py。

用法:
    # A. 原始 txt 目录（先用 --dry-run 核对文件匹配）
    python scripts/prepare_mammal.py --src /data/BamaPig3D/label_mix --dry-run
    python scripts/prepare_mammal.py --src /data/BamaPig3D/label_mix \\
        --max-frame 1400 --out data/poses_bamapig_train.npz

    # B. 精简版 pkl（推荐，只要 481MB）
    python scripts/prepare_mammal.py --src /data/BamaPig3D_pure_pickle/label_3d.pkl \\
        --max-frame 1400 --out data/poses_bamapig_train.npz
"""
import argparse
import glob
import os
import pickle
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 官方 23 -> 19 的有效关节索引（utils.py 中的 g_all_parts）
G_ALL_PARTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
JOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
               "l_elbow", "r_elbow", "l_paw", "r_paw", "l_hip", "r_hip", "l_knee",
               "r_knee", "l_foot", "r_foot", "tail", "center"]
# 官方 bones（19 关节版），可用于可视化检查
BONES_19 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (0, 18), (18, 17), (18, 5), (5, 7),
            (7, 9), (18, 6), (6, 8), (8, 10), (17, 11), (11, 13), (13, 15), (17, 12),
            (12, 14), (14, 16)]


def _to_19(mat: np.ndarray) -> np.ndarray:
    """(..., 23, 3) -> (..., 19, 3)；(..., 19, 3) 原样返回。"""
    if mat.shape[-2] == 23:
        return mat[..., G_ALL_PARTS, :]
    if mat.shape[-2] == 19:
        return mat
    raise ValueError(f"关节维应为 23 或 19，实际形状 {mat.shape}")


def load_from_txt_dir(src_dir: str):
    """读取 label_3d / label_mix 目录下的 pig_{i}_frame_{k:06d}.txt。"""
    files = sorted(glob.glob(os.path.join(src_dir, "pig_*_frame_*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(src_dir, "**", "pig_*_frame_*.txt"),
                                 recursive=True))
    if not files:
        raise FileNotFoundError(
            f"在 {src_dir} 未找到 'pig_*_frame_*.txt'。\n"
            f"  请确认指向 BamaPig3D 的 label_3d 或 label_mix 目录"
            f"（官方命名形如 pig_0_frame_000000.txt）。\n"
            f"  若你只有精简版 pure_pickle，请把 --src 指到 label_3d.pkl / label_mix.pkl。")

    poses, meta = [], []
    for f in files:
        m = re.search(r"pig_(\d+)_frame_(\d+)", os.path.basename(f))
        if not m:
            continue
        arr = np.atleast_2d(np.loadtxt(f))
        if arr.shape[-1] != 3:
            print(f"[!] 跳过 {os.path.basename(f)}：形状 {arr.shape}")
            continue
        poses.append(_to_19(arr))
        meta.append((int(m.group(1)), int(m.group(2))))
    if not poses:
        raise RuntimeError("没有成功解析任何标注文件，请检查格式")
    return np.asarray(poses, np.float64), meta


def load_from_pickle(src_path: str):
    """读取 pure_pickle 的 label_3d.pkl / label_mix.pkl -> (M,19,3)。"""
    with open(src_path, "rb") as f:
        data = pickle.load(f)
    arr = np.asarray(data)
    if arr.ndim != 4:
        raise ValueError(f"{src_path} 期望 4 维 (帧, 个体, 关节, 3)，实际 {arr.shape}")
    n_frames, n_pigs = arr.shape[0], arr.shape[1]
    arr = _to_19(arr)
    poses = arr.reshape(-1, arr.shape[-2], 3)
    step = 25   # 官方每 25 帧标注一次
    meta = [(p, k * step) for k in range(n_frames) for p in range(n_pigs)]
    return poses.astype(np.float64), meta


def main():
    ap = argparse.ArgumentParser(description="BamaPig3D -> SteerPose poses.npz")
    ap.add_argument("--src", type=str, required=True,
                    help="label_3d/label_mix 目录，或 label_3d.pkl/label_mix.pkl 文件")
    ap.add_argument("--out", type=str, default="data/poses_bamapig_train.npz")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="只保留帧号 <= 该值的姿态（论文训练用 0-1400，可给 1400）")
    ap.add_argument("--min-frame", type=int, default=None, help="只保留帧号 >= 该值")
    ap.add_argument("--min-valid-joints", type=int, default=12,
                    help="每个姿态至少有多少个非零关节才保留（遮挡严重的丢弃，默认 12）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = ap.parse_args()

    if args.src.lower().endswith(".pkl"):
        poses, meta = load_from_pickle(args.src)
        src_desc = os.path.basename(args.src)
    else:
        poses, meta = load_from_txt_dir(args.src)
        src_desc = args.src
    M0 = len(poses)
    print(f"[i] 解析到 {M0} 个姿态 x {poses.shape[1]} 关节（来源 {src_desc}）")

    frames = np.array([m[1] for m in meta])
    pigs = np.array([m[0] for m in meta])
    print(f"[i] 帧号范围 {frames.min()}~{frames.max()}，个体 ID {sorted(set(pigs.tolist()))}")

    keep = np.ones(M0, bool)
    if args.max_frame is not None:
        keep &= frames <= args.max_frame
        print(f"[i] 保留帧号 <= {args.max_frame}：{int(keep.sum())}/{M0}")
    if args.min_frame is not None:
        keep &= frames >= args.min_frame
        print(f"[i] 保留帧号 >= {args.min_frame}：{int(keep.sum())}/{M0}")
    poses = poses[keep]
    meta = [m for m, k in zip(meta, keep) if k]

    # 有效关节数：官方标注中未标注（遮挡）的关节为 0
    nz = (np.abs(poses).sum(axis=2) > 1e-9).sum(axis=1)
    print(f"[i] 有效关节数: min {nz.min()}  中位 {int(np.median(nz))}  max {nz.max()}")
    ok = nz >= args.min_valid_joints
    print(f"[i] 按 --min-valid-joints={args.min_valid_joints} 保留 {int(ok.sum())}/{len(poses)} 个姿态")
    if int(ok.sum()) < 20:
        print("[!] 保留姿态过少（<20），训练会不稳定：可降低 --min-valid-joints，"
              "或改用 label_mix（它用 label_mesh 补齐了缺失关节）")
    poses = poses[ok]
    meta = [m for m, k in zip(meta, ok) if k]

    if args.dry_run:
        print("[dry-run] 未写文件。前 3 个姿态的 center 关节（索引 18）坐标:")
        for i in range(min(3, len(poses))):
            print(f"    #{i} meta(pig,frame)={meta[i]} center={np.round(poses[i, 18], 4)}")
        return

    span = poses.reshape(-1, 3).max(0) - poses.reshape(-1, 3).min(0)
    print(f"[i] 坐标跨度(整体): {np.round(span, 4)}（单位米，训练时会逐姿态居中）")
    poses = poses - poses.mean(axis=1, keepdims=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out,
                        poses3d=poses.astype(np.float32),
                        joint_names=np.array(JOINT_NAMES),
                        meta_pig=np.array([m[0] for m in meta], np.int32),
                        meta_frame=np.array([m[1] for m in meta], np.int32),
                        source=np.array([f"MAMMAL/BamaPig3D:{src_desc}"]))
    print(f"[ok] 已保存 -> {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB, "
          f"{poses.shape[0]} 个姿态)")
    print(f"     训练: bash scripts/run_h100.sh {args.out} quick"
          f"   # 按目标总步数自动折算 epoch")
    print(f"     或手动: python -m steerpose.train --poses3d {args.out} "
          f"--epochs 2000 --batch 256 --workers 8 --device cuda --out ckpt/bamapig.pt")
    print(f"     ⚠️ 决定成败的是**总优化步数**（每轮批数 × epoch），不是 epoch 数；"
          f"训练日志会打印总步数与平凡基线")


if __name__ == "__main__":
    main()
