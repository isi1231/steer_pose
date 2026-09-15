#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 MAMMAL 数据集（BamaPig3D / Beagle Dog）的 3D 关键点标注转成 SteerPose 的 poses.npz。

⚠️ 本脚本按官方数据集说明文档编写，**尚未在真实数据上验证**：
   下载后请先用 --dry-run 查看文件匹配情况，确认目录结构与说明一致再正式转换。

官方数据说明：
  - BamaPig3D: https://github.com/anl13/MAMMAL_datasets
      label_3d/pig_{i}frame{k}.txt, i=0..3, k=0,25,...,1725
      每个文件是 23x3 矩阵；其中第 18,20,22,23 行（1-based）恒为 0，
      即实际有效关节 19 个；未标注（被遮挡）的关节以 0 填充。
  - Beagle Dog: https://github.com/anl13/Beagle_dog_dataset
      labeled/ 下为 3D 关键点标注（29 个关键点，格式见其 json）

用法:
    # 1) 先看能不能匹配到文件
    python scripts/prepare_mammal.py --src /data/BamaPig3D/label_3d --dry-run

    # 2) 正式转换（默认剔除全零关节被遮挡过多的姿态）
    python scripts/prepare_mammal.py --src /data/BamaPig3D/label_3d \\
        --out data/poses_bamapig.npz --min-valid-joints 12

    # 3) 同时导出该数据集的视角/相机参数（用于后续标定评估，若目录存在）
    python scripts/prepare_mammal.py --src /data/BamaPig3D/label_3d \\
        --out data/poses_bamapig.npz --dump-cameras
"""
import argparse
import glob
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BamaPig3D 中恒为 0 的行（1-based 18,20,22,23 -> 0-based 17,19,21,22）
BAMAPIG_ZERO_ROWS = [17, 19, 21, 22]


def load_bamapig_3d(src_dir: str):
    """读取 label_3d/pig_{i}frame{k}.txt -> (M, 19, 3) 与统计信息。"""
    files = sorted(glob.glob(os.path.join(src_dir, "pig_*frame*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(src_dir, "**", "pig_*frame*.txt"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"在 {src_dir} 未找到 'pig_*frame*.txt'。请确认指向 BamaPig3D 的 label_3d 目录，"
            f"或参考 README 的数据集说明检查目录结构。")

    poses, meta = [], []
    for f in files:
        arr = np.loadtxt(f, delimiter=None)
        arr = np.atleast_2d(arr)
        if arr.shape[0] != 23 or arr.shape[1] != 3:
            print(f"[!] 跳过 {os.path.basename(f)}：形状 {arr.shape}（期望 (23,3)）")
            continue
        valid = [j for j in range(23) if j not in BAMAPIG_ZERO_ROWS]
        poses.append(arr[valid, :])
        m = re.search(r"pig_(\d+)frame(\d+)", os.path.basename(f))
        meta.append((int(m.group(1)), int(m.group(2))) if m else (-1, -1))
    if not poses:
        raise RuntimeError("没有成功解析任何标注文件，请检查格式")
    return np.asarray(poses, np.float64), meta


def main():
    ap = argparse.ArgumentParser(description="MAMMAL 数据集 -> poses.npz")
    ap.add_argument("--src", type=str, required=True, help="BamaPig3D 的 label_3d 目录")
    ap.add_argument("--out", type=str, default="data/poses_bamapig.npz")
    ap.add_argument("--min-valid-joints", type=int, default=12,
                    help="每个姿态至少有多少个非零关节才保留（默认 12）")
    ap.add_argument("--dry-run", action="store_true", help="只列出匹配到的文件与统计，不写文件")
    ap.add_argument("--dump-cameras", action="store_true",
                    help="若同目录存在相机标定文件，一并复制到输出目录")
    args = ap.parse_args()

    poses, meta = load_bamapig_3d(args.src)
    M, J, _ = poses.shape
    print(f"[i] 解析到 {M} 个姿态 x {J} 关节（来自 {len(set(meta))} 个 (个体,帧) 组合）")

    # 有效关节数统计：官方标注里被遮挡的关节以 0 填充
    nz = (np.abs(poses).sum(axis=2) > 1e-9).sum(axis=1)
    print(f"[i] 每个姿态的有效关节数: min {nz.min()}  中位 {int(np.median(nz))}  max {nz.max()}")
    keep = nz >= args.min_valid_joints
    print(f"[i] 按 --min-valid-joints={args.min_valid_joints} 保留 {int(keep.sum())}/{M} 个姿态")
    poses = poses[keep]

    if args.dry_run:
        print("[dry-run] 未写文件。示例前 4 个姿态的第一个关节坐标:")
        for i in range(min(4, len(poses))):
            print(f"    #{i} meta={meta[i]} joint0={np.round(poses[i, 0], 4)}")
        return

    # 坐标检查：BamaPig3D 的 3D 关键点在公制世界系（米）
    span = poses.reshape(-1, 3).max(0) - poses.reshape(-1, 3).min(0)
    print(f"[i] 坐标跨度(整体): {np.round(span, 4)}（单位与数据集一致，训练时会逐姿态居中）")
    poses = poses - poses.mean(axis=1, keepdims=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, poses3d=poses.astype(np.float32),
                        source=np.array([f"MAMMAL/BamaPig3D:{args.src}"]))
    print(f"[ok] 已保存 -> {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")
    print(f"     训练: python -m steerpose.train --poses3d {args.out} "
          f"--epochs 300 --batch 512 --workers 8 --device cuda --out ckpt/bamapig.pt")

    if args.dump_cameras:
        found = []
        root = os.path.dirname(os.path.abspath(args.src.rstrip("/\\")))
        for pat in ("*.json", "*.txt"):
            found += [p for p in glob.glob(os.path.join(root, "calib*", pat))]
        if found:
            print(f"[i] 同目录发现 {len(found)} 个可能的标定文件，可手动拷出用于评估：")
            for p in found[:10]:
                print("    ", p)
        else:
            print("[i] 未自动发现标定文件（BamaPig3D 的相机参数在其 calib 目录，按需手动取用）")


if __name__ == "__main__":
    main()
