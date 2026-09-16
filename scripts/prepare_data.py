#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把任意来源的 3D 姿态数据整理成 SteerPose 训练所需的 poses.npz。

标准格式（训练脚本要求）:
    poses3d : (M, J, 3) float32/64   —— M 个 3D 姿态，每个 J 个关节，世界系坐标
    （可选）joint_names : (J,)       —— 关节名，仅用于记录

用法示例:
    # 1) 无数据集时生成演示数据（程序化四足骨架）
    python scripts/prepare_data.py --demo --out data/poses_demo.npz

    # 2) 从已有 npz/npy 整理（自动识别键名与维度）
    python scripts/prepare_data.py --input raw/mammal_pig.npz --out data/poses_quadruped.npz

    # 3) 指定键名、只保留/重排部分关节（顺序即骨架定义顺序）
    python scripts/prepare_data.py --input raw/animal3d.npz --key joints3d \\
        --joints 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \\
        --out data/poses_animal3d.npz

    # 4) 从 .mat 读取（需要 scipy）
    python scripts/prepare_data.py --input raw/pose.mat --key pose3d --out data/poses.npz

整理的检查项:
    - 维度必须是 (..., J, 3)，多余的批次/帧/个体维度会被展平成 M
    - NaN/Inf 会被报告并剔除
    - 每个姿态会被居中（去掉全局平移，训练时同样处理）
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from steerpose.geometry import make_demo_dataset  # noqa: E402


def load_raw(path: str, key: str = None):
    """读取 npz / npy / mat，返回候选数组字典 {key: array}。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return {"array": np.load(path, allow_pickle=True)}
    if ext == ".npz":
        f = np.load(path, allow_pickle=True)
        if key:
            if key not in f:
                raise KeyError(f"{path} 中无键 '{key}'，现有: {list(f.keys())}")
            return {key: f[key]}
        return {k: f[k] for k in f.keys()}
    if ext == ".mat":
        from scipy.io import loadmat
        f = loadmat(path)
        d = {k: v for k, v in f.items() if not k.startswith("__")}
        if key:
            if key not in d:
                raise KeyError(f"{path} 中无键 '{key}'，现有: {list(d.keys())}")
            return {key: d[key]}
        return d
    raise ValueError(f"不支持的格式: {ext}（支持 .npz/.npy/.mat）")


def pick_array(cands: dict, key: str = None) -> tuple:
    """自动挑选形状形如 (..., J, 3) 的数组。返回 (name, array)。"""
    if key:
        return key, np.asarray(cands[key])
    for k, v in cands.items():
        v = np.asarray(v)
        if v.ndim >= 3 and v.shape[-1] == 3 and v.dtype.kind in "fiu":
            return k, v
    raise ValueError(f"未找到形如 (..., J, 3) 的数组，候选: "
                     f"{ {k: np.asarray(v).shape for k, v in cands.items()} }，"
                     f"请用 --key 指定")


def main():
    ap = argparse.ArgumentParser(description="整理 3D 姿态数据为 poses.npz")
    ap.add_argument("--input", type=str, default=None, help="原始文件 (.npz/.npy/.mat)")
    ap.add_argument("--key", type=str, default=None, help="数组键名（默认自动识别）")
    ap.add_argument("--demo", action="store_true", help="生成程序化演示数据")
    ap.add_argument("--demo-anims", type=int, default=200)
    ap.add_argument("--demo-frames", type=int, default=20)
    ap.add_argument("--joints", type=str, default=None,
                    help="保留/重排的关节索引，如 0,1,2,...（默认全部）")
    ap.add_argument("--no-center", action="store_true", help="不做逐姿态居中")
    ap.add_argument("--out", type=str, required=True, help="输出 npz 路径")
    args = ap.parse_args()

    if args.demo or not args.input:
        if not args.demo:
            print("[i] 未提供 --input，生成演示数据")
        poses = make_demo_dataset(num_anims=args.demo_anims,
                                  num_frames=args.demo_frames, seed=1)
        src = "demo(程序化四足骨架)"
    else:
        cands = load_raw(args.input, args.key)
        name, arr = pick_array(cands, args.key)
        arr = np.asarray(arr)
        print(f"[i] 读取 {args.input} 键 '{name}'，原始形状 {arr.shape}")
        # 展平除最后两维之外的所有维度 -> (M, J, 3)
        poses = arr.reshape(-1, arr.shape[-2], arr.shape[-1]).astype(np.float64)
        src = f"{os.path.basename(args.input)}:{name}"

    # 关节选择/重排
    if args.joints:
        sel = [int(x) for x in args.joints.split(",") if x.strip() != ""]
        if max(sel) >= poses.shape[1]:
            raise ValueError(f"关节索引 {max(sel)} 超出自有 {poses.shape[1]} 个关节")
        poses = poses[:, sel, :]

    # 质量检查
    M, J, D = poses.shape
    bad = ~np.isfinite(poses).all(axis=(1, 2))
    if bad.any():
        print(f"[!] 发现 {int(bad.sum())} 个姿态含 NaN/Inf，已剔除")
        poses = poses[~bad]
        M = poses.shape[0]
    if D != 3:
        raise ValueError(f"最后一维应为 3，实际 {D}")
    span = poses.reshape(-1, 3).max(0) - poses.reshape(-1, 3).min(0)
    print(f"[i] 有效数据: {M} 个姿态 x {J} 关节")
    print(f"[i] 坐标范围(整体): {np.round(span, 3)}  (尺度任意，训练时会居中)")

    if not args.no_center:
        poses = poses - poses.mean(axis=1, keepdims=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, poses3d=poses.astype(np.float32),
                        source=np.array([src]))
    print(f"[ok] 已保存 -> {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")
    print(f"\n     训练命令（注意：决定成败的是**总优化步数** = 每轮批数 × epoch，")
    print(f"     不是 epoch 数。这个模型只有 ~69K 参数，epoch 很便宜，先给足步数）：")
    print(f"     python -m steerpose.train --poses3d {args.out} \\")
    print(f"         --epochs 2000 --batch 256 --workers 4 --device cuda --out ckpt/run.pt")
    print(f"     或直接用脚本按目标步数自动折算：")
    print(f"     bash scripts/run_h100.sh {args.out} quick")


if __name__ == "__main__":
    main()
