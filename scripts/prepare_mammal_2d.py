#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 BamaPig3D 构造 SteerPose 两视角标定实验的输入：P、P' 与真值相对旋转 R_gt。

数据来源（官方 pure_pickle 精简版，481MB 那个即可）：
  - label_keypoints2d.pkl : (70, 10, 4, 19, 3)
        轴含义 [帧, 相机序号, 个体, 关节, (x, y, valid)]
        相机序号对应官方 camids = [0,1,2,5,6,7,8,9,10,11]
        x/y 是**已去畸变图像**(label_images)上的像素坐标
  - label_3d.pkl / label_mix.pkl : (70, 4, 19, 3) 或 (70, 4, 23, 3)
  - extrinsic_camera_params/{camid:02d}.txt : 6 个浮点数
        前 3 个为相机旋转（axis-angle），后 3 个为平移 xyz，单位米
        约定 x_cam = R @ X_world + t（与官方 visualize_BamaPig3D.py 一致）
  - intrinsic_camera_params/distortion_info.pkl : 内含 newcameramtx
        ★ label_images 已经去畸变，所以标签对应的相机矩阵是 newcameramtx，
          不是 undistortion.py 里那个带畸变的原始 K。本脚本自动读取。

输出 npz（直接喂给 steerpose.calibrate）：
    P    : (B1, 19, 2)  视角 A 的 2D 姿态（像素坐标）
    P2   : (B2, 19, 2)  视角 B 的 2D 姿态（像素坐标）
    R    : (3, 3)       真值相对旋转（A -> B，满足 R = R_b @ R_a^T）
    t    : (3,)         真值相对平移（仅方向有意义）
    K    : (3, 3)       内参（newcameramtx）—— calibrate 会直接用它，
                        等价于给 --focal fx 与主点 (cx, cy)
    meta_* : 帧号 / 个体 / 相机 ID，便于溯源

用法:
    # 论文 Table 7：标定用帧 1400-1750，取相机 0 与 6
    python scripts/prepare_mammal_2d.py --root /data/BamaPig3D_pure_pickle \\
        --cam-a 0 --cam-b 6 --min-frame 1400 \\
        --out data/two_view_pig_c0_c6.npz

    # 也可以分别指定文件
    python scripts/prepare_mammal_2d.py \\
        --kp2d /data/BamaPig3D_pure_pickle/label_keypoints2d.pkl \\
        --extrinsics /data/BamaPig3D/extrinsic_camera_params \\
        --cam-a 0 --cam-b 6 --min-frame 1400 --out data/two_view_pig_c0_c6.npz

    # 模拟论文"GT 2D + 高斯噪声 σ=3px"的设置
    python scripts/prepare_mammal_2d.py --root ... --noise-px 3 --seed 0 \\
        --out data/two_view_noisy.npz

随后标定:
    python -m steerpose.calibrate --ckpt ckpt/bamapig_best.pt \\
        --poses-npz data/two_view_pig_c0_c6.npz --out out/two_view_pig.npz
"""
import argparse
import glob
import os
import pickle
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_mammal_data import load_intrinsics  # noqa: E402  （同目录下的复用）

CAMIDS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]   # 官方 10 个视角的顺序
NUM_JOINTS = 19


def load_kp2d(path: str) -> np.ndarray:
    """读取 label_keypoints2d.pkl -> (F, C, P, J, 3)。"""
    with open(path, "rb") as f:
        arr = np.asarray(pickle.load(f))
    if arr.ndim != 5:
        raise ValueError(f"{path} 期望 5 维 (帧, 相机, 个体, 关节, 3)，实际 {arr.shape}")
    if arr.shape[3] != NUM_JOINTS:
        raise ValueError(f"关节数应为 {NUM_JOINTS}，实际 {arr.shape[3]}（请确认用的是官方 pkl）")
    return arr


def load_extrinsics(folder: str, camids) -> dict:
    """读取 extrinsic_camera_params/{camid}.txt -> {camid: (R(3,3), t(3,))}。"""
    out = {}
    for cid in camids:
        cands = [os.path.join(folder, f"{cid:02d}.txt"),
                 os.path.join(folder, f"{cid}.txt"),
                 os.path.join(folder, f"cam{cid}.txt")]
        path = next((p for p in cands if os.path.isfile(p)), None)
        if path is None:
            found = sorted(glob.glob(os.path.join(folder, "*.txt")))
            m = [p for p in found if re.search(rf"(^|\D)0*{cid}(\D|$)", os.path.basename(p))]
            if not m:
                raise FileNotFoundError(
                    f"在 {folder} 未找到相机 {cid} 的外参文件（试过 {cands}）。\n"
                    f"  该目录现有 {len(found)} 个 txt：{ [os.path.basename(p) for p in found[:10]] }")
            path = m[0]
        vals = np.loadtxt(path).ravel()
        if vals.size < 6:
            raise ValueError(f"{path} 应含 6 个浮点数（axis-angle 旋转 + 平移），实际 {vals.size}")
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec(vals[:3]).as_matrix()
        t = vals[3:6]
        out[cid] = (R, t)
    return out


def extract_view(kp: np.ndarray, cam_idx: int, frame_idx: int, min_valid: int):
    """取某帧某相机的所有个体 2D 姿态 -> (poses (B,19,2), pigs, valid_counts)。"""
    poses, pigs = [], []
    n_pigs = kp.shape[2]
    for p in range(n_pigs):
        xy = kp[frame_idx, cam_idx, p, :, :2].astype(np.float64)
        valid = kp[frame_idx, cam_idx, p, :, 2] > 0.5
        if valid.sum() < min_valid:
            continue
        # 缺失关节用该姿态有效关节的质心填充（避免 0 坐标把归一化尺度带偏）
        if (~valid).any():
            xy[~valid] = xy[valid].mean(axis=0)
        poses.append(xy)
        pigs.append(p)
    return np.asarray(poses), np.asarray(pigs)


def main():
    ap = argparse.ArgumentParser(description="BamaPig3D 两视角标定数据准备")
    ap.add_argument("--root", type=str, default=None,
                    help="BamaPig3D_pure_pickle 目录（自动找 label_keypoints2d.pkl / "
                         "extrinsic_camera_params / intrinsic_camera_params）")
    ap.add_argument("--kp2d", type=str, default=None, help="label_keypoints2d.pkl 路径")
    ap.add_argument("--extrinsics", type=str, default=None,
                    help="extrinsic_camera_params 目录")
    ap.add_argument("--cam-a", type=int, default=0, help="视角 A 的相机 ID")
    ap.add_argument("--cam-b", type=int, default=6, help="视角 B 的相机 ID")
    ap.add_argument("--frame", type=int, default=None, help="只取某一帧（默认见 --min-frame）")
    ap.add_argument("--min-frame", type=int, default=1400,
                    help="帧号 >= 该值（论文标定用 1400-1750；设 -1 表示不限制）")
    ap.add_argument("--max-frame", type=int, default=None, help="帧号 <= 该值")
    ap.add_argument("--min-valid-joints", type=int, default=12,
                    help="每个姿态至少多少有效关节才保留")
    ap.add_argument("--noise-px", type=float, default=0.0,
                    help="给 2D 坐标加的高斯噪声（像素）。论文合成设置用 σ=3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/two_view_bamapig.npz")
    args = ap.parse_args()

    # ---- 定位输入文件 ----
    kp2d_path, extr_dir = args.kp2d, args.extrinsics
    if args.root:
        kp2d_path = kp2d_path or os.path.join(args.root, "label_keypoints2d.pkl")
        extr_dir = extr_dir or os.path.join(args.root, "extrinsic_camera_params")
    if not kp2d_path or not os.path.isfile(kp2d_path):
        raise SystemExit(f"[!] 找不到 label_keypoints2d.pkl：{kp2d_path}\n"
                         f"    请给 --root <BamaPig3D_pure_pickle 目录> 或 --kp2d <文件>")
    if not extr_dir or not os.path.isdir(extr_dir):
        raise SystemExit(f"[!] 找不到 extrinsic_camera_params 目录：{extr_dir}\n"
                         f"    请给 --root 或 --extrinsics")

    kp = load_kp2d(kp2d_path)
    F = kp.shape[0]
    # 帧号映射：官方每 25 帧标注一次（第 k 个标注帧 = 25k）
    frame_ids = np.arange(F) * 25
    if args.frame is not None:
        sel = [int(np.argmin(np.abs(frame_ids - args.frame)))]
    else:
        sel = list(range(F))
        if args.min_frame is not None and args.min_frame >= 0:
            sel = [i for i in sel if frame_ids[i] >= args.min_frame]
        if args.max_frame is not None:
            sel = [i for i in sel if frame_ids[i] <= args.max_frame]
    print(f"[i] label_keypoints2d 形状 {kp.shape}，选取 {len(sel)} 个标注帧: "
          f"{[int(frame_ids[i]) for i in sel][:8]}{' ...' if len(sel) > 8 else ''}")

    cams = CAMIDS
    ia, ib = cams.index(args.cam_a), cams.index(args.cam_b)
    extr = load_extrinsics(extr_dir, [args.cam_a, args.cam_b])
    Ra, ta = extr[args.cam_a]
    Rb, tb = extr[args.cam_b]
    R_rel = Rb @ Ra.T                       # 世界->相机约定下的 A->B 相对旋转
    t_rel = tb - R_rel @ ta                 # 对应的相对平移（方向有意义）
    print(f"[i] 相机 {args.cam_a} 与 {args.cam_b} 的真值相对旋转已由外参算出")

    # ---- 内参：label_images 已去畸变 -> 用 newcameramtx ----
    K = None
    if args.root:
        K, ksrc, _ = load_intrinsics(args.root, verbose=False)
        print(f"[i] 内参（{ksrc}）fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
              f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        print("     注意：label_images 已去畸变，标签对应的就是 newcameramtx，无需再处理畸变")
    else:
        print("[i] 未给 --root，未写入内参；calibrate 时请用 --focal 指定像素焦距")

    rng = np.random.default_rng(args.seed)
    Ps, P2s, metas = [], [], []
    for f in sel:
        pa, pigs_a = extract_view(kp, ia, f, args.min_valid_joints)
        pb, pigs_b = extract_view(kp, ib, f, args.min_valid_joints)
        if len(pa) == 0 or len(pb) == 0:
            print(f"[!] 帧 {frame_ids[f]}：某视角无有效姿态（A {len(pa)} / B {len(pb)}），跳过")
            continue
        # 只保留两个视角都有效的个体（便于用序号一致性评估匹配精度）
        common = sorted(set(pigs_a.tolist()) & set(pigs_b.tolist()))
        if not common:
            print(f"[!] 帧 {frame_ids[f]}：两视角没有共同个体，跳过")
            continue
        pa = pa[[list(pigs_a).index(p) for p in common]]
        pb = pb[[list(pigs_b).index(p) for p in common]]
        Ps.append(pa); P2s.append(pb)
        metas.append((frame_ids[f], common))
        if len(sel) <= 12:
            print(f"    帧 {int(frame_ids[f])}: {len(common)} 个共同个体 {common}")

    if not Ps:
        raise RuntimeError("没有得到任何可用帧，请放宽 --min-valid-joints 或检查帧范围")

    P = np.concatenate(Ps, axis=0)
    P2 = np.concatenate(P2s, axis=0)
    if args.noise_px > 0:
        P = P + rng.normal(0, args.noise_px, P.shape)
        P2 = P2 + rng.normal(0, args.noise_px, P2.shape)
        print(f"[i] 已加高斯噪声 σ = {args.noise_px} px")
    print(f"[i] 视角 A {P.shape}，视角 B {P2.shape}（含 {len(Ps)} 帧）")

    out_kw = {}
    if K is not None:
        out_kw["K"] = K.astype(np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, P=P.astype(np.float32), P2=P2.astype(np.float32),
             R=R_rel.astype(np.float64), t=t_rel.astype(np.float64),
             frames=np.array([m[0] for m in metas], np.int32),
             cam_a=args.cam_a, cam_b=args.cam_b, **out_kw)
    print(f"[ok] 已保存 -> {args.out}"
          f"{'（含 K）' if K is not None else ''}")
    print(f"     标定: python -m steerpose.calibrate --ckpt ckpt/bamapig_best.pt "
          f"--poses-npz {args.out} --out out/two_view_c{args.cam_a}_c{args.cam_b}.npz")
    print(f"     先跑数据体检: python scripts/check_mammal_data.py --root "
          f"{args.root or '<pure_pickle 目录>'}")


if __name__ == "__main__":
    main()
