#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BamaPig3D 数据体检 + 投影一致性验证（不需要训练、不需要任何权重，秒级）。

为什么先跑这个
--------------
在 train / calibrate 之前先确认"数据本身是自洽的"，可以一次性排除掉几乎所有
低级错误。最有价值的一项检查是 **3D GT 用官方外参+内参投影回图像，与官方 2D 标注
比对**：如果重投影误差很小（几个像素），说明

  * 骨架顺序（23→19 的 g_all_parts 映射）是对的
  * 外参约定 x_cam = R @ X_world + t 是对的
  * 内参的口径（去畸变后的 newcameramtx）是对的
  * 2D 标注的坐标系（原点在左上角、x 向右 y 向下）是对的

这四条恰好是后面 calibrate 全部几何计算的前提，且任何一条错了都不会报错、
只会让结果变得没意义。所以**先跑这个，再去训练**。

同时也定量报告数据的可用性：每台相机的重投影误差、每个关节的可见率、
哪些相机/关节特别差。

用法:
    python scripts/check_mammal_data.py --root /data/BamaPig3D_pure_pickle
    python scripts/check_mammal_data.py --root /data/BamaPig3D_pure_pickle --save-json out/report.json
    python scripts/check_mammal_data.py --root /tmp/mock_bamapig        # 先用假数据自测

输出:
    终端报告 + 可选 JSON + 可选验证图（--plot out/reproj_check.png）
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAMIDS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
G_ALL_PARTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
JOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
               "l_elbow", "r_elbow", "l_paw", "r_paw", "l_hip", "r_hip", "l_knee",
               "r_knee", "l_foot", "r_foot", "tail", "center"]
# 官方 undistortion.py 里给出的原始内参（含畸变）
OFFICIAL_K = np.array([[1625.30923, 0.0, 963.88710],
                       [0.0, 1625.34802, 523.45901],
                       [0.0, 0.0, 1.0]])
OFFICIAL_COEFF = np.array([-0.35582, 0.14595, -0.00031, -0.00004, 0.00000])
IMG_W, IMG_H = 1920, 1080


def _find(root, *cands):
    for c in cands:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def _disp(p):
    """路径只用于展示：统一分隔符，避免 Windows 上出现 /a/b\\c 的混搭。"""
    return os.path.normpath(p) if p else ""


def load_intrinsics(root, verbose=True):
    """读内参。优先官方 distortion_info.pkl / intrinsic_params.pkl 的 newcameramtx。"""
    pk = _find(root, "intrinsic_camera_params/distortion_info.pkl",
               "intrinsic_camera_params/intrinsic_params.pkl",
               "distortion_info.pkl")
    if pk:
        with open(pk, "rb") as f:
            d = pickle.load(f)
        if isinstance(d, dict):
            for key in ("newcameramtx", "new_camera_mtx", "K_new", "P"):
                if key in d:
                    if verbose:
                        print(f"[i] 内参来源：{os.path.normpath(pk)} 的 '{key}'")
                    return np.asarray(d[key], np.float64), os.path.normpath(pk), list(d.keys())
        if verbose:
            print(f"[!] {pk} 里没找到 newcameramtx 之类的键，改用官方硬编码内参。"
                  f"现有键：{list(d.keys()) if isinstance(d, dict) else type(d)}")
    # 兜底：用 undistortion.py 的原始 K + 官方去畸变流程（label_images 已去畸变）
    try:
        import cv2
        nm, roi = cv2.getOptimalNewCameraMatrix(OFFICIAL_K.astype(np.float32),
                                                OFFICIAL_COEFF.astype(np.float32),
                                                (IMG_W, IMG_H), 1, (IMG_W, IMG_H))
        if verbose:
            print("[i] 未找到内参 pkl，按 undistortion.py 的流程计算 newcameramtx "
                  "(alpha=1)")
        return nm.astype(np.float64), "cv2.getOptimalNewCameraMatrix(alpha=1)", None
    except ImportError:
        if verbose:
            print("[!] 没有 cv2，回退到原始 K（含畸变，重投影误差会偏大）")
        return OFFICIAL_K.copy(), "undistortion.py 的原始 K（未去畸变）", None


def load_extrinsics(folder):
    """{camid: (R(3,3), t(3,))}，约定 x_cam = R @ X_world + t。"""
    from scipy.spatial.transform import Rotation
    out = {}
    for cid in CAMIDS:
        p = _find(folder, f"{cid:02d}.txt", f"{cid}.txt")
        if p is None:
            continue
        v = np.loadtxt(p).ravel()
        if v.size < 6:
            print(f"[!] {p} 只有 {v.size} 个数，应为 6")
            continue
        out[cid] = (Rotation.from_rotvec(v[:3]).as_matrix(), v[3:6])
    return out


def main():
    ap = argparse.ArgumentParser(description="BamaPig3D 数据体检")
    ap.add_argument("--root", type=str, required=True,
                    help="BamaPig3D_pure_pickle 目录（或含同名文件的目录）")
    ap.add_argument("--label", type=str, default="label_mix",
                    help="用哪套 3D 标注：label_mix（官方实验用，默认）/ label_3d")
    ap.add_argument("--save-json", type=str, default=None)
    ap.add_argument("--plot", type=str, default=None, help="输出验证图路径（可选）")
    ap.add_argument("--plot-cam", type=int, default=0, help="验证图用哪个相机")
    ap.add_argument("--plot-frame-idx", type=int, default=35,
                    help="验证图用第几个标注帧（0-69，35 约等于帧 875）")
    args = ap.parse_args()

    root = args.root
    print("=" * 72)
    print(f"BamaPig3D 数据体检：{root}")
    print("=" * 72)

    # ---------- 1. 文件齐全性 ----------
    print("\n[1/5] 文件检查")
    paths = {
        "3D 标注": _find(root, f"{args.label}.pkl", "label_mix.pkl", "label_3d.pkl"),
        "2D 标注": _find(root, "label_keypoints2d.pkl"),
        "外参目录": _find(root, "extrinsic_camera_params"),
    }
    missing = [k for k, v in paths.items() if v is None]
    for k, v in paths.items():
        print(f"     {'OK ' if v else '缺失'} {k:8s} {_disp(v)}")
    if missing:
        print(f"\n[!] 缺少 {missing}。")
        print("    BamaPig3D_pure_pickle（481MB）应当同时含 label_mix.pkl、"
              "label_keypoints2d.pkl、extrinsic_camera_params/、intrinsic_camera_params/。")
        print("    若你只下载了 BamaPig2D，那是 2D 检测训练集，不含 3D 标注与外参。")
        return 1

    # ---------- 2. 载入 ----------
    print("\n[2/5] 数据载入")
    with open(paths["3D 标注"], "rb") as f:
        poses = np.asarray(pickle.load(f), np.float64)
    print(f"     3D {args.label:9s} {poses.shape}")
    if poses.ndim != 4:
        print(f"[!] 3D 标注应为 4 维 (帧,个体,关节,3)，实际 {poses.shape}")
        return 1
    nj = poses.shape[-2]
    if nj == 23:
        poses = poses[..., G_ALL_PARTS, :]
        print("     关节维 23 -> 取 g_all_parts -> 19")
    elif nj == 19:
        print("     关节维已是 19")
    else:
        print(f"[!] 关节维应为 19 或 23，实际 {nj}")
        return 1
    F, P = poses.shape[0], poses.shape[1]
    frame_ids = np.arange(F) * 25

    with open(paths["2D 标注"], "rb") as f:
        kp2d = np.asarray(pickle.load(f), np.float64)
    print(f"     2D 标注          {kp2d.shape}  (帧, 相机, 个体, 关节, (x,y,valid))")
    if kp2d.ndim != 5 or kp2d.shape[3] != 19:
        print(f"[!] 2D 标注期望 (帧,相机,个体,19,3)，实际 {kp2d.shape}")
        return 1

    extr = load_extrinsics(paths["外参目录"])
    print(f"     外参             {len(extr)} 台相机 {sorted(extr)}")
    K, ksrc, kkeys = load_intrinsics(root)
    print(f"     K  fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    # ---------- 3. 核心：3D GT 重投影 vs 2D 标注 ----------
    print("\n[3/5] 投影一致性（3D GT + 外参 + 内参 投影回图像 vs 官方 2D 标注）")
    per_cam, all_err = {}, []
    for ci, cid in enumerate(CAMIDS):
        if cid not in extr or ci >= kp2d.shape[1]:
            continue
        R, t = extr[cid]
        errs = []
        for k in range(F):
            for pid in range(P):
                X = poses[k, pid]                      # (19,3) 世界系
                valid = kp2d[k, ci, pid, :, 2] > 0.5
                obs = kp2d[k, ci, pid, :, :2]
                # 跳过未标注的 3D 关节（官方用 0 表示缺失）
                has3d = np.abs(X).sum(1) > 1e-9
                m = valid & has3d
                if m.sum() < 3:
                    continue
                xc = (R @ X[m].T).T + t
                z = np.where(np.abs(xc[:, 2]) < 1e-6, 1e-6, xc[:, 2])
                uv = (xc[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
                errs.append(np.linalg.norm(uv - obs[m], axis=1))
        if errs:
            e = np.concatenate(errs)
            per_cam[cid] = {"n": int(e.size), "median": float(np.median(e)),
                            "p90": float(np.percentile(e, 90)), "max": float(e.max())}
            all_err.append(e)
    if not all_err:
        print("[!] 没有可比较的样本（检查 2D valid 标记或 3D 非零关节数）")
        return 1
    E = np.concatenate(all_err)
    print(f"     样本数 {E.size}")
    print(f"     重投影误差  中位 {np.median(E):6.2f} px   均值 {E.mean():6.2f} px   "
          f"P90 {np.percentile(E, 90):6.2f} px   max {E.max():7.2f} px")
    print("\n     按相机：")
    for cid in sorted(per_cam):
        d = per_cam[cid]
        flag = "  <-- 偏大" if d["median"] > 10 else ""
        print(f"       cam{cid:2d}  n={d['n']:6d}  中位 {d['median']:6.2f} px  "
              f"P90 {d['p90']:6.2f} px{flag}")

    med = float(np.median(E))
    if med < 5:
        verdict = "优秀"
    elif med < 10:
        verdict = "可接受"
    elif med < 30:
        verdict = "偏大（可能是内参口径不对，见下方提示）"
    else:
        verdict = "异常（大概率是内参/外参/骨架顺序之一错了）"
    print(f"\n     判定：{verdict}")
    if med >= 10:
        print("     排查顺序：① 内参是否用了去畸变后的 newcameramtx（label_images 已去畸变）；"
              "\n               ② 2D 标注是否来自 label_images 而不是 image；"
              "\n               ③ 外参约定是否为 x_cam = R @ X_world + t；"
              "\n               ④ 关节顺序是否为 g_all_parts = [0..16,18,20]。")

    # ---------- 4. 可用性统计 ----------
    print("\n[4/5] 数据可用性")
    vis = kp2d[..., 2] > 0.5
    print(f"     2D 总可见率 {(vis.mean()*100):.1f}%（官方统计约 60%~70%）")
    print("     每个关节可见视角数（占 70 帧 × 4 头的比例）：")
    order = np.argsort(vis.mean(axis=(0, 1, 2)))[::-1]
    for j in order:
        r = vis[:, :, :, j].mean()
        bar = "#" * int(round(r * 24))
        print(f"       {j:2d} {JOINT_NAMES[j]:10s} {r*100:5.1f}%  {bar}")
    n3d = (np.abs(poses).sum(axis=-1) > 1e-9)
    print(f"     3D 标注非零率 {(n3d.mean()*100):.1f}%")
    nz = n3d.sum(axis=-1).ravel()
    print(f"     每个姿态有效 3D 关节数：min {nz.min()}  中位 {int(np.median(nz))}  "
          f"max {nz.max()}")
    span = poses.reshape(-1, 3).max(0) - poses.reshape(-1, 3).min(0)
    print(f"     3D 坐标跨度 {np.round(span, 3)} （米；单头猪身长应约 0.8~1.2 m）")
    print(f"     帧号 {frame_ids[0]} ~ {frame_ids[-1]}（步长 25，共 {F} 帧）")

    # ---------- 5. 可选验证图 ----------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            k, ci, cid = args.plot_frame_idx, 0, args.plot_cam
            ci = CAMIDS.index(cid) if cid in CAMIDS else 0
            R, t = extr.get(cid, (np.eye(3), np.zeros(3)))
            n_pig = P
            fig, axes = plt.subplots(1, n_pig, figsize=(4 * n_pig, 4.5))
            axes = np.atleast_1d(axes)
            for pid in range(n_pig):
                ax = axes[pid]
                obs = kp2d[k, ci, pid, :, :2]
                v = kp2d[k, ci, pid, :, 2] > 0.5
                X = poses[k, pid]
                xc = (R @ X.T).T + t
                z = np.where(np.abs(xc[:, 2]) < 1e-6, 1e-6, xc[:, 2])
                proj = (xc[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
                has3d = np.abs(X).sum(1) > 1e-9
                m = v & has3d
                ax.scatter(obs[v, 0], obs[v, 1], s=42, facecolors="none",
                           edgecolors="tab:green", label="2D label")
                ax.scatter(proj[m, 0], proj[m, 1], s=12, c="tab:red", marker="x",
                           label="3D GT 投影")
                for j in np.where(m)[0]:
                    ax.plot([obs[j, 0], proj[j, 0]], [obs[j, 1], proj[j, 1]],
                            "-", color="gray", lw=0.5)
                ax.set_title(f"pig {pid}  frame {frame_ids[k]}", fontsize=10)
                ax.invert_yaxis(); ax.set_aspect("equal"); ax.grid(alpha=0.2)
                if pid == 0:
                    ax.legend(fontsize=8)
            fig.suptitle(f"cam{cid}: 绿色=官方 2D 标注, 红色=3D GT 重投影 "
                         f"(中位误差 {per_cam.get(cid, {}).get('median', float('nan')):.2f} px)",
                         fontsize=11)
            fig.tight_layout()
            os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
            fig.savefig(args.plot, dpi=130)
            print(f"\n[ok] 验证图 -> {args.plot}")
        except ImportError:
            print("\n[!] 没装 matplotlib，跳过 --plot")

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"root": root, "label": args.label, "shape_3d": list(poses.shape),
                       "shape_2d": list(kp2d.shape),
                       "intrinsics": {"source": ksrc, "fx": float(K[0, 0]),
                                      "fy": float(K[1, 1]), "cx": float(K[0, 2]),
                                      "cy": float(K[1, 2])},
                       "reproj_median_px": med, "per_camera": per_cam,
                       "verdict": verdict}, f, ensure_ascii=False, indent=2)
        print(f"[ok] 报告 -> {args.save_json}")

    print("\n下一步：")
    print(f"  # 1) 训练用 3D 姿态（论文 Table 7：帧 0-1400）")
    print(f"  python scripts/prepare_mammal.py --src {_disp(paths['3D 标注'])} "
          f"--max-frame 1400 --out data/poses_bamapig_train.npz")
    print(f"  # 2) 两视角标定数据（论文：帧 1400-1750，相机 0 <-> 6）")
    print(f"  python scripts/prepare_mammal_2d.py --root {_disp(root)} "
          f"--cam-a 0 --cam-b 6 --min-frame 1400 --out data/two_view_pig_c0_c6.npz")
    print(f"  # 3) 训练完成后先自检推断链，再标定")
    print(f"  python scripts/selftest_infer.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
