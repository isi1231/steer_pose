#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BamaPig3D -> 单目 3D 提升器（`steerpose.lifting.PoseLifter`）训练数据。

它补的是 RA-L2022 链路的**第一环**
----------------------------------
SteerPose 原论文只有一条 2D→2D 的网络（用于跨视角匹配），没有 3D。但 RA-L2022
（`Extrinsic Camera Calibration From a Moving Person`，同作者前作）的整条标定链路
是 **2D → 单目 3D → 骨方向 → 外参**，缺了这一环后面全部无从谈起。本脚本把
BamaPig3D 的官方标注整理成提升器的监督信号。

一条样本 = **（某一帧、某一台相机、某一头猪）**
------------------------------------------------
为什么"每台相机各训一份"而不是只训一份相机无关的：
单目 3D 提升器的输出在**相机坐标系**里，而同一头猪、同一帧、在不同相机下的
相机系 3D 是**不同的**（差一个外参）。输入是"按内参归一化"的 2D（相机无关），
输出是相机系 3D（相机相关）—— 但输入本身已经隐含了视角，所以
"归一化 2D -> 相机系 3D"是一个**跨相机一致的映射**，一份网络就能吃所有相机。
这正是 RA-L2022 的用法：**每台相机各自**跑同一个提升器。

监督信号的两个关键约定（都写在 docs/ral_backend.md）
----------------------------------------------------
1. **用 `newcameramtx` 归一化 2D**：`(x-cx)/fx, (y-cy)/fy`。
   `label_images` 是**已去畸变**图像，所以标签对应的内参是 `newcameramtx`
   （fx≈1340），不是 `undistortion.py` 里那个带畸变的原始 K（fx≈1625）。
   用归一化坐标的好处：换相机/分辨率不用重训，而且与几何后端的射线口径
   `(u,v,1)` 完全一致。
2. **3D 目标 = 相机系 + 根相对 + 骨长归一化**：
   * 相机系：世界系 GT 用官方外参转过去 `X_cam = R_c @ X_world + t_c`；
   * 根相对：减去 root(=18, "center")；
   * 骨长归一化：除以**有效骨的中位骨长**。
   单目 3D 的绝对深度/尺度本来就不可观测，强行回归绝对量纲只会让网络学不出来。
   归一化后主损失用**骨方向余弦**（对尺度、平移都免疫），MPJPE 只作辅助。

输出
----
`--out` 写训练数据 npz：
    X2       (N,19,2) float32   按内参归一化的 2D
    Mask     (N,19)   float32   1=可见（2D 有效 且 3D 有效）
    Y3       (N,19,3) float32   相机系 / 根相对 / 骨长归一化的 3D
    FrameId  (N,)     int32     标注帧号（= 索引 * 25），**按帧划分 train/val 用它**
    CamId    (N,)     int32     相机 ID（0,1,2,5,...,11）
    PigId    (N,)     int32     个体 ID（0..3）
    K        (3,3)    float64   newcameramtx（归一化已做完，这里只留档）
    joint_names       (19,)     关节名
    bones             (19,2)    骨索引（与 lifting.PIG19_BONES 一致）

`--dump-calib` 另外写一份**多视角标定输入包**（给 `steerpose.calib_ral` 用）：
    p2d      (C,N,19,2) float64 像素坐标（★几何后端要的是像素，不是归一化）
    w2d      (C,N,19)   float64 2D 有效性
    K        (3,3)      float64 newcameramtx
    R_gt     (C,3,3)    float64 真值外参旋转（世界->相机）
    t_gt     (C,3)      float64 真值外参平移
    camids   (C,)       int32
    frame_ids (N,)      int32   每一条样本属于哪一帧
    ★ N 只保留"所有 C 台相机都看得见"的 (帧,个体)，因为
      `calibrate_from_lifter` 的 `visible_from_all` 要求跨相机样本一一对齐。

用法
----
    # ⓪ 先体检（秒级，纯 numpy，不需要权重）
    python scripts/check_mammal_data.py --root /data/BamaPig3D_pure_pickle

    # ① 训练数据（论文 Table 7：训练用帧 0-1400）
    python scripts/prepare_mammal_lifter.py --root /data/BamaPig3D_pure_pickle \\
        --max-frame 1400 --out data/lifter_bamapig_train.npz

    # ② 标定输入包（论文 Table 7：标定用帧 1400-1750）
    #    ★ 用**相机对**：几何后端要求"同一根骨在所有相机都可见"，
    #      存活率 ≈ 单相机可见率^相机数；10 台会掉到 ~1%，2 台约 42%
    python scripts/prepare_mammal_lifter.py --root /data/BamaPig3D_pure_pickle \\
        --min-frame 1400 --calib-cams 0,6 --dump-calib data/lifter_calib_c0_c6.npz

    # ③ 先用假数据把链路跑通（不做任何 GPU 计算，纯 numpy 合成）
    python scripts/make_mock_mammal.py --out /tmp/mock_bamapig
    python scripts/prepare_mammal_lifter.py --root /tmp/mock_bamapig \\
        --max-frame 1400 --out data/lifter_mock.npz --dump-calib data/lifter_mock_calib.npz

随后训练：
    python scripts/train_lifter.py --npz data/lifter_bamapig_train.npz \\
        --out ckpt/lifter_bamapig.pt --device cuda --workers 8
"""
import argparse
import os
import pickle
import re
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_mammal_data import load_intrinsics   # noqa: E402  同目录复用，避免口径漂移
from steerpose.skeleton import get_skeleton     # noqa: E402

CAMIDS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]   # 官方 10 个视角的顺序（= kp2d 第 2 维）
NUM_JOINTS = 19
G_ALL_PARTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
FRAME_STEP = 25      # 官方每 25 帧标注一次（第 k 个标注帧 = 25k）
ZERO_EPS = 1e-9      # 官方用 0 填充"未标注（遮挡）"的 3D 关节


# ============================ 读数据 ============================

def load_kp2d(path: str) -> np.ndarray:
    """label_keypoints2d.pkl -> (F, C, P, J, 3)，最后一维是 (x, y, valid)。"""
    with open(path, "rb") as f:
        arr = np.asarray(pickle.load(f), np.float64)
    if arr.ndim != 5:
        raise ValueError(f"{path} 期望 5 维 (帧,相机,个体,关节,3)，实际 {arr.shape}")
    if arr.shape[3] != NUM_JOINTS:
        raise ValueError(f"关节数应为 {NUM_JOINTS}，实际 {arr.shape[3]}（请确认是官方 pkl）")
    return arr


def load_3d(path: str) -> np.ndarray:
    """label_mix.pkl / label_3d.pkl -> (F, P, 19, 3)（世界系，单位米）。

    23 关节版自动按 g_all_parts 取到 19；未标注关节以 0 填充（官方约定）。
    """
    with open(path, "rb") as f:
        arr = np.asarray(pickle.load(f), np.float64)
    if arr.ndim != 4:
        raise ValueError(f"{path} 期望 4 维 (帧,个体,关节,3)，实际 {arr.shape}")
    nj = arr.shape[-2]
    if nj == 23:
        arr = arr[..., G_ALL_PARTS, :]
    elif nj != NUM_JOINTS:
        raise ValueError(f"关节维应为 19 或 23，实际 {nj}")
    return arr


def load_extrinsics(folder: str, camids) -> dict:
    """extrinsic_camera_params/{camid}.txt -> {camid: (R(3,3), t(3,))}。

    约定 `x_cam = R @ X_world + t`（与官方 visualize_BamaPig3D.py 一致）。
    """
    from scipy.spatial.transform import Rotation
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
                    f"  该目录现有 {len(found)} 个 txt："
                    f"{[os.path.basename(p) for p in found[:10]]}")
            path = m[0]
        v = np.loadtxt(path).ravel()
        if v.size < 6:
            raise ValueError(f"{path} 应含 6 个浮点数（axis-angle + 平移），实际 {v.size}")
        out[cid] = (Rotation.from_rotvec(v[:3]).as_matrix(), v[3:6].copy())
    return out


# ============================ 3D 目标的归一化 ============================

def _bone_pairs(bones) -> np.ndarray:
    b = np.asarray(list(bones), np.int64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"bones 应为 (B,2)，实际 {b.shape}")
    return b


def robust_bone_scale(y3: np.ndarray, bones, valid: np.ndarray,
                      eps: float = 1e-9) -> float:
    """单目尺度基准 = **两端都有效的骨**的中位骨长。

    比"所有骨的平均长度"稳：头部小骨（鼻/眼/耳）在 2D 上只有几个像素，单目 3D
    估出来长度噪声极大，放进尺度里会把整个姿态的量纲带偏。中位数对少数坏骨免疫。
    若一根有效骨都没有，回退到"根相对后的关节 RMS 半径"，保证不返回 0。
    """
    b = _bone_pairs(bones)
    ok = valid[b[:, 0]] & valid[b[:, 1]]
    if ok.any():
        d = np.linalg.norm(y3[b[ok, 0]] - y3[b[ok, 1]], axis=-1)
        s = float(np.median(d))
        if s > eps:
            return s
    r = float(np.sqrt((y3 ** 2).sum(axis=-1).mean()))
    return r if r > eps else 1.0


def normalize_target(X_cam: np.ndarray, bones, root: int,
                     valid: np.ndarray) -> np.ndarray:
    """相机系世界坐标 -> 根相对 + 骨长归一化（提升器的回归目标）。"""
    y = X_cam - X_cam[root: root + 1]
    s = robust_bone_scale(y, bones, valid)
    return y / s


# ============================ 主流程 ============================

def build_samples(kp, poses3d, extr, K, camids, args, sk):
    """逐 (帧, 相机, 个体) 生成提升器样本。

    返回 dict（各字段见模块 docstring）+ 统计信息。
    """
    F, C_kp, P = kp.shape[0], kp.shape[1], kp.shape[2]
    bones = sk["bones"]
    root = sk["root"]
    b = _bone_pairs(bones)

    # 内参归一化的除数与中心（label_images 已去畸变 -> 用 newcameramtx）
    cxy = K[:2, 2]
    fxy = np.array([K[0, 0], K[1, 1]], np.float64)

    frame_ids = np.arange(F) * FRAME_STEP
    keep_frames = np.ones(F, bool)
    if args.min_frame is not None and args.min_frame >= 0:
        keep_frames &= frame_ids >= args.min_frame
    if args.max_frame is not None:
        keep_frames &= frame_ids <= args.max_frame

    X2, Mask, Y3 = [], [], []
    fids, cids, pids = [], [], []
    n_skip_valid, n_skip_bone = 0, 0

    for ci, cid in enumerate(CAMIDS):
        if ci >= C_kp or cid not in extr:
            print(f"[!] 相机 {cid} 在 2D 标注/外参里缺失，跳过")
            continue
        R, t = extr[cid]
        for k in range(F):
            if not keep_frames[k]:
                continue
            Xw_all = poses3d[k]                       # (P,19,3) 世界系
            for p in range(P):
                xy = kp[k, ci, p, :, :2].copy()
                v2d = kp[k, ci, p, :, 2] > 0.5
                Xw = Xw_all[p]
                v3d = np.abs(Xw).sum(axis=1) > ZERO_EPS     # 官方 0 = 未标注

                # 骨架内部的骨骼完整性：一根骨要么两端都有效，要么不计入监督
                valid = v2d & v3d
                n_bone_ok = int((valid[b[:, 0]] & valid[b[:, 1]]).sum())
                if valid.sum() < args.min_valid_joints:
                    n_skip_valid += 1
                    continue
                if n_bone_ok < args.min_valid_bones:
                    n_skip_bone += 1
                    continue

                # 缺失关节用"有效关节的质心"填充，避免 0 坐标把网络输入带偏
                # （掩码会告诉网络这些位置不可信，损失也不看它们）
                if (~v2d).any():
                    xy[~v2d] = xy[v2d].mean(axis=0)

                X_cam = (R @ Xw.T).T + t               # 世界 -> 相机
                y3 = normalize_target(X_cam, bones, root, valid)
                y3 = np.where(valid[:, None], y3, 0.0)  # 无效关节目标置 0

                X2.append((xy - cxy) / fxy)
                Mask.append(valid.astype(np.float32))
                Y3.append(y3)
                fids.append(frame_ids[k])
                cids.append(cid)
                pids.append(p)

    if not X2:
        raise RuntimeError(
            "没有生成任何样本。请检查：① --min-frame/--max-frame 是否把帧全滤掉了；"
            "② --min-valid-joints 是否过高；③ 2D 标注的关节数是否为 19。")

    out = {
        "X2": np.asarray(X2, np.float32),
        "Mask": np.asarray(Mask, np.float32),
        "Y3": np.asarray(Y3, np.float32),
        "FrameId": np.asarray(fids, np.int32),
        "CamId": np.asarray(cids, np.int32),
        "PigId": np.asarray(pids, np.int32),
        "K": np.asarray(K, np.float64),
        "joint_names": np.array(sk["names"]),
        "bones": _bone_pairs(bones).astype(np.int32),
        "calib_bones": _bone_pairs(sk["calib_bones"]).astype(np.int32),
        "root": np.int32(root),
    }
    stats = {"n_skip_valid": n_skip_valid, "n_skip_bone": n_skip_bone,
             "keep_frames": keep_frames, "frame_ids": frame_ids}
    return out, stats


def build_calib_bundle(kp, extr, K, camids, args, sk):
    """构造 `steerpose.calib_ral.calibrate_from_lifter` 的多视角输入。

    ★ N 必须是"所有相机都有效"的 (帧,个体) —— 因为几何后端会做
      `visible_from_all(mask)` 把样本对齐到"全视角可见"，各相机样本数不一致
      会在那儿被静默截断，标定结果无法复现。这里一次性筛好。

    ★★ 关于相机个数（实测很重要）
      `oriented_points` / `normalized_rays` 都要求"同一根骨在所有相机都可见"
      （`~isnan(nrm).any(axis=0)`），所以存活率 ≈ (单相机可见率)^C。
      BamaPig3D 的关节可见率约 60%~70%（猪会被栏杆/同伴遮挡）：
          C=2  -> 0.65² ≈ 42%   可用
          C=10 -> 0.65¹⁰ ≈ 1.3% 基本没有样本
      这正是参考实现（RA-L2022）以**相机对**为单位标定的原因。
      所以本函数默认允许用 `--calib-cams` 指定相机子集，**推荐两两配对**。
    """
    F, C_kp, P = kp.shape[0], kp.shape[1], kp.shape[2]
    b = _bone_pairs(sk["bones"])

    frame_ids = np.arange(F) * FRAME_STEP
    keep = np.ones(F, bool)
    if args.min_frame is not None and args.min_frame >= 0:
        keep &= frame_ids >= args.min_frame
    if args.max_frame is not None:
        keep &= frame_ids <= args.max_frame

    # 选用的相机（按 --calib-cams 过滤，保持 CAMIDS 顺序）
    want = camids
    if args.calib_cams:
        want = [int(x) for x in str(args.calib_cams).replace(" ", "").split(",") if x != ""]
        unknown = [c for c in want if c not in CAMIDS]
        if unknown:
            raise SystemExit(f"[!] --calib-cams 里有未知相机 {unknown}，"
                             f"合法值：{CAMIDS}")
    used_idx, used_cid = [], []
    for ci, cid in enumerate(CAMIDS):
        if ci < C_kp and cid in extr and cid in want:
            used_idx.append(ci)
            used_cid.append(cid)
    if len(used_cid) < 2:
        raise RuntimeError(f"标定至少需要 2 台相机，当前只有 {used_cid}")
    if args.calib_cams is None and len(used_cid) > 2:
        print(f"[!] 未指定 --calib-cams，默认用全部 {len(used_cid)} 台相机。"
              f"\n    注意：几何后端要求'同一根骨在所有相机都可见'，存活率 ≈ "
              f"(单相机可见率)^{len(used_cid)}。")
        print(f"    BamaPig3D 真实数据关节可见率约 0.65 -> 估计只剩 "
              f"{0.65 ** len(used_cid) * 100:.2f}% 的样本。")
        print(f"    ★ 强烈建议用相机对（参考实现的做法）：--calib-cams 0,6")

    # 单相机的关节平均可见率（用于预估全视角存活率）
    vis_single = (kp[..., 2] > 0.5).mean()
    est = vis_single ** len(used_cid)

    sel = []          # [(k, p), ...]
    for k in range(F):
        if not keep[k]:
            continue
        for p in range(P):
            ok = True
            for ci in used_idx:
                v2d = kp[k, ci, p, :, 2] > 0.5
                if v2d.sum() < args.min_valid_joints:
                    ok = False
                    break
                if (v2d[b[:, 0]] & v2d[b[:, 1]]).sum() < args.min_valid_bones:
                    ok = False
                    break
            if ok:
                sel.append((k, p))
    if not sel:
        raise RuntimeError("标定帧里没有任何 '全视角可见' 的 (帧,个体)，"
                           "请放宽 --min-valid-joints/--min-valid-bones，"
                           "或用更少的相机（--calib-cams 0,6）")

    C, N, J = len(used_cid), len(sel), NUM_JOINTS
    p2d = np.zeros((C, N, J, 2), np.float64)
    w2d = np.zeros((C, N, J), np.float64)
    R_gt = np.zeros((C, 3, 3), np.float64)
    t_gt = np.zeros((C, 3), np.float64)
    for i, cid in enumerate(used_cid):
        R_gt[i], t_gt[i] = extr[cid]
    fid = np.zeros(N, np.int32)
    pig = np.zeros(N, np.int32)
    for n, (k, p) in enumerate(sel):
        fid[n], pig[n] = frame_ids[k], p
        for i, ci in enumerate(used_idx):
            xy = kp[k, ci, p, :, :2]
            v = kp[k, ci, p, :, 2] > 0.5
            p2d[i, n] = xy
            w2d[i, n] = v.astype(np.float64)

    # 实际存活率：几何后端最终能用的 (样本,骨) 数量（决定标定是否可解）
    vis_all = w2d.astype(bool).all(axis=0)                       # (N,J)
    bb = _bone_pairs(sk["calib_bones"])
    bone_ok = vis_all[:, bb[:, 0]] & vis_all[:, bb[:, 1]]
    diag = {"n_sel": N, "n_frames": int(len(np.unique(fid))),
            "n_cams": C, "vis_single": float(vis_single),
            "est_survival": float(est),
            "joint_survival": float(vis_all.mean()),
            "bone_survival": float(bone_ok.mean()),
            "n_bone_obs": int(bone_ok.sum()),
            "camids": used_cid}

    return {"p2d": p2d, "w2d": w2d, "K": np.asarray(K, np.float64),
            "R_gt": R_gt, "t_gt": t_gt,
            "camids": np.asarray(used_cid, np.int32),
            "frame_ids": fid, "pigids": pig}, diag


def main():
    ap = argparse.ArgumentParser(
        description="BamaPig3D -> 单目 3D 提升器训练数据 / RA-L2022 标定输入")
    ap.add_argument("--root", type=str, default=None,
                    help="BamaPig3D_pure_pickle 目录（自动找 label_keypoints2d.pkl / "
                         "label_mix.pkl / extrinsic_camera_params / intrinsic_camera_params）")
    ap.add_argument("--kp2d", type=str, default=None, help="label_keypoints2d.pkl 路径")
    ap.add_argument("--label", type=str, default="label_mix",
                    help="3D 标注文件名（默认 label_mix；官方实验用 label_mix）")
    ap.add_argument("--poses3d", type=str, default=None, help="3D 标注 pkl 路径（覆盖 --label）")
    ap.add_argument("--extrinsics", type=str, default=None, help="extrinsic_camera_params 目录")
    ap.add_argument("--out", type=str, default="data/lifter_bamapig_train.npz",
                    help="提升器训练数据输出（--dump-calib 单独指定标定包时可用 --no-train-out 跳过）")
    ap.add_argument("--no-train-out", action="store_true",
                    help="只导标定包，不写训练数据 npz")
    ap.add_argument("--dump-calib", type=str, default=None,
                    help="额外导出多视角标定输入包 npz 的路径")
    ap.add_argument("--calib-cams", type=str, default=None,
                    help="标定用哪些相机（逗号分隔，如 '0,6'）。默认全部 10 台。"
                         "★ 强烈建议用相机对：几何后端要求'同一根骨在所有相机都可见'，"
                         "存活率 ≈ 单相机可见率^相机数，10 台会掉到 ~1%")
    ap.add_argument("--min-frame", type=int, default=None,
                    help="只保留帧号 >= 该值（标定用 1400）")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="只保留帧号 <= 该值（训练用 1400，论文 Table 7）")
    ap.add_argument("--min-valid-joints", type=int, default=12,
                    help="每条样本至少多少有效关节（默认 12）")
    ap.add_argument("--min-valid-bones", type=int, default=8,
                    help="每条样本至少多少根'两端都有效'的骨（默认 8；19 根骨里 14 根是关键骨）")
    ap.add_argument("--skeleton", type=str, default="pig19", help="骨架名（默认 pig19）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计不写文件（纯 numpy，秒级）")
    args = ap.parse_args()

    sk = get_skeleton(args.skeleton)

    # ---- 定位输入文件 ----
    kp2d_path, extr_dir, poses3d_path = args.kp2d, args.extrinsics, args.poses3d
    if args.root:
        kp2d_path = kp2d_path or os.path.join(args.root, "label_keypoints2d.pkl")
        extr_dir = extr_dir or os.path.join(args.root, "extrinsic_camera_params")
        poses3d_path = poses3d_path or os.path.join(args.root, f"{args.label}.pkl")
    for name, p in (("label_keypoints2d.pkl", kp2d_path),
                    (f"{args.label}.pkl", poses3d_path),
                    ("extrinsic_camera_params", extr_dir)):
        if not p or not os.path.exists(p):
            raise SystemExit(f"[!] 找不到 {name}：{p}\n"
                             f"    请给 --root <BamaPig3D_pure_pickle 目录>，或分别用 "
                             f"--kp2d/--poses3d/--extrinsics 指定。")

    print("=" * 72)
    print("BamaPig3D -> 单目 3D 提升器训练数据")
    print("=" * 72)

    kp = load_kp2d(kp2d_path)
    poses3d = load_3d(poses3d_path)
    extr = load_extrinsics(extr_dir, CAMIDS)
    if args.root:
        K, ksrc, _ = load_intrinsics(args.root, verbose=False)
    else:
        raise SystemExit("[!] 需要 --root 才能读到 newcameramtx 内参"
                         "（label_images 已去畸变，必须用 newcameramtx，不能用原始 K）")
    print(f"[i] 2D 标注   {kp.shape}  (帧, 相机, 个体, 关节, (x,y,valid))")
    print(f"[i] 3D 标注   {poses3d.shape}  (帧, 个体, 19, 3)  世界系，米")
    print(f"[i] 外参      {len(extr)} 台相机 {sorted(extr)}")
    print(f"[i] 内参      {os.path.basename(ksrc)}  fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    print(f"              ★ label_images 已去畸变，标签对应 newcameramtx"
          f"（fx≈1340，不是原始 K 的 fx≈1625）")

    if poses3d.shape[0] != kp.shape[0]:
        print(f"[!] 2D 与 3D 的帧数不一致（{kp.shape[0]} vs {poses3d.shape[0]}），"
              f"取较小者 {min(kp.shape[0], poses3d.shape[0])}")
        n = min(kp.shape[0], poses3d.shape[0])
        kp, poses3d = kp[:n], poses3d[:n]

    # ---- 样本 ----
    data, stats = build_samples(kp, poses3d, extr, K, CAMIDS, args, sk)
    keep = stats["keep_frames"]
    fid_all = stats["frame_ids"]
    print(f"\n[i] 帧筛选：保留 {int(keep.sum())}/{len(keep)} 个标注帧，"
          f"帧号 {fid_all[keep][0] if keep.any() else '-'} ~ "
          f"{fid_all[keep][-1] if keep.any() else '-'}（步长 {FRAME_STEP}）")
    print(f"[i] 剔除：有效关节不足 {stats['n_skip_valid']} 条，"
          f"有效骨不足 {stats['n_skip_bone']} 条")
    N = data["X2"].shape[0]
    print(f"[i] 生成 {N} 条样本 = {len(np.unique(data['FrameId']))} 帧 × "
          f"{len(np.unique(data['CamId']))} 相机（+ 个体维）")
    if N < 500:
        print(f"[!] 样本数偏少（{N}<500），提升器容易过拟合。"
              f"可放宽 --min-valid-joints，或用 --max-frame 放宽帧范围。")

    m = data["Mask"]
    print(f"[i] 关节可见率 {(m.mean()*100):.1f}%")
    per_j = m.mean(axis=0)
    worst = np.argsort(per_j)[:5]
    print(f"     最不可见的 5 个关节：" +
          "  ".join(f"{sk['names'][j]}({per_j[j]*100:.0f}%)" for j in worst))
    # 目标自检：根关节必须归零，且有效骨的骨长中位应为 1（归一化的定义）
    print(f"[i] 目标自检：root 关节目标最大绝对值 "
          f"{np.abs(data['Y3'][:, sk['root'], :]).max():.2e}（应为 0）")
    b = _bone_pairs(sk["bones"])
    ok_pair = (m[:, b[:, 0]] > 0.5) & (m[:, b[:, 1]] > 0.5)
    bl = np.linalg.norm(data["Y3"][:, b[:, 0], :] - data["Y3"][:, b[:, 1], :], axis=-1)
    print(f"             有效骨骨长中位 {np.median(bl[ok_pair]):.4f}（应≈1）")

    if args.dry_run:
        print("\n[dry-run] 未写文件。")
        return 0

    # ---- 写训练数据 ----
    if not args.no_train_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out, **data)
        size = os.path.getsize(args.out) / 1e6
        print(f"\n[ok] 训练数据 -> {args.out} ({size:.2f} MB)")
        print(f"     训练: python scripts/train_lifter.py --npz {args.out} "
              f"--out ckpt/lifter_bamapig.pt --device cuda --workers 8")

    # ---- 写标定输入包 ----
    if args.dump_calib:
        bundle, diag = build_calib_bundle(kp, extr, K, CAMIDS, args, sk)
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_calib)), exist_ok=True)
        np.savez_compressed(args.dump_calib, **bundle)
        print(f"\n[ok] 标定输入包 -> {args.dump_calib}")
        print(f"     p2d {bundle['p2d'].shape}（C,N,J,2，像素）  w2d {bundle['w2d'].shape}")
        print(f"     相机 {diag['camids']}（{diag['n_cams']} 台），"
              f"全视角可见的 (帧,个体) N={diag['n_sel']}，覆盖 {diag['n_frames']} 帧")
        print(f"     单相机关节可见率 {diag['vis_single']:.3f}  ->  "
              f"预估全视角存活率 {diag['est_survival']*100:.2f}%")
        print(f"     实际：关节 {diag['joint_survival']*100:.1f}%，"
              f"标定骨 {diag['bone_survival']*100:.1f}%"
              f"（= {diag['n_bone_obs']} 个骨观测）")
        if diag["n_bone_obs"] < 100:
            print(f"     [!] 骨观测 < 100，线性解会很不稳。请减少相机数"
                  f"（--calib-cams 0,6）或放宽 --min-valid-joints。")
        print(f"     ★ 这一步已经把 '全视角可见' 筛好，标定时直接用 "
              f"calibrate_from_lifter({diag['n_cams']} 视角, visible_all=True)。")
    elif args.dump_calib is None and not args.no_train_out:
        print(f"     （如需标定输入包，加 --min-frame 1400 --calib-cams 0,6 "
              f"--dump-calib data/lifter_calib_c0_c6.npz）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
