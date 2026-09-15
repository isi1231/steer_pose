#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成"官方格式"的 BamaPig3D 假数据，用于在没有 8GB 数据集时跑通/自测整条链路。

生成的文件完全按官方命名与结构（见 anl13/MAMMAL_datasets）：
    <out>/label_3d/pig_{pig}_frame_{frame:06d}.txt     23x3，第 18/20/22/23 行恒为 0
    <out>/label_3d.pkl                                  (70, 4, 23, 3)  与官方精简版同构
    <out>/label_keypoints2d.pkl                         (70, 10, 4, 19, 3)  (x, y, valid)
    <out>/extrinsic_camera_params/{camid:02d}.txt       6 个数：axis-angle + 平移（米）
    <out>/camids.txt                                    10 个相机 ID

骨架按官方 19 关节定义（nose, eyes, ears, shoulders… tail, center），
姿态是程序化生成的四足行走动作，**仅用于验证代码链路，不能用于任何精度结论**。

用法:
    python scripts/make_mock_mammal.py --out /tmp/mock_bamapig
    # 之后即可完整走一遍：
    python scripts/prepare_mammal.py --src /tmp/mock_bamapig/label_3d --max-frame 1400 \\
        --out data/poses_mock.npz
    python -m steerpose.train --poses3d data/poses_mock.npz --epochs 50 --batch 256 \\
        --workers 0 --device cuda --out ckpt/mock.pt
    python scripts/prepare_mammal_2d.py --kp2d /tmp/mock_bamapig/label_keypoints2d.pkl \\
        --extrinsics /tmp/mock_bamapig/extrinsic_camera_params --cam-a 0 --cam-b 6 \\
        --out data/two_view_mock.npz
    python -m steerpose.calibrate --ckpt ckpt/mock_best.pt --poses-npz data/two_view_mock.npz
"""
import argparse
import math
import os
import pickle
import numpy as np

CAMIDS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
NUM_FRAMES = 70          # 官方标注帧数
FRAME_STEP = 25          # 每 25 帧标注一次
NUM_PIGS = 4
G_ALL_PARTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]


def make_pig_pose_19(phase: float, body_len: float, leg_len: float, height: float,
                     y_off: float) -> np.ndarray:
    """程序化生成 19 关节猪姿态（顺序与官方 19 关节一致）。"""
    p = np.zeros((19, 3))
    sw = math.sin(phase)
    # center(18) 作为躯干中心，肩/髋围绕它
    p[18] = [0.0, y_off, height]                      # center
    p[0] = [0.55 * body_len, y_off, height + 0.05]    # nose
    p[1] = [0.42 * body_len, y_off + 0.05, height + 0.07]   # l_eye
    p[2] = [0.42 * body_len, y_off - 0.05, height + 0.07]   # r_eye
    p[3] = [0.34 * body_len, y_off + 0.09, height + 0.06]   # l_ear
    p[4] = [0.34 * body_len, y_off - 0.09, height + 0.06]   # r_ear
    p[5] = [0.22 * body_len, y_off + 0.13, height - 0.02]   # l_shoulder
    p[6] = [0.22 * body_len, y_off - 0.13, height - 0.02]   # r_shoulder
    p[17] = [-0.42 * body_len, y_off, height + 0.02]        # tail
    p[11] = [-0.22 * body_len, y_off + 0.13, height - 0.02] # l_hip
    p[12] = [-0.22 * body_len, y_off - 0.13, height - 0.02]# r_hip
    # 四条腿：前左(5)、前右(6)、后左(11)、后右(12) 各接 elbow(7/8/13/14) 与 paw(9/10/15/16)
    for sh, el, pw, sign, swing in ((5, 7, 9, +1, sw), (6, 8, 10, -1, -sw),
                                    (11, 13, 15, +1, -sw), (12, 14, 16, -1, sw)):
        base = p[sh].copy()
        p[el] = base + [0.06 * swing, 0.0, -0.5 * leg_len]
        p[pw] = base + [0.12 * swing, 0.0, -0.95 * leg_len]
    return p


def rotation_matrix(axis_angle) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(axis_angle).as_matrix()


def main():
    ap = argparse.ArgumentParser(description="生成官方格式的 BamaPig3D 假数据")
    ap.add_argument("--out", type=str, required=True, help="输出目录")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cameras", type=int, default=10, help="生成多少个视角（最多 10）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "label_3d"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "extrinsic_camera_params"), exist_ok=True)

    cams = CAMIDS[: args.cameras]
    # ---- 相机外参：环绕场景一圈，半径 4m，朝向原点 ----
    extr = {}
    for i, cid in enumerate(cams):
        ang = 2 * math.pi * i / len(cams)
        eye = np.array([4.0 * math.cos(ang), 4.0 * math.sin(ang), 2.2])
        z = -eye / np.linalg.norm(eye)
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=0)
        t = -R @ eye
        extr[cid] = (R, t)
        # 官方格式：6 个浮点数 = axis-angle 旋转(3) + 平移 xyz(3)，单位米
        from scipy.spatial.transform import Rotation
        np.savetxt(os.path.join(args.out, "extrinsic_camera_params", f"{cid:02d}.txt"),
                   np.concatenate([Rotation.from_matrix(R).as_rotvec(), t]))
    with open(os.path.join(args.out, "camids.txt"), "w") as f:
        f.write("\n".join(str(c) for c in cams))

    # ---- 生成 3D 姿态：(70, 4, 19, 3) ----
    poses19 = np.zeros((NUM_FRAMES, NUM_PIGS, 19, 3))
    for k in range(NUM_FRAMES):
        t = k * FRAME_STEP / 25.0     # 时间（秒）
        for pid in range(NUM_PIGS):
            phase = 2 * math.pi * 1.2 * t + pid * math.pi / 2
            poses19[k, pid] = make_pig_pose_19(
                phase,
                body_len=rng.uniform(0.9, 1.1),
                leg_len=rng.uniform(0.8, 1.0),
                height=rng.uniform(0.55, 0.7),
                y_off=(pid - 1.5) * 0.8)

    # ---- 写 label_3d/*.txt（23 行版，4 行为 0）----
    zero_rows = [r for r in range(23) if r not in G_ALL_PARTS]
    for k in range(NUM_FRAMES):
        fid = k * FRAME_STEP
        for pid in range(NUM_PIGS):
            full = np.zeros((23, 3))
            full[G_ALL_PARTS] = poses19[k, pid]
            # 模拟遮挡：随机若干关节置 0（官方未标注的关节也是 0）
            for j in rng.choice(19, size=rng.integers(0, 3), replace=False):
                full[G_ALL_PARTS[j]] = 0.0
            np.savetxt(os.path.join(args.out, "label_3d",
                                    f"pig_{pid}_frame_{fid:06d}.txt"), full)

    # ---- label_3d.pkl（官方精简版同构）----
    with open(os.path.join(args.out, "label_3d.pkl"), "wb") as f:
        pickle.dump(poses19.copy(), f)
    with open(os.path.join(args.out, "label_mix.pkl"), "wb") as f:
        pickle.dump(poses19.copy(), f)

    # ---- label_keypoints2d.pkl：(70, 10, 4, 19, 3) ----
    kp2d = np.zeros((NUM_FRAMES, len(cams), NUM_PIGS, 19, 3), np.float64)
    for i, cid in enumerate(cams):
        R, t = extr[cid]
        for k in range(NUM_FRAMES):
            for pid in range(NUM_PIGS):
                X = poses19[k, pid]
                xc = (R @ X.T).T + t                     # 相机系
                xc[:, 2] = np.where(np.abs(xc[:, 2]) < 1e-6, 1e-6, xc[:, 2])
                uv = xc[:, :2] / xc[:, 2:3] * 1200.0 + np.array([960.0, 540.0])
                kp2d[k, i, pid, :, :2] = uv
                kp2d[k, i, pid, :, 2] = 1.0              # valid 标记
    # 模拟检测失败：随机把少数点标为无效
    miss = rng.random(kp2d.shape[:4]) < 0.03
    kp2d[..., 2][miss] = 0.0
    with open(os.path.join(args.out, "label_keypoints2d.pkl"), "wb") as f:
        pickle.dump(kp2d, f)

    print(f"[ok] 假数据已生成 -> {args.out}")
    print(f"     3D 姿态 {poses19.shape}，2D 标注 {kp2d.shape}，{len(cams)} 个视角 {cams}")
    print(f"     ⚠️ 仅用于验证代码链路（骨架为程序化生成），不可用于任何精度结论")


if __name__ == "__main__":
    main()
