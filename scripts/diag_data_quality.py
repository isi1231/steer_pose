# -*- coding: utf-8 -*-
"""比较"自编 demo 数据"与"真实 BamaPig3D 数据"的 3D 结构质量。

核心指标：把每个姿态中心化后做 SVD，得到 s1>=s2>=s3。
  s3/s1 很小  -> 姿态几乎是一张平面上的图（planar），正交投影下"旋转"几乎不可观测，
                 导致 Lkp 有很大的**不可约下界**，网络再怎么训也降不下去。
  s3/s1 接近 1 -> 姿态是真正立体的，任务可解。

同时看"姿态在时间上的变化是否只发生在一个平面内"。
"""
import io
import pickle
import sys
import zipfile

import numpy as np

sys.path.insert(0, r"C:\Users\23271\Desktop\passage\steer_pose_repo")
from steerpose.geometry import make_demo_dataset  # noqa: E402

P3D = r"D:\新建文件夹\BamaPig3D_pure_pickle.zip"
G_ALL_PARTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]


def sv_ratios(poses):
    """(M,J,3) -> 每个姿态的 s2/s1, s3/s1。"""
    c = poses - poses.mean(axis=1, keepdims=True)
    # 批量 SVD
    out2, out3 = [], []
    for p in c:
        s = np.linalg.svd(p, compute_uv=False)
        s = np.sort(s)[::-1]
        out3.append(s[2] / (s[0] + 1e-12))
        out2.append(s[1] / (s[0] + 1e-12))
    return np.array(out2), np.array(out3)


def report(name, poses, note=""):
    print("=" * 74)
    print(f"{name}   shape={poses.shape}  {note}")
    s2, s3 = sv_ratios(poses)
    print(f"  s2/s1  中位 {np.median(s2):.3f}   min {s2.min():.3f}")
    print(f"  s3/s1  中位 {np.median(s3):.3f}   min {s3.min():.3f}   "
          f"<0.15 的比例 {(s3 < 0.15).mean()*100:.1f}%")
    span = poses.reshape(-1, 3).max(0) - poses.reshape(-1, 3).min(0)
    print(f"  坐标跨度 {np.round(span, 3)}（米）")
    # 单个体在时间上的运动方向分布
    if poses.shape[0] >= 3:
        vel = np.diff(poses, axis=0)                       # (M-1,J,3)
        v = vel.reshape(-1, 3)
        v = v[np.linalg.norm(v, axis=1) > 1e-6]
        if len(v) > 10:
            sv = np.linalg.svd(v - v.mean(0), compute_uv=False)
            sv = sv / sv[0]
            print(f"  运动方向主轴占比 s2/s1 {sv[1]:.3f}  s3/s1 {sv[2]:.3f} "
                  f"→ 运动{'基本平面内' if sv[2] < 0.1 else '是立体的'}")
    return float(np.median(s3))


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    demo = make_demo_dataset(num_anims=200, num_frames=20, seed=1)
    r_demo = report("A. 自编 demo 数据（20 关节，四足骨架）", demo)

    z = zipfile.ZipFile(P3D)
    with z.open("BamaPig3D_pure_pickle/label_mix.pkl") as f:
        mix = pickle.load(io.BytesIO(f.read()))
    z.close()
    real = mix[..., G_ALL_PARTS, :]              # (70,4,19,3)
    nz = (np.abs(real).sum(-1) > 1e-9)
    real = real[nz.all(-1)] if nz.ndim == 3 and nz.all(-1).any() else real.reshape(-1, 19, 3)
    r_real = report("B. 真实 BamaPig3D label_mix（19 关节）", real)

    # 单个个体在 70 帧上的时间序列（真实数据里同一头猪）
    pig0 = mix[:, 0][..., G_ALL_PARTS, :]
    report("C. 真实数据：同一头猪 70 帧的时间序列", pig0)

    print()
    print("=" * 74)
    print(f"结论：demo 的 s3/s1 中位 = {r_demo:.3f}，真实数据 = {r_real:.3f}")
    print("      s3/s1 越小 → 姿态越接近一张平面图 → 正交投影下视角旋转几乎不可观测")
    print("      → 网络输出对 R 不敏感 → Lkp 有很高的不可约下界（这正是 demo 上 Lkp")
    print("        卡在 0.64、且与'直接输出平均姿态'的 0.6481 几乎相同的原因）")


if __name__ == "__main__":
    main()
