# -*- coding: utf-8 -*-
"""几何工具：视角合成、Rodrigues 旋转、投影、度量指标（论文附录 D.1）.

训练数据生成流程:
  1. Fibonacci 球面法在上半单位球面均匀取 100 个视点（朝向原点）；
  2. 每个视点绕光轴均匀采 20 个滚转角；
  3. 随机抽 num_pairs 对相机，对 3D 姿态做正交投影 -> (P, P') 与真值相对旋转 R。
"""
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def fibonacci_hemisphere(n: int = 100) -> np.ndarray:
    """上半单位球面均匀取 n 个视点 (n, 3)。"""
    pts = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = max(1.0 - (i + 0.5) / n * 1.0, 0.05)   # 只取上半球
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = i * golden
        pts.append([r * math.cos(phi), r * math.sin(phi), z])
    return np.asarray(pts, dtype=np.float64)


def look_at_origin(eye: np.ndarray) -> np.ndarray:
    """构造朝向原点的相机旋转 R_w2c (3,3)：相机 z 轴指向原点。"""
    z = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


def roll_about_axis(R: np.ndarray, angle: float) -> np.ndarray:
    """绕相机光轴（z）滚转 angle。"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R


def ortho_project(pose3d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """正交投影：世界系 3D 姿态 (M,3) 经 R_w2c 后取前两维 -> (M,2)。"""
    return (R @ pose3d.T).T[:, :2]


def perspective_project(pose3d: np.ndarray, R: np.ndarray, t: np.ndarray,
                        focal: float = 1.0) -> np.ndarray:
    """透视投影 -> 归一化图像坐标（主点=0）。

    注意：训练数据合成用的是正交投影（论文附录 D.1），而推理阶段的 Lgeom 用的是
    文献[19]的透视共线/共面约束，所以几何自检和 --demo 必须用透视投影才能与 Lgeom 自洽。
    """
    K, J, _ = pose3d.shape
    Xc = (np.einsum("ij,kj->ki", R, pose3d.reshape(-1, 3)) + t).reshape(K, J, 3)
    return Xc[..., :2] / (Xc[..., 2:3] + 1e-9) * focal


def make_two_view_scene(n_obj: int = 12, seed: int = 0, jitter: float = 0.35,
                        rot_deg: float = None, baseline: float = None,
                        xy: float = 1.2, depth=(3.0, 6.0)) -> dict:
    """合成"两视角 + 已知真值相对位姿"的场景，用于几何自检与 calibrate --demo。

    相机 A 固定在原点（R=I, t=0）；相机 B 用随机朝向与基线。目标在视锥内随机摆放，
    这样 t 才可辨识（所有目标都挤在一条视线上时 t 是退化的）。

    返回 dict: P, P2, R, t, poses3d（P/P2 为**归一化图像坐标**，可直接喂 calibrate）
    """
    rng = np.random.default_rng(seed)
    base = make_demo_dataset(num_anims=n_obj, num_frames=1, seed=seed)
    base = base - base.mean(axis=1, keepdims=True)
    base = base / (np.abs(base).max() + 1e-9) * jitter
    pos = rng.uniform([-xy, -xy, depth[0]], [xy, xy, depth[1]], size=(n_obj, 1, 3))
    posed3d = base + pos

    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    deg = rot_deg if rot_deg is not None else rng.uniform(20.0, 90.0)
    R = Rotation.from_rotvec(axis * np.deg2rad(deg)).as_matrix()
    t = rng.normal(size=3); t = t / np.linalg.norm(t) * \
        (baseline if baseline is not None else rng.uniform(0.8, 2.5))

    P = perspective_project(posed3d, np.eye(3), np.zeros(3))
    P2 = perspective_project(posed3d, R, t)
    return {"P": P, "P2": P2, "R": R, "t": t, "poses3d": posed3d,
            "focal": 1.0, "principal_point": np.zeros(2)}


# ---------------- Rodrigues 旋转（可微 / numpy 两版） ----------------

def rodrigues_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """旋转向量 (...,3) -> 旋转矩阵 (...,3,3)，对 rotvec 可微。"""
    theta = torch.norm(rotvec, dim=-1, keepdim=True) + 1e-12
    k = rotvec / theta
    K = torch.zeros(*rotvec.shape[:-1], 3, 3, device=rotvec.device, dtype=rotvec.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    return eye + torch.sin(theta).unsqueeze(-1) * K \
        + (1.0 - torch.cos(theta).unsqueeze(-1)) * (K @ K)


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> Rodrigues 向量 (3,)。"""
    return Rotation.from_matrix(R).as_rotvec()


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """旋转误差（度），对应论文指标 ER。"""
    return float(np.degrees(np.linalg.norm(
        Rotation.from_matrix(R_est @ R_gt.T).as_rotvec())))


def translation_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """平移方向误差（度），对应论文指标 Et（平移只恢复方向，尺度不可观测）。"""
    a, b = np.asarray(t_est).ravel(), np.asarray(t_gt).ravel()
    a, b = a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(math.acos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


# ---------------- 视角合成器（论文附录 D.1） ----------------

class ViewSynthesizer:
    def __init__(self, num_views: int = 100, num_rolls: int = 20,
                 num_pairs: int = 3000, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.cameras = []
        for e in fibonacci_hemisphere(num_views):
            R0 = look_at_origin(e)
            for k in range(num_rolls):
                self.cameras.append(roll_about_axis(R0, 2.0 * math.pi * k / num_rolls))
        self.num_pairs = num_pairs

    def sample_pair(self):
        """随机抽一对相机，返回 (R_a, R_b, R_rel = R_b @ R_a^T)。"""
        ia, ib = self.rng.choice(len(self.cameras), size=2, replace=False)
        Ra, Rb = self.cameras[ia], self.cameras[ib]
        return Ra, Rb, Rb @ Ra.T

    def make_training_set(self, poses3d: np.ndarray) -> dict:
        """(M, J, 3) 3D 姿态 -> {'P','P2','rotvec'} 合成训练对。"""
        M, J, _ = poses3d.shape
        poses_c = poses3d - poses3d.mean(axis=1, keepdims=True)   # 每个姿态居中
        Ps, P2s, RVs = [], [], []
        for _ in range(self.num_pairs):
            Ra, Rb, Rrel = self.sample_pair()
            idx = self.rng.integers(M)
            Ps.append(ortho_project(poses_c[idx], Ra))
            P2s.append(ortho_project(poses_c[idx], Rb))
            RVs.append(matrix_to_rodrigues(Rrel))
        return {"P": np.asarray(Ps, np.float32),
                "P2": np.asarray(P2s, np.float32),
                "rotvec": np.asarray(RVs, np.float32)}


# ---------------- 无数据集时的演示数据 ----------------

def make_demo_dataset(num_anims: int = 200, num_frames: int = 20,
                      num_joints: int = 20, seed: int = 1) -> np.ndarray:
    """程序化生成"四足动物"3D 姿态 (M, J, 3)，仅用于跑通流程自测。

    骨架：脊柱 4 + 头颈 2 + 尾 2 + 四条腿各 3 = 20 关节；个体间有体型差异。
    """
    rng = np.random.default_rng(seed)
    spine, headtail = 4, 4
    need = spine + headtail + 4 * 3
    if num_joints != need:
        raise ValueError(
            f"演示骨架固定为 {need} 个关节，收到 num_joints={num_joints}。"
            f"真实数据请用 scripts/prepare_*.py 生成 poses.npz。")
    num_joints = need
    M = num_anims * num_frames
    poses = np.zeros((M, num_joints, 3), np.float64)
    for a in range(num_anims):
        phase = rng.uniform(0, 2 * math.pi)
        freq = rng.uniform(0.8, 1.6)
        body_y = rng.uniform(-0.2, 0.2)
        s_body = rng.uniform(0.7, 1.3)
        s_leg = rng.uniform(0.7, 1.3)
        h_body = rng.uniform(0.35, 0.75)
        for f in range(num_frames):
            t = f * 0.1
            m = a * num_frames + f
            swing = math.sin(2 * math.pi * freq * t + phase)
            for i in range(spine):
                poses[m, i] = [0.35 * s_body * (i - 1.5), body_y, h_body + 0.05 * math.sin(t + i)]
            poses[m, spine + 0] = [0.85 * s_body, body_y + 0.05 * swing, h_body + 0.07]
            poses[m, spine + 1] = [1.05 * s_body, body_y + 0.10 * swing, h_body + 0.11]
            poses[m, spine + 2] = [-0.75 * s_body, body_y - 0.03 * swing, h_body + 0.05]
            poses[m, spine + 3] = [-0.95 * s_body, body_y - 0.06 * swing, h_body + 0.09]
            for li in range(4):
                s = 1.0 if li < 2 else -1.0
                d = 1.0 if li % 2 == 0 else -1.0
                sw = swing * (1.0 if li % 2 == 0 else -1.0)
                base = spine + headtail + li * 3
                poses[m, base + 0] = [0.45 * s_body * s, 0.25 * d, h_body - 0.05]
                poses[m, base + 1] = [0.45 * s_body * s + 0.15 * sw, 0.32 * d, h_body - 0.05 - 0.22 * s_leg]
                poses[m, base + 2] = [0.45 * s_body * s + 0.30 * sw, 0.34 * d, h_body - 0.05 - 0.45 * s_leg]
    poses += rng.normal(0, 0.02, poses.shape)
    return poses
