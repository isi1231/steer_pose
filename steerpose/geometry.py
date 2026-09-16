# -*- coding: utf-8 -*-
"""几何工具：视角合成、Rodrigues 旋转、投影、度量指标（论文附录 D.1）.

训练数据生成流程:
  1. Fibonacci 球面法**在完整单位球面上**均匀取 100 个视点（朝向原点），再对每个视点
     绕光轴均匀采 20 个滚转角，共 2000 台相机；
  2. 随机抽 num_pairs 对相机，对 3D 姿态做正交投影 -> (P, P') 与真值相对旋转 R。

⚠️ 视点必须覆盖**完整球面**而不是只取上半球：只取上半球等价于只覆盖一半旋转群，
   训练集里不存在"镜像视图"，推理时会静默失效（见 make_two_view_scene 的说明）。
"""
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def fibonacci_sphere(n: int = 100) -> np.ndarray:
    """单位球面（含下半球）均匀取 n 个视点 (n, 3)，用于覆盖完整的 SO(3)。

    ⚠️ 早期实现只取上半球（z>=0.05），这会让训练相机集合缺失一半朝向。
    后果很隐蔽：网络只见过"相机 z 轴落在位姿系上半球"的视图，遇到镜像视图
    （等价于从下半球看）时会输出完全错误的结果。详见 make_two_view_scene 的说明。
    """
    pts = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n          # [-1, 1]，含负值
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = i * golden
        pts.append([r * math.cos(phi), r * math.sin(phi), z])
    return np.asarray(pts, dtype=np.float64)


def fibonacci_hemisphere(n: int = 100) -> np.ndarray:
    """上半单位球面均匀取 n 个视点 (n, 3)。

    保留是为了向后兼容；训练默认已改用 fibonacci_sphere（见 ViewSynthesizer）。
    只用上半球等价于只覆盖一半旋转群，会引入"镜像视图"这一分布外输入。
    """
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
                        xy: float = 1.2, depth=(3.0, 6.0),
                        camera_a: str = "training") -> dict:
    """合成"两视角 + 已知真值相对位姿"的场景，用于几何自检与 calibrate --demo。

    返回 dict: P, P2, R, t, poses3d, camera_a（P/P2 为**归一化图像坐标**，可直接喂 calibrate）

    ⚠️ 为什么相机 A 不能随便放（曾经踩过的坑，别改回去也别误判）
    ----------------------------------------------------------------
    1. 对**标定问题本身**，相机 A 的朝向只是世界系的规范选择：整体旋转同时作用于姿态与
       两台相机时，P、P'、(R, t) 全都不变。所以相机 A 放哪里都不影响 ER/Et 的定义。
    2. 但 P 是**网络的输入**，它的 2D 形状（含手性）取决于相机 A 相对"姿态规范系"的朝向。
       论文附录 D.1 把 100 台相机放在**单位半球面**上（原文："100 cameras uniformly over
       the unit hemisphere"），即网络只见过"从一侧看"的 2D 形状；镜像视角（正交投影下
       等价于从另一侧看）它没见过。实测：把相机 A 放在 `np.eye(3)`（沿世界 +z 看）时，
       同一权重的 Lkp 由 0.23 跳到 0.60、12 个目标 **0 个**被正确匹配 → Lmatch 降不下来
       → Sinkhorn 近均匀指派 → 可靠匹配不足 3 对 → **Lgeom 恒为 0**。
       症状全在推理端，训练曲线完全看不出来。
    3. 因此默认 `camera_a="training"`：相机 A 取自训练同源的相机集合（用上半球那批，
       它是 sphere / hemisphere 两种训练配置的公共区域，对两者都在分布内）。
       `camera_a="identity"` 保留旧行为，仅用于复现"分布外输入"这一现象；
       也可以直接传一个 (3,3) 旋转矩阵。
    4. 相机 B 随便放不影响网络输入（它只决定真值 R、t 与目标 P2），但必须保证**物体在
       B 的前方**，否则透视投影的深度变号、场景物理上不成立。这里按"物体到 B 的距离
       ~ depth"反解 B 的位置，天然满足。

    目标在视锥内随机摆放，这样 t 才可辨识（所有目标都挤在一条视线上时 t 是退化的）。
    """
    rng = np.random.default_rng(seed)
    base = make_demo_dataset(num_anims=n_obj, num_frames=1, seed=seed)
    base = base - base.mean(axis=1, keepdims=True)
    base = base / (np.abs(base).max() + 1e-9) * jitter

    # ---- 相机 A ----
    if camera_a == "training":
        # 上半球相机集合：sphere / hemisphere 两种训练配置的公共区域
        cams = ViewSynthesizer(num_views=100, num_rolls=20, num_pairs=1,
                               seed=seed, full_sphere=False).cameras
        Ra = cams[int(rng.integers(len(cams)))]
    elif camera_a == "identity":
        Ra = np.eye(3)
    else:
        Ra = np.asarray(camera_a, np.float64)
        if Ra.shape != (3, 3):
            raise ValueError(f"camera_a 应为 'training' / 'identity' 或 (3,3) 旋转矩阵，收到 {Ra.shape}")

    # ---- 目标摆放：在**相机 A 系**里摆（相机 A 在原点、沿 +z 看），保证都在视锥内 ----
    pos_a = rng.uniform([-xy, -xy, depth[0]], [xy, xy, depth[1]], size=(n_obj, 1, 3))
    posed3d_a = base + pos_a
    posed3d_w = posed3d_a @ Ra          # 世界系坐标：X_a = Ra @ X_w

    # ---- 相机 B：相对旋转 R，相对平移 t（由"物体在 B 前方"反解）----
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    deg = rot_deg if rot_deg is not None else rng.uniform(20.0, 90.0)
    R = Rotation.from_rotvec(axis * np.deg2rad(deg)).as_matrix()
    fwd_b = R[2]                                  # 相机 B 光轴（在 A 系中）方向
    d0 = float(posed3d_a[..., 2].mean())          # 物体中心到相机 A 的距离
    d_b = float(rng.uniform(*depth))              # 物体到相机 B 的距离
    lat = rng.normal(size=3); lat -= float(lat @ fwd_b) * fwd_b
    lat /= np.linalg.norm(lat) + 1e-12
    b = baseline if baseline is not None else rng.uniform(0.8, 2.5)
    c_b = np.array([0.0, 0.0, d0]) - fwd_b * d_b + lat * b   # 相机 B 中心（A 系）
    t = -R @ c_b                                  # X_b = R X_a + t

    P = perspective_project(posed3d_w, Ra, np.zeros(3))
    P2 = perspective_project(posed3d_w, R @ Ra, t)
    return {"P": P, "P2": P2, "R": R, "t": t, "poses3d": posed3d_w,
            "camera_a": Ra, "focal": 1.0, "principal_point": np.zeros(2)}


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
                 num_pairs: int = 3000, seed: int = 0, full_sphere: bool = True):
        """full_sphere=True 用完整球面覆盖 SO(3)（推荐）。

        只用上半球（full_sphere=False）时，相机 z 轴恒落在位姿系的上半球，
        等价于只覆盖一半旋转群：训练集里不存在"镜像视图"。标定时若真实相机
        相对位姿系的朝向落在下半球，2D 形状就是镜像的，网络会给出完全错误的
        预测（实测 Lkp 0.23 -> 0.60，正确配对无一被挑出）。所以默认用完整球面。
        """
        self.rng = np.random.default_rng(seed)
        self.cameras = []
        directions = fibonacci_sphere(num_views) if full_sphere \
            else fibonacci_hemisphere(num_views)
        for e in directions:
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
