# -*- coding: utf-8 -*-
"""数据准备：3D 姿态加载、2D 姿态归一化、训练对构建与增强（论文附录 D.1）.

标准数据格式：一个 .npz 文件，内含数组
    poses3d : (M, J, 3)  世界系 3D 关节坐标（任意单位/尺度，训练时会逐姿态居中）
可选数组
    joint_names : (J,)   关节名（仅用于日志/可视化，不参与训练）

增强（论文附录 D.1）：
    - 输入 2D 关节位置加高斯噪声（模拟关键点检测误差）
    - 随机掩码 10%~30% 的关节 token（模拟遮挡）
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import ViewSynthesizer


def normalize_pose(p: np.ndarray) -> np.ndarray:
    """2D 姿态归一化到 [-1, 1]：包围盒中心为原点、最长边映射为 2。"""
    c = (p.max(axis=0) + p.min(axis=0)) / 2.0
    scale = (p.max(axis=0) - p.min(axis=0)).max() / 2.0 + 1e-8
    return (p - c) / scale


def load_poses3d(path: str, key: str = "poses3d") -> np.ndarray:
    """读取 3D 姿态 npz -> float32 (M, J, 3)。"""
    arr = np.load(path, allow_pickle=False)
    if key not in arr:
        raise KeyError(f"{path} 中缺少数组 '{key}'，现有: {list(arr.keys())}")
    poses = np.asarray(arr[key], np.float32)
    if poses.ndim != 3 or poses.shape[2] != 3:
        raise ValueError(f"poses3d 形状应为 (M, J, 3)，实际 {poses.shape}")
    return poses


class PosePairDataset(Dataset):
    """由合成训练对构成的 Dataset：返回归一化后的 (P, P2, rotvec, mask)。

    按论文附录 D.1，训练时对输入加噪声（默认 0.02，归一化坐标下）并随机掩码
    10%~30% 关节；验证时噪声更小（0.01），保持一致的掩码比例。
    """

    def __init__(self, pairs: dict, indices: np.ndarray, noise_std: float = 0.02,
                 mask_ratio: tuple = (0.1, 0.3), seed: int = 0):
        self.P = pairs["P"][indices]
        self.P2 = pairs["P2"][indices]
        self.rotvec = pairs["rotvec"][indices]
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.rng = np.random.default_rng(seed)
        self.num_joints = self.P.shape[1]

    def __len__(self) -> int:
        return self.P.shape[0]

    def __getitem__(self, i: int):
        p = normalize_pose(self.P[i].astype(np.float64)).astype(np.float32)
        p2 = normalize_pose(self.P2[i].astype(np.float64)).astype(np.float32)
        if self.noise_std > 0:
            p = p + self.rng.normal(0, self.noise_std, p.shape).astype(np.float32)
        mask = np.ones(self.num_joints, np.float32)
        lo, hi = self.mask_ratio
        k = int(self.num_joints * (lo + self.rng.random() * (hi - lo)))
        if k > 0:
            mask[self.rng.choice(self.num_joints, k, replace=False)] = 0.0
        return (torch.from_numpy(p), torch.from_numpy(p2),
                torch.from_numpy(self.rotvec[i].astype(np.float32)),
                torch.from_numpy(mask))


def build_pairs(poses3d: np.ndarray, num_pairs: int = 3000, num_views: int = 100,
                num_rolls: int = 20, seed: int = 0, full_sphere: bool = True) -> dict:
    """按论文附录 D.1 合成 (P, P', R) 训练对。

    full_sphere=True（默认）用完整球面视点，覆盖全部 SO(3)；
    只用上半球（False）会让训练集缺失一半朝向，推理时遇到"镜像视角"
    （等价于从下半球看）会给出完全错误的结果。详见 geometry.ViewSynthesizer。
    """
    vs = ViewSynthesizer(num_views=num_views, num_rolls=num_rolls,
                         num_pairs=num_pairs, seed=seed, full_sphere=full_sphere)
    return vs.make_training_set(poses3d)


def split_indices(num_pairs: int, ratios=(0.7, 0.2, 0.1)) -> dict:
    """按 7:2:1 划分训练/验证/测试（论文附录 D.1）。"""
    n_tr = int(num_pairs * ratios[0])
    n_va = int(num_pairs * ratios[1])
    return {"train": np.arange(0, n_tr),
            "val": np.arange(n_tr, n_tr + n_va),
            "test": np.arange(n_tr + n_va, num_pairs)}
