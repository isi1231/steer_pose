# -*- coding: utf-8 -*-
"""骨架定义：关节顺序、骨连接（bones）、根关节（root）。

为什么单独一个模块
------------------
SteerPose 的 2D→2D 网络与物种无关（只吃 2D 关节坐标和旋转），但另外两件事必须知道骨架：

1. **单目 3D 提升器**（`steerpose.lifting`）：输出根相对 3D 需要指定根关节；骨长用于
   把 3D 归一化到一个与单目尺度无关的量纲；
2. **RA-L2022 几何标定后端**（`steerpose.calib_ral`）：它的核心观测量就是
   **oriented points = 骨方向单位向量**（对单目深度误差鲁棒，只依赖姿态朝向），
   所以必须知道"哪两个关节构成一根骨头"。

BamaPig3D 官方骨架（`utils.py` 的 `g_all_parts`，23 → 19 关节）
--------------------------------------------------------------
23 关节里只有 19 个被官方使用：索引 `[0..16, 18, 20]`（第 17/19/21/22 行恒为 0）。
这 19 个关节的名字/顺序在下面 `PIG19` 里给出，与 `label_keypoints2d.pkl` 的关节维一致。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# BamaPig3D 官方 19 关节
# ---------------------------------------------------------------------------
PIG19_NAMES = [
    "nose",        # 0
    "l_eye",       # 1
    "r_eye",       # 2
    "l_ear",       # 3
    "r_ear",       # 4
    "l_shoulder",  # 5
    "r_shoulder",  # 6
    "l_elbow",     # 7
    "r_elbow",     # 8
    "l_paw",       # 9   前左爪
    "r_paw",       # 10  前右爪
    "l_hip",       # 11
    "r_hip",       # 12
    "l_knee",      # 13
    "r_knee",      # 14
    "l_foot",      # 15  后左脚
    "r_foot",      # 16  后右脚
    "tail",        # 17
    "center",      # 18  躯干中心（连接 nose / tail / 双肩）
]

# 官方 19 骨（与 scripts/prepare_mammal.py 的 BONES_19 一致，用于可视化/骨长统计）
PIG19_BONES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),          # 头部小骨
    (0, 18), (18, 17),                                # 头—躯干、躯干—尾
    (18, 5), (5, 7), (7, 9),                          # 左前腿
    (18, 6), (6, 8), (8, 10),                         # 右前腿
    (17, 11), (11, 13), (13, 15),                     # 左后腿
    (17, 12), (12, 14), (14, 16),                     # 右后腿
]

# 标定用的"可靠骨"：剔除头部小骨（眼/耳/鼻只有几个像素，2D 噪声下方向极不可靠）。
# 这是 RA-L2022 式 oriented points 的默认输入（14 根）。
PIG19_CALIB_BONES = [
    (0, 18), (18, 17),
    (18, 5), (5, 7), (7, 9),
    (18, 6), (6, 8), (8, 10),
    (17, 11), (11, 13), (13, 15),
    (17, 12), (12, 14), (14, 16),
]

PIG19_ROOT = 18          # "center"：躯干中心，是 2D/3D 的标准根关节

SKELETONS = {
    "pig19": {
        "names": PIG19_NAMES,
        "bones": PIG19_BONES,
        "calib_bones": PIG19_CALIB_BONES,
        "root": PIG19_ROOT,
        "num_joints": 19,
    }
}


def get_skeleton(name: str = "pig19") -> dict:
    """取骨架定义（dict，含 names / bones / calib_bones / root / num_joints）。

    想换成别的物种/骨架：往 `SKELETONS` 里加一条即可，其余代码都按 dict 取值。
    也可以直接自己构造一个同样字段的 dict 传进来。
    """
    if name not in SKELETONS:
        raise KeyError(f"未知骨架 '{name}'，可选：{list(SKELETONS)}")
    sk = SKELETONS[name]
    return {"names": list(sk["names"]),
            "bones": [tuple(b) for b in sk["bones"]],
            "calib_bones": [tuple(b) for b in sk["calib_bones"]],
            "root": int(sk["root"]),
            "num_joints": int(sk["num_joints"])}


def bones_to_index_array(bones) -> np.ndarray:
    """[(i,j), ...] -> (B,2) int 数组（下标是 0-based 关节索引）。"""
    b = np.asarray(list(bones), dtype=np.int64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"bones 应为 (B,2) 的关节索引对，实际 {b.shape}")
    return b


def bone_lengths(p3d: np.ndarray, bones) -> np.ndarray:
    """(...,J,3) + bones -> (...,B) 骨长。"""
    b = bones_to_index_array(bones)
    d = p3d[..., b[:, 0], :] - p3d[..., b[:, 1], :]
    return np.linalg.norm(d, axis=-1)
