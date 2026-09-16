# -*- coding: utf-8 -*-
"""SteerPose 独立复现包.

论文: SteerPose: Simultaneous Extrinsic Camera Calibration and Matching from
Articulation (Lee, Nishino, Nobuhara; BMVC 2025, arXiv:2506.01691)

官方代码 github.com/kcvl-public/steerpose 截至 2026-09 尚未公开，
本包按论文正文与附录 D 独立实现。

除 SteerPose 本体（2D→2D 匹配/标定）外，本包还移植了同作者前作
**RA-L2022 `Extrinsic Camera Calibration From a Moving Person`** 的几何标定后端，
补上 SteerPose 缺失的"2D→单目 3D→外参"这一环（多物种/畜牧场景用）：

    steerpose.skeleton : 骨架定义（关节序 / 骨连接 / 根关节）
    steerpose.lifting  : 单目 3D 提升器 PoseLifter（RA-L2022 链路第一环）
    steerpose.calib_ral: RA-L2022 几何标定后端（旋转/平移线性解 + BA）

分工见 docs/ral_backend.md。
"""
from .model import SteerPose
from .losses import (lkp_mean, matching_loss, geometric_loss, sinkhorn, similarity,
                     to_ray_coords, guess_focal, build_linear_system, solve_translation,
                     confident_pairs, geom_conf_threshold)
from .geometry import (ViewSynthesizer, make_demo_dataset, rodrigues_to_matrix,
                       matrix_to_rodrigues, ortho_project, rotation_error_deg,
                       translation_error_deg)
from .calibrate import calibrate, calibrate_with_retries

# --- 骨架 / 提升器 / RA-L2022 几何后端 ---
from .skeleton import (get_skeleton, bones_to_index_array, bone_lengths,
                       PIG19_NAMES, PIG19_BONES, PIG19_CALIB_BONES, PIG19_ROOT)
from .lifting import (PoseLifter, LifterDataset, normalize_2d_intrinsics,
                      normalize_3d, root_center, pose_scale, bone_vectors,
                      bone_direction_loss, bone_direction_error_deg, mpjpe,
                      procrustes_mpjpe, bone_length_mean, scale_invariant_mpjpe,
                      build_frame_split, build_split_indices)
from .calib_ral import (calibrate_from_lifter, calib_ral, oriented_points,
                        normalized_rays, solve_rotation, solve_translation,
                        z_test_sign, triangulate, dlt_batch, ba_refine,
                        visible_from_all, camera_centers, alignment_error,
                        similarity_align, relative_rotation_error_deg,
                        translation_direction_error_deg, so3_angle_deg,
                        reprojection_errors)

__all__ = [
    # SteerPose 本体
    "SteerPose", "lkp_mean", "matching_loss", "geometric_loss", "sinkhorn",
    "similarity", "to_ray_coords", "guess_focal", "build_linear_system", "solve_translation",
    "confident_pairs", "geom_conf_threshold",
    "ViewSynthesizer", "make_demo_dataset", "rodrigues_to_matrix",
    "matrix_to_rodrigues", "ortho_project", "rotation_error_deg",
    "translation_error_deg", "calibrate", "calibrate_with_retries",
    # 骨架
    "get_skeleton", "bones_to_index_array", "bone_lengths",
    "PIG19_NAMES", "PIG19_BONES", "PIG19_CALIB_BONES", "PIG19_ROOT",
    # 单目 3D 提升器
    "PoseLifter", "LifterDataset", "normalize_2d_intrinsics", "normalize_3d",
    "root_center", "pose_scale", "bone_vectors", "bone_direction_loss",
    "bone_direction_error_deg", "mpjpe", "procrustes_mpjpe", "bone_length_mean",
    "scale_invariant_mpjpe", "build_frame_split", "build_split_indices",
    # RA-L2022 几何标定后端
    "calibrate_from_lifter", "calib_ral", "oriented_points", "normalized_rays",
    "solve_rotation", "solve_translation", "z_test_sign", "triangulate",
    "dlt_batch", "ba_refine", "visible_from_all", "camera_centers",
    "alignment_error", "similarity_align", "relative_rotation_error_deg",
    "translation_direction_error_deg", "so3_angle_deg", "reprojection_errors",
]
__version__ = "0.2.0"
