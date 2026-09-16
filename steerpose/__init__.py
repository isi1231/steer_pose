# -*- coding: utf-8 -*-
"""SteerPose 独立复现包.

论文: SteerPose: Simultaneous Extrinsic Camera Calibration and Matching from
Articulation (Lee, Nishino, Nobuhara; BMVC 2025, arXiv:2506.01691)

官方代码 github.com/kcvl-public/steerpose 截至 2026-09 尚未公开，
本包按论文正文与附录 D 独立实现。
"""
from .model import SteerPose
from .losses import (lkp_mean, matching_loss, geometric_loss, sinkhorn, similarity,
                     to_ray_coords, guess_focal, build_linear_system, solve_translation,
                     confident_pairs, geom_conf_threshold)
from .geometry import (ViewSynthesizer, make_demo_dataset, rodrigues_to_matrix,
                       matrix_to_rodrigues, ortho_project, rotation_error_deg,
                       translation_error_deg)
from .calibrate import calibrate, calibrate_with_retries

__all__ = [
    "SteerPose", "lkp_mean", "matching_loss", "geometric_loss", "sinkhorn",
    "similarity", "to_ray_coords", "guess_focal", "build_linear_system", "solve_translation",
    "confident_pairs", "geom_conf_threshold",
    "ViewSynthesizer", "make_demo_dataset", "rodrigues_to_matrix",
    "matrix_to_rodrigues", "ortho_project", "rotation_error_deg",
    "translation_error_deg", "calibrate", "calibrate_with_retries",
]
__version__ = "0.1.2"
