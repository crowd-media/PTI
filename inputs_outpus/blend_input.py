from dataclasses import dataclass

import numpy as np


@dataclass
class BlendInput:
    frames: np.ndarray
    affine_matrices: np.ndarray
    inverse_affine_matrices: np.ndarray
    mask: np.ndarray
    crop_coordinates: np.ndarray
    faces: np.ndarray