import numpy as np
import torch
from colmap.parse_model import qvec2rotmat  # <- move import here

def build_camera_dict(cam_info):
    # Intrinsics
    model = cam_info["model"]
    w, h = cam_info["width"], cam_info["height"]
    params = cam_info["params"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0, 1]])

    elif model == "SIMPLE_RADIAL":
        f, cx, cy, _ = params  # ignore distortion param (k)
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
    else:
        raise NotImplementedError(f"Unsupported camera model: {model}")

    # Extrinsics
    R = qvec2rotmat(cam_info["qvec"])
    T = np.array(cam_info["tvec"])

    return {
        "K": torch.tensor(K, dtype=torch.float32),
        "R": torch.tensor(R, dtype=torch.float32),
        "T": torch.tensor(T, dtype=torch.float32),
        "width": w,
        "height": h
    }
