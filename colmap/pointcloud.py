from .parse_model import read_model
import numpy as np
import torch

def load_points3d(sparse_model_dir):
    result = read_model(sparse_model_dir, ext='.bin')

    if result is None:
        raise RuntimeError(f"[✗] Failed to read COLMAP model from: {sparse_model_dir}")

    _, _, points3D = result

    if points3D is None:
        raise RuntimeError(f"[✗] Failed to load 3D points from: {sparse_model_dir}")

    positions = []
    colors = []

    for _, pt in points3D.items():
        positions.append(pt.xyz)
        colors.append([c / 255.0 for c in pt.rgb])  # Normalize to [0, 1]

    return torch.from_numpy(np.array(positions)).float(), torch.from_numpy(np.array(colors)).float()
