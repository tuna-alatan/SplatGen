import torch

def generate_dummy_gaussians(positions, colors):
    N = positions.shape[0]

    # Dummy values
    scales = torch.full((N, 3), 0.01)              # Small, isotropic Gaussians
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(N, 1)  # Identity quaternions
    opacities = torch.full((N,), 1.0)              # Fully opaque

    return {
        "positions": positions,
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities
    }
