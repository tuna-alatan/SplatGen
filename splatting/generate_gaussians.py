import torch

def generate_dummy_gaussians(positions, colors):
    N = positions.shape[0]

    # Compute density-based scales
    distances = torch.cdist(positions, positions)  # Pairwise distances
    density = torch.sum(distances < 0.1, dim=1)    # Count neighbors within a radius
    scales = 0.01 / (density + 1e-5)               # Inverse density scaling
    scales = scales.unsqueeze(1).repeat(1, 3)     # Isotropic scaling

    # Compute normals for rotations (dummy normals for now)
    normals = torch.zeros_like(positions)
    normals[:, 2] = 1.0  # Assume all normals point up
    rotations = torch.cat([torch.ones((N, 1)), normals], dim=1)  # Quaternion (w, x, y, z)

    # Compute opacity based on density
    opacities = torch.clamp(1.0 / (density + 1e-5), max=1.0)  # Higher density → lower opacity

    return {
        "positions": positions,
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities
    }
