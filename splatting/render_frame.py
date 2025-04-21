import torch
from gsplat.rendering import rasterization
from torchvision.utils import save_image

def render_frame(gaussians, camera, save_path):
    # Construct the 4x4 world-to-camera transformation matrix
    R = camera["R"]  # Rotation matrix [3, 3]
    T = camera["T"].reshape(3, 1)  # Translation vector [3, 1]
    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = T.squeeze()
    viewmat = viewmat.unsqueeze(0)  # Shape: [1, 4, 4]

    # Prepare camera intrinsics
    K = camera["K"].unsqueeze(0)  # Shape: [1, 3, 3]

    # Set image dimensions
    width = camera["width"]
    height = camera["height"]

    # Set background color
    backgrounds = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)  # Shape: [1, 3]

    # Perform rasterization
    image, alpha, _ = rasterization(
        means=gaussians["positions"],       # [N, 3]
        quats=gaussians["rotations"],       # [N, 4]
        scales=gaussians["scales"],         # [N, 3]
        opacities=gaussians["opacities"],   # [N]
        colors=gaussians["colors"],         # [N, 3] or [N, K, 3] if using SH
        viewmats=viewmat,                   # [1, 4, 4]
        Ks=K,                               # [1, 3, 3]
        width=width,
        height=height,
        backgrounds=backgrounds,            # [1, 3]
        render_mode='RGB',                  # Options: 'RGB', 'D', 'ED', 'RGB+D', 'RGB+ED'
        rasterize_mode='classic',           # Options: 'classic', 'antialiased'
        packed=True                         # Memory-efficient mode
    )

    # Save the rendered image
    save_image(image.squeeze(0), save_path)
    print(f"[✓] Rendered and saved: {save_path}")
