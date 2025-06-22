import torch
import torch.nn.functional as F

def render_gaussians_2d(positions, colors, camera, scales, opacities, point_radius=3, image_res=None):
    device = torch.device("mps")  # Use Metal backend for macOS
    positions = positions.to(device)
    colors = colors.to(device)

    # Intrinsics and view matrix
    K = camera["K"].clone().to(device)
    R = camera["R"].to(device)
    T = camera["T"].to(device)

    width = camera["width"]
    height = camera["height"]
    if image_res is None:
        image_res = (height, width)
    else:
        # Scale intrinsics if resizing
        scale_x = image_res[1] / width
        scale_y = image_res[0] / height
        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y
        width = image_res[1]
        height = image_res[0]

    # Convert world to camera space
    cam_positions = (R @ positions.T).T + T

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    zs = cam_positions[:, 2]
    xs = (cam_positions[:, 0] / zs) * fx + cx
    ys = (cam_positions[:, 1] / zs) * fy + cy

    valid = (zs > 0) & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    cs = colors[valid]

    img = torch.ones(3, height, width, device=device)  # Black background

    # Create a grid of pixel coordinates
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    yy = yy.unsqueeze(0).float()
    xx = xx.unsqueeze(0).float()

    # Compute distances and render all Gaussians in parallel
    for x, y, c, scale, opacity in zip(xs, ys, cs, scales, opacities):
        dx = xx - x
        dy = yy - y
        dist = torch.sqrt(dx**2 + dy**2)
        adjusted_radius = point_radius * scale.mean()  # Adjust radius by scale
        alpha = torch.clamp(1 - (dist / adjusted_radius), min=0) * opacity  # Modulate alpha by opacity
        alpha = alpha.unsqueeze(0)  # Add channel dimension
        img = img * (1 - alpha) + c.view(3, 1, 1) * alpha

    return img.clamp(0, 1)
