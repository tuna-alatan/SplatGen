import torch
import numpy as np

def generate_orbit_path(center, radius, height, num_frames, intrinsics, image_res):
    cameras = []

    center = center.to(torch.float32)
    intrinsics = intrinsics.to(torch.float32)

    for i in range(num_frames):
        theta = 2 * np.pi * i / num_frames

        # Orbit camera position
        cam_pos = center + torch.tensor([
            radius * np.cos(theta),
            radius * np.sin(theta),
            height
        ], dtype=torch.float32)

        # Camera look-at
        forward = (center - cam_pos)
        forward = forward / torch.norm(forward)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        right = torch.cross(up, forward, dim=0)
        up = torch.cross(forward, right, dim=0)

        R = torch.stack([right, up, forward], dim=0).T  # World-to-camera

        cameras.append({
            "R": R,
            "T": cam_pos,
            "K": intrinsics,
            "width": image_res[0],
            "height": image_res[1]
        })

    return cameras

def generate_orbit_from_reference(start_R, start_T, center, num_frames, intrinsics, image_res):
    cameras = []

    center = center.to(torch.float32)
    intrinsics = intrinsics.to(torch.float32)

    # Compute initial relative vector (from center to camera)
    offset = start_T - center
    radius = torch.norm(offset)
    height = offset[2]  # Maintain same height

    for i in range(num_frames):
        theta = 2 * np.pi * i / num_frames

        # Rotate offset around Z axis
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = height
        cam_pos = center + torch.tensor([x, y, z], dtype=torch.float32)

        # Compute look-at rotation
        forward = (center - cam_pos)
        forward = forward / torch.norm(forward)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        right = torch.cross(up, forward, dim=0)
        up = torch.cross(forward, right, dim=0)
        R = torch.stack([right, up, forward], dim=0).T  # World-to-camera

        cameras.append({
            "R": R,
            "T": cam_pos,
            "K": intrinsics,
            "width": image_res[0],
            "height": image_res[1]
        })

    return cameras
