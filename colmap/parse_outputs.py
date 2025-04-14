from .parse_model import read_model

def load_camera_data(sparse_model_dir):
    result = read_model(sparse_model_dir, ext='.bin')

    if result is None:
        raise RuntimeError(f"[✗] Failed to read COLMAP model from: {sparse_model_dir}")

    cameras, images, points3D = result

    print(f"[✓] Loaded {len(images)} registered images")
    print(f"[✓] Loaded {len(cameras)} cameras")
    print(f"[✓] Loaded {len(points3D)} 3D points")

    camera_data = []

    for image_id, image in images.items():
        cam = cameras[image.camera_id]

        # Extrinsics
        qw, qx, qy, qz = image.qvec
        tx, ty, tz = image.tvec

        # Intrinsics
        model = cam.model
        width = cam.width
        height = cam.height
        params = cam.params  # varies by model type

        camera_data.append({
            "image_name": image.name,
            "qvec": image.qvec,   # quaternion rotation
            "tvec": image.tvec,   # translation vector
            "model": model,
            "width": width,
            "height": height,
            "params": params
        })

    return camera_data
