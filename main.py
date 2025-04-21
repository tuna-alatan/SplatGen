from cli.parser import get_args
from colmap.run_colmap import run_colmap_pipeline
from colmap.parse_outputs import load_camera_data
from splatting.render_utils import build_camera_dict
from splatting.render_frame import render_frame
from colmap.pointcloud import load_points3d
from splatting.generate_gaussians import generate_dummy_gaussians

import os

def main():
    args = get_args()
    output_path = os.path.join(args.output_dir, args.project_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"[✓] Input images loaded from: {args.input_folder}")
    print(f"[✓] Output will be saved to: {output_path}")

    # Step 2: Run COLMAP
    run_colmap_pipeline(args.input_folder, output_path)

    # Step 3: Parse COLMAP output
    sparse_model_dir = os.path.join(output_path, "sparse", "0")
    camera_data = load_camera_data(sparse_model_dir)

    positions, colors = load_points3d(sparse_model_dir)
    gaussians = generate_dummy_gaussians(positions, colors)

    camera = build_camera_dict(camera_data[0])
    image_name = camera_data[0]["image_name"]
    render_path = os.path.join(output_path, f"render_{image_name}")
    render_frame(gaussians, camera, render_path)

    print("[→] First camera info:")
    print(camera_data[0])

if __name__ == "__main__":
    main()
