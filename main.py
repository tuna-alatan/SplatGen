from cli.parser import get_args
from colmap.run_colmap import run_colmap_pipeline
from colmap.parse_outputs import load_camera_data
from splatting.render_utils import build_camera_dict
from colmap.pointcloud import load_points3d
from splatting.generate_gaussians import generate_dummy_gaussians
from splatting.renderer import render_gaussians_2d

from torchvision.utils import save_image
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

    # Step 4: Load 3D points and create dummy Gaussians
    positions, colors = load_points3d(sparse_model_dir)
    gaussians = generate_dummy_gaussians(positions, colors)

    # Step 5: Render from a single camera (camera ID 40)
    camera = build_camera_dict(camera_data[39])
    print(camera_data[39])

    image = render_gaussians_2d(
        positions=gaussians["positions"],
        colors=gaussians["colors"],
        camera=camera,
        point_radius=10
    )

    output_image_path = os.path.join(output_path, "single_view_render.png")
    save_image(image, output_image_path)
    print(f"[✓] Rendered and saved single frame from camera 40 → {output_image_path}")

if __name__ == "__main__":
    main()
