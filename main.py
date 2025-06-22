from cli.parser import get_args
from colmap.run_colmap import run_colmap_pipeline
from colmap.parse_outputs import load_camera_data
from splatting.render_utils import build_camera_dict
from colmap.pointcloud import load_points3d
from splatting.generate_gaussians import generate_dummy_gaussians
from splatting.renderer import render_gaussians_2d

import torch
from torchvision.utils import save_image
import os

def main():
    args = get_args()
    output_path = os.path.join(args.output_dir, args.project_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"[✓] Input images loaded from: {args.input_folder}")
    print(f"[✓] Output will be saved to: {output_path}")

    # Step 2: Run COLMAP
    #run_colmap_pipeline(args.input_folder, output_path)

    # Step 3: Parse COLMAP output
    sparse_model_dir = os.path.join(output_path, "sparse", "0")
    camera_data = load_camera_data(sparse_model_dir)

    # Step 4: Load 3D points
    positions, colors = load_points3d(sparse_model_dir)

    # Step 5: Initialize learnable Gaussian parameters
    positions = positions.to("mps")
    colors = colors.to("mps")
    scales = torch.full((positions.shape[0], 3), 0.01, device="mps").detach().requires_grad_()
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="mps").repeat(positions.shape[0], 1).detach().requires_grad_()
    opacities = torch.full((positions.shape[0],), 1.0, device="mps").detach().requires_grad_()

    # Step 6: Define optimizer and loss function
    optimizer = torch.optim.Adam([scales, rotations, opacities], lr=0.01)
    target_image = torch.ones(3, camera_data[39]["height"], camera_data[39]["width"], device="mps")  # Dummy target

    # Step 7: Optimization loop
    camera = build_camera_dict(camera_data[39])

    for epoch in range(100):  # Number of optimization steps
        optimizer.zero_grad()

        # Render image in batches
        batch_size = 5000  # Adjust batch size based on memory
        num_batches = (positions.shape[0] + batch_size - 1) // batch_size
        image = torch.zeros(3, 600, 800, device="mps")  # Initialize black background

        for i in range(num_batches):
            batch_positions = positions[i * batch_size:(i + 1) * batch_size]
            batch_colors = colors[i * batch_size:(i + 1) * batch_size]
            batch_scales = scales[i * batch_size:(i + 1) * batch_size]
            batch_opacities = opacities[i * batch_size:(i + 1) * batch_size]

            batch_image = render_gaussians_2d(
                positions=batch_positions,
                colors=batch_colors,
                camera=camera,
                scales=batch_scales,
                opacities=batch_opacities,
                point_radius=10,
                image_res=(600, 800)  # Reduced resolution
            )
            image += batch_image  # Accumulate batch results

        # Compute loss
        loss = torch.nn.functional.mse_loss(image, target_image)
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Save the optimized image
    output_image_path = os.path.join(output_path, "optimized_render.png")
    save_image(image, output_image_path)
    print(f"[✓] Rendered and saved optimized frame from camera 40 → {output_image_path}")

if __name__ == "__main__":
    main()
