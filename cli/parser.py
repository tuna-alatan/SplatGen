import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D Gaussian Splatting scene from input images"
    )

    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing input images",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store all outputs (COLMAP, Gaussians, video, etc.)",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="scene1",
        help="Name of the project (used for folder naming)",
    )

    args = parser.parse_args()

    # Validate input folder
    if not os.path.isdir(args.input_folder):
        parser.error(f"Input folder {args.input_folder} does not exist or is not a directory.")

    return args
