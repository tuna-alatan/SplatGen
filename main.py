from colmap.run_colmap import run_colmap_pipeline
from colmap.parse_outputs import load_camera_data
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

    print(f"[→] First camera info:")
    print(camera_data[0])
