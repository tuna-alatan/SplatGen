import os
import subprocess

def run_colmap_pipeline(images_path, output_dir):
    db_path = os.path.join(output_dir, "database.db")
    sparse_path = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_path, exist_ok=True)

    print("[→] Running COLMAP Feature Extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_path,
    ], check=True)

    print("[→] Running COLMAP Exhaustive Matcher...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", db_path,
    ], check=True)

    print("[→] Running COLMAP Sparse Mapper...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", db_path,
        "--image_path", images_path,
        "--output_path", sparse_path,
    ], check=True)

    # Dense Reconstruction
    dense_path = os.path.join(output_dir, "dense")
    os.makedirs(dense_path, exist_ok=True)

    print("[→] Running COLMAP Stereo Matcher...")
    subprocess.run([
        "colmap", "stereo_matcher",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--DenseStereo.geom_consistency", "true",
    ], check=True)

    print("[→] Running COLMAP Stereo Fusion...")
    subprocess.run([
        "colmap", "stereo_fusion",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", os.path.join(dense_path, "fused.ply"),
    ], check=True)

    print("[✓] COLMAP pipeline completed.")
    print(f"[✓] Sparse results saved to: {sparse_path}")
    print(f"[✓] Dense results saved to: {dense_path}")
