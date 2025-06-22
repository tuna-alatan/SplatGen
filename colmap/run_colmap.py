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



    print("[✓] COLMAP pipeline completed.")
    print(f"[✓] Results saved to: {sparse_path}")
