# ============================================================
# Parallel mesh summary measurements
#
# Measures:
#   OMM_vol.stl          -> OMM_vol_um3
#   OMM_surf.stl         -> OMM_surf_um2
#   CM_surf.stl          -> CM_surf_um2
#   CM_surf_highk1.stl   -> CM_highk1_um2
#   CM_surf.stl +
#   IBM_surf.stl         -> IMM_surf_um2
#
# Output:
#   ../mesh_summary.csv
#
# Run with normal Python:
#   python batch_mesh_summary_parallel.py
# ============================================================

import os

# Limit internal thread storms when using process pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from project_paths import WORKERS, BLENDER_WORKERS
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback
import re
import csv

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
NOTEBOOK_DIR = Path(__file__).resolve().parent
ROOT = NOTEBOOK_DIR.parent

OUT_CSV = ROOT / "mesh_summary.csv"

MAX_WORKERS = WORKERS

SUMMARY_LOG_NAME = "mesh_summary_processing_log.csv"
FAILURE_LOG_NAME = "mesh_summary_failures.csv"

ROUND_DECIMALS = 3

print("Notebook dir:", NOTEBOOK_DIR)
print("Root dir:", ROOT)
print("Output CSV:", OUT_CSV)

# ============================================================
# HELPERS
# ============================================================
def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Not a Trimesh object: {path}")

    return mesh

def parse_metadata_from_folder(folder_path):
    """
    Example folder:
    .../Control/2022-10-08 Control 1

    Returns:
        Date, Condition, Mito
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    m = re.match(
        r"(\d{4}-\d{2}-\d{2})\s+([A-Za-z]+)\s+(\d+)$",
        folder_name,
    )

    if not m:
        raise ValueError(f"Could not parse metadata from folder name: {folder_name}")

    date_str = m.group(1)
    condition = m.group(2)
    mito = int(m.group(3))

    return date_str, condition, mito

def safe_volume(mesh_path):
    mesh = load_mesh(mesh_path)
    return round(float(mesh.volume), ROUND_DECIMALS)

def safe_area(mesh_path):
    mesh = load_mesh(mesh_path)
    return round(float(mesh.area), ROUND_DECIMALS)

def discover_jobs(root):
    meshes_dirs = sorted([
        p for p in root.rglob("meshes")
        if p.is_dir()
    ])

    jobs = []

    for meshes_dir in meshes_dirs:
        sample_dir = meshes_dir.parent.parent

        jobs.append({
            "meshes_dir": str(meshes_dir),
            "sample_dir": str(sample_dir),
        })

    return jobs

# ============================================================
# WORKER
# ============================================================
def process_one_folder(job):
    meshes_dir = Path(job["meshes_dir"])
    sample_dir = Path(job["sample_dir"])

    result = {
        "status": "ok",
        "sample_dir": str(sample_dir),
        "meshes_dir": str(meshes_dir),
        "Date": "",
        "Condition": "",
        "Mito": "",
        "OMM_vol_um3": np.nan,
        "OMM_surf_um2": np.nan,
        "CM_surf_um2": np.nan,
        "CM_highk1_um2": np.nan,
        "IMM_surf_um2": np.nan,
        "missing_files": "",
        "error": "",
        "traceback": "",
    }

    try:
        date_str, condition, mito = parse_metadata_from_folder(sample_dir)

        result["Date"] = date_str
        result["Condition"] = condition
        result["Mito"] = mito

        omm_vol_path = meshes_dir / "OMM_vol.stl"
        omm_surf_path = meshes_dir / "OMM_surf.stl"
        cm_surf_path = meshes_dir / "CM_surf.stl"
        cm_highk1_path = meshes_dir / "CM_surf_highk1.stl"
        ibm_surf_path = meshes_dir / "IBM_surf.stl"

        missing = []

        if omm_vol_path.exists():
            result["OMM_vol_um3"] = safe_volume(omm_vol_path)
        else:
            missing.append("OMM_vol.stl")

        if omm_surf_path.exists():
            result["OMM_surf_um2"] = safe_area(omm_surf_path)
        else:
            missing.append("OMM_surf.stl")

        if cm_surf_path.exists():
            result["CM_surf_um2"] = safe_area(cm_surf_path)
        else:
            missing.append("CM_surf.stl")

        if cm_highk1_path.exists():
            result["CM_highk1_um2"] = safe_area(cm_highk1_path)
        else:
            missing.append("CM_surf_highk1.stl")

        if ibm_surf_path.exists():
            ibm_surf_um2 = safe_area(ibm_surf_path)
        else:
            ibm_surf_um2 = np.nan
            missing.append("IBM_surf.stl")

        if np.isfinite(result["CM_surf_um2"]) and np.isfinite(ibm_surf_um2):
            result["IMM_surf_um2"] = round(
                result["CM_surf_um2"] + ibm_surf_um2,
                ROUND_DECIMALS,
            )

        result["missing_files"] = "; ".join(missing)

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result

# ============================================================
# CSV LOGGING
# ============================================================
def write_log_csv(path, rows):
    if not rows:
        return

    fieldnames = [
        "status",
        "sample_dir",
        "meshes_dir",
        "Date",
        "Condition",
        "Mito",
        "OMM_vol_um3",
        "OMM_surf_um2",
        "CM_surf_um2",
        "CM_highk1_um2",
        "IMM_surf_um2",
        "missing_files",
        "error",
        "traceback",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in fieldnames
            })

# ============================================================
# MAIN
# ============================================================
def main():
    jobs = discover_jobs(ROOT)

    print("Notebook dir:", NOTEBOOK_DIR)
    print("Root dir:", ROOT)
    print("Output CSV:", OUT_CSV)
    print(f"Found {len(jobs)} mesh folders.")
    print(f"Using up to {MAX_WORKERS} worker processes.")

    if not jobs:
        print("No mesh folders found.")
        return

    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_folder, job) for job in jobs]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing meshes",
        ):
            results.append(future.result())

    successful = [result for result in results if result["status"] == "ok"]
    failed = [result for result in results if result["status"] == "failed"]

    summary_columns = [
        "Date",
        "Condition",
        "Mito",
        "OMM_vol_um3",
        "OMM_surf_um2",
        "CM_surf_um2",
        "CM_highk1_um2",
        "IMM_surf_um2",
    ]

    summary_df = pd.DataFrame(successful, columns=summary_columns)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Date", "Condition", "Mito"])

    summary_df.to_csv(OUT_CSV, index=False)

    write_log_csv(NOTEBOOK_DIR / SUMMARY_LOG_NAME, results)
    write_log_csv(NOTEBOOK_DIR / FAILURE_LOG_NAME, failed)

    print(f"Completed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    mp.freeze_support()
    main()
