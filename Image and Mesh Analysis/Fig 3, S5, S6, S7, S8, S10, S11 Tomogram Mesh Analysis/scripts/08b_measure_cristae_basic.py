# ============================================================
# Parallel mesh summary measurements
#
# Measures:
#   OMM_vol.stl          -> OMM_vol_um3
#   CM_surf.stl          -> CM_surf_um2
#   CM_surf_highk1.stl   -> CM_highk1_um2
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
        "CM_surf_um2": np.nan,
        "CM_highk1_um2": np.nan,
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
        cm_surf_path = meshes_dir / "CM_surf.stl"
        cm_highk1_path = meshes_dir / "CM_surf_highk1.stl"

        missing = []

        if omm_vol_path.exists():
            result["OMM_vol_um3"] = safe_volume(omm_vol_path)
        else:
            missing.append("OMM_vol.stl")

        if cm_surf_path.exists():
            result["CM_surf_um2"] = safe_area(cm_surf_path)
        else:
            missing.append("CM_surf.stl")

        if cm_highk1_path.exists():
            result["CM_highk1_um2"] = safe_area(cm_highk1_path)
        else:
            missing.append("CM_surf_highk1.stl")

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
        "CM_surf_um2",
        "CM_highk1_um2",
        "missing_files",
        "error",
        "traceback",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in fieldnames
            })

# ============================================================
# MAIN
# ============================================================
def main():
    jobs = discover_jobs(ROOT)

    print("")
    print(f"Found {len(jobs)} meshes folders.")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"ROUND_DECIMALS: {ROUND_DECIMALS}")
    print("")

    if len(jobs) == 0:
        print("No meshes folders found.")
        return

    results = []
    rows = []

    ok_count = 0
    failed_count = 0

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=ctx,
    ) as executor:

        futures = [
            executor.submit(process_one_folder, job)
            for job in jobs
        ]

        with tqdm(
            total=len(futures),
            desc="Measuring meshes",
            unit="folder",
            dynamic_ncols=True,
        ) as pbar:

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "status": "failed",
                        "sample_dir": "",
                        "meshes_dir": "",
                        "Date": "",
                        "Condition": "",
                        "Mito": "",
                        "OMM_vol_um3": np.nan,
                        "CM_surf_um2": np.nan,
                        "CM_highk1_um2": np.nan,
                        "missing_files": "",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

                results.append(result)

                if result["status"] == "ok":
                    ok_count += 1

                    rows.append({
                        "Date": result["Date"],
                        "Condition": result["Condition"],
                        "Mito": result["Mito"],
                        "OMM_vol_um3": result["OMM_vol_um3"],
                        "CM_surf_um2": result["CM_surf_um2"],
                        "CM_highk1_um2": result["CM_highk1_um2"],
                    })

                else:
                    failed_count += 1

                pbar.update(1)

    df = pd.DataFrame(rows)

    if not df.empty:
        measure_cols = [
            "OMM_vol_um3",
            "CM_surf_um2",
            "CM_highk1_um2",
        ]

        for col in measure_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(ROUND_DECIMALS)

        df = df.sort_values(
            ["Condition", "Date", "Mito"]
        ).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)

    summary_log_path = NOTEBOOK_DIR / SUMMARY_LOG_NAME
    failure_log_path = NOTEBOOK_DIR / FAILURE_LOG_NAME

    write_log_csv(summary_log_path, results)

    failure_rows = [
        row for row in results
        if row["status"] != "ok"
    ]

    write_log_csv(failure_log_path, failure_rows)

    print("")
    print("Done.")
    print(f"Successful folders: {ok_count}")
    print(f"Failed folders:     {failed_count}")
    print("Combined output:", OUT_CSV)
    print("Summary log:", summary_log_path)

    if failure_rows:
        print("Failure log:", failure_log_path)

    print("")
    print(df)

if __name__ == "__main__":
    mp.freeze_support()
    main()
