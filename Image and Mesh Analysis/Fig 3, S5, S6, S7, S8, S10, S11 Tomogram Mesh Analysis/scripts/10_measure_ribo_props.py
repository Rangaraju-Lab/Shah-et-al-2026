# ============================================================
# Parallel Ribo mesh summary measurements
#
# Measures:
#   Ribo_vol.stl
#
# For each Control / TTX mesh folder:
#   - If Ribo_vol.stl is missing:
#       Ribo_vol_nm3 = 0
#       Retained_Ribo_components = 0
#       Smallest_Ribo_vol_nm3 = 0
#
#   - If Ribo_vol.stl exists:
#       Ribo_vol_nm3 = total Ribo_vol volume in nm3
#       Split into loose / connected components using corrected STL logic
#       Retain components > 2000 nm3
#       Retained_Ribo_components = number retained
#       Smallest_Ribo_vol_nm3 = smallest retained component volume in nm3
#
# After all rows are measured:
#   - Average Smallest_Ribo_vol_nm3 across rows, excluding zeros
#   - Average_Smallest_Ribo_vol_nm3 = that average value
#   - n_ribosomes = Ribo_vol_nm3 / Average_Smallest_Ribo_vol_nm3
#
# Output:
#   ../Ribo_mesh_summary.csv
#
# Run with normal Python:
#   python batch_Ribo_mesh_summary_parallel.py
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
import warnings

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
NOTEBOOK_DIR = Path(__file__).resolve().parent
ROOT = NOTEBOOK_DIR.parent

ROOT_NAMES = ["Control", "TTX"]

OUT_CSV = ROOT / "Ribo_mesh_summary.csv"

MAX_WORKERS = WORKERS

SUMMARY_LOG_NAME = "Ribo_mesh_summary_processing_log.csv"
FAILURE_LOG_NAME = "Ribo_mesh_summary_failures.csv"

ROUND_DECIMALS = 3

# Mesh coordinates are assumed to be in um.
# 1 um3 = 1e9 nm3.
UM3_TO_NM3 = 1_000_000_000.0

MIN_RIBO_COMPONENT_VOL_NM3 = 2000.0
MIN_RIBO_COMPONENT_VOL_UM3 = MIN_RIBO_COMPONENT_VOL_NM3 / UM3_TO_NM3

print("Notebook dir:", NOTEBOOK_DIR)
print("Root dir:", ROOT)
print("Output CSV:", OUT_CSV)

# ============================================================
# HELPERS
# ============================================================
def load_mesh(path):
    """
    Load Ribo STL for connected-component splitting.

    Correct STL logic:
      - process=True
      - merge duplicate vertices
      - remove duplicate / degenerate faces if possible
      - remove unreferenced vertices

    This prevents STL triangle-duplicated vertices from causing split()
    to break the mesh into individual faces.
    """
    path = Path(path)

    mesh = trimesh.load(path, force="mesh", process=True)

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Not a Trimesh object: {path}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no vertices/faces: {path}")

    mesh.merge_vertices()

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

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

def mesh_volume_nm3(mesh):
    """
    Same basic volume logic as OMM volume measurement:
        abs(float(mesh.volume)) converted from um3 to nm3.

    Invalid or non-finite total volume returns 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        try:
            vol_um3 = abs(float(mesh.volume))
        except Exception:
            return 0.0

    if not np.isfinite(vol_um3):
        return 0.0

    vol_nm3 = vol_um3 * UM3_TO_NM3
    return round(vol_nm3, ROUND_DECIMALS)

def component_volume_nm3(mesh):
    """
    Component volume using mesh.volume.

    Returns None for invalid, NaN, infinite, or zero volume.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        try:
            vol_um3 = abs(float(mesh.volume))
        except Exception:
            return None

    if not np.isfinite(vol_um3):
        return None

    if vol_um3 <= 0:
        return None

    return vol_um3 * UM3_TO_NM3

def measure_ribo_components(ribo_mesh):
    """
    Split Ribo_vol into connected / loose components.

    Correct splitting logic assumes ribo_mesh was loaded with:
      process=True
      merge_vertices()

    Retains components with volume > 2000 nm3.

    Returns:
        total_ribo_vol_nm3
        retained_component_count
        smallest_retained_component_vol_nm3
    """
    total_ribo_vol_nm3 = mesh_volume_nm3(ribo_mesh)

    components = ribo_mesh.split(only_watertight=False)

    retained_volumes_nm3 = []

    for comp in components:
        comp_vol_nm3 = component_volume_nm3(comp)

        if comp_vol_nm3 is None:
            continue

        if comp_vol_nm3 > MIN_RIBO_COMPONENT_VOL_NM3:
            retained_volumes_nm3.append(comp_vol_nm3)

    retained_component_count = int(len(retained_volumes_nm3))

    if retained_component_count > 0:
        smallest_retained_component_vol_nm3 = round(
            float(np.min(retained_volumes_nm3)),
            ROUND_DECIMALS
        )
    else:
        smallest_retained_component_vol_nm3 = 0.0

    return (
        round(total_ribo_vol_nm3, ROUND_DECIMALS),
        retained_component_count,
        smallest_retained_component_vol_nm3,
    )

def discover_jobs(root):
    jobs = []

    for root_name in ROOT_NAMES:
        condition_root = root / root_name

        if not condition_root.exists():
            continue

        sample_dirs = sorted([
            p for p in condition_root.iterdir()
            if p.is_dir()
        ])

        for sample_dir in sample_dirs:
            meshes_dir = sample_dir / "scaled" / "meshes"

            if not meshes_dir.exists():
                continue

            jobs.append({
                "condition_root": root_name,
                "sample_dir": str(sample_dir),
                "meshes_dir": str(meshes_dir),
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
        "condition_root": job["condition_root"],
        "sample_dir": str(sample_dir),
        "meshes_dir": str(meshes_dir),
        "Date": "",
        "Condition": "",
        "Mito": "",
        "Ribo_vol_nm3": 0.0,
        "Retained_Ribo_components": 0,
        "Smallest_Ribo_vol_nm3": 0.0,
        "missing_files": "",
        "error": "",
        "traceback": "",
    }

    try:
        date_str, condition, mito = parse_metadata_from_folder(sample_dir)

        result["Date"] = date_str
        result["Condition"] = condition
        result["Mito"] = mito

        ribo_vol_path = meshes_dir / "Ribo_vol.stl"

        if not ribo_vol_path.exists():
            result["missing_files"] = "Ribo_vol.stl"
            return result

        ribo_mesh = load_mesh(ribo_vol_path)

        (
            ribo_vol_nm3,
            retained_component_count,
            smallest_ribo_vol_nm3,
        ) = measure_ribo_components(ribo_mesh)

        result["Ribo_vol_nm3"] = ribo_vol_nm3
        result["Retained_Ribo_components"] = retained_component_count
        result["Smallest_Ribo_vol_nm3"] = smallest_ribo_vol_nm3

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
        "condition_root",
        "sample_dir",
        "meshes_dir",
        "Date",
        "Condition",
        "Mito",
        "Ribo_vol_nm3",
        "Retained_Ribo_components",
        "Smallest_Ribo_vol_nm3",
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
    print(f"MIN_RIBO_COMPONENT_VOL_NM3: {MIN_RIBO_COMPONENT_VOL_NM3}")
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
            desc="Measuring Ribo meshes",
            unit="folder",
            dynamic_ncols=True,
        ) as pbar:

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "status": "failed",
                        "condition_root": "",
                        "sample_dir": "",
                        "meshes_dir": "",
                        "Date": "",
                        "Condition": "",
                        "Mito": "",
                        "Ribo_vol_nm3": 0.0,
                        "Retained_Ribo_components": 0,
                        "Smallest_Ribo_vol_nm3": 0.0,
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
                        "Ribo_vol_nm3": result["Ribo_vol_nm3"],
                        "Retained_Ribo_components": result["Retained_Ribo_components"],
                        "Smallest_Ribo_vol_nm3": result["Smallest_Ribo_vol_nm3"],
                    })

                else:
                    failed_count += 1

                pbar.update(1)

    df = pd.DataFrame(rows)

    if not df.empty:
        measure_cols = [
            "Ribo_vol_nm3",
            "Smallest_Ribo_vol_nm3",
        ]

        for col in measure_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(ROUND_DECIMALS)

        df["Retained_Ribo_components"] = pd.to_numeric(
            df["Retained_Ribo_components"],
            errors="coerce"
        ).fillna(0).astype(int)

        nonzero_smallest = df.loc[
            df["Smallest_Ribo_vol_nm3"] > 0,
            "Smallest_Ribo_vol_nm3"
        ]

        if len(nonzero_smallest) > 0:
            average_smallest_ribo_vol_nm3 = float(nonzero_smallest.mean())
        else:
            average_smallest_ribo_vol_nm3 = 0.0

        df["Average_Smallest_Ribo_vol_nm3"] = round(
            average_smallest_ribo_vol_nm3,
            ROUND_DECIMALS
        )

        if average_smallest_ribo_vol_nm3 > 0:
            df["n_ribosomes"] = (
                df["Ribo_vol_nm3"] / average_smallest_ribo_vol_nm3
            ).round(0).astype(int)
        else:
            df["n_ribosomes"] = 0

        df = df[
            [
                "Date",
                "Condition",
                "Mito",
                "Ribo_vol_nm3",
                "Retained_Ribo_components",
                "Smallest_Ribo_vol_nm3",
                "Average_Smallest_Ribo_vol_nm3",
                "n_ribosomes",
            ]
        ]

        df = df.sort_values(
            ["Condition", "Date", "Mito"]
        ).reset_index(drop=True)

    else:
        df = pd.DataFrame(columns=[
            "Date",
            "Condition",
            "Mito",
            "Ribo_vol_nm3",
            "Retained_Ribo_components",
            "Smallest_Ribo_vol_nm3",
            "Average_Smallest_Ribo_vol_nm3",
            "n_ribosomes",
        ])

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
