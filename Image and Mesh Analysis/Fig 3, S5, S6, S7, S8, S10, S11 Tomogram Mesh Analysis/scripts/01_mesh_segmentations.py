# ============================================================
# Parallel mesh reconstruction from scaled uint8 TIFF stacks
# Works on:
#   ../Control/<direct subfolder>/scaled
#   ../TTX/<direct subfolder>/scaled
#
# Does NOT recursively process scaled/scaled or nested folders.
# ============================================================

import os

# Limit internal thread storms when using multiple processes
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
import csv

import numpy as np
import tifffile as tiff
from skimage import measure
import trimesh
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
ROOT_NAMES = ["Control", "TTX"]

VOXEL_SIZE = (0.002, 0.002, 0.002)   # µm, axis order = (z, y, x)
ISOLEVEL = 127
STEP_SIZE = 1
ALLOW_DEGENERATE = False

PAD_WIDTH = 1
EXTRUDE_SLICES_BELOW = 5
EXTRUDE_SLICES_ABOVE = 5

SCALED_FOLDER_NAME = "scaled"
MESHES_FOLDER_NAME = "meshes"

# On network drives, too many workers can slow things down.
# Try 4, 6, 8.
MAX_WORKERS = WORKERS

# If True, skips an STL if it already exists.
SKIP_EXISTING = False

FAILURE_LOG_NAME = "mesh_reconstruction_failures.csv"

# ============================================================
# PATH HELPERS
# ============================================================
def get_script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def discover_scaled_folders(root_name, root):
    """
    Find only:
        root / direct_subfolder / scaled

    Does not recursively search.
    Therefore, it will not process:
        root / direct_subfolder / scaled / scaled
    """
    jobs = []

    if not root.exists():
        return jobs

    direct_subfolders = sorted([
        p for p in root.iterdir()
        if p.is_dir()
    ])

    for folder in direct_subfolders:
        scaled_dir = folder / SCALED_FOLDER_NAME

        if scaled_dir.exists() and scaled_dir.is_dir():
            tif_files = sorted([
                p for p in scaled_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
            ])

            if len(tif_files) > 0:
                jobs.append({
                    "root_name": root_name,
                    "parent_folder": str(folder),
                    "scaled_dir": str(scaled_dir),
                })

    return jobs

# ============================================================
# IMAGE / MESH HELPERS
# ============================================================
def load_stack(path):
    arr = tiff.imread(path)
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3D stack, got shape {arr.shape}")

    return arr

def is_8bit_tif(arr):
    return arr.dtype == np.uint8

def extrude_by_copying_end_slices(mask_u8, slices_below=5, slices_above=5):
    """
    Literally extrude in z by copying:
    - the first slice `slices_below` times before the stack
    - the last slice `slices_above` times after the stack

    Returns a new uint8 stack with larger z dimension.
    """
    if mask_u8.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {mask_u8.shape}")

    first_slice = mask_u8[0:1]   # shape (1, y, x)
    last_slice = mask_u8[-1:]    # shape (1, y, x)

    below = np.repeat(first_slice, slices_below, axis=0)
    above = np.repeat(last_slice, slices_above, axis=0)

    extruded = np.concatenate([below, mask_u8, above], axis=0)
    return extruded

def pad_with_black(arr, pad_width=1):
    """
    Pad a 3D volume with black voxels (0) on all sides.
    Axis order assumed to be (z, y, x).
    """
    return np.pad(
        arr,
        pad_width=((pad_width, pad_width),
                   (pad_width, pad_width),
                   (pad_width, pad_width)),
        mode="constant",
        constant_values=0
    )

def clean_mesh_faces(mesh):
    mesh.remove_unreferenced_vertices()

    try:
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    return mesh

def stack_to_mesh(
    mask_u8,
    voxel_size=(0.002, 0.002, 0.002),
    level=127,
    pad_width=1,
    extrude_below=5,
    extrude_above=5,
):
    """
    Build mesh from a uint8 segmentation volume.

    Pipeline:
    1. Copy first slice below and last slice above
    2. Pad with black border
    3. Marching cubes
    4. Shift vertices so original stack remains aligned in its original coordinates

    Assumes axis order is (z, y, x).
    """
    if mask_u8.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {mask_u8.shape}")

    if np.max(mask_u8) == 0:
        raise ValueError("Empty volume; cannot reconstruct mesh.")

    extruded = extrude_by_copying_end_slices(
        mask_u8,
        slices_below=extrude_below,
        slices_above=extrude_above
    )

    if np.max(extruded) <= level:
        raise ValueError("No voxels above marching cubes level after extrusion.")

    padded = pad_with_black(extruded, pad_width=pad_width)

    verts, faces, normals, values = measure.marching_cubes(
        volume=padded,
        level=level,
        spacing=voxel_size,
        step_size=STEP_SIZE,
        allow_degenerate=ALLOW_DEGENERATE,
        method="lewiner",
    )

    # Shift back by padding AND by the added slices below
    shift = np.array([
        (pad_width + extrude_below) * voxel_size[0],
        pad_width * voxel_size[1],
        pad_width * voxel_size[2],
    ], dtype=float)

    verts = verts - shift

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh = clean_mesh_faces(mesh)

    return mesh, extruded.shape, padded.shape

# ============================================================
# WORKER
# ============================================================
def process_scaled_folder(job):
    """
    Process one scaled folder.
    No printing inside worker.
    """
    root_name = job["root_name"]
    parent_folder = Path(job["parent_folder"])
    scaled_dir = Path(job["scaled_dir"])

    saved_count = 0
    skipped_count = 0
    failed_files = []

    try:
        meshes_dir = scaled_dir / MESHES_FOLDER_NAME
        meshes_dir.mkdir(exist_ok=True)

        tif_files = sorted([
            p for p in scaled_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
        ])

        for tif_path in tif_files:
            try:
                out_path = meshes_dir / f"{tif_path.stem}.stl"

                if SKIP_EXISTING and out_path.exists():
                    skipped_count += 1
                    continue

                arr = load_stack(tif_path)

                if not is_8bit_tif(arr):
                    skipped_count += 1
                    continue

                if np.max(arr) == 0:
                    skipped_count += 1
                    continue

                mesh, extruded_shape, padded_shape = stack_to_mesh(
                    arr,
                    voxel_size=VOXEL_SIZE,
                    level=ISOLEVEL,
                    pad_width=PAD_WIDTH,
                    extrude_below=EXTRUDE_SLICES_BELOW,
                    extrude_above=EXTRUDE_SLICES_ABOVE,
                )

                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    skipped_count += 1
                    continue

                mesh.export(out_path)
                saved_count += 1

            except Exception as e:
                failed_files.append({
                    "root_name": root_name,
                    "parent_folder": str(parent_folder),
                    "scaled_dir": str(scaled_dir),
                    "file": str(tif_path),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        return {
            "status": "ok",
            "root_name": root_name,
            "parent_folder": str(parent_folder),
            "scaled_dir": str(scaled_dir),
            "saved_count": saved_count,
            "skipped_count": skipped_count,
            "failed_count": len(failed_files),
            "failed_files": failed_files,
        }

    except Exception as e:
        failed_files.append({
            "root_name": root_name,
            "parent_folder": str(parent_folder),
            "scaled_dir": str(scaled_dir),
            "file": str(scaled_dir),
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

        return {
            "status": "failed",
            "root_name": root_name,
            "parent_folder": str(parent_folder),
            "scaled_dir": str(scaled_dir),
            "saved_count": saved_count,
            "skipped_count": skipped_count,
            "failed_count": len(failed_files),
            "failed_files": failed_files,
        }

# ============================================================
# FAILURE LOG
# ============================================================
def write_failure_log(path, failed_files):
    if not failed_files:
        return

    fieldnames = [
        "root_name",
        "parent_folder",
        "scaled_dir",
        "file",
        "error",
        "traceback",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in failed_files:
            writer.writerow(row)

# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = get_script_dir()

    roots = {
        root_name: (script_dir.parent / root_name).resolve()
        for root_name in ROOT_NAMES
    }

    jobs = []

    for root_name, root in roots.items():
        root_jobs = discover_scaled_folders(root_name, root)
        jobs.extend(root_jobs)

    print("Script dir:", script_dir)
    for root_name, root in roots.items():
        print(f"{root_name} root:", root)

    print("")
    print(f"Total scaled folders to process: {len(jobs)}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"SKIP_EXISTING: {SKIP_EXISTING}")
    print("")

    if len(jobs) == 0:
        print("No scaled folders with TIFF files were found.")
        return

    n_ok = 0
    n_failed_folders = 0
    total_saved = 0
    total_skipped = 0
    total_failed_files = 0
    all_failed_files = []

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=ctx,
    ) as executor:

        futures = [executor.submit(process_scaled_folder, job) for job in jobs]

        with tqdm(
            total=len(futures),
            desc="Processing scaled folders",
            unit="folder",
            dynamic_ncols=True,
        ) as pbar:

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "status": "failed",
                        "root_name": "unknown",
                        "parent_folder": "unknown",
                        "scaled_dir": "unknown",
                        "saved_count": 0,
                        "skipped_count": 0,
                        "failed_count": 1,
                        "failed_files": [{
                            "root_name": "unknown",
                            "parent_folder": "unknown",
                            "scaled_dir": "unknown",
                            "file": "unknown",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }],
                    }

                if result["status"] == "ok":
                    n_ok += 1
                else:
                    n_failed_folders += 1

                total_saved += result["saved_count"]
                total_skipped += result["skipped_count"]
                total_failed_files += result["failed_count"]
                all_failed_files.extend(result["failed_files"])

                pbar.update(1)

    failure_log_path = script_dir / FAILURE_LOG_NAME
    write_failure_log(failure_log_path, all_failed_files)

    print("")
    print("Done.")
    print(f"Scaled folders processed successfully: {n_ok}")
    print(f"Scaled folders failed:                 {n_failed_folders}")
    print(f"Meshes saved:                          {total_saved}")
    print(f"TIFF files skipped:                    {total_skipped}")
    print(f"TIFF/file failures:                    {total_failed_files}")

    if all_failed_files:
        print(f"Failure log saved to: {failure_log_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()