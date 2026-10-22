# ============================================================
# Fast batch Matrix volume generation
#
# Matrix_vol.stl = IBM_vol.stl - IMS_vol.stl
#
# Processes only:
#   ../Control/<direct subfolder>/scaled/meshes
#   ../TTX/<direct subfolder>/scaled/meshes
#
# Does NOT recursively search nested meshes folders.
#
# Run with normal Python:
#   python batch_make_matrix_vol.py
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
import csv

import trimesh
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
ROOT_NAMES = ["Control", "TTX"]

SCALED_FOLDER_NAME = "scaled"
MESHES_FOLDER_NAME = "meshes"

IBM_NAME = "IBM_vol.stl"
IMS_NAME = "IMS_vol.stl"
OUT_NAME = "Matrix_vol.stl"

MIN_COMPONENT_VERTICES = 100

# If True, skip folders where Matrix_vol.stl already exists
SKIP_EXISTING = False

# On a network drive, try 4, 6, 8 and compare.
MAX_WORKERS = WORKERS

FAILURE_LOG_NAME = "make_matrix_vol_failures.csv"
SUMMARY_LOG_NAME = "make_matrix_vol_summary.csv"

# ============================================================
# PATH HELPERS
# ============================================================
def get_script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def discover_jobs(script_dir):
    roots = {
        root_name: (script_dir.parent / root_name).resolve()
        for root_name in ROOT_NAMES
    }

    jobs = []

    for root_name, root in roots.items():
        if not root.exists():
            continue

        direct_subfolders = sorted([
            p for p in root.iterdir()
            if p.is_dir()
        ])

        for folder in direct_subfolders:
            meshes_dir = folder / SCALED_FOLDER_NAME / MESHES_FOLDER_NAME

            if not meshes_dir.exists():
                continue

            ibm_path = meshes_dir / IBM_NAME
            ims_path = meshes_dir / IMS_NAME
            out_path = meshes_dir / OUT_NAME

            if not ibm_path.exists():
                continue

            if not ims_path.exists():
                continue

            if SKIP_EXISTING and out_path.exists():
                continue

            jobs.append({
                "root_name": root_name,
                "folder": str(folder),
                "meshes_dir": str(meshes_dir),
                "ibm_path": str(ibm_path),
                "ims_path": str(ims_path),
                "out_path": str(out_path),
            })

    return jobs, roots

# ============================================================
# MESH HELPERS
# ============================================================
def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Not a Trimesh object: {path}")

    return mesh

def clean_mesh(mesh):
    mesh = mesh.copy()

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

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

def subtract_meshes(mesh_a, mesh_b):
    """
    Return mesh_a - mesh_b using manifold boolean difference.
    """
    out = trimesh.boolean.difference(
        [mesh_a, mesh_b],
        engine="manifold",
        check_volume=False
    )

    if out is None:
        raise RuntimeError("Boolean difference returned None")

    if isinstance(out, list):
        if len(out) == 0:
            raise RuntimeError("Boolean difference returned empty list")
        out = out[0]

    if out.is_empty:
        raise RuntimeError("Boolean difference returned empty mesh")

    return clean_mesh(out)

def remove_small_components_by_vertices(mesh, min_vertices=100):
    """
    Split mesh into disconnected components and keep only those
    with at least `min_vertices` vertices.
    """
    parts = mesh.split(only_watertight=False)

    if len(parts) == 0:
        raise RuntimeError("Mesh split returned no components")

    kept = []
    removed = 0

    for part in parts:
        part = clean_mesh(part)
        n_vertices = len(part.vertices)

        if n_vertices >= min_vertices:
            kept.append(part)
        else:
            removed += 1

    if len(kept) == 0:
        raise RuntimeError(
            f"All components were removed by min_vertices={min_vertices}"
        )

    merged = trimesh.util.concatenate(kept)
    merged = clean_mesh(merged)

    return merged, len(parts), len(kept), removed

# ============================================================
# WORKER
# ============================================================
def process_one_folder(job):
    ibm_path = Path(job["ibm_path"])
    ims_path = Path(job["ims_path"])
    out_path = Path(job["out_path"])

    try:
        ibm = clean_mesh(load_mesh(ibm_path))
        ims = clean_mesh(load_mesh(ims_path))

        matrix = subtract_meshes(ibm, ims)

        matrix, n_parts, n_kept, n_removed = remove_small_components_by_vertices(
            matrix,
            min_vertices=MIN_COMPONENT_VERTICES
        )

        matrix.export(out_path)

        return {
            "status": "ok",
            "root_name": job["root_name"],
            "folder": job["folder"],
            "meshes_dir": job["meshes_dir"],
            "ibm_path": str(ibm_path),
            "ims_path": str(ims_path),
            "out_path": str(out_path),
            "n_parts": int(n_parts),
            "n_kept": int(n_kept),
            "n_removed": int(n_removed),
            "vertices": int(len(matrix.vertices)),
            "faces": int(len(matrix.faces)),
            "error": "",
            "traceback": "",
        }

    except Exception as e:
        failed = dict(job)
        failed.update({
            "status": "failed",
            "n_parts": "",
            "n_kept": "",
            "n_removed": "",
            "vertices": "",
            "faces": "",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        return failed

# ============================================================
# CSV LOGGING
# ============================================================
def write_csv(path, rows):
    if not rows:
        return

    fieldnames = [
        "status",
        "root_name",
        "folder",
        "meshes_dir",
        "ibm_path",
        "ims_path",
        "out_path",
        "n_parts",
        "n_kept",
        "n_removed",
        "vertices",
        "faces",
        "error",
        "traceback",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = get_script_dir()
    jobs, roots = discover_jobs(script_dir)

    print("Script dir:", script_dir)

    for root_name, root in roots.items():
        print(f"{root_name} root:", root)

    print("")
    print(f"Found folders to process: {len(jobs)}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"SKIP_EXISTING: {SKIP_EXISTING}")
    print(f"MIN_COMPONENT_VERTICES: {MIN_COMPONENT_VERTICES}")
    print("")

    if len(jobs) == 0:
        print("No folders with IBM_vol.stl and IMS_vol.stl were found.")
        return

    results = []
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
            desc="Making Matrix_vol",
            unit="folder",
            dynamic_ncols=True,
        ) as pbar:

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "status": "failed",
                        "root_name": "",
                        "folder": "",
                        "meshes_dir": "",
                        "ibm_path": "",
                        "ims_path": "",
                        "out_path": "",
                        "n_parts": "",
                        "n_kept": "",
                        "n_removed": "",
                        "vertices": "",
                        "faces": "",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

                results.append(result)

                if result["status"] == "ok":
                    ok_count += 1
                else:
                    failed_count += 1

                pbar.update(1)

    summary_log_path = script_dir / SUMMARY_LOG_NAME
    write_csv(summary_log_path, results)

    failure_rows = [
        row for row in results
        if row["status"] != "ok"
    ]

    failure_log_path = script_dir / FAILURE_LOG_NAME
    write_csv(failure_log_path, failure_rows)

    print("")
    print("Done.")
    print(f"Successful Matrix_vol.stl files: {ok_count}")
    print(f"Failed folders:                 {failed_count}")
    print(f"Summary log saved to:           {summary_log_path}")

    if failure_rows:
        print(f"Failure log saved to:           {failure_log_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()