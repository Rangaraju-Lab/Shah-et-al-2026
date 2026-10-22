# ============================================================
# Fast batch crop/intersect remeshed STL meshes with thickness box
#
# Input:
#   ../Control/<direct subfolder>/scaled/meshes/*_remeshed.stl
#   ../TTX/<direct subfolder>/scaled/meshes/*_remeshed.stl
#
# Output:
#   *_vol.stl
#
# Run with normal Python, not Blender:
#   python batch_make_vol_meshes.py
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

import numpy as np
import trimesh
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
ROOT_NAMES = ["Control", "TTX"]

SCALED_FOLDER_NAME = "scaled"
MESHES_FOLDER_NAME = "meshes"

# Only these objects will be processed
TARGET_STEMS = [
    "OMM",
    "IMS",
    "IBM",
    "ER",
    "Ribo",
]

THICK_AXIS = 0
THICK_MIN = 0.0
THICK_MAX = 0.12            # 60 slices * 0.002 µm
MARGIN = 0.01

IN_SUFFIX = "_remeshed"
OUT_SUFFIX = "_vol"

# If True, skips output if *_vol.stl already exists
SKIP_EXISTING = False

# On network drive, test 4, 6, 8
MAX_WORKERS = WORKERS

FAILURE_LOG_NAME = "make_vol_mesh_failures.csv"

# Vertex fusion tolerance equivalent to the 1e-6 µm test that fixed the mesh.
# trimesh.merge_vertices(digits_vertex=6) merges vertices after rounding
# vertex coordinates to 6 decimal places.
FUSE_VERTICES_DIGITS = 6

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

            for stem in TARGET_STEMS:
                in_path = meshes_dir / f"{stem}{IN_SUFFIX}.stl"
                out_path = meshes_dir / f"{stem}{OUT_SUFFIX}.stl"

                if not in_path.exists():
                    continue

                if SKIP_EXISTING and out_path.exists():
                    continue

                jobs.append({
                    "root_name": root_name,
                    "folder": str(folder),
                    "meshes_dir": str(meshes_dir),
                    "stem": stem,
                    "in_path": str(in_path),
                    "out_path": str(out_path),
                })

    return jobs, roots

# ============================================================
# MESH HELPERS
# ============================================================
def load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Not a Trimesh object: {mesh_path}")

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

def fuse_close_vertices(mesh, digits_vertex=6):
    """
    Fuse vertices that are extremely close together.

    This is the same cleanup that fixed the IMS_vol.stl test:
      - before: nonmanifold_edges = 3, is_volume = False
      - after vertex fusion at ~1e-6: nonmanifold_edges = 0, is_volume = True

    This does not add faces or fill holes.
    It only merges near-duplicate vertices and removes unreferenced vertices.
    """
    mesh = mesh.copy()

    try:
        mesh.merge_vertices(digits_vertex=digits_vertex)
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    return mesh

def make_crop_box(
    mesh,
    thick_axis=0,
    thick_min=0.0,
    thick_max=0.12,
    margin=0.01,
):
    bounds = mesh.bounds.copy()
    mins = bounds[0].copy()
    maxs = bounds[1].copy()

    # Expand non-thickness axes slightly
    for ax in range(3):
        if ax != thick_axis:
            mins[ax] -= margin
            maxs[ax] += margin

    # Restrict thickness axis
    mins[thick_axis] = thick_min
    maxs[thick_axis] = thick_max

    extents = maxs - mins

    if np.any(extents <= 0):
        raise ValueError(f"Invalid crop box extents: {extents}")

    center = (mins + maxs) / 2.0

    transform = np.eye(4, dtype=float)
    transform[:3, 3] = center

    box = trimesh.creation.box(extents=extents, transform=transform)
    return box

def intersect_with_box(mesh, box):
    out = trimesh.boolean.intersection(
        [mesh, box],
        engine="manifold",
        check_volume=False,
    )

    if out is None:
        raise RuntimeError("Intersection returned None")

    if isinstance(out, list):
        if len(out) == 0:
            raise RuntimeError("Intersection returned empty list")
        out = out[0]

    if out.is_empty:
        raise RuntimeError("Intersection returned empty mesh")

    return clean_mesh(out)

# ============================================================
# WORKER
# ============================================================
def process_one_mesh(job):
    in_path = Path(job["in_path"])
    out_path = Path(job["out_path"])

    try:
        mesh = load_mesh(in_path)
        mesh = clean_mesh(mesh)

        crop_box = make_crop_box(
            mesh,
            thick_axis=THICK_AXIS,
            thick_min=THICK_MIN,
            thick_max=THICK_MAX,
            margin=MARGIN,
        )

        out = intersect_with_box(mesh, crop_box)

        # --------------------------------------------------------
        # Added cleanup before saving:
        # fuse near-duplicate vertices in the saved *_vol.stl
        # --------------------------------------------------------
        out = fuse_close_vertices(
            out,
            digits_vertex=FUSE_VERTICES_DIGITS,
        )

        out.export(out_path)

        return {
            "status": "ok",
            "root_name": job["root_name"],
            "folder": job["folder"],
            "meshes_dir": job["meshes_dir"],
            "stem": job["stem"],
            "in_path": str(in_path),
            "out_path": str(out_path),
            "vertices": int(len(out.vertices)),
            "faces": int(len(out.faces)),
            "error": "",
            "traceback": "",
        }

    except Exception as e:
        failed = dict(job)
        failed["vertices"] = ""
        failed["faces"] = ""
        failed["error"] = str(e)
        failed["traceback"] = traceback.format_exc()

        return {
            "status": "failed",
            **failed,
        }

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
        "stem",
        "in_path",
        "out_path",
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
    print(f"Found remeshed meshes to process: {len(jobs)}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"SKIP_EXISTING: {SKIP_EXISTING}")
    print(f"THICK_AXIS: {THICK_AXIS}")
    print(f"THICK_MIN: {THICK_MIN}")
    print(f"THICK_MAX: {THICK_MAX}")
    print(f"MARGIN: {MARGIN}")
    print(f"FUSE_VERTICES_DIGITS: {FUSE_VERTICES_DIGITS}")
    print("")

    if len(jobs) == 0:
        print("No matching *_remeshed.stl files found.")
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
            executor.submit(process_one_mesh, job)
            for job in jobs
        ]

        with tqdm(
            total=len(futures),
            desc="Cropping/intersecting meshes",
            unit="mesh",
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
                        "stem": "",
                        "in_path": "",
                        "out_path": "",
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

    failure_rows = [
        row for row in results
        if row["status"] != "ok"
    ]

    failure_log_path = script_dir / FAILURE_LOG_NAME
    write_csv(failure_log_path, failure_rows)

    print("")
    print("Done.")
    print(f"Successful *_vol.stl meshes: {ok_count}")
    print(f"Failed meshes:               {failed_count}")

    if failure_rows:
        print(f"Failure log saved to: {failure_log_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()