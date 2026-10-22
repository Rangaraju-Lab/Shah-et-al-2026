# ============================================================
# Parallel CJ detection from IBM_surf.stl and CM_surf.stl
#
# For each meshes folder:
#   IBM_surf.stl + CM_surf.stl -> CJ.csv
#
# Combined output:
#   ../CJ_all.csv
#
# Run with normal Python:
#   python batch_detect_CJ_parallel.py
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
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback
import itertools
import re
import csv

import numpy as np
import pandas as pd
import trimesh
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
ROOT_NAMES = ["Control", "TTX"]

SCALED_FOLDER_NAME = "scaled"
MESHES_FOLDER_NAME = "meshes"

IBM_NAME = "IBM_surf.stl"
CM_NAME = "CM_surf.stl"

PER_FOLDER_OUT = "CJ.csv"

MIN_COMPONENT_SIZE = 15
MIN_PATH_LENGTH = 10
DISTANCE_THRESHOLD = 0.01
MIN_FACES = 500

ROUND_DECIMALS = 9

MAX_WORKERS = WORKERS

SUMMARY_LOG_NAME = "CJ_summary.csv"
FAILURE_LOG_NAME = "CJ_failures.csv"

# ============================================================
# PATH HELPERS
# ============================================================
def get_script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def discover_jobs(script_dir):
    root = script_dir.parent
    roots = {
        root_name: (root / root_name).resolve()
        for root_name in ROOT_NAMES
    }

    jobs = []

    for root_name, condition_root in roots.items():
        if not condition_root.exists():
            continue

        # Only direct subfolders under Control/TTX
        sample_dirs = sorted([
            p for p in condition_root.iterdir()
            if p.is_dir()
        ])

        for sample_dir in sample_dirs:
            meshes_dir = sample_dir / SCALED_FOLDER_NAME / MESHES_FOLDER_NAME

            if not meshes_dir.exists():
                continue

            ibm_path = meshes_dir / IBM_NAME
            cm_path = meshes_dir / CM_NAME

            if not ibm_path.exists():
                continue

            if not cm_path.exists():
                continue

            jobs.append({
                "root_name": root_name,
                "sample_dir": str(sample_dir),
                "meshes_dir": str(meshes_dir),
                "ibm_path": str(ibm_path),
                "cm_path": str(cm_path),
                "out_path": str(meshes_dir / PER_FOLDER_OUT),
            })

    return jobs, roots

# ============================================================
# HELPERS
# ============================================================
def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Not a Trimesh object: {path}")

    return mesh

def round_point(p, decimals=9):
    return tuple(np.round(np.asarray(p, dtype=float), decimals=decimals))

def get_edges_from_mesh(mesh, decimals=9):
    """
    Build a set of undirected edges represented by rounded endpoint coordinates.
    """
    edges = set()
    verts = np.asarray(mesh.vertices, dtype=float)

    for face in mesh.faces:
        pts = verts[face]

        for i in range(3):
            p1 = round_point(pts[i], decimals=decimals)
            p2 = round_point(pts[(i + 1) % 3], decimals=decimals)
            edge = tuple(sorted((p1, p2)))
            edges.add(edge)

    return edges

def group_edges_into_components(edges):
    """
    Group connected shared edges into edge-components.
    """
    v2e = defaultdict(set)

    for e in edges:
        for v in e:
            v2e[v].add(e)

    visited = set()
    comps = []

    for e in edges:
        if e in visited:
            continue

        q = deque([e])
        comp = set()

        while q:
            cur = q.popleft()

            if cur in visited:
                continue

            visited.add(cur)
            comp.add(cur)

            q.extend(v2e[cur[0]] | v2e[cur[1]])

        comps.append(comp)

    return comps

def split_component_by_geometry(
    edges,
    min_path_length=5,
    distance_threshold=0.005,
):
    """
    Split an edge component if two graph-distant vertices are physically close.

    This version uses cKDTree to find only physically-close vertex pairs,
    instead of checking every possible pair.
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    nodes = list(G.nodes)

    if len(nodes) < 2:
        return [edges]

    pts = np.asarray(nodes, dtype=float)
    tree = cKDTree(pts)

    close_pairs = tree.query_pairs(distance_threshold)

    for i, j in close_pairs:
        v1 = nodes[i]
        v2 = nodes[j]

        if G.has_edge(v1, v2):
            continue

        try:
            path = nx.shortest_path(G, v1, v2)
        except nx.NetworkXNoPath:
            continue

        if len(path) - 1 >= min_path_length:
            path_edges = {
                tuple(sorted((path[k], path[k + 1])))
                for k in range(len(path) - 1)
            }

            # Add closure edge for consistency with previous logic.
            path_edges.add(tuple(sorted((v1, v2))))

            return [
                path_edges,
                edges - path_edges,
            ]

    return [edges]

def shared_edge_junction_centroids(
    mesh1,
    mesh2,
    min_component_size=15,
    min_path_length=10,
    distance_threshold=0.01,
    decimals=9,
):
    """
    Return shared-edge junction centroids between two meshes.
    """
    edges1 = get_edges_from_mesh(mesh1, decimals=decimals)
    edges2 = get_edges_from_mesh(mesh2, decimals=decimals)

    shared_edges = edges1 & edges2

    if not shared_edges:
        return []

    valid_components = []

    for comp in group_edges_into_components(shared_edges):
        if len(comp) < min_component_size:
            continue

        subcomponents = split_component_by_geometry(
            comp,
            min_path_length=min_path_length,
            distance_threshold=distance_threshold,
        )

        for sub in subcomponents:
            if len(sub) >= min_component_size:
                valid_components.append(sub)

    if not valid_components:
        return []

    centroids = []

    for comp in valid_components:
        comp_vertices = np.asarray(
            list({v for e in comp for v in e}),
            dtype=float,
        )

        centroids.append(np.mean(comp_vertices, axis=0))

    return centroids

def refine_junctions_by_cm_component(cm_mesh, centroids, min_faces=500):
    """
    Assign each centroid to the nearest connected component of cm_mesh and
    keep only those whose matched component has at least min_faces.
    """
    centroids = np.asarray(centroids, dtype=float)

    if len(centroids) == 0 or len(cm_mesh.faces) == 0:
        return np.empty((0, 3), dtype=float)

    comps = cm_mesh.split(only_watertight=False)

    if len(comps) == 0:
        return np.empty((0, 3), dtype=float)

    trees = []

    for comp in comps:
        if len(comp.vertices) == 0:
            continue

        trees.append((comp, cKDTree(comp.vertices)))

    if len(trees) == 0:
        return np.empty((0, 3), dtype=float)

    refined = []

    for pt in centroids:
        comp = min(
            trees,
            key=lambda ct: ct[1].query(pt)[0],
        )[0]

        if len(comp.faces) >= min_faces:
            refined.append(pt)

    if len(refined) == 0:
        return np.empty((0, 3), dtype=float)

    return np.asarray(refined, dtype=float)

def parse_metadata_from_folder(folder_path):
    """
    Example:
    .../Control/2022-10-08 Control 1
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    m = re.match(
        r"(\d{4}-\d{2}-\d{2})\s+([A-Za-z]+)\s+(\d+)$",
        folder_name,
    )

    if not m:
        raise ValueError(f"Could not parse metadata from folder name: {folder_name}")

    return m.group(1), m.group(2), int(m.group(3))

# ============================================================
# WORKER
# ============================================================
def process_one_folder(job):
    sample_dir = Path(job["sample_dir"])
    meshes_dir = Path(job["meshes_dir"])
    ibm_path = Path(job["ibm_path"])
    cm_path = Path(job["cm_path"])
    out_path = Path(job["out_path"])

    result = {
        "status": "ok",
        "root_name": job["root_name"],
        "sample_dir": str(sample_dir),
        "meshes_dir": str(meshes_dir),
        "junctions": 0,
        "cj_csv_saved": False,
        "error": "",
        "traceback": "",
        "rows": [],
    }

    try:
        date_str, condition, mito = parse_metadata_from_folder(sample_dir)

        ibm_mesh = load_mesh(ibm_path)
        cm_mesh = load_mesh(cm_path)

        centroids = shared_edge_junction_centroids(
            cm_mesh,
            ibm_mesh,
            min_component_size=MIN_COMPONENT_SIZE,
            min_path_length=MIN_PATH_LENGTH,
            distance_threshold=DISTANCE_THRESHOLD,
            decimals=ROUND_DECIMALS,
        )

        refined = refine_junctions_by_cm_component(
            cm_mesh,
            np.asarray(centroids, dtype=float),
            min_faces=MIN_FACES,
        )

        if len(refined) == 0:
            cj_df = pd.DataFrame(columns=["x", "y", "z"])
        else:
            cj_df = pd.DataFrame(refined, columns=["x", "y", "z"])

        cj_df.to_csv(out_path, index=False)

        result["junctions"] = int(len(cj_df))
        result["cj_csv_saved"] = True

        rows = []

        for i, row in cj_df.iterrows():
            rows.append({
                "Date": date_str,
                "Condition": condition,
                "Mito": mito,
                "Folder": str(sample_dir),
                "Junction_ID": int(i + 1),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            })

        result["rows"] = rows

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["rows"] = []
        return result

# ============================================================
# CSV LOGGING
# ============================================================
def write_summary_csv(path, rows):
    if not rows:
        return

    fieldnames = [
        "status",
        "root_name",
        "sample_dir",
        "meshes_dir",
        "junctions",
        "cj_csv_saved",
        "error",
        "traceback",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            clean = {
                key: row.get(key, "")
                for key in fieldnames
            }

            writer.writerow(clean)

# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = get_script_dir()
    root = script_dir.parent

    combined_out = root / "CJ_all.csv"

    jobs, roots = discover_jobs(script_dir)

    print("Script dir:", script_dir)
    print("Root dir:", root)

    for root_name, root_path in roots.items():
        print(f"{root_name} root:", root_path)

    print("")
    print(f"Found folders to process: {len(jobs)}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"MIN_COMPONENT_SIZE: {MIN_COMPONENT_SIZE}")
    print(f"MIN_PATH_LENGTH: {MIN_PATH_LENGTH}")
    print(f"DISTANCE_THRESHOLD: {DISTANCE_THRESHOLD}")
    print(f"MIN_FACES: {MIN_FACES}")
    print(f"ROUND_DECIMALS: {ROUND_DECIMALS}")
    print("")

    if len(jobs) == 0:
        print("No folders found.")
        return

    results = []
    all_rows = []

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
            desc="Detecting CJs",
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
                        "sample_dir": "",
                        "meshes_dir": "",
                        "junctions": 0,
                        "cj_csv_saved": False,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "rows": [],
                    }

                results.append(result)
                all_rows.extend(result.get("rows", []))

                if result["status"] == "ok":
                    ok_count += 1
                else:
                    failed_count += 1

                pbar.update(1)

    # --------------------------------------------------------
    # Save combined table
    # --------------------------------------------------------
    all_df = pd.DataFrame(all_rows)

    if not all_df.empty:
        all_df = all_df.sort_values(
            ["Condition", "Date", "Mito", "Junction_ID"]
        ).reset_index(drop=True)

    all_df.to_csv(combined_out, index=False)

    # --------------------------------------------------------
    # Save logs
    # --------------------------------------------------------
    summary_path = script_dir / SUMMARY_LOG_NAME
    failure_path = script_dir / FAILURE_LOG_NAME

    write_summary_csv(summary_path, results)

    failure_rows = [
        row for row in results
        if row["status"] != "ok"
    ]

    write_summary_csv(failure_path, failure_rows)

    print("")
    print("Done.")
    print(f"Successful folders: {ok_count}")
    print(f"Failed folders:     {failed_count}")
    print("Combined output:", combined_out)
    print("Total junctions:", len(all_df))
    print("Summary log:", summary_path)

    if failure_rows:
        print("Failure log:", failure_path)

if __name__ == "__main__":
    mp.freeze_support()
    main()
