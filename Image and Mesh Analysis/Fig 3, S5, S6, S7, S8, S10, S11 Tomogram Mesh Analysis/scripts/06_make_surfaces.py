# ============================================================
# Fast batch with safe vertex-order curvature mapping:
#
# 1) OMM_vol / ER_vol / IBM_vol / Matrix_vol / smooth_k1
#       -> OMM_surf / ER_surf / ERMCS_surf / nonERMCS_surf
#       -> IBM_surf / CM_surf / CM_surf_smooth_k1.pkl
#
# 2) CM_surf + CM_surf_smooth_k1.pkl
#       -> CM_surf_highk1.stl
#
# Processes only:
#   ../Control/<direct subfolder>/scaled/meshes
#   ../TTX/<direct subfolder>/scaled/meshes
#
# Run with normal Python:
#   python batch_make_surfs_and_highk1.py
#
# IMPORTANT:
#   smooth_k1 should be the updated payload from Blender:
#       {
#           "vertices": Blender vertex coordinates,
#           "smooth_k1": per-vertex K1 values
#       }
#
# This avoids assuming Blender and trimesh have identical STL vertex ordering.
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
import pickle

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
ROOT_NAMES = ["Control", "TTX"]

SCALED_FOLDER_NAME = "scaled"
MESHES_FOLDER_NAME = "meshes"

THICK_AXIS = 0
THICK_MIN = 0.0
THICK_MAX = 0.12
XY_MARGIN = 0.01

BOX_FACE_THRESHOLD = 0.005
IBM_VERTEX_THRESHOLD = 0.008

# 20 nm = 0.02 um, assuming mesh coordinates are in um.
ERMCS_DISTANCE_UM = 0.02

# Tolerance for vertex remapping after STL import/export.
# If STL precision causes failures, try 1e-5.
REMAP_TOL = 1e-6

CURV_THRESHOLD = 70.0

SKIP_EXISTING_SURFS = False
SKIP_EXISTING_HIGHK1 = False

MAX_WORKERS = WORKERS

SUMMARY_LOG_NAME = "make_surfs_and_highk1_summary.csv"
FAILURE_LOG_NAME = "make_surfs_and_highk1_failures.csv"

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

            omm_vol_path = meshes_dir / "OMM_vol.stl"
            ibm_vol_path = meshes_dir / "IBM_vol.stl"
            matrix_vol_path = meshes_dir / "Matrix_vol.stl"
            smooth_path = meshes_dir / "smooth_k1"

            # Minimum needed to do the combined workflow.
            # ER_vol is optional; ER outputs are made when ER_vol exists.
            if not omm_vol_path.exists():
                continue

            if not ibm_vol_path.exists():
                continue

            if not matrix_vol_path.exists():
                continue

            if not smooth_path.exists():
                continue

            jobs.append({
                "root_name": root_name,
                "folder": str(folder),
                "meshes_dir": str(meshes_dir),
            })

    return jobs, roots

# ============================================================
# BASIC IO
# ============================================================
def load_mesh(path):
    """
    Load STL without trimesh processing.

    process=False is important because processing can merge/reorder vertices,
    which breaks per-vertex curvature alignment.
    """
    mesh = trimesh.load(path, force="mesh", process=False)

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Not a Trimesh object: {path}")

    return mesh

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# ============================================================
# WARNING HELPER
# ============================================================
def append_warning(result, key, msg):
    if result.get(key):
        result[key] = result[key] + " " + msg
    else:
        result[key] = msg

# ============================================================
# GEOMETRY HELPERS
# ============================================================
def slab_bounds_from_mesh(mesh):
    mins = mesh.bounds[0].copy()
    maxs = mesh.bounds[1].copy()

    for ax in range(3):
        if ax != THICK_AXIS:
            mins[ax] -= XY_MARGIN
            maxs[ax] += XY_MARGIN

    mins[THICK_AXIS] = THICK_MIN
    maxs[THICK_AXIS] = THICK_MAX

    return mins, maxs

def face_mask_near_box(mesh, threshold):
    cents = mesh.triangles_center
    mins, maxs = slab_bounds_from_mesh(mesh)

    dists = np.stack([
        np.abs(cents[:, 0] - mins[0]),
        np.abs(cents[:, 0] - maxs[0]),
        np.abs(cents[:, 1] - mins[1]),
        np.abs(cents[:, 1] - maxs[1]),
        np.abs(cents[:, 2] - mins[2]),
        np.abs(cents[:, 2] - maxs[2]),
    ], axis=1)

    return np.min(dists, axis=1) <= threshold

def vertex_mask_near_mesh_vertices(query_vertices, ref_vertices, threshold):
    tree = cKDTree(ref_vertices)
    distances, _ = tree.query(query_vertices, k=1)
    return distances < threshold

def reference_points_for_surface_distance(mesh):
    """
    Reference points for fast approximate distance to a mesh surface.

    Uses mesh vertices plus triangle centers. This is faster and avoids
    optional trimesh/rtree proximity dependencies.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)

    if len(mesh.faces) > 0:
        centers = np.asarray(mesh.triangles_center, dtype=float)
        return np.vstack([vertices, centers])

    return vertices

def vertex_mask_near_mesh_surface_fast(query_vertices, ref_mesh, threshold):
    """
    Fast approximate test for whether query vertices are within threshold
    of ref_mesh surface.

    Uses cKDTree on ref_mesh vertices + triangle centers.
    """
    query_vertices = np.asarray(query_vertices, dtype=float)
    ref_points = reference_points_for_surface_distance(ref_mesh)

    tree = cKDTree(ref_points)
    distances, _ = tree.query(query_vertices, k=1)

    return distances < threshold, distances

def submesh_with_mapping_fast(mesh, face_mask):
    """
    Create a submesh and return mapping from new vertices back to old vertices.

    Returns:
        new_mesh
        kept_face_idx
        kept_vertex_idx

    Mapping:
        new_mesh.vertices[i] == mesh.vertices[kept_vertex_idx[i]]

    IMPORTANT:
        Do not call remove_unreferenced_vertices, process=True, or other cleaning.
        Any cleaning can alter vertex count/order and break curvature mapping.
    """
    face_mask = np.asarray(face_mask, dtype=bool)
    kept_face_idx = np.flatnonzero(face_mask)

    if kept_face_idx.size == 0:
        raise RuntimeError("No faces selected for submesh.")

    kept_faces_old = mesh.faces[kept_face_idx]

    kept_vertex_idx, inverse = np.unique(
        kept_faces_old.reshape(-1),
        return_inverse=True
    )

    new_faces = inverse.reshape((-1, 3))
    new_vertices = mesh.vertices[kept_vertex_idx]

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )

    return new_mesh, kept_face_idx, kept_vertex_idx

def submesh_from_vertex_mask(mesh, vertex_mask):
    """
    Keep faces where ALL three vertices satisfy the vertex mask.
    Used for CM_surf_highk1.stl.
    """
    vertex_mask = np.asarray(vertex_mask, dtype=bool)

    if len(vertex_mask) != len(mesh.vertices):
        raise ValueError(
            f"vertex_mask length ({len(vertex_mask)}) must match "
            f"mesh vertex count ({len(mesh.vertices)})"
        )

    keep_faces = np.all(vertex_mask[mesh.faces], axis=1)

    if not np.any(keep_faces):
        raise RuntimeError("No faces remain after thresholding.")

    kept_faces_old = mesh.faces[keep_faces]

    kept_vertex_idx, inverse = np.unique(
        kept_faces_old.reshape(-1),
        return_inverse=True
    )

    new_faces = inverse.reshape((-1, 3))
    new_vertices = mesh.vertices[kept_vertex_idx]

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )

    return new_mesh

# ============================================================
# ER / OMM CONTACT SURFACES
# ============================================================
def split_er_mcs_surfaces(
    er_surf,
    omm_surf,
    ermcs_path,
    nonermcs_path,
    distance_um=0.02,
):
    """
    Split ER_surf into:
        ERMCS_surf: faces with at least one ER vertex within distance_um of OMM_surf.
        nonERMCS_surf: all remaining ER_surf faces.

    Distance is computed approximately using ER vertices against OMM vertices
    plus OMM triangle centers.
    """
    er_vertex_near_omm, er_vertex_distances = vertex_mask_near_mesh_surface_fast(
        query_vertices=er_surf.vertices,
        ref_mesh=omm_surf,
        threshold=distance_um,
    )

    ermcs_face_mask = np.any(
        er_vertex_near_omm[er_surf.faces],
        axis=1
    )

    nonermcs_face_mask = ~ermcs_face_mask

    result = {
        "er_vertices_within_ermcs_distance": int(np.sum(er_vertex_near_omm)),
        "er_faces_within_ermcs_distance": int(np.sum(ermcs_face_mask)),
        "ermcs_saved": False,
        "ermcs_vertices": "",
        "ermcs_faces": "",
        "nonermcs_saved": False,
        "nonermcs_vertices": "",
        "nonermcs_faces": "",
        "min_er_to_omm_distance": float(np.min(er_vertex_distances)) if len(er_vertex_distances) > 0 else "",
        "median_er_to_omm_distance": float(np.median(er_vertex_distances)) if len(er_vertex_distances) > 0 else "",
    }

    if np.any(ermcs_face_mask):
        ermcs_surf, _, _ = submesh_with_mapping_fast(
            er_surf,
            ermcs_face_mask
        )
        ermcs_surf.export(ermcs_path)

        result["ermcs_saved"] = True
        result["ermcs_vertices"] = int(len(ermcs_surf.vertices))
        result["ermcs_faces"] = int(len(ermcs_surf.faces))

    if np.any(nonermcs_face_mask):
        nonermcs_surf, _, _ = submesh_with_mapping_fast(
            er_surf,
            nonermcs_face_mask
        )
        nonermcs_surf.export(nonermcs_path)

        result["nonermcs_saved"] = True
        result["nonermcs_vertices"] = int(len(nonermcs_surf.vertices))
        result["nonermcs_faces"] = int(len(nonermcs_surf.faces))

    return result

# ============================================================
# CURVATURE REMAPPING
# ============================================================
def remap_curvature_by_coordinates(
    source_vertices,
    source_curvatures,
    target_vertices,
    tol=1e-6,
    label="curvature remap",
):
    """
    Map per-vertex curvature from source_vertices to target_vertices by nearest
    coordinate.

    This is the safety step that handles STL vertex reorder/merge differences.
    """
    source_vertices = np.asarray(source_vertices, dtype=float)
    source_curvatures = np.asarray(source_curvatures, dtype=float)
    target_vertices = np.asarray(target_vertices, dtype=float)

    if len(source_vertices) != len(source_curvatures):
        raise ValueError(
            f"{label}: source length mismatch. "
            f"source_vertices={len(source_vertices)}, "
            f"source_curvatures={len(source_curvatures)}"
        )

    if len(target_vertices) == 0:
        return np.asarray([], dtype=float)

    tree = cKDTree(source_vertices)
    distances, idx = tree.query(target_vertices, k=1)

    max_dist = float(np.max(distances))

    if max_dist > tol:
        raise RuntimeError(
            f"{label}: vertex remap exceeded tolerance. "
            f"max_dist={max_dist}, tol={tol}"
        )

    return source_curvatures[idx]

def load_matrix_vertex_curvature(smooth_path, matrix_mesh):
    """
    Load smooth_k1 and return curvature aligned to matrix_mesh.vertices.

    New safe format:
        {
            "vertices": Blender vertices,
            "smooth_k1": Blender curvature values
        }

    Old raw format:
        accepted only if length matches matrix_mesh.vertices,
        but warning is returned because vertex ordering is not guaranteed.
    """
    data = load_pickle(smooth_path)

    # New safe format from updated Blender worker
    if isinstance(data, dict):
        if "vertices" not in data or "smooth_k1" not in data:
            raise ValueError(
                f"{smooth_path} is a dict but missing 'vertices' or 'smooth_k1'."
            )

        blender_vertices = np.asarray(data["vertices"], dtype=float)
        blender_k1 = np.asarray(data["smooth_k1"], dtype=float)

        matrix_k1 = remap_curvature_by_coordinates(
            source_vertices=blender_vertices,
            source_curvatures=blender_k1,
            target_vertices=matrix_mesh.vertices,
            tol=REMAP_TOL,
            label="Blender smooth_k1 -> trimesh Matrix_vol"
        )

        return matrix_k1, "safe_payload_with_vertices"

    # Old raw array format
    raw_k1 = np.asarray(data, dtype=float)

    if len(raw_k1) != len(matrix_mesh.vertices):
        raise ValueError(
            f"Old raw smooth_k1 cannot be safely used. "
            f"smooth_k1 length={len(raw_k1)}, "
            f"Matrix_vol vertices={len(matrix_mesh.vertices)}. "
            f"Recompute smooth_k1 with the updated Blender worker that saves "
            f"both vertices and smooth_k1."
        )

    return raw_k1, "old_raw_array_length_matched_warning"

# ============================================================
# WORKER
# ============================================================
def process_one_folder(job):
    meshes_dir = Path(job["meshes_dir"])

    omm_vol_path = meshes_dir / "OMM_vol.stl"
    er_vol_path = meshes_dir / "ER_vol.stl"
    ibm_vol_path = meshes_dir / "IBM_vol.stl"
    matrix_vol_path = meshes_dir / "Matrix_vol.stl"
    smooth_path = meshes_dir / "smooth_k1"

    omm_surf_path = meshes_dir / "OMM_surf.stl"
    er_surf_path = meshes_dir / "ER_surf.stl"
    ermcs_surf_path = meshes_dir / "ERMCS_surf.stl"
    nonermcs_surf_path = meshes_dir / "nonERMCS_surf.stl"
    ibm_surf_path = meshes_dir / "IBM_surf.stl"
    cm_surf_path = meshes_dir / "CM_surf.stl"
    cm_smooth_path = meshes_dir / "CM_surf_smooth_k1.pkl"
    highk1_path = meshes_dir / "CM_surf_highk1.stl"

    result = {
        "status": "ok",
        "root_name": job["root_name"],
        "folder": job["folder"],
        "meshes_dir": job["meshes_dir"],
        "omm_surf_saved": False,
        "er_surf_saved": False,
        "ermcs_surf_saved": False,
        "nonermcs_surf_saved": False,
        "ibm_surf_saved": False,
        "cm_surf_saved": False,
        "cm_smooth_saved": False,
        "highk1_saved": False,
        "curvature_source": "",
        "omm_vertices": "",
        "omm_faces": "",
        "er_vertices": "",
        "er_faces": "",
        "ermcs_distance_um": ERMCS_DISTANCE_UM,
        "er_vertices_within_ermcs_distance": "",
        "er_faces_within_ermcs_distance": "",
        "ermcs_vertices": "",
        "ermcs_faces": "",
        "nonermcs_vertices": "",
        "nonermcs_faces": "",
        "min_er_to_omm_distance": "",
        "median_er_to_omm_distance": "",
        "matrix_vertices": "",
        "matrix_k1_values": "",
        "matrix_box_vertices": "",
        "matrix_box_k1_values": "",
        "cm_vertices": "",
        "cm_faces": "",
        "cm_k1_values": "",
        "vertices_above_threshold": "",
        "highk1_vertices": "",
        "highk1_faces": "",
        "warning": "",
        "er_warning": "",
        "error": "",
        "traceback": "",
    }

    try:
        # --------------------------------------------------------
        # OMM_vol -> OMM_surf
        # --------------------------------------------------------
        if (not SKIP_EXISTING_SURFS) or (not omm_surf_path.exists()):
            omm_vol = load_mesh(omm_vol_path)
            omm_keep_faces = ~face_mask_near_box(
                omm_vol,
                BOX_FACE_THRESHOLD
            )

            omm_surf, _, _ = submesh_with_mapping_fast(
                omm_vol,
                omm_keep_faces
            )

            omm_surf.export(omm_surf_path)
            result["omm_surf_saved"] = True

        # Need OMM_surf for ERMCS/nonERMCS splitting.
        omm_surf_for_er = load_mesh(omm_surf_path)

        # --------------------------------------------------------
        # ER_vol -> ER_surf, ERMCS_surf, nonERMCS_surf
        # --------------------------------------------------------
        if er_vol_path.exists():
            need_er_outputs = (
                (not SKIP_EXISTING_SURFS)
                or (not er_surf_path.exists())
                or (not ermcs_surf_path.exists())
                or (not nonermcs_surf_path.exists())
            )

            if need_er_outputs:
                if (not SKIP_EXISTING_SURFS) or (not er_surf_path.exists()):
                    er_vol = load_mesh(er_vol_path)
                    er_keep_faces = ~face_mask_near_box(
                        er_vol,
                        BOX_FACE_THRESHOLD
                    )

                    er_surf, _, _ = submesh_with_mapping_fast(
                        er_vol,
                        er_keep_faces
                    )

                    er_surf.export(er_surf_path)
                    result["er_surf_saved"] = True
                else:
                    er_surf = load_mesh(er_surf_path)

                er_split = split_er_mcs_surfaces(
                    er_surf=er_surf,
                    omm_surf=omm_surf_for_er,
                    ermcs_path=ermcs_surf_path,
                    nonermcs_path=nonermcs_surf_path,
                    distance_um=ERMCS_DISTANCE_UM,
                )

                result["er_vertices_within_ermcs_distance"] = er_split["er_vertices_within_ermcs_distance"]
                result["er_faces_within_ermcs_distance"] = er_split["er_faces_within_ermcs_distance"]
                result["ermcs_surf_saved"] = er_split["ermcs_saved"]
                result["ermcs_vertices"] = er_split["ermcs_vertices"]
                result["ermcs_faces"] = er_split["ermcs_faces"]
                result["nonermcs_surf_saved"] = er_split["nonermcs_saved"]
                result["nonermcs_vertices"] = er_split["nonermcs_vertices"]
                result["nonermcs_faces"] = er_split["nonermcs_faces"]
                result["min_er_to_omm_distance"] = er_split["min_er_to_omm_distance"]
                result["median_er_to_omm_distance"] = er_split["median_er_to_omm_distance"]

                if not er_split["ermcs_saved"]:
                    append_warning(
                        result,
                        "er_warning",
                        "No ER faces were within ERMCS distance; ERMCS_surf.stl not created."
                    )

                if not er_split["nonermcs_saved"]:
                    append_warning(
                        result,
                        "er_warning",
                        "All ER faces were within ERMCS distance; nonERMCS_surf.stl not created."
                    )

        else:
            result["er_warning"] = "ER_vol.stl missing; ER_surf.stl, ERMCS_surf.stl, and nonERMCS_surf.stl not created."

        # --------------------------------------------------------
        # IBM_vol -> temporary IBM_surf
        # --------------------------------------------------------
        ibm_vol = load_mesh(ibm_vol_path)
        ibm_keep_faces = ~face_mask_near_box(
            ibm_vol,
            BOX_FACE_THRESHOLD
        )

        ibm_surf_temp, _, _ = submesh_with_mapping_fast(
            ibm_vol,
            ibm_keep_faces
        )

        # --------------------------------------------------------
        # Matrix_vol -> CM_surf and final IBM_surf
        # --------------------------------------------------------
        need_cm_outputs = (
            (not SKIP_EXISTING_SURFS)
            or (not cm_surf_path.exists())
            or (not cm_smooth_path.exists())
            or (not ibm_surf_path.exists())
        )

        if need_cm_outputs:
            matrix_vol = load_mesh(matrix_vol_path)

            # Load curvature safely and align to matrix_vol.vertices
            matrix_k1, curvature_source = load_matrix_vertex_curvature(
                smooth_path=smooth_path,
                matrix_mesh=matrix_vol
            )

            result["curvature_source"] = curvature_source
            result["matrix_vertices"] = int(len(matrix_vol.vertices))
            result["matrix_k1_values"] = int(len(matrix_k1))

            if curvature_source == "old_raw_array_length_matched_warning":
                result["warning"] = (
                    "Old raw smooth_k1 format used. Length matched, but vertex "
                    "order is not guaranteed. Recompute smooth_k1 with updated "
                    "Blender worker for safest results."
                )

            if len(matrix_k1) != len(matrix_vol.vertices):
                raise ValueError(
                    f"Internal matrix_k1 mismatch. "
                    f"matrix_k1={len(matrix_k1)}, "
                    f"Matrix_vol vertices={len(matrix_vol.vertices)}"
                )

            # 1) Box trim Matrix_vol
            near_box = face_mask_near_box(
                matrix_vol,
                BOX_FACE_THRESHOLD
            )

            matrix_box_keep = ~near_box

            matrix_box_trimmed, _, matrix_box_vertex_idx = submesh_with_mapping_fast(
                matrix_vol,
                matrix_box_keep
            )

            # Mapping:
            # matrix_box_trimmed.vertices[i]
            #     == matrix_vol.vertices[matrix_box_vertex_idx[i]]
            matrix_box_k1 = matrix_k1[matrix_box_vertex_idx]

            result["matrix_box_vertices"] = int(len(matrix_box_trimmed.vertices))
            result["matrix_box_k1_values"] = int(len(matrix_box_k1))

            if len(matrix_box_k1) != len(matrix_box_trimmed.vertices):
                raise ValueError(
                    f"matrix_box_k1 mismatch. "
                    f"k1={len(matrix_box_k1)}, "
                    f"vertices={len(matrix_box_trimmed.vertices)}"
                )

            # 2) Fast IBM proximity on trimmed Matrix vertices
            close_vertices_mask = vertex_mask_near_mesh_vertices(
                matrix_box_trimmed.vertices,
                ibm_surf_temp.vertices,
                IBM_VERTEX_THRESHOLD
            )

            near_ibm_faces_trimmed = np.any(
                close_vertices_mask[matrix_box_trimmed.faces],
                axis=1
            )

            cm_keep_trimmed = ~near_ibm_faces_trimmed
            ibm_keep_trimmed = near_ibm_faces_trimmed

            # 3) Build CM_surf in-memory and preserve mapping
            cm_surf, _, cm_vertex_idx_from_box = submesh_with_mapping_fast(
                matrix_box_trimmed,
                cm_keep_trimmed
            )

            # Mapping:
            # cm_surf.vertices[i]
            #     == matrix_box_trimmed.vertices[cm_vertex_idx_from_box[i]]
            cm_k1_in_memory = matrix_box_k1[cm_vertex_idx_from_box]

            if len(cm_k1_in_memory) != len(cm_surf.vertices):
                raise ValueError(
                    f"CM in-memory k1 mismatch. "
                    f"k1={len(cm_k1_in_memory)}, "
                    f"vertices={len(cm_surf.vertices)}"
                )

            cm_surf.export(cm_surf_path)
            result["cm_surf_saved"] = True

            # 4) Reload exported CM_surf and remap curvature to match reloaded STL exactly
            cm_surf_reloaded = load_mesh(cm_surf_path)

            cm_smooth_reloaded = remap_curvature_by_coordinates(
                source_vertices=cm_surf.vertices,
                source_curvatures=cm_k1_in_memory,
                target_vertices=cm_surf_reloaded.vertices,
                tol=REMAP_TOL,
                label="CM in-memory -> exported/reloaded CM_surf"
            )

            if len(cm_smooth_reloaded) != len(cm_surf_reloaded.vertices):
                raise ValueError(
                    f"CM reloaded k1 mismatch. "
                    f"k1={len(cm_smooth_reloaded)}, "
                    f"vertices={len(cm_surf_reloaded.vertices)}"
                )

            save_pickle(cm_smooth_path, cm_smooth_reloaded)
            result["cm_smooth_saved"] = True

            result["cm_vertices"] = int(len(cm_surf_reloaded.vertices))
            result["cm_faces"] = int(len(cm_surf_reloaded.faces))
            result["cm_k1_values"] = int(len(cm_smooth_reloaded))

            # 5) Final IBM_surf from removed Matrix faces
            ibm_surf, _, _ = submesh_with_mapping_fast(
                matrix_box_trimmed,
                ibm_keep_trimmed
            )

            ibm_surf.export(ibm_surf_path)
            result["ibm_surf_saved"] = True

        # --------------------------------------------------------
        # CM_surf + CM_surf_smooth_k1.pkl -> CM_surf_highk1.stl
        # --------------------------------------------------------
        if (not SKIP_EXISTING_HIGHK1) or (not highk1_path.exists()):
            cm_surf = load_mesh(cm_surf_path)
            cm_curv = np.asarray(load_pickle(cm_smooth_path), dtype=float)

            if len(cm_curv) != len(cm_surf.vertices):
                raise ValueError(
                    f"Curvature length ({len(cm_curv)}) must match "
                    f"CM_surf vertex count ({len(cm_surf.vertices)})"
                )

            high_mask = cm_curv > CURV_THRESHOLD
            n_high = int(np.sum(high_mask))

            if n_high == 0:
                raise RuntimeError(
                    f"No vertices above threshold {CURV_THRESHOLD}"
                )

            cm_high = submesh_from_vertex_mask(cm_surf, high_mask)
            cm_high.export(highk1_path)

            result["highk1_saved"] = True
            result["vertices_above_threshold"] = n_high
            result["highk1_vertices"] = int(len(cm_high.vertices))
            result["highk1_faces"] = int(len(cm_high.faces))

        # Read final OMM stats if available
        if omm_surf_path.exists():
            omm_tmp = load_mesh(omm_surf_path)
            result["omm_vertices"] = int(len(omm_tmp.vertices))
            result["omm_faces"] = int(len(omm_tmp.faces))

        # Read final ER stats if available
        if er_surf_path.exists():
            er_tmp = load_mesh(er_surf_path)
            result["er_vertices"] = int(len(er_tmp.vertices))
            result["er_faces"] = int(len(er_tmp.faces))

        # Read final ERMCS stats if available
        if ermcs_surf_path.exists():
            ermcs_tmp = load_mesh(ermcs_surf_path)
            result["ermcs_vertices"] = int(len(ermcs_tmp.vertices))
            result["ermcs_faces"] = int(len(ermcs_tmp.faces))

        # Read final nonERMCS stats if available
        if nonermcs_surf_path.exists():
            nonermcs_tmp = load_mesh(nonermcs_surf_path)
            result["nonermcs_vertices"] = int(len(nonermcs_tmp.vertices))
            result["nonermcs_faces"] = int(len(nonermcs_tmp.faces))

        # Read final CM stats if available
        if cm_surf_path.exists():
            cm_tmp = load_mesh(cm_surf_path)
            result["cm_vertices"] = int(len(cm_tmp.vertices))
            result["cm_faces"] = int(len(cm_tmp.faces))

        if cm_smooth_path.exists():
            cm_tmp_curv = np.asarray(load_pickle(cm_smooth_path), dtype=float)
            result["cm_k1_values"] = int(len(cm_tmp_curv))

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result

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
        "omm_surf_saved",
        "er_surf_saved",
        "ermcs_surf_saved",
        "nonermcs_surf_saved",
        "ibm_surf_saved",
        "cm_surf_saved",
        "cm_smooth_saved",
        "highk1_saved",
        "curvature_source",
        "omm_vertices",
        "omm_faces",
        "er_vertices",
        "er_faces",
        "ermcs_distance_um",
        "er_vertices_within_ermcs_distance",
        "er_faces_within_ermcs_distance",
        "ermcs_vertices",
        "ermcs_faces",
        "nonermcs_vertices",
        "nonermcs_faces",
        "min_er_to_omm_distance",
        "median_er_to_omm_distance",
        "matrix_vertices",
        "matrix_k1_values",
        "matrix_box_vertices",
        "matrix_box_k1_values",
        "cm_vertices",
        "cm_faces",
        "cm_k1_values",
        "vertices_above_threshold",
        "highk1_vertices",
        "highk1_faces",
        "warning",
        "er_warning",
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
    print(f"SKIP_EXISTING_SURFS: {SKIP_EXISTING_SURFS}")
    print(f"SKIP_EXISTING_HIGHK1: {SKIP_EXISTING_HIGHK1}")
    print(f"CURV_THRESHOLD: {CURV_THRESHOLD}")
    print(f"REMAP_TOL: {REMAP_TOL}")
    print("")

    if len(jobs) == 0:
        print("No complete folders found.")
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
            desc="Making surf + highK1",
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
                        "omm_surf_saved": False,
                        "er_surf_saved": False,
                        "ermcs_surf_saved": False,
                        "nonermcs_surf_saved": False,
                        "ibm_surf_saved": False,
                        "cm_surf_saved": False,
                        "cm_smooth_saved": False,
                        "highk1_saved": False,
                        "curvature_source": "",
                        "omm_vertices": "",
                        "omm_faces": "",
                        "er_vertices": "",
                        "er_faces": "",
                        "ermcs_distance_um": ERMCS_DISTANCE_UM,
                        "er_vertices_within_ermcs_distance": "",
                        "er_faces_within_ermcs_distance": "",
                        "ermcs_vertices": "",
                        "ermcs_faces": "",
                        "nonermcs_vertices": "",
                        "nonermcs_faces": "",
                        "min_er_to_omm_distance": "",
                        "median_er_to_omm_distance": "",
                        "matrix_vertices": "",
                        "matrix_k1_values": "",
                        "matrix_box_vertices": "",
                        "matrix_box_k1_values": "",
                        "cm_vertices": "",
                        "cm_faces": "",
                        "cm_k1_values": "",
                        "vertices_above_threshold": "",
                        "highk1_vertices": "",
                        "highk1_faces": "",
                        "warning": "",
                        "er_warning": "",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

                results.append(result)

                if result["status"] == "ok":
                    ok_count += 1
                else:
                    failed_count += 1

                pbar.update(1)

    summary_path = script_dir / SUMMARY_LOG_NAME
    write_csv(summary_path, results)

    failure_rows = [
        row for row in results
        if row["status"] != "ok"
    ]

    failure_path = script_dir / FAILURE_LOG_NAME
    write_csv(failure_path, failure_rows)

    print("")
    print("Done.")
    print(f"Successful folders: {ok_count}")
    print(f"Failed folders:     {failed_count}")
    print(f"Summary log saved:  {summary_path}")

    if failure_rows:
        print(f"Failure log saved:  {failure_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
