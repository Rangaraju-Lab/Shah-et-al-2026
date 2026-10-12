"""Numerical and rendering routines extracted from the supplied analysis; sample orchestration is in run_sample.py."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

os.environ.setdefault("MKL_NUM_THREADS", "1")

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing as mp

import contextlib

import traceback

import time

import gc

import numpy as np

import pandas as pd

import vtk

from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from tqdm import tqdm

TARGET_AGGLO_ID = None

SKIP_EXISTING = False

SMALL_MESH_MAX_GB = 0.35

MAX_SMALL_WORKERS = 2

PROCESS_START_METHOD = "spawn"

TASK_PRIORITY_MODE = "mesh_size_id"

SUPPRESS_INTERNAL_OUTPUT = True

ENCLOSED_POINT_TOLERANCE = 1e-6

CHECK_MITO_SURFACE_CLOSED = False

TQDM_MININTERVAL_SEC = 1.0

SWC_COLUMNS = [
    "node_id", "type", "x", "y", "z", "radius", "parent_id",
    "synapse_id", "role_code", "path_order",
    "syn_base_dist_nm", "base_dend_dist_nm",
]

ROLE_SYNAPSE = 1

ROLE_INTERIOR = 2

ROLE_BASE = 3

OUTPUT_COLUMNS = [
    "agglo_id",
    "branch_id",
    "synapse_id",
    "base_node_id",
    "base_x_nm",
    "base_y_nm",
    "base_z_nm",
    "inside_mito",
    "signed_distance_nm",
    "absolute_distance_nm",
    "synapse_to_mito_distance_nm",
    "nearest_surface_x_nm",
    "nearest_surface_y_nm",
    "nearest_surface_z_nm",
    "nearest_surface_cell_id",
    "mito_component_id",
    "base_nearest_mito_volume_um3",
    "synapse_nearest_mito_volume_um3",
    "mapped_vertex_idx",
    "mapped_vertex_x_nm",
    "mapped_vertex_y_nm",
    "mapped_vertex_z_nm",
    "base_to_mapped_vertex_distance_nm",
    "mito_width_raw_nm",
    "mito_width_normalized_z",
    "width_normalized_per_component",
    "syn_base_dist_nm",
    "base_dend_dist_nm",
]

def load_width_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing width file: {path}")

    with np.load(path) as data:
        required = [
            "vertices",
            "faces",
            "diameter_nm",
            "display_value",
            "component_id",
        ]

        missing = [
            name
            for name in required
            if name not in data.files
        ]

        if missing:
            raise ValueError(
                f"Width file is missing arrays: {missing}\n"
                f"Available arrays: {list(data.files)}"
            )

        output = {
            "vertices": np.asarray(data["vertices"], dtype=np.float64),
            "faces": np.asarray(data["faces"], dtype=np.int64),
            "diameter_nm": np.asarray(data["diameter_nm"], dtype=float),
            "display_value": np.asarray(data["display_value"], dtype=float),
            "component_id": np.asarray(data["component_id"], dtype=np.int64),
        }

        if "normalize_per_component_display" in data.files:
            output["normalize_per_component_display"] = bool(
                int(
                    np.asarray(
                        data["normalize_per_component_display"]
                    ).ravel()[0]
                )
            )
        else:
            output["normalize_per_component_display"] = True

    n_vertices = len(output["vertices"])

    if len(output["diameter_nm"]) != n_vertices:
        raise ValueError("diameter_nm length does not match vertices.")

    if len(output["display_value"]) != n_vertices:
        raise ValueError("display_value length does not match vertices.")

    if len(output["component_id"]) != n_vertices:
        raise ValueError("component_id length does not match vertices.")

    if len(output["faces"]) == 0:
        raise ValueError("The mitochondrial mesh has no faces.")

    if (
        np.min(output["faces"]) < 0
        or np.max(output["faces"]) >= n_vertices
    ):
        raise ValueError("The mitochondrial mesh has invalid face indices.")

    return output

def load_spine_bases(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing spine SWC: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=SWC_COLUMNS,
        engine="python",
    )

    if df.empty:
        raise ValueError(f"The SWC contains no nodes: {path}")

    for column in SWC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "node_id", "x", "y", "z",
            "parent_id", "role_code", "path_order",
        ]
    ).copy()

    if df.empty:
        raise ValueError("No valid SWC nodes remain.")

    for column in [
        "node_id",
        "type",
        "parent_id",
        "role_code",
        "path_order",
    ]:
        df[column] = df[column].astype(int)

    df["_branch_id"] = df["parent_id"].eq(-1).cumsum().astype(int)

    base_df = df[df["role_code"] == ROLE_BASE].copy()

    if base_df.empty:
        raise ValueError("No role_code == 3 spine-base nodes were found.")

    base_df = (
        base_df
        .sort_values(["_branch_id", "path_order"])
        .groupby("_branch_id", sort=False, as_index=False)
        .tail(1)
        .copy()
    )

    synapse_df = df[df["role_code"] == ROLE_SYNAPSE].copy()

    if synapse_df.empty:
        raise ValueError("No role_code == 1 synapse nodes were found.")

    synapse_df = (
        synapse_df
        .sort_values(["_branch_id", "path_order"])
        .groupby("_branch_id", sort=False, as_index=False)
        .head(1)[["_branch_id", "x", "y", "z"]]
        .rename(columns={
            "x": "synapse_x",
            "y": "synapse_y",
            "z": "synapse_z",
        })
    )

    base_df = (
        base_df
        .merge(
            synapse_df,
            on="_branch_id",
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    base_points = base_df[
        ["x", "y", "z"]
    ].to_numpy(dtype=np.float64)

    synapse_points = base_df[
        ["synapse_x", "synapse_y", "synapse_z"]
    ].to_numpy(dtype=np.float64)

    finite = (
        np.all(np.isfinite(base_points), axis=1)
        & np.all(np.isfinite(synapse_points), axis=1)
    )

    if not np.all(finite):
        base_df = base_df.loc[finite].reset_index(drop=True)
        base_points = base_df[
            ["x", "y", "z"]
        ].to_numpy(dtype=np.float64)

    if base_df.empty:
        raise ValueError(
            "No branches with finite base and synapse coordinates remain."
        )

    return base_df, base_points

def make_triangle_cell_array(faces):
    faces = np.asarray(faces, dtype=np.int64)

    connectivity = np.ascontiguousarray(
        faces.ravel(),
        dtype=np.int64,
    )

    offsets = np.arange(
        0,
        3 * (len(faces) + 1),
        3,
        dtype=np.int64,
    )

    cells = vtk.vtkCellArray()

    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )

    return cells

def make_mito_polydata(vertices, faces):
    points = vtk.vtkPoints()

    points.SetData(
        numpy_to_vtk(
            np.asarray(vertices, dtype=np.float64),
            deep=True,
        )
    )

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(make_triangle_cell_array(faces))

    return polydata

def derive_face_component_ids(faces, vertex_component_ids):
    vertex_ids = vertex_component_ids[faces]

    a = vertex_ids[:, 0]
    b = vertex_ids[:, 1]
    c = vertex_ids[:, 2]

    return np.where(
        a == b,
        a,
        np.where(
            a == c,
            a,
            np.where(b == c, b, a),
        ),
    ).astype(np.int64)

def component_mesh_volume_um3(
    vertices,
    faces,
    face_component_ids,
    component_id,
):
    """
    Extract one mitochondrial component mesh and calculate its
    complete enclosed volume.
    """
    component_face_indices = np.flatnonzero(
        face_component_ids == component_id
    )

    if len(component_face_indices) == 0:
        raise ValueError(
            f"Mito component {component_id} has no faces."
        )

    component_global_faces = faces[
        component_face_indices
    ]

    # Extract only vertices used by this mitochondrial object.
    component_vertex_indices, local_face_indices = np.unique(
        component_global_faces,
        return_inverse=True,
    )

    component_vertices = vertices[
        component_vertex_indices
    ]

    component_faces = local_face_indices.reshape(
        -1,
        3,
    )

    # Translation near the origin improves numerical precision
    # without changing volume.
    component_vertices = (
        component_vertices
        - component_vertices.mean(axis=0)
    )

    triangles = component_vertices[
        component_faces
    ]

    signed_tetrahedron_volumes_nm3 = (
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(
                triangles[:, 1],
                triangles[:, 2],
            ),
        )
        / 6.0
    )

    volume_nm3 = abs(
        float(
            np.sum(
                signed_tetrahedron_volumes_nm3,
                dtype=np.float64,
            )
        )
    )

    volume_um3 = volume_nm3 / 1e9

    if not np.isfinite(volume_um3):
        raise ValueError(
            f"Non-finite volume for mito component {component_id}."
        )

    return volume_um3

def measure_needed_component_volumes(
    vertices,
    faces,
    face_component_ids,
    needed_component_ids,
):
    """
    Each requested mitochondrial object is extracted and measured
    exactly once. The resulting dictionary acts as the cache.
    """
    volume_cache = {}

    for component_id in np.unique(
        np.asarray(
            needed_component_ids,
            dtype=np.int64,
        )
    ):
        volume_cache[int(component_id)] = (
            component_mesh_volume_um3(
                vertices=vertices,
                faces=faces,
                face_component_ids=face_component_ids,
                component_id=int(component_id),
            )
        )

    return volume_cache

def classify_points_inside_surface(points, surface_poly):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))

    point_poly = vtk.vtkPolyData()
    point_poly.SetPoints(vtk_points)

    selector = vtk.vtkSelectEnclosedPoints()
    selector.SetInputData(point_poly)
    selector.SetSurfaceData(surface_poly)
    selector.SetTolerance(float(ENCLOSED_POINT_TOLERANCE))

    if CHECK_MITO_SURFACE_CLOSED:
        selector.CheckSurfaceOn()
    else:
        selector.CheckSurfaceOff()

    selector.Update()

    selected_array = (
        selector
        .GetOutput()
        .GetPointData()
        .GetArray("SelectedPoints")
    )

    if selected_array is None:
        raise RuntimeError(
            "vtkSelectEnclosedPoints did not return SelectedPoints."
        )

    return np.asarray(
        vtk_to_numpy(selected_array),
        dtype=np.uint8,
    ).astype(bool)

def query_closest_surface_points(points, surface_poly):
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface_poly)
    locator.BuildLocator()

    closest_surface_points = np.empty(
        (len(points), 3),
        dtype=np.float64,
    )

    closest_cell_ids = np.empty(
        len(points),
        dtype=np.int64,
    )

    absolute_distances_nm = np.empty(
        len(points),
        dtype=np.float64,
    )

    for point_index, query_point in enumerate(points):
        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(-1)
        sub_id = vtk.reference(0)
        distance_squared = vtk.reference(np.inf)

        locator.FindClosestPoint(
            query_point,
            closest_point,
            cell_id,
            sub_id,
            distance_squared,
        )

        found_cell_id = int(cell_id)

        if found_cell_id < 0:
            raise RuntimeError(
                "No closest mitochondrial triangle was found "
                f"for query index {point_index}."
            )

        closest_surface_points[point_index] = np.asarray(
            closest_point,
            dtype=np.float64,
        )

        closest_cell_ids[point_index] = found_cell_id

        absolute_distances_nm[point_index] = np.sqrt(
            max(float(distance_squared), 0.0)
        )

    return (
        closest_surface_points,
        closest_cell_ids,
        absolute_distances_nm,
    )

def map_surface_points_to_triangle_vertices(
    base_points,
    closest_surface_points,
    closest_cell_ids,
    mito_vertices,
    mito_faces,
):
    closest_faces = mito_faces[
        closest_cell_ids
    ]

    triangle_vertices = mito_vertices[
        closest_faces
    ]

    difference = (
        triangle_vertices
        - closest_surface_points[:, None, :]
    )

    local_vertex_indices = np.argmin(
        np.sum(difference**2, axis=2),
        axis=1,
    )

    row_indices = np.arange(
        len(base_points),
        dtype=np.int64,
    )

    mapped_vertex_indices = closest_faces[
        row_indices,
        local_vertex_indices,
    ].astype(np.int64)

    mapped_vertex_points = mito_vertices[
        mapped_vertex_indices
    ]

    base_to_mapped_vertex_distance_nm = np.linalg.norm(
        mapped_vertex_points - base_points,
        axis=1,
    )

    return (
        mapped_vertex_indices,
        mapped_vertex_points,
        base_to_mapped_vertex_distance_nm,
    )

def measure_spine_bases(
    agglo_id,
    width,
    base_df,
    base_points,
):
    mito_vertices = width["vertices"]
    mito_faces = width["faces"]

    surface_poly = make_mito_polydata(
        mito_vertices,
        mito_faces,
    )

    inside_mask = classify_points_inside_surface(
        base_points,
        surface_poly,
    )

    synapse_points = base_df[
        ["synapse_x", "synapse_y", "synapse_z"]
    ].to_numpy(dtype=np.float64)

    # One locator query for bases and synapses together.
    all_query_points = np.vstack(
        [
            base_points,
            synapse_points,
        ]
    )

    (
        all_closest_surface_points,
        all_closest_cell_ids,
        all_absolute_distances_nm,
    ) = query_closest_surface_points(
        all_query_points,
        surface_poly,
    )

    n_bases = len(base_points)

    base_surface_points = (
        all_closest_surface_points[:n_bases]
    )

    base_surface_cell_ids = (
        all_closest_cell_ids[:n_bases]
    )

    synapse_surface_cell_ids = (
        all_closest_cell_ids[n_bases:]
    )

    base_absolute_distance_nm = (
        all_absolute_distances_nm[:n_bases]
    )

    synapse_to_mito_distance_nm = (
        all_absolute_distances_nm[n_bases:]
    )

    signed_distance_nm = (
        base_absolute_distance_nm.copy()
    )

    signed_distance_nm[
        inside_mask
    ] *= -1.0

    (
        mapped_vertex_indices,
        mapped_vertex_points,
        base_to_mapped_vertex_distance_nm,
    ) = map_surface_points_to_triangle_vertices(
        base_points,
        base_surface_points,
        base_surface_cell_ids,
        mito_vertices,
        mito_faces,
    )

    face_component_ids = derive_face_component_ids(
        mito_faces,
        width["component_id"],
    )

    # The closest triangles directly identify the mitochondrial
    # objects required for the two volume measurements.
    base_nearest_component_ids = face_component_ids[
        base_surface_cell_ids
    ]

    synapse_nearest_component_ids = face_component_ids[
        synapse_surface_cell_ids
    ]

    needed_component_ids = np.unique(
        np.concatenate(
            [
                base_nearest_component_ids,
                synapse_nearest_component_ids,
            ]
        )
    )

    # Only these components are extracted. Each is measured once.
    volume_cache = measure_needed_component_volumes(
        vertices=mito_vertices,
        faces=mito_faces,
        face_component_ids=face_component_ids,
        needed_component_ids=needed_component_ids,
    )

    base_nearest_mito_volume_um3 = np.asarray(
        [
            volume_cache[int(component_id)]
            for component_id
            in base_nearest_component_ids
        ],
        dtype=np.float64,
    )

    synapse_nearest_mito_volume_um3 = np.asarray(
        [
            volume_cache[int(component_id)]
            for component_id
            in synapse_nearest_component_ids
        ],
        dtype=np.float64,
    )

    raw_width_nm = width["diameter_nm"][
        mapped_vertex_indices
    ]

    normalized_width = width["display_value"][
        mapped_vertex_indices
    ]

    output = pd.DataFrame({
        "agglo_id": str(agglo_id),
        "branch_id": base_df["_branch_id"].to_numpy(dtype=int),
        "synapse_id": base_df["synapse_id"].to_numpy(),
        "base_node_id": base_df["node_id"].to_numpy(dtype=int),
        "base_x_nm": base_points[:, 0],
        "base_y_nm": base_points[:, 1],
        "base_z_nm": base_points[:, 2],
        "inside_mito": inside_mask,
        "signed_distance_nm": signed_distance_nm,
        "absolute_distance_nm": base_absolute_distance_nm,
        "synapse_to_mito_distance_nm": synapse_to_mito_distance_nm,
        "nearest_surface_x_nm": base_surface_points[:, 0],
        "nearest_surface_y_nm": base_surface_points[:, 1],
        "nearest_surface_z_nm": base_surface_points[:, 2],
        "nearest_surface_cell_id": base_surface_cell_ids,
        "mito_component_id": base_nearest_component_ids,
        "base_nearest_mito_volume_um3": base_nearest_mito_volume_um3,
        "synapse_nearest_mito_volume_um3": synapse_nearest_mito_volume_um3,
        "mapped_vertex_idx": mapped_vertex_indices,
        "mapped_vertex_x_nm": mapped_vertex_points[:, 0],
        "mapped_vertex_y_nm": mapped_vertex_points[:, 1],
        "mapped_vertex_z_nm": mapped_vertex_points[:, 2],
        "base_to_mapped_vertex_distance_nm": (
            base_to_mapped_vertex_distance_nm
        ),
        "mito_width_raw_nm": raw_width_nm,
        "mito_width_normalized_z": normalized_width,
        "width_normalized_per_component": bool(
            width["normalize_per_component_display"]
        ),
        "syn_base_dist_nm": base_df[
            "syn_base_dist_nm"
        ].to_numpy(dtype=float),
        "base_dend_dist_nm": base_df[
            "base_dend_dist_nm"
        ].to_numpy(dtype=float),
    })

    return (
        output[OUTPUT_COLUMNS]
        .sort_values(
            ["synapse_id", "branch_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

def save_measurements_atomic(dataframe, output_path):
    output_path = Path(output_path)
    temporary_path = Path(str(output_path) + ".tmp")

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(output_path)
