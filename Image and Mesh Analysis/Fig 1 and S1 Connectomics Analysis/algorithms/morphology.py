"""Numerical and rendering routines extracted from the supplied analysis; sample orchestration is in run_sample.py."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

os.environ.setdefault("MKL_NUM_THREADS", "1")

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import argparse

import gc

import re

import traceback

import numpy as np

import pandas as pd

import matplotlib.colors as mcolors

import vtk

from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from tqdm import tqdm

MAX_SNAPS = 100

MITO_MAX_DISTANCE_NM = 2000.0

MITO_IN_SPINE_MAX_DISTANCE_NM = 500.0

MITO_IN_SPINE_MIN_VOLUME_UM3 = 0.01

WIDENING_WIDTH_THRESHOLD = 1.0

PROTRUDE_WIDTH_THRESHOLD = 1.5

PROTRUDE_MAX_SIGNED_DISTANCE_NM = 200.0

SORT_BY_WIDTH_DESCENDING = True

SKIP_EXISTING = True

SHOW_ONLY_MAPPED_MITO_COMPONENT = True

SHOW_NEURON_MESH = True

SHOW_MITO_MESH = True

SHOW_SPINE_PATH = False

SHOW_SYNAPSE_POINT = False

SHOW_SPINE_BASE_POINT = False

DISPLAY_VALUE_VMIN = 0.0

DISPLAY_VALUE_VMAX = 2.0

CLIP_VALUES_FOR_COLOR = True

IMAGE_WIDTH = 1400

IMAGE_HEIGHT = 1400

BACKGROUND_COLOR = (0.0, 0.0, 0.0)

VOXEL_SIZE_NM = np.asarray((11.24, 11.24, 28.0), dtype=float)

MIN_PARALLEL_SCALE_NM = 900.0

ZOOM_MARGIN_FACTOR = 1.60

MITO_AXIS_RADIUS_NM = 3000.0

MIN_MITO_AXIS_POINTS = 20

MAX_CAMERA_MITO_POINTS = 50000

MIN_CAMERA_HALF_DEPTH_NM = 2500.0

CAMERA_DEPTH_MARGIN_FACTOR = 1.50

CAMERA_DISTANCE_NM = 100000.0

NEURON_MESH_COLOR = (1.0, 1.0, 1.0)

NEURON_MESH_OPACITY = 0.10

MITO_MESH_OPACITY = 0.90

NAN_COLOR_RGBA_255 = (80, 80, 80, 80)

SPINE_PATH_COLOR = (0.0, 1.0, 1.0)

SPINE_PATH_LINE_WIDTH = 7.0

LTP_COLOR = (0.0, 1.0, 0.0)

LTD_COLOR = (1.0, 0.0, 0.0)

CONTROL_COLOR = (0.0, 0.25, 1.0)

NO_PAIR_COLOR = (0.55, 0.55, 0.55)

POINT_OPACITY = 1.0

PAIR_POINT_SIZES = [
    210 / 2,
    235 / 2,
    260 / 2,
    285 / 2,
    310 / 2,
    335 / 2,
]

NO_PAIR_POINT_SIZE = 180 / 4

BASE_POINT_SIZE_FACTOR = 1.5

CATEGORIES = ("protrusion", "budding", "recruitment")

AGGLO_COLUMN = "aggloIds_2"

PAIR_ID_COLUMN = "pairIdx"

SYNAPSE_ID_COLUMN = "id"

CONTOUR_COLUMN = "contour"

PAIRED_CONTOURS = [
    "LTP",
    "LTD",
    "control",
]

SWC_COLUMNS = [
    "node_id",
    "type",
    "x",
    "y",
    "z",
    "radius",
    "parent_id",
    "synapse_id",
    "role_code",
    "path_order",
    "syn_base_dist_nm",
    "base_dend_dist_nm",
]

ROLE_SYNAPSE = 1

ROLE_INTERIOR = 2

ROLE_BASE = 3

REQUIRED_MEASUREMENT_COLUMNS = {
    "agglo_id",
    "branch_id",
    "synapse_id",
    "base_x_nm",
    "base_y_nm",
    "base_z_nm",
    "signed_distance_nm",
    "nearest_surface_x_nm",
    "nearest_surface_y_nm",
    "nearest_surface_z_nm",
    "mito_component_id",
    "mito_width_normalized_z",
    "absolute_distance_nm",
    "synapse_to_mito_distance_nm",
    "base_nearest_mito_volume_um3",
    "synapse_nearest_mito_volume_um3",
}

SYNAPSE_COMPONENT_COLUMNS = (
    "synapse_nearest_mito_component_id",
    "synapse_mito_component_id",
    "synapse_nearest_component_id",
)

def normalize_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        numeric = float(text)
        if np.isfinite(numeric):
            if numeric.is_integer():
                return str(int(numeric))
            return format(numeric, ".15g")
    except Exception:
        pass
    return text

def safe_filename(value):
    text = normalize_id(value)
    if text == "":
        text = "unknown_synapse"
    text = re.sub(
        r'[<>:"/\\|?*\x00-\x1F]',
        "_",
        text,
    )
    text = text.strip(" .")
    if text == "":
        text = "unknown_synapse"
    return text

def synapse_coordinate_filename(synapse_id, synapse_point):
    coordinates = []
    for value in np.asarray(synapse_point, dtype=float) / VOXEL_SIZE_NM:
        if not np.isfinite(value):
            raise ValueError("Synapse coordinates must be finite.")
        coordinates.append(str(int(np.rint(value))))
    return f"{safe_filename(synapse_id)}_{','.join(coordinates)}.png"

def normalize_vector(vector, fallback=None):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if np.isfinite(length) and length > 1e-9:
        return vector / length
    if fallback is None:
        fallback = np.asarray(
            [0.0, 0.0, 1.0],
            dtype=float,
        )
    fallback = np.asarray(
        fallback,
        dtype=float,
    )
    fallback_length = np.linalg.norm(fallback)
    if not np.isfinite(fallback_length) or fallback_length <= 1e-9:
        raise ValueError("Cannot normalize a zero-length vector.")
    return fallback / fallback_length

def is_valid_png(path):
    path = Path(path)
    if not path.exists():
        return False
    if path.stat().st_size <= 8:
        return False
    try:
        with path.open("rb") as file:
            signature = file.read(8)
        return signature == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False

def select_measurements(measurements, max_snaps):
    component_column = next(
        (column for column in SYNAPSE_COMPONENT_COLUMNS if column in measurements), None
    )
    measurements["_synapse_mito_component_id"] = (
        measurements[component_column] if component_column else np.nan
    )
    numeric_columns = [
        "branch_id",
        "base_x_nm",
        "base_y_nm",
        "base_z_nm",
        "signed_distance_nm",
        "nearest_surface_x_nm",
        "nearest_surface_y_nm",
        "nearest_surface_z_nm",
        "mito_component_id",
        "mito_width_normalized_z",
        "absolute_distance_nm",
        "synapse_to_mito_distance_nm",
        "base_nearest_mito_volume_um3",
        "synapse_nearest_mito_volume_um3",
        "_synapse_mito_component_id",
    ]
    for column in numeric_columns:
        measurements[column] = pd.to_numeric(
            measurements[column],
            errors="coerce",
        )
    measurements["_agglo_key"] = measurements["agglo_id"].map(normalize_id)
    measurements["_synapse_key"] = measurements["synapse_id"].map(normalize_id)
    valid_mask = measurements["_agglo_key"].ne("") & measurements["_synapse_key"].ne("")
    measurements = measurements[valid_mask].copy()
    grouped = measurements.groupby(["_agglo_key", "_synapse_key"], as_index=False, dropna=False)[
        numeric_columns
    ].mean()
    render_columns = [
        "branch_id",
        "base_x_nm",
        "base_y_nm",
        "base_z_nm",
        "nearest_surface_x_nm",
        "nearest_surface_y_nm",
        "nearest_surface_z_nm",
        "mito_component_id",
    ]
    render_valid = np.ones(len(grouped), dtype=bool)
    for column in render_columns:
        render_valid &= np.isfinite(grouped[column])
    grouped = grouped.loc[render_valid].copy()
    grouped["_file_stem"] = grouped["_synapse_key"].map(safe_filename)
    valid_width = np.isfinite(grouped["mito_width_normalized_z"])
    valid_signed = np.isfinite(grouped["signed_distance_nm"])
    valid_absolute = np.isfinite(grouped["absolute_distance_nm"])
    valid_synapse_distance = np.isfinite(grouped["synapse_to_mito_distance_nm"])
    valid_base_volume = np.isfinite(grouped["base_nearest_mito_volume_um3"])
    valid_synapse_volume = np.isfinite(grouped["synapse_nearest_mito_volume_um3"])
    has_mito = valid_absolute & (grouped["absolute_distance_nm"] <= MITO_MAX_DISTANCE_NM)
    different_volume = (
        valid_base_volume
        & valid_synapse_volume
        & (grouped["base_nearest_mito_volume_um3"] != grouped["synapse_nearest_mito_volume_um3"])
    )
    mito_in_spine = (
        valid_synapse_distance
        & (grouped["synapse_to_mito_distance_nm"] <= MITO_IN_SPINE_MAX_DISTANCE_NM)
        & (grouped["synapse_nearest_mito_volume_um3"] >= MITO_IN_SPINE_MIN_VOLUME_UM3)
        & different_volume
    )
    widening = (
        has_mito & valid_width & (grouped["mito_width_normalized_z"] > WIDENING_WIDTH_THRESHOLD)
    )
    protrusion = (
        widening
        & valid_signed
        & (grouped["mito_width_normalized_z"] > PROTRUDE_WIDTH_THRESHOLD)
        & (grouped["signed_distance_nm"] < PROTRUDE_MAX_SIGNED_DISTANCE_NM)
    )
    recruitment = widening & ~protrusion
    category_masks = {
        "protrusion": protrusion,
        "budding": mito_in_spine,
        "recruitment": recruitment,
    }
    selected_categories = []
    for category in CATEGORIES:
        selected = grouped.loc[category_masks[category]].copy()
        selected = selected.sort_values(
            ["mito_width_normalized_z", "signed_distance_nm", "_agglo_key", "_synapse_key"],
            ascending=[not SORT_BY_WIDTH_DESCENDING, True, True, True],
            na_position="last",
            kind="stable",
        )
        duplicate_filenames = selected["_file_stem"].duplicated(keep="first")
        if duplicate_filenames.any():
            print(
                f"Warning: {int(duplicate_filenames.sum()):,} duplicate {category} synapse filenames were removed."
            )
            selected = selected.loc[~duplicate_filenames].copy()
        if max_snaps is not None:
            if max_snaps < 1:
                raise ValueError("max_snaps must be at least 1 or None.")
            selected = selected.head(max_snaps).copy()
        selected = selected.reset_index(drop=True)
        selected["_category"] = category
        selected["_rank"] = np.arange(1, len(selected) + 1, dtype=int)
        selected["_output_dir"] = str(output_dir / category)
        selected_categories.append(selected)
    return (
        pd.concat(selected_categories, ignore_index=True)
        if selected_categories
        else grouped.iloc[0:0].copy()
    )

def load_synapse_table():
    if not synapse_table_path.exists():
        print(
            "Warning: synapse metadata table was not found.\n"
            "All synapses will use the no_pair style.\n"
            f"{synapse_table_path}"
        )
        return pd.DataFrame()
    dataframe = pd.read_csv(synapse_table_path, low_memory=False)
    required = {
        AGGLO_COLUMN,
        SYNAPSE_ID_COLUMN,
        CONTOUR_COLUMN,
    }
    missing = required - set(dataframe.columns)
    if missing:
        print(
            "Warning: synapse metadata is missing columns "
            f"{sorted(missing)}. All synapses will use "
            "the no_pair style."
        )
        return pd.DataFrame()
    if PAIR_ID_COLUMN not in dataframe.columns:
        dataframe[PAIR_ID_COLUMN] = pd.NA
    dataframe["_agglo_key"] = dataframe[AGGLO_COLUMN].map(normalize_id)
    dataframe["_synapse_key"] = dataframe[SYNAPSE_ID_COLUMN].map(normalize_id)
    dataframe["_contour_norm"] = dataframe[CONTOUR_COLUMN].fillna("no_pair").astype(str).str.strip()
    dataframe["_pair_id_string"] = dataframe[PAIR_ID_COLUMN].map(normalize_id)
    dataframe["_pair_key"] = dataframe["_contour_norm"] + "::" + dataframe["_pair_id_string"]
    return dataframe

def color_for_contour(contour):
    contour = str(contour)
    if contour == "LTP":
        return LTP_COLOR
    if contour == "LTD":
        return LTD_COLOR
    if contour == "control":
        return CONTROL_COLOR
    return NO_PAIR_COLOR

def build_style_map(synapse_table, agglo_id):
    if len(synapse_table) == 0:
        return {}
    sub = synapse_table[synapse_table["_agglo_key"] == normalize_id(agglo_id)].copy()
    if len(sub) == 0:
        return {}
    paired = sub[sub["_contour_norm"].isin(PAIRED_CONTOURS) & sub["_pair_id_string"].ne("")].copy()
    pair_keys = paired["_pair_key"].drop_duplicates().tolist()
    pair_size_map = {
        pair_key: float(PAIR_POINT_SIZES[index % len(PAIR_POINT_SIZES)])
        for index, pair_key in enumerate(pair_keys)
    }
    style_map = {}
    for _, row in sub.iterrows():
        synapse_key = row["_synapse_key"]
        if synapse_key == "" or synapse_key in style_map:
            continue
        contour = row["_contour_norm"]
        if contour not in (PAIRED_CONTOURS + ["no_pair"]):
            contour = "no_pair"
        if contour == "no_pair":
            point_size = float(NO_PAIR_POINT_SIZE)
        else:
            point_size = pair_size_map.get(
                row["_pair_key"],
                float(PAIR_POINT_SIZES[0]),
            )
        style_map[synapse_key] = {
            "contour": contour,
            "color": color_for_contour(contour),
            "synapse_size": float(point_size),
            "base_size": float(point_size * BASE_POINT_SIZE_FACTOR),
        }
    return style_map

def validate_mesh(vertices, faces, label):
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{label}: vertices must have shape (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{label}: faces must have shape (M, 3).")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"{label}: mesh is empty.")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError(f"{label}: face indices are invalid.")

def load_npz_mesh(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh: {path}")
    with np.load(path) as data:
        required = {
            "vertices",
            "faces",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Mesh file {path.name} is missing " f"arrays: {sorted(missing)}")
        vertices = np.asarray(
            data["vertices"],
            dtype=np.float64,
        )
        faces = np.asarray(
            data["faces"],
            dtype=np.int64,
        )
    validate_mesh(
        vertices,
        faces,
        path.name,
    )
    return {
        "vertices": vertices,
        "faces": faces,
    }

def load_width_mesh(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing mitochondrial width mesh: {path}")
    with np.load(path) as data:
        required = {
            "vertices",
            "faces",
            "display_value",
            "component_id",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Width mesh {path.name} is missing " f"arrays: {sorted(missing)}")
        result = {
            "vertices": np.asarray(
                data["vertices"],
                dtype=np.float64,
            ),
            "faces": np.asarray(
                data["faces"],
                dtype=np.int64,
            ),
            "display_value": np.asarray(
                data["display_value"],
                dtype=float,
            ),
            "component_id": np.asarray(
                data["component_id"],
                dtype=np.int64,
            ),
        }
    validate_mesh(
        result["vertices"],
        result["faces"],
        path.name,
    )
    n_vertices = len(result["vertices"])
    if len(result["display_value"]) != n_vertices:
        raise ValueError("display_value length does not match " "the mitochondrial vertex count.")
    if len(result["component_id"]) != n_vertices:
        raise ValueError("component_id length does not match " "the mitochondrial vertex count.")
    return result

def load_spine_paths(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing spine SWC: {path}")
    dataframe = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=SWC_COLUMNS,
        engine="python",
    )
    if len(dataframe) == 0:
        raise ValueError(f"The SWC contains no nodes: {path}")
    for column in SWC_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
    dataframe = dataframe.dropna(
        subset=[
            "node_id",
            "x",
            "y",
            "z",
            "parent_id",
            "role_code",
            "path_order",
        ]
    ).copy()
    integer_columns = [
        "node_id",
        "type",
        "parent_id",
        "role_code",
        "path_order",
    ]
    for column in integer_columns:
        dataframe[column] = dataframe[column].astype(int)
    dataframe["_branch_id"] = dataframe["parent_id"].eq(-1).cumsum().astype(int)
    branches = {}
    for branch_id, branch in dataframe.groupby(
        "_branch_id",
        sort=False,
    ):
        branch = branch.sort_values("path_order").copy()
        points = branch[["x", "y", "z"]].to_numpy(dtype=np.float64)
        finite = np.all(
            np.isfinite(points),
            axis=1,
        )
        branch = branch.loc[finite].copy()
        points = points[finite]
        if len(points) == 0:
            continue
        synapse_rows = branch[branch["role_code"] == ROLE_SYNAPSE]
        if len(synapse_rows) > 0:
            synapse_point = synapse_rows[["x", "y", "z"]].iloc[0].to_numpy(dtype=float)
        else:
            synapse_point = points[0].copy()
        base_rows = branch[branch["role_code"] == ROLE_BASE]
        if len(base_rows) > 0:
            base_point = base_rows[["x", "y", "z"]].iloc[-1].to_numpy(dtype=float)
        else:
            base_point = points[-1].copy()
        synapse_id = normalize_id(branch["synapse_id"].iloc[0])
        branches[int(branch_id)] = {
            "points": points,
            "synapse_point": synapse_point,
            "base_point": base_point,
            "synapse_id": synapse_id,
        }
    if len(branches) == 0:
        raise ValueError(f"No usable spine paths were found in {path}")
    return branches

def make_vtk_points(vertices):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_to_vtk(
            np.ascontiguousarray(
                vertices,
                dtype=np.float64,
            ),
            deep=True,
        )
    )
    return vtk_points

def make_triangle_cells(faces):
    faces = np.ascontiguousarray(
        faces,
        dtype=np.int64,
    )
    offsets = np.arange(
        0,
        3 * len(faces) + 1,
        3,
        dtype=np.int64,
    )
    connectivity = faces.reshape(-1)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            offsets,
            deep=True,
        ),
        numpy_to_vtkIdTypeArray(
            connectivity,
            deep=True,
        ),
    )
    return cells

def make_line_cells(n_points):
    offsets = np.asarray(
        [0, n_points],
        dtype=np.int64,
    )
    connectivity = np.arange(
        n_points,
        dtype=np.int64,
    )
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            offsets,
            deep=True,
        ),
        numpy_to_vtkIdTypeArray(
            connectivity,
            deep=True,
        ),
    )
    return cells

def make_vertex_cells(n_points):
    offsets = np.arange(
        n_points + 1,
        dtype=np.int64,
    )
    connectivity = np.arange(
        n_points,
        dtype=np.int64,
    )
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            offsets,
            deep=True,
        ),
        numpy_to_vtkIdTypeArray(
            connectivity,
            deep=True,
        ),
    )
    return cells

def make_plain_mesh_actor(
    vertices,
    faces,
    color,
    opacity,
):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(make_vtk_points(vertices))
    polydata.SetPolys(make_triangle_cells(faces))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(
        float(color[0]),
        float(color[1]),
        float(color[2]),
    )
    actor.GetProperty().SetOpacity(float(opacity))
    actor.GetProperty().SetInterpolationToPhong()
    return actor

def values_to_green_yellow_rgba(values):
    values = np.asarray(
        values,
        dtype=float,
    )
    finite = np.isfinite(values)
    rgba = np.zeros(
        (len(values), 4),
        dtype=np.uint8,
    )
    rgba[:] = np.asarray(
        NAN_COLOR_RGBA_255,
        dtype=np.uint8,
    )
    if not np.any(finite):
        return rgba
    finite_values = values[finite].copy()
    if CLIP_VALUES_FOR_COLOR:
        finite_values = np.clip(
            finite_values,
            DISPLAY_VALUE_VMIN,
            DISPLAY_VALUE_VMAX,
        )
    normalizer = mcolors.Normalize(
        vmin=DISPLAY_VALUE_VMIN,
        vmax=DISPLAY_VALUE_VMAX,
        clip=CLIP_VALUES_FOR_COLOR,
    )
    cmap = mcolors.LinearSegmentedColormap.from_list("green_yellow", ["#00FF00", "#FFFF00"])
    rgba_float = cmap(normalizer(finite_values))
    rgba_uint8 = np.round(rgba_float * 255).astype(np.uint8)
    rgba_uint8[:, 3] = int(
        np.clip(
            MITO_MESH_OPACITY,
            0.0,
            1.0,
        )
        * 255
    )
    rgba[finite] = rgba_uint8
    return rgba

def make_colored_mito_actor(
    vertices,
    faces,
    values,
    uniform_green=False,
):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(make_vtk_points(vertices))
    polydata.SetPolys(make_triangle_cells(faces))
    rgba = values_to_green_yellow_rgba(np.zeros(len(vertices)) if uniform_green else values)
    color_array = numpy_to_vtk(
        np.ascontiguousarray(rgba),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    color_array.SetName("mito_width_rgba")
    color_array.SetNumberOfComponents(4)
    polydata.GetPointData().SetScalars(color_array)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetInterpolationToPhong()
    return actor

def make_path_actor(points):
    points = np.asarray(
        points,
        dtype=np.float64,
    )
    if len(points) < 2:
        return None
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(make_vtk_points(points))
    polydata.SetLines(make_line_cells(len(points)))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*SPINE_PATH_COLOR)
    actor.GetProperty().SetLineWidth(float(SPINE_PATH_LINE_WIDTH))
    actor.GetProperty().SetOpacity(1.0)
    return actor

def make_point_actor(
    point,
    size,
    color,
):
    point = np.asarray(
        point,
        dtype=np.float64,
    ).reshape(
        1,
        3,
    )
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(make_vtk_points(point))
    polydata.SetVerts(make_vertex_cells(1))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(
        float(color[0]),
        float(color[1]),
        float(color[2]),
    )
    actor.GetProperty().SetOpacity(float(POINT_OPACITY))
    actor.GetProperty().SetPointSize(float(size))
    actor.GetProperty().RenderPointsAsSpheresOn()
    return actor

def extract_mito_component(
    width_mesh,
    component_id,
):
    component_id = int(component_id)
    vertex_component_ids = width_mesh["component_id"]
    faces = width_mesh["faces"]
    face_mask = np.all(
        vertex_component_ids[faces] == component_id,
        axis=1,
    )
    selected_faces = faces[face_mask]
    if len(selected_faces) == 0:
        raise ValueError(f"Mito component {component_id} has no faces.")
    used_vertices = np.unique(selected_faces.reshape(-1))
    old_to_new = np.full(
        len(width_mesh["vertices"]),
        -1,
        dtype=np.int64,
    )
    old_to_new[used_vertices] = np.arange(
        len(used_vertices),
        dtype=np.int64,
    )
    remapped_faces = old_to_new[selected_faces]
    return {
        "vertices": width_mesh["vertices"][used_vertices],
        "faces": remapped_faces,
        "display_value": width_mesh["display_value"][used_vertices],
    }

def build_mito_surface_locator(width_mesh):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(make_vtk_points(width_mesh["vertices"]))
    polydata.SetPolys(make_triangle_cells(width_mesh["faces"]))
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    return polydata, locator

def resolve_budding_component(row, width_mesh, surface_locator, synapse_point):
    base_component_id = int(row["mito_component_id"])
    supplied_component_id = row["_synapse_mito_component_id"]
    if np.isfinite(supplied_component_id):
        component_id = int(supplied_component_id)
        if component_id == base_component_id:
            raise ValueError(
                "Budding synapse-nearest component is the same component as the base-nearest mitochondrion."
            )
        if not np.any(width_mesh["component_id"] == component_id):
            raise ValueError(
                f"Synapse-nearest mito component {component_id} is absent from the width mesh."
            )
        return component_id
    if surface_locator is None:
        raise ValueError("A mitochondrial surface locator is required for budding snapshots.")
    closest_point = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    distance_squared = vtk.reference(0.0)
    surface_locator.FindClosestPoint(
        np.asarray(synapse_point, dtype=float), closest_point, cell_id, sub_id, distance_squared
    )
    face = width_mesh["faces"][int(cell_id)]
    face_component_ids = width_mesh["component_id"][face]
    if not np.all(face_component_ids == face_component_ids[0]):
        raise ValueError("The synapse-nearest mitochondrial face crosses component labels.")
    component_id = int(face_component_ids[0])
    if component_id == base_component_id:
        raise ValueError(
            "The synapse-nearest mitochondrial component is the same component as the base-nearest mitochondrion."
        )
    return component_id

def configure_camera(
    renderer,
    base_point,
    mito_point,
    synapse_point,
    path_points,
    mito_vertices,
):
    base_point = np.asarray(
        base_point,
        dtype=float,
    )
    mito_point = np.asarray(
        mito_point,
        dtype=float,
    )
    synapse_point = np.asarray(
        synapse_point,
        dtype=float,
    )
    path_points = np.asarray(
        path_points,
        dtype=float,
    )
    mito_vertices = np.asarray(
        mito_vertices,
        dtype=float,
    )
    finite_mito = np.all(
        np.isfinite(mito_vertices),
        axis=1,
    )
    mito_vertices = mito_vertices[finite_mito]
    if len(mito_vertices) < 3:
        raise ValueError(
            "At least three finite mitochondrial vertices " "are required to orient the camera."
        )
    # Subsampling is used only for calculating orientation.
    if len(mito_vertices) > MAX_CAMERA_MITO_POINTS:
        sample_indices = np.linspace(
            0,
            len(mito_vertices) - 1,
            MAX_CAMERA_MITO_POINTS,
            dtype=np.int64,
        )
        orientation_vertices = mito_vertices[sample_indices]
    else:
        orientation_vertices = mito_vertices
    # Find vertices near the mapped mitochondrial point.
    distance_from_contact = np.linalg.norm(
        orientation_vertices - mito_point,
        axis=1,
    )
    local_mito_vertices = orientation_vertices[distance_from_contact <= MITO_AXIS_RADIUS_NM]
    if len(local_mito_vertices) < MIN_MITO_AXIS_POINTS:
        local_mito_vertices = orientation_vertices
    # PCA of the local mitochondrion.
    local_mito_center = np.mean(
        local_mito_vertices,
        axis=0,
    )
    centered_local_mito = local_mito_vertices - local_mito_center
    mito_covariance = (centered_local_mito.T @ centered_local_mito) / max(
        len(centered_local_mito) - 1,
        1,
    )
    (
        mito_eigenvalues,
        mito_eigenvectors,
    ) = np.linalg.eigh(mito_covariance)
    # The first PCA axis becomes screen left-to-right.
    right_axis = normalize_vector(
        mito_eigenvectors[
            :,
            int(np.argmax(mito_eigenvalues)),
        ]
    )
    # PCA axis signs are arbitrary. Make the result deterministic.
    dominant_coordinate = int(np.argmax(np.abs(right_axis)))
    if right_axis[dominant_coordinate] < 0:
        right_axis = -right_axis
    # Preferred direction of the spine in the image plane.
    spine_axis = synapse_point - base_point
    if np.linalg.norm(spine_axis) <= 1e-9:
        spine_axis = path_points[0] - path_points[-1]
    spine_axis = normalize_vector(
        spine_axis,
        fallback=np.asarray(
            [0.0, 0.0, 1.0],
            dtype=float,
        ),
    )
    # Remove the part parallel to the mitochondrial horizontal
    # axis. This keeps the spine in the image plane.
    up_candidate = (
        spine_axis
        - np.dot(
            spine_axis,
            right_axis,
        )
        * right_axis
    )
    if np.linalg.norm(up_candidate) <= 1e-9:
        base_to_mito = mito_point - base_point
        up_candidate = (
            base_to_mito
            - np.dot(
                base_to_mito,
                right_axis,
            )
            * right_axis
        )
    if np.linalg.norm(up_candidate) <= 1e-9:
        coordinate_axes = np.eye(
            3,
            dtype=float,
        )
        reference_axis = coordinate_axes[np.argmin(np.abs(coordinate_axes @ right_axis))]
        up_candidate = (
            reference_axis
            - np.dot(
                reference_axis,
                right_axis,
            )
            * right_axis
        )
    up_axis = normalize_vector(up_candidate)
    # Keep the synapse above the base.
    if (
        np.dot(
            synapse_point - base_point,
            up_axis,
        )
        < 0
    ):
        up_axis = -up_axis
    # Camera direction is perpendicular to the plane containing
    # the mitochondrial and spine directions.
    view_axis = normalize_vector(
        np.cross(
            right_axis,
            up_axis,
        )
    )
    # Ensure the basis is exactly orthogonal.
    right_axis = normalize_vector(
        np.cross(
            up_axis,
            view_axis,
        )
    )
    # Original framing objects are unchanged. The mitochondrial
    # vertices used for PCA are not added to the zoom calculation.
    context_points = np.vstack(
        [
            path_points,
            base_point.reshape(
                1,
                3,
            ),
            mito_point.reshape(
                1,
                3,
            ),
            synapse_point.reshape(
                1,
                3,
            ),
        ]
    )
    # Keep the spine base at the center.
    focal_point = base_point.copy()
    centered = context_points - focal_point
    right_extent = np.max(np.abs(centered @ right_axis))
    up_extent = np.max(np.abs(centered @ up_axis))
    image_aspect = float(IMAGE_WIDTH) / float(IMAGE_HEIGHT)
    required_vertical_scale = max(
        up_extent,
        right_extent / image_aspect,
    )
    # Use the same physical field of view for every image.
    parallel_scale = 2500.0
    depth_extent = np.max(np.abs(centered @ view_axis))
    camera_half_depth = max(
        MIN_CAMERA_HALF_DEPTH_NM,
        depth_extent * CAMERA_DEPTH_MARGIN_FACTOR,
    )
    camera_position = focal_point + view_axis * CAMERA_DISTANCE_NM
    camera = renderer.GetActiveCamera()
    camera.SetPosition(
        float(camera_position[0]),
        float(camera_position[1]),
        float(camera_position[2]),
    )
    camera.SetFocalPoint(
        float(focal_point[0]),
        float(focal_point[1]),
        float(focal_point[2]),
    )
    camera.SetViewUp(
        float(up_axis[0]),
        float(up_axis[1]),
        float(up_axis[2]),
    )
    camera.ParallelProjectionOn()
    camera.SetParallelScale(float(parallel_scale))
    camera.SetClippingRange(
        max(
            1.0,
            CAMERA_DISTANCE_NM - camera_half_depth,
        ),
        CAMERA_DISTANCE_NM + camera_half_depth,
    )

def create_offscreen_renderer():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(
        float(BACKGROUND_COLOR[0]),
        float(BACKGROUND_COLOR[1]),
        float(BACKGROUND_COLOR[2]),
    )
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.1)
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(
        int(IMAGE_WIDTH),
        int(IMAGE_HEIGHT),
    )
    render_window.SetMultiSamples(0)
    render_window.SetAlphaBitPlanes(1)
    render_window.AddRenderer(renderer)
    return (
        renderer,
        render_window,
    )

def save_render_window_png(
    render_window,
    output_path,
):
    output_path = Path(output_path)
    temporary_path = output_path.with_name(output_path.stem + ".tmp.png")
    if temporary_path.exists():
        temporary_path.unlink()
    render_window.Render()
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(temporary_path))
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.Write()
    if not is_valid_png(temporary_path):
        raise RuntimeError(f"VTK did not produce a valid PNG: " f"{temporary_path}")
    temporary_path.replace(output_path)

def render_agglo_group(
    agglo_id,
    rows,
    synapse_table,
    overwrite,
    progress_bar,
):
    width_path = width_mesh_dir / f"{agglo_id}.npz"
    neuron_path = neuron_mesh_dir / f"{agglo_id}.npz"
    swc_path = spine_swc_dir / f"{agglo_id}.swc"
    width_mesh = load_width_mesh(width_path)
    branches = load_spine_paths(swc_path)
    style_map = build_style_map(
        synapse_table,
        agglo_id,
    )
    (
        renderer,
        render_window,
    ) = create_offscreen_renderer()
    persistent_actors = []
    # Original neuron rendering is unchanged.
    if SHOW_NEURON_MESH:
        if neuron_path.exists():
            neuron_mesh = load_npz_mesh(neuron_path)
            neuron_actor = make_plain_mesh_actor(
                vertices=neuron_mesh["vertices"],
                faces=neuron_mesh["faces"],
                color=NEURON_MESH_COLOR,
                opacity=NEURON_MESH_OPACITY,
            )
            renderer.AddActor(neuron_actor)
            persistent_actors.append(neuron_actor)
        else:
            print(f"\nWarning: neuron mesh not found " f"for agglo {agglo_id}: " f"{neuron_path}")
    # This cache is used only for orientation calculations.
    component_vertex_cache = {}
    # Original rendered mitochondrial actors are unchanged.
    component_actor_cache = {}
    full_mito_actor_cache = {}
    surface_polydata = None
    surface_locator = None
    saved = 0
    skipped = 0
    failed = []
    try:
        for _, row in rows.iterrows():
            synapse_key = row["_synapse_key"]
            category = row["_category"]
            rank = int(row["_rank"])
            progress_bar.set_postfix_str(
                (f"category={category} " f"rank={rank} " f"agglo={agglo_id} " f"syn={synapse_key}"),
                refresh=False,
            )
            target_actors = []
            try:
                branch_id = int(row["branch_id"])
                if branch_id not in branches:
                    raise KeyError(f"Branch {branch_id} was not found " f"in {swc_path.name}")
                branch = branches[branch_id]
                if branch["synapse_id"] != "" and branch["synapse_id"] != synapse_key:
                    print(
                        f"\nWarning: branch {branch_id} "
                        f"contains synapse "
                        f"{branch['synapse_id']}, but the "
                        f"measurement row contains "
                        f"{synapse_key}."
                    )
                path_points = branch["points"]
                synapse_point = branch["synapse_point"]
                output_path = Path(row["_output_dir"]) / synapse_coordinate_filename(
                    synapse_key, synapse_point
                )
                if not overwrite and SKIP_EXISTING and is_valid_png(output_path):
                    skipped += 1
                    continue
                base_point = np.asarray(
                    [
                        row["base_x_nm"],
                        row["base_y_nm"],
                        row["base_z_nm"],
                    ],
                    dtype=float,
                )
                mito_point = np.asarray(
                    [
                        row["nearest_surface_x_nm"],
                        row["nearest_surface_y_nm"],
                        row["nearest_surface_z_nm"],
                    ],
                    dtype=float,
                )
                style = style_map.get(
                    synapse_key,
                    {
                        "contour": "no_pair",
                        "color": NO_PAIR_COLOR,
                        "synapse_size": float(NO_PAIR_POINT_SIZE),
                        "base_size": float(NO_PAIR_POINT_SIZE * BASE_POINT_SIZE_FACTOR),
                    },
                )
                component_id = int(row["mito_component_id"])
                uniform_green = True
                color_mode = "green" if uniform_green else "width"
                render_component_ids = [component_id]
                if category == "budding":
                    if (
                        row["base_nearest_mito_volume_um3"]
                        == row["synapse_nearest_mito_volume_um3"]
                    ):
                        raise ValueError(
                            "Budding requires different base-nearest and synapse-nearest mitochondrial volumes."
                        )
                    if (
                        not np.isfinite(row["_synapse_mito_component_id"])
                        and surface_locator is None
                    ):
                        surface_polydata, surface_locator = build_mito_surface_locator(width_mesh)
                    budding_component_id = resolve_budding_component(
                        row, width_mesh, surface_locator, synapse_point
                    )
                    render_component_ids.append(budding_component_id)
                if component_id not in component_vertex_cache:
                    component_vertex_mask = width_mesh["component_id"] == component_id
                    component_vertex_cache[component_id] = width_mesh["vertices"][
                        component_vertex_mask
                    ]
                if SHOW_MITO_MESH:
                    if SHOW_ONLY_MAPPED_MITO_COMPONENT:
                        mito_actors = []
                        for render_component_id in render_component_ids:
                            actor_key = (render_component_id, color_mode)
                            if actor_key not in component_actor_cache:
                                component = extract_mito_component(width_mesh, render_component_id)
                                component_actor_cache[actor_key] = make_colored_mito_actor(
                                    vertices=component["vertices"],
                                    faces=component["faces"],
                                    values=component["display_value"],
                                    uniform_green=uniform_green,
                                )
                            mito_actors.append(component_actor_cache[actor_key])
                    else:
                        if color_mode not in full_mito_actor_cache:
                            full_mito_actor_cache[color_mode] = make_colored_mito_actor(
                                vertices=width_mesh["vertices"],
                                faces=width_mesh["faces"],
                                values=width_mesh["display_value"],
                                uniform_green=uniform_green,
                            )
                        mito_actors = [full_mito_actor_cache[color_mode]]
                    for mito_actor in mito_actors:
                        renderer.AddActor(mito_actor)
                        target_actors.append(mito_actor)
                # Original spine rendering is unchanged.
                if SHOW_SPINE_PATH:
                    path_actor = make_path_actor(path_points)
                    if path_actor is not None:
                        renderer.AddActor(path_actor)
                        target_actors.append(path_actor)
                # Original synapse rendering is unchanged.
                if SHOW_SYNAPSE_POINT:
                    synapse_actor = make_point_actor(
                        point=synapse_point,
                        size=style["synapse_size"],
                        color=style["color"],
                    )
                    renderer.AddActor(synapse_actor)
                    target_actors.append(synapse_actor)
                # Original spine-base rendering is unchanged.
                if SHOW_SPINE_BASE_POINT:
                    base_actor = make_point_actor(
                        point=base_point,
                        size=style["base_size"],
                        color=style["color"],
                    )
                    renderer.AddActor(base_actor)
                    target_actors.append(base_actor)
                configure_camera(
                    renderer=renderer,
                    base_point=base_point,
                    mito_point=mito_point,
                    synapse_point=synapse_point,
                    path_points=path_points,
                    mito_vertices=component_vertex_cache[component_id],
                )
                save_render_window_png(
                    render_window,
                    output_path,
                )
                saved += 1
                progress_bar.write(
                    (
                        f"Saved {category} rank {rank}: "
                        f"{output_path.name} | "
                        f"agglo={agglo_id} | "
                        f"width_z="
                        f"{row['mito_width_normalized_z']:.3f} | "
                        f"signed_distance="
                        f"{row['signed_distance_nm']:.3f} nm"
                    )
                )
            except Exception as exc:
                failed.append(
                    {
                        "rank": rank,
                        "category": category,
                        "agglo_id": agglo_id,
                        "synapse_id": synapse_key,
                        "error": str(exc),
                    }
                )
                progress_bar.write(
                    (
                        f"FAILED {category} rank {rank}: "
                        f"agglo={agglo_id}, "
                        f"synapse={synapse_key}: "
                        f"{exc}"
                    )
                )
            finally:
                for actor in target_actors:
                    renderer.RemoveViewProp(actor)
                progress_bar.update(1)
    finally:
        render_window.Finalize()
        del component_vertex_cache
        del component_actor_cache
        del full_mito_actor_cache
        del surface_polydata
        del surface_locator
        del persistent_actors
        del renderer
        del render_window
        gc.collect()
    return {
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
    }
