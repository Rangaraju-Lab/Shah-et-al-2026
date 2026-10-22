# ============================================================
# Batch ERMCS screenshots for Control and TTX
#
# Scans:
#   ../Control/<sample>/scaled/meshes
#   ../TTX/<sample>/scaled/meshes
#
# Loads per sample:
#   CM_surf.stl
#   OMM_surf.stl
#   IBM_surf.stl
#   ERMCS_surf.stl
#   nonERMCS_surf.stl
#   Ribo_vol.stl when available
#
# Does NOT load or use curvature.
#
# CM/cristae color:
#   RGB = (0.9686, 0.7686, 0.2627)
#
# ERMCS color:
#   RGB = (0.0, 0.447, 0.698)
#
# nonERMCS color:
#   RGB = (0.0, 0.447, 0.698)
#   opacity = 0.2
#
# Ribo color:
#   RGB = (0.780, 0.082, 0.522)
#
# Camera behavior:
#   - choose plane: XY, XZ, or YZ
#   - selected plane faces camera
#   - longest axis within selected plane is rotated into landscape orientation
#   - image center is based on selected center points
#   - horizontal field of view is fixed, default 2.0 um
#   - vertical field of view is determined by 16:9 aspect ratio
#
# Saves all TIF screenshots to:
#   ../ERMCS snaps/
#
# Output image names:
#   Control 1.tif
#   Control 2.tif
#   ...
#   TTX 1.tif
#   TTX 2.tif
#   ...
#
# Also saves:
#   ../ERMCS snaps/ERMCS_snaps_manifest.csv
#
# Run from Notebooks folder:
#   python batch_ERMCS_snaps_fixed_fov.py --plane yz --horizontal-fov-um 2.0 --overwrite
# ============================================================

from project_paths import WORKERS, BLENDER_WORKERS
from pathlib import Path
import argparse
import re
import traceback
import gc

import numpy as np
import pandas as pd
import trimesh
import pyvista as pv
from tqdm import tqdm

# ============================================================
# DEFAULTS
# ============================================================
ROOT_NAMES = ["Control", "TTX"]

MESH_SUBPATH = Path("scaled") / "meshes"

CM_SURF_NAME = "CM_surf.stl"
OMM_SURF_NAME = "OMM_surf.stl"
IBM_SURF_NAME = "IBM_surf.stl"
ERMCS_SURF_NAME = "ERMCS_surf.stl"
NONERMCS_SURF_NAME = "nonERMCS_surf.stl"
RIBO_VOL_NAME = "Ribo_vol.stl"

OUT_FOLDER_NAME = "ERMCS snaps"
MANIFEST_NAME = "ERMCS_snaps_manifest.csv"
FAILURES_NAME = "ERMCS_snaps_failures.csv"

CRISTAE_COLOR_RGB = (0.9686, 0.7686, 0.2627)
ERMCS_COLOR_RGB = (0.0, 0.447, 0.698)
NONERMCS_COLOR_RGB = (0.0, 0.447, 0.698)
RIBO_COLOR_RGB = (0.780, 0.082, 0.522)

# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-render fixed-FOV ERMCS screenshots for Control and TTX."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="",
        help=(
            "Project root. If omitted, uses the package root. "
            "Paths are independent of the working directory."
        ),
    )

    parser.add_argument(
        "--plane",
        type=str,
        default="yz",
        choices=["xy", "xz", "yz"],
        help="Plane facing the camera. Choose xy, xz, or yz.",
    )

    parser.add_argument(
        "--camera-sign",
        type=str,
        default="+",
        choices=["+", "-"],
        help="Which side of the plane normal to view from. Use + or - if image appears flipped.",
    )

    parser.add_argument(
        "--horizontal-fov-um",
        type=float,
        default=2.0,
        help="Horizontal field of view in mesh units. Default is 2.0 um.",
    )

    parser.add_argument(
        "--omm-opacity",
        type=float,
        default=0.25,
        help="OMM opacity.",
    )

    parser.add_argument(
        "--ibm-opacity",
        type=float,
        default=0.25,
        help="IBM opacity.",
    )

    parser.add_argument(
        "--cm-opacity",
        type=float,
        default=1.0,
        help="Cristae/CM opacity.",
    )

    parser.add_argument(
        "--ermcs-opacity",
        type=float,
        default=1.0,
        help="ERMCS opacity.",
    )

    parser.add_argument(
        "--nonermcs-opacity",
        type=float,
        default=0.3,
        help="nonERMCS opacity.",
    )

    parser.add_argument(
        "--ribo-opacity",
        type=float,
        default=1.0,
        help="Ribo_vol opacity.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Screenshot width. Default is 1920 for 4:3.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1440,
        help="Screenshot height. Default is 1440 for 4:3."
    )

    parser.add_argument(
        "--center-to",
        type=str,
        default="all",
        choices=["all", "mito", "cm", "er", "ermcs"],
        help=(
            "Center fixed-FOV image. "
            "all = OMM+IBM+CM+ERMCS+nonERMCS; "
            "mito = OMM+IBM+CM; "
            "cm = CM only; "
            "er = ERMCS+nonERMCS; "
            "ermcs = ERMCS only."
        ),
    )

    parser.add_argument(
        "--use-combined-orientation",
        action="store_true",
        help="Use OMM+IBM+CM together to compute landscape orientation. Default uses CM only.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing screenshots. Default is True.",
    )
    return parser.parse_args()

# ============================================================
# PATH / METADATA HELPERS
# ============================================================
def get_default_root():
    return Path(__file__).resolve().parent.parent

def parse_metadata_from_folder(sample_dir):
    """
    Example:
        2025-12-09 TTX 21

    Returns:
        date_str, condition, mito_number
    """
    sample_dir = Path(sample_dir)
    folder_name = sample_dir.name

    m = re.match(
        r"(\d{4}-\d{2}-\d{2})\s+([A-Za-z]+)\s+(\d+)$",
        folder_name,
    )

    if not m:
        raise ValueError(f"Could not parse metadata from folder name: {folder_name}")

    return m.group(1), m.group(2), int(m.group(3))

def sample_sort_key(sample_dir):
    try:
        date_str, condition, mito = parse_metadata_from_folder(sample_dir)
        return date_str, mito, sample_dir.name
    except Exception:
        return "9999-99-99", 999999, sample_dir.name

def safe_filename(label):
    label = str(label)
    label = re.sub(r"[\\/:\*\?\"<>\|]", "_", label)
    return label

def discover_jobs(root, out_dir, overwrite=False):
    """
    Find direct samples under Control and TTX.

    Output labels are sequential per condition:
        Control 1, Control 2, ...
        TTX 1, TTX 2, ...
    """
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()

    jobs = []

    for root_name in ROOT_NAMES:
        condition_root = root / root_name

        if not condition_root.exists():
            print(f"Missing condition folder: {condition_root}")
            continue

        sample_dirs = sorted(
            [p for p in condition_root.iterdir() if p.is_dir()],
            key=sample_sort_key,
        )

        label_idx = 1

        for sample_dir in sample_dirs:
            meshes_dir = sample_dir / MESH_SUBPATH

            if not meshes_dir.exists():
                continue

            cm_surf_path = meshes_dir / CM_SURF_NAME
            omm_surf_path = meshes_dir / OMM_SURF_NAME
            ibm_surf_path = meshes_dir / IBM_SURF_NAME
            ermcs_surf_path = meshes_dir / ERMCS_SURF_NAME
            nonermcs_surf_path = meshes_dir / NONERMCS_SURF_NAME

            required = [
                cm_surf_path,
                omm_surf_path,
                ibm_surf_path,
                ermcs_surf_path,
                nonermcs_surf_path,
            ]

            missing = [str(p.name) for p in required if not p.exists()]

            label = f"{root_name} {label_idx}"
            out_path = out_dir / f"{safe_filename(label)}.tif"

            if missing:
                jobs.append({
                    "status": "missing",
                    "label": label,
                    "condition_root": root_name,
                    "sample_dir": sample_dir,
                    "meshes_dir": meshes_dir,
                    "out_path": out_path,
                    "missing_files": "; ".join(missing),
                })

                label_idx += 1
                continue

            if out_path.exists() and not overwrite:
                jobs.append({
                    "status": "skip_existing",
                    "label": label,
                    "condition_root": root_name,
                    "sample_dir": sample_dir,
                    "meshes_dir": meshes_dir,
                    "out_path": out_path,
                    "missing_files": "",
                })
            else:
                jobs.append({
                    "status": "ready",
                    "label": label,
                    "condition_root": root_name,
                    "sample_dir": sample_dir,
                    "meshes_dir": meshes_dir,
                    "out_path": out_path,
                    "missing_files": "",
                })

            label_idx += 1

    return jobs

# ============================================================
# FILE HELPERS
# ============================================================
def save_tif(path, image):
    path = Path(path)

    try:
        import tifffile
        tifffile.imwrite(path, image)
        return
    except Exception:
        pass

    try:
        import imageio.v3 as iio
        iio.imwrite(path, image)
        return
    except Exception:
        pass

    raise RuntimeError(
        "Could not save TIF. Install tifffile with: pip install tifffile"
    )

# ============================================================
# MESH LOADING
# ============================================================
def load_trimesh_surface(path):
    """
    Load STL with trimesh without processing.

    process=False avoids unnecessary vertex merging/reordering.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing mesh file: {path}")

    mesh = trimesh.load(path, force="mesh", process=False)

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Not a Trimesh object: {path}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no vertices/faces: {path}")

    return mesh

def trimesh_to_pyvista(mesh):
    """
    Convert trimesh.Trimesh to pyvista.PolyData while preserving vertex order.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    pv_faces = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64),
        faces,
    ]).ravel()

    pv_mesh = pv.PolyData(vertices, pv_faces)

    if pv_mesh.n_points == 0 or pv_mesh.n_cells == 0:
        raise ValueError("Converted PyVista mesh is empty.")

    return pv_mesh

def load_surface_mesh(path):
    tri_mesh = load_trimesh_surface(path)
    pv_mesh = trimesh_to_pyvista(tri_mesh)
    return pv_mesh, tri_mesh

def load_optional_surface_mesh(path):
    path = Path(path)

    if not path.exists():
        return None

    pv_mesh, _ = load_surface_mesh(path)
    return pv_mesh

def load_visualization_inputs(meshes_dir):
    meshes_dir = Path(meshes_dir)

    cm_surf_path = meshes_dir / CM_SURF_NAME
    omm_surf_path = meshes_dir / OMM_SURF_NAME
    ibm_surf_path = meshes_dir / IBM_SURF_NAME
    ermcs_surf_path = meshes_dir / ERMCS_SURF_NAME
    nonermcs_surf_path = meshes_dir / NONERMCS_SURF_NAME
    ribo_vol_path = meshes_dir / RIBO_VOL_NAME

    cm_surf, _ = load_surface_mesh(cm_surf_path)
    omm_surf, _ = load_surface_mesh(omm_surf_path)
    ibm_surf, _ = load_surface_mesh(ibm_surf_path)
    ermcs_surf, _ = load_surface_mesh(ermcs_surf_path)
    nonermcs_surf, _ = load_surface_mesh(nonermcs_surf_path)
    ribo_surf = load_optional_surface_mesh(ribo_vol_path)

    return omm_surf, ibm_surf, cm_surf, ermcs_surf, nonermcs_surf, ribo_surf

# ============================================================
# CAMERA ORIENTATION / FIXED FOV
# ============================================================
def get_plane_config(plane):
    """
    Returns:
        axis_a, axis_b, normal_axis

    axis_a and axis_b define the displayed plane.
    normal_axis is the camera normal.
    """
    plane = plane.lower()

    if plane == "xy":
        return 0, 1, 2, "X", "Y", "Z"

    if plane == "xz":
        return 0, 2, 1, "X", "Z", "Y"

    if plane == "yz":
        return 1, 2, 0, "Y", "Z", "X"

    raise ValueError(f"Unknown plane: {plane}")

def finite_points(points):
    points = np.asarray(points, dtype=float)
    return points[np.all(np.isfinite(points), axis=1)]

def points_from_meshes(meshes):
    arrays = []

    for mesh in meshes:
        if mesh is None:
            continue

        pts = np.asarray(mesh.points, dtype=float)
        pts = finite_points(pts)

        if len(pts) > 0:
            arrays.append(pts)

    if not arrays:
        raise RuntimeError("No valid points found.")

    return np.vstack(arrays)

def get_points_for_orientation(
    omm_surf,
    ibm_surf,
    cm_surf,
    ermcs_surf,
    nonermcs_surf,
    ribo_surf,
    use_combined=False,
):
    """
    Return points used to calculate the longest in-plane landscape axis.
    Default uses CM only because CM is the main object being shown.
    """
    if use_combined:
        pts = points_from_meshes([
            omm_surf,
            ibm_surf,
            cm_surf,
        ])
    else:
        pts = np.asarray(cm_surf.points, dtype=float)
        pts = finite_points(pts)

    if len(pts) < 3:
        raise RuntimeError("Not enough points to compute orientation.")

    return pts

def get_points_for_center(
    omm_surf,
    ibm_surf,
    cm_surf,
    ermcs_surf,
    nonermcs_surf,
    ribo_surf,
    center_to="all",
):
    """
    Return points used to center the fixed-FOV image.

    center_to='all'   -> OMM + IBM + CM + ERMCS + nonERMCS
    center_to='mito'  -> OMM + IBM + CM
    center_to='cm'    -> CM only
    center_to='er'    -> ERMCS + nonERMCS
    center_to='ermcs' -> ERMCS only

    Ribo_vol is rendered when available but does not change centering.
    """
    if center_to == "all":
        pts = points_from_meshes([
            omm_surf,
            ibm_surf,
            cm_surf,
            ermcs_surf,
            nonermcs_surf,
        ])

    elif center_to == "mito":
        pts = points_from_meshes([
            omm_surf,
            ibm_surf,
            cm_surf,
        ])

    elif center_to == "cm":
        pts = np.asarray(cm_surf.points, dtype=float)
        pts = finite_points(pts)

    elif center_to == "er":
        pts = points_from_meshes([
            ermcs_surf,
            nonermcs_surf,
        ])

    elif center_to == "ermcs":
        pts = np.asarray(ermcs_surf.points, dtype=float)
        pts = finite_points(pts)

    else:
        raise ValueError(f"Unknown center_to: {center_to}")

    if len(pts) < 3:
        raise RuntimeError("Not enough points to center camera.")

    return pts

def longest_axis_in_plane_pca(points, axis_a, axis_b):
    """
    Find longest axis within selected plane using PCA.

    Returns:
        world-space unit vector representing screenshot-horizontal direction.
    """
    plane_coords = np.asarray(points[:, [axis_a, axis_b]], dtype=float)
    plane_coords = plane_coords - np.mean(plane_coords, axis=0)

    cov = np.cov(plane_coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    longest_2d = eigvecs[:, np.argmax(eigvals)]
    longest_2d = longest_2d / np.linalg.norm(longest_2d)

    if longest_2d[0] < 0:
        longest_2d = -longest_2d

    right = np.zeros(3, dtype=float)
    right[axis_a] = float(longest_2d[0])
    right[axis_b] = float(longest_2d[1])
    right = right / np.linalg.norm(right)

    return right, longest_2d

def apply_plane_landscape_camera_fixed_fov(
    plotter,
    orientation_points,
    center_points,
    plane="xy",
    camera_sign="+",
    horizontal_fov_um=2.0,
    aspect=16.0 / 9.0,
):
    """
    Orient camera so:
    - selected plane faces the camera
    - longest in-plane axis appears horizontal
    - image is centered on center_points
    - horizontal field of view is fixed to horizontal_fov_um
    - vertical field of view is horizontal_fov_um / aspect
    """
    orientation_points = np.asarray(orientation_points, dtype=float)
    center_points = np.asarray(center_points, dtype=float)

    axis_a, axis_b, normal_axis, axis_a_name, axis_b_name, normal_name = get_plane_config(plane)

    # Start from center of requested object points.
    center = np.mean(center_points, axis=0)

    bounds_min = np.min(center_points, axis=0)
    bounds_max = np.max(center_points, axis=0)
    extents = bounds_max - bounds_min

    plane_extent = max(extents[axis_a], extents[axis_b])
    normal_extent = extents[normal_axis]

    # Distance does not define scale because parallel projection is used.
    # It only needs to be large enough for clipping.
    distance = max(plane_extent * 3.0, normal_extent * 10.0, horizontal_fov_um * 5.0, 1.0)

    normal = np.zeros(3, dtype=float)
    normal[normal_axis] = 1.0

    right, longest_2d = longest_axis_in_plane_pca(
        orientation_points,
        axis_a=axis_a,
        axis_b=axis_b,
    )

    if camera_sign == "+":
        camera_position = center + normal * distance
        direction = -normal
    else:
        camera_position = center - normal * distance
        direction = normal

    # screen right = direction x view_up
    # therefore view_up = right x direction
    view_up = np.cross(right, direction)
    view_up = view_up / np.linalg.norm(view_up)

    # --------------------------------------------------------
    # Re-center exactly in screen coordinates
    # --------------------------------------------------------
    centered = center_points - center

    x_proj = centered @ right
    y_proj = centered @ view_up

    x_mid = 0.5 * (float(np.min(x_proj)) + float(np.max(x_proj)))
    y_mid = 0.5 * (float(np.min(y_proj)) + float(np.max(y_proj)))

    new_center = center + right * x_mid + view_up * y_mid
    new_camera_position = camera_position + right * x_mid + view_up * y_mid

    plotter.camera_position = [
        tuple(new_camera_position),
        tuple(new_center),
        tuple(view_up),
    ]

    plotter.camera.parallel_projection = True

    # In VTK/PyVista, parallel_scale is half the vertical field of view.
    vertical_fov_um = float(horizontal_fov_um) / float(aspect)
    plotter.camera.parallel_scale = vertical_fov_um / 2.0

    plotter.reset_camera_clipping_range()
    plotter.render()

    x_extent = float(np.max(x_proj) - np.min(x_proj))
    y_extent = float(np.max(y_proj) - np.min(y_proj))

    angle_deg = np.degrees(np.arctan2(longest_2d[1], longest_2d[0]))

    return {
        "plane": plane.upper(),
        "camera_side": f"{camera_sign}{normal_name}",
        "landscape_axes": f"{axis_a_name}{axis_b_name}",
        "longest_axis_angle_deg": float(angle_deg),
        "center_x_extent": x_extent,
        "center_y_extent": y_extent,
        "horizontal_fov_um": float(horizontal_fov_um),
        "vertical_fov_um": float(vertical_fov_um),
        "parallel_scale": float(plotter.camera.parallel_scale),
    }

# ============================================================
# DISPLAY / SAVE
# ============================================================
def make_ermcs_scene(
    omm_surf,
    ibm_surf,
    cm_surf,
    ermcs_surf,
    nonermcs_surf,
    ribo_surf=None,
    cm_color=CRISTAE_COLOR_RGB,
    ermcs_color=ERMCS_COLOR_RGB,
    nonermcs_color=NONERMCS_COLOR_RGB,
    ribo_color=RIBO_COLOR_RGB,
    omm_color="#C8C8C8",
    ibm_color="#8A8A8A",
    omm_opacity=0.25,
    ibm_opacity=0.25,
    cm_opacity=1.0,
    ermcs_opacity=1.0,
    nonermcs_opacity=0.2,
    ribo_opacity=1.0,
    window_size=(1920, 1080),
    plane="xy",
    camera_sign="+",
    horizontal_fov_um=2.0,
    center_to="all",
    use_combined_orientation=False,
    off_screen=True,
):
    """
    Create PyVista scene with white background and selected plane-facing fixed-FOV camera.
    """
    plotter = pv.Plotter(
        notebook=False,
        off_screen=off_screen,
        window_size=window_size,
    )

    plotter.background_color = "white"

    # Keep rest as before
    plotter.add_mesh(
        omm_surf,
        color=omm_color,
        opacity=omm_opacity,
        smooth_shading=True,
    )

    plotter.add_mesh(
        ibm_surf,
        color=ibm_color,
        opacity=ibm_opacity,
        smooth_shading=True,
    )

    plotter.add_mesh(
        cm_surf,
        color=cm_color,
        opacity=cm_opacity,
        smooth_shading=True,
    )

    # Add ER overlays
    plotter.add_mesh(
        nonermcs_surf,
        color=nonermcs_color,
        opacity=nonermcs_opacity,
        smooth_shading=True,
    )

    plotter.add_mesh(
        ermcs_surf,
        color=ermcs_color,
        opacity=ermcs_opacity,
        smooth_shading=True,
    )

    # Add Ribo_vol when available
    if ribo_surf is not None:
        plotter.add_mesh(
            ribo_surf,
            color=ribo_color,
            opacity=ribo_opacity,
            smooth_shading=True,
        )

    plotter.add_light(
        pv.Light(position=(1, 1, 1), light_type="headlight", intensity=0.4)
    )

    orientation_points = get_points_for_orientation(
        omm_surf=omm_surf,
        ibm_surf=ibm_surf,
        cm_surf=cm_surf,
        ermcs_surf=ermcs_surf,
        nonermcs_surf=nonermcs_surf,
        ribo_surf=ribo_surf,
        use_combined=use_combined_orientation,
    )

    center_points = get_points_for_center(
        omm_surf=omm_surf,
        ibm_surf=ibm_surf,
        cm_surf=cm_surf,
        ermcs_surf=ermcs_surf,
        nonermcs_surf=nonermcs_surf,
        ribo_surf=ribo_surf,
        center_to=center_to,
    )

    aspect = window_size[0] / window_size[1]

    camera_info = apply_plane_landscape_camera_fixed_fov(
        plotter=plotter,
        orientation_points=orientation_points,
        center_points=center_points,
        plane=plane,
        camera_sign=camera_sign,
        horizontal_fov_um=horizontal_fov_um,
        aspect=aspect,
    )

    return plotter, camera_info

def save_scene_screenshot(plotter, out_path):
    """
    Save current PyVista scene to TIF.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter.render()
    image = plotter.screenshot(return_img=True, transparent_background=True)

    save_tif(out_path, image)

def render_one_job(job, args):
    """
    Render one sample.
    """
    row = {
        "Label": job["label"],
        "ConditionRoot": job["condition_root"],
        "SampleFolder": str(job["sample_dir"]),
        "MeshesFolder": str(job["meshes_dir"]),
        "OutputTIF": str(job["out_path"]),
        "Status": "",
        "Date": "",
        "Condition": "",
        "Mito": "",
        "MissingFiles": job.get("missing_files", ""),
        "RiboRendered": False,
        "Error": "",
        "Traceback": "",
        "Plane": "",
        "CameraSide": "",
        "LandscapeAxes": "",
        "LongestAxisAngleDeg": "",
        "CenterXExtent": "",
        "CenterYExtent": "",
        "HorizontalFOV_um": "",
        "VerticalFOV_um": "",
        "ParallelScale": "",
        "CristaeColorR": CRISTAE_COLOR_RGB[0],
        "CristaeColorG": CRISTAE_COLOR_RGB[1],
        "CristaeColorB": CRISTAE_COLOR_RGB[2],
        "ERMCSColorR": ERMCS_COLOR_RGB[0],
        "ERMCSColorG": ERMCS_COLOR_RGB[1],
        "ERMCSColorB": ERMCS_COLOR_RGB[2],
        "nonERMCSColorR": NONERMCS_COLOR_RGB[0],
        "nonERMCSColorG": NONERMCS_COLOR_RGB[1],
        "nonERMCSColorB": NONERMCS_COLOR_RGB[2],
        "nonERMCSOpacity": args.nonermcs_opacity,
        "RiboColorR": RIBO_COLOR_RGB[0],
        "RiboColorG": RIBO_COLOR_RGB[1],
        "RiboColorB": RIBO_COLOR_RGB[2],
        "RiboOpacity": args.ribo_opacity,
    }

    if job["status"] == "missing":
        row["Status"] = "missing_files"
        return row

    if job["status"] == "skip_existing":
        row["Status"] = "skipped_existing"

        try:
            date_str, condition, mito = parse_metadata_from_folder(job["sample_dir"])
            row["Date"] = date_str
            row["Condition"] = condition
            row["Mito"] = mito
        except Exception:
            pass

        return row

    try:
        date_str, condition, mito = parse_metadata_from_folder(job["sample_dir"])
        row["Date"] = date_str
        row["Condition"] = condition
        row["Mito"] = mito

        omm_surf, ibm_surf, cm_surf, ermcs_surf, nonermcs_surf, ribo_surf = load_visualization_inputs(
            job["meshes_dir"]
        )

        row["RiboRendered"] = ribo_surf is not None

        plotter, camera_info = make_ermcs_scene(
            omm_surf=omm_surf,
            ibm_surf=ibm_surf,
            cm_surf=cm_surf,
            ermcs_surf=ermcs_surf,
            nonermcs_surf=nonermcs_surf,
            ribo_surf=ribo_surf,
            cm_color=CRISTAE_COLOR_RGB,
            ermcs_color=ERMCS_COLOR_RGB,
            nonermcs_color=NONERMCS_COLOR_RGB,
            ribo_color=RIBO_COLOR_RGB,
            omm_opacity=args.omm_opacity,
            ibm_opacity=args.ibm_opacity,
            cm_opacity=args.cm_opacity,
            ermcs_opacity=args.ermcs_opacity,
            nonermcs_opacity=args.nonermcs_opacity,
            ribo_opacity=args.ribo_opacity,
            window_size=(args.width, args.height),
            plane=args.plane,
            camera_sign=args.camera_sign,
            horizontal_fov_um=args.horizontal_fov_um,
            center_to=args.center_to,
            use_combined_orientation=args.use_combined_orientation,
            off_screen=True,
        )

        save_scene_screenshot(plotter, job["out_path"])

        plotter.close()
        pv.close_all()

        row["Status"] = "saved"
        row["Plane"] = camera_info["plane"]
        row["CameraSide"] = camera_info["camera_side"]
        row["LandscapeAxes"] = camera_info["landscape_axes"]
        row["LongestAxisAngleDeg"] = round(camera_info["longest_axis_angle_deg"], 3)
        row["CenterXExtent"] = round(camera_info["center_x_extent"], 3)
        row["CenterYExtent"] = round(camera_info["center_y_extent"], 3)
        row["HorizontalFOV_um"] = round(camera_info["horizontal_fov_um"], 3)
        row["VerticalFOV_um"] = round(camera_info["vertical_fov_um"], 3)
        row["ParallelScale"] = round(camera_info["parallel_scale"], 3)

        del omm_surf, ibm_surf, cm_surf, ermcs_surf, nonermcs_surf, ribo_surf, plotter
        gc.collect()

        return row

    except Exception as e:
        row["Status"] = "failed"
        row["Error"] = str(e)
        row["Traceback"] = traceback.format_exc()

        try:
            pv.close_all()
        except Exception:
            pass

        gc.collect()

        return row

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    if args.root:
        root = Path(args.root).resolve()
    else:
        root = get_default_root()

    out_dir = root / OUT_FOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / MANIFEST_NAME
    failures_path = out_dir / FAILURES_NAME

    aspect = args.width / args.height
    target_aspect = 4.0 / 3.0

    if abs(aspect - target_aspect) > 1e-3:
        raise ValueError(
            f"Screenshot must be 16:9. Got width/height={args.width}/{args.height} "
            f"= {aspect:.4f}. Use 1920x1080, 2560x1440, or 3840x2160."
        )

    if args.horizontal_fov_um <= 0:
        raise ValueError("--horizontal-fov-um must be > 0.")

    vertical_fov_um = args.horizontal_fov_um / aspect

    jobs = discover_jobs(
        root=root,
        out_dir=out_dir,
        overwrite=args.overwrite,
    )

    print("")
    print("Root:", root)
    print("Output folder:", out_dir)
    print("Found jobs:", len(jobs))
    print("Plane:", args.plane.upper())
    print("Camera sign:", args.camera_sign)
    print("Window:", f"{args.width} x {args.height}")
    print("Horizontal FOV:", f"{args.horizontal_fov_um:.4f} um")
    print("Vertical FOV:", f"{vertical_fov_um:.4f} um")
    print("Center to:", args.center_to)
    print("Cristae color RGB:", CRISTAE_COLOR_RGB)
    print("Overwrite:", args.overwrite)
    print("")

    rows = []

    for job in tqdm(jobs, desc="Rendering fixed-FOV ERMCS snaps", unit="mesh", dynamic_ncols=True):
        row = render_one_job(job, args)
        rows.append(row)

        if row["Status"] == "saved":
            tqdm.write(f"Saved {row['Label']} -> {row['OutputTIF']}")
        elif row["Status"] == "skipped_existing":
            tqdm.write(f"Skipped existing {row['Label']}")
        else:
            tqdm.write(f"{row['Status']}: {row['Label']}")

    manifest_df = pd.DataFrame(rows)

    if not manifest_df.empty:
        manifest_df.to_csv(manifest_path, index=False)

        failures_df = manifest_df[
            manifest_df["Status"].isin(["failed", "missing_files"])
        ].copy()

        if not failures_df.empty:
            failures_df.to_csv(failures_path, index=False)

    saved_count = int((manifest_df["Status"] == "saved").sum()) if not manifest_df.empty else 0
    skipped_count = int((manifest_df["Status"] == "skipped_existing").sum()) if not manifest_df.empty else 0
    failed_count = int((manifest_df["Status"] == "failed").sum()) if not manifest_df.empty else 0
    missing_count = int((manifest_df["Status"] == "missing_files").sum()) if not manifest_df.empty else 0

    print("")
    print("Done.")
    print("Saved:", saved_count)
    print("Skipped existing:", skipped_count)
    print("Missing files:", missing_count)
    print("Failed:", failed_count)
    print("Manifest:", manifest_path)

    if failed_count > 0 or missing_count > 0:
        print("Failures:", failures_path)

if __name__ == "__main__":
    main()
