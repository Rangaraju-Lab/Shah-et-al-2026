# ============================================================
# Render Control 1 and Control 2 together as one ERMCS screenshot
#
# Output:
#   ../ERMCS snaps/Control 1 and 2.tif
#
# Combined from Control 1 + Control 2:
#   OMM_surf.stl
#   IBM_surf.stl
#   CM_surf.stl
#   ERMCS_surf.stl
#   Ribo_vol.stl if available
#
# From Control 1 only:
#   nonERMCS_surf.stl
#
# Orientation:
#   computed from Control 1 only
#
# Centering:
#   computed from the rendered combined scene
#
# Run from Notebooks folder:
#   python render_Control_1_and_2_ERMCS.py --plane yz --horizontal-fov-um 2.0 --overwrite
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

# ============================================================
# DEFAULTS
# ============================================================
MESH_SUBPATH = Path("scaled") / "meshes"

CM_SURF_NAME = "CM_surf.stl"
OMM_SURF_NAME = "OMM_surf.stl"
IBM_SURF_NAME = "IBM_surf.stl"
ERMCS_SURF_NAME = "ERMCS_surf.stl"
NONERMCS_SURF_NAME = "nonERMCS_surf.stl"
RIBO_VOL_NAME = "Ribo_vol.stl"

OUT_FOLDER_NAME = "ERMCS snaps"
OUT_TIF_NAME = "Control 1 and 2.tif"
MANIFEST_NAME = "Control_1_and_2_ERMCS_manifest.csv"
FAILURES_NAME = "Control_1_and_2_ERMCS_failures.csv"

CRISTAE_COLOR_RGB = (0.9686, 0.7686, 0.2627)
ERMCS_COLOR_RGB = (0.0, 0.447, 0.698)
NONERMCS_COLOR_RGB = (0.0, 0.447, 0.698)
RIBO_COLOR_RGB = (0.780, 0.082, 0.522)

# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Render Control 1 and Control 2 together as one fixed-FOV ERMCS screenshot.")

    parser.add_argument("--root", type=str, default="", help="Project root. If omitted, uses the package root. Paths are independent of the working directory.")
    parser.add_argument("--plane", type=str, default="yz", choices=["xy", "xz", "yz"], help="Plane facing the camera.")
    parser.add_argument("--camera-sign", type=str, default="+", choices=["+", "-"], help="Which side of the plane normal to view from.")
    parser.add_argument("--horizontal-fov-um", type=float, default=2.0, help="Horizontal field of view in mesh units.")

    parser.add_argument("--omm-opacity", type=float, default=0.25)
    parser.add_argument("--ibm-opacity", type=float, default=0.25)
    parser.add_argument("--cm-opacity", type=float, default=1.0)
    parser.add_argument("--ermcs-opacity", type=float, default=1.0)
    parser.add_argument("--nonermcs-opacity", type=float, default=0.3)
    parser.add_argument("--ribo-opacity", type=float, default=1.0)

    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1440)

    parser.add_argument("--center-to", type=str, default="all", choices=["all", "mito", "cm", "er", "ermcs"])
    parser.add_argument("--use-combined-orientation", action="store_true", help="Use Control 1 OMM+IBM+CM for orientation. Default uses Control 1 CM only.")
    parser.add_argument("--overwrite", action="store_true", default=True)

    return parser.parse_args()

# ============================================================
# PATH / METADATA HELPERS
# ============================================================
def get_default_root():
    return Path(__file__).resolve().parent.parent

def parse_metadata_from_folder(sample_dir):
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

def mesh_paths(meshes_dir):
    meshes_dir = Path(meshes_dir)
    return {
        "cm": meshes_dir / CM_SURF_NAME,
        "omm": meshes_dir / OMM_SURF_NAME,
        "ibm": meshes_dir / IBM_SURF_NAME,
        "ermcs": meshes_dir / ERMCS_SURF_NAME,
        "nonermcs": meshes_dir / NONERMCS_SURF_NAME,
        "ribo": meshes_dir / RIBO_VOL_NAME,
    }

def discover_control_1_and_2(root, out_dir, overwrite=True):
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()

    control_root = root / "Control"
    if not control_root.exists():
        raise FileNotFoundError(f"Missing Control folder: {control_root}")

    sample_dirs = sorted(
        [p for p in control_root.iterdir() if p.is_dir()],
        key=sample_sort_key,
    )

    if len(sample_dirs) < 2:
        raise RuntimeError(f"Need at least two direct sample folders under: {control_root}")

    control1_dir = sample_dirs[0]
    control2_dir = sample_dirs[1]

    control1_meshes_dir = control1_dir / MESH_SUBPATH
    control2_meshes_dir = control2_dir / MESH_SUBPATH

    out_path = out_dir / OUT_TIF_NAME

    job = {
        "label": "Control 1 and 2",
        "condition_root": "Control",
        "sample_dirs": [control1_dir, control2_dir],
        "meshes_dirs": [control1_meshes_dir, control2_meshes_dir],
        "out_path": out_path,
        "status": "ready",
        "missing_files": "",
    }

    missing = []

    if not control1_meshes_dir.exists():
        missing.append("Control 1: meshes folder missing")
    if not control2_meshes_dir.exists():
        missing.append("Control 2: meshes folder missing")

    if not missing:
        p1 = mesh_paths(control1_meshes_dir)
        p2 = mesh_paths(control2_meshes_dir)

        required = [
            p1["cm"],
            p1["omm"],
            p1["ibm"],
            p1["ermcs"],
            p1["nonermcs"],
            p2["cm"],
            p2["omm"],
            p2["ibm"],
            p2["ermcs"],
        ]

        for p in required:
            if not p.exists():
                if p.parent == control1_meshes_dir:
                    missing.append(f"Control 1: {p.name}")
                elif p.parent == control2_meshes_dir:
                    missing.append(f"Control 2: {p.name}")
                else:
                    missing.append(str(p))

    if missing:
        job["status"] = "missing"
        job["missing_files"] = "; ".join(missing)
    elif out_path.exists() and not overwrite:
        job["status"] = "skip_existing"

    return job

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

    raise RuntimeError("Could not save TIF. Install tifffile with: pip install tifffile")

# ============================================================
# MESH LOADING
# ============================================================
def load_trimesh_surface(path):
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
    return pv_mesh

def load_optional_surface_mesh(path):
    path = Path(path)

    if not path.exists():
        return None

    return load_surface_mesh(path)

def combine_pyvista_meshes(meshes):
    meshes = [m for m in meshes if m is not None]

    if len(meshes) == 0:
        return None

    if len(meshes) == 1:
        return meshes[0]

    combined = pv.MultiBlock(meshes).combine(merge_points=False)

    if combined.n_points == 0 or combined.n_cells == 0:
        raise ValueError("Combined PyVista mesh is empty.")

    return combined

def load_control_1_and_2_inputs(meshes_dirs):
    control1_meshes_dir = Path(meshes_dirs[0])
    control2_meshes_dir = Path(meshes_dirs[1])

    p1 = mesh_paths(control1_meshes_dir)
    p2 = mesh_paths(control2_meshes_dir)

    omm_1 = load_surface_mesh(p1["omm"])
    omm_2 = load_surface_mesh(p2["omm"])

    ibm_1 = load_surface_mesh(p1["ibm"])
    ibm_2 = load_surface_mesh(p2["ibm"])

    cm_1 = load_surface_mesh(p1["cm"])
    cm_2 = load_surface_mesh(p2["cm"])

    ermcs_1 = load_surface_mesh(p1["ermcs"])
    ermcs_2 = load_surface_mesh(p2["ermcs"])

    # Requested: nonERMCS only from Control 1
    nonermcs_1 = load_surface_mesh(p1["nonermcs"])

    ribo_1 = load_optional_surface_mesh(p1["ribo"])
    ribo_2 = load_optional_surface_mesh(p2["ribo"])

    # Rendered scene meshes
    omm_surf = combine_pyvista_meshes([omm_1, omm_2])
    ibm_surf = combine_pyvista_meshes([ibm_1, ibm_2])
    cm_surf = combine_pyvista_meshes([cm_1, cm_2])
    ermcs_surf = combine_pyvista_meshes([ermcs_1, ermcs_2])
    nonermcs_surf = nonermcs_1
    ribo_surf = combine_pyvista_meshes([ribo_1, ribo_2])

    # Orientation meshes: Control 1 only
    orientation_omm_surf = omm_1
    orientation_ibm_surf = ibm_1
    orientation_cm_surf = cm_1

    return (
        omm_surf,
        ibm_surf,
        cm_surf,
        ermcs_surf,
        nonermcs_surf,
        ribo_surf,
        orientation_omm_surf,
        orientation_ibm_surf,
        orientation_cm_surf,
    )

# ============================================================
# CAMERA ORIENTATION / FIXED FOV
# ============================================================
def get_plane_config(plane):
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
    use_combined=False,
):
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
    aspect=4.0 / 3.0,
):
    orientation_points = np.asarray(orientation_points, dtype=float)
    center_points = np.asarray(center_points, dtype=float)

    axis_a, axis_b, normal_axis, axis_a_name, axis_b_name, normal_name = get_plane_config(plane)

    center = np.mean(center_points, axis=0)

    bounds_min = np.min(center_points, axis=0)
    bounds_max = np.max(center_points, axis=0)
    extents = bounds_max - bounds_min

    plane_extent = max(extents[axis_a], extents[axis_b])
    normal_extent = extents[normal_axis]

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

    view_up = np.cross(right, direction)
    view_up = view_up / np.linalg.norm(view_up)

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
    orientation_omm_surf=None,
    orientation_ibm_surf=None,
    orientation_cm_surf=None,
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
    window_size=(1920, 1440),
    plane="xy",
    camera_sign="+",
    horizontal_fov_um=2.0,
    center_to="all",
    use_combined_orientation=False,
    off_screen=True,
):
    plotter = pv.Plotter(
        notebook=False,
        off_screen=off_screen,
        window_size=window_size,
    )

    plotter.background_color = "white"

    plotter.add_mesh(omm_surf, color=omm_color, opacity=omm_opacity, smooth_shading=True)
    plotter.add_mesh(ibm_surf, color=ibm_color, opacity=ibm_opacity, smooth_shading=True)
    plotter.add_mesh(cm_surf, color=cm_color, opacity=cm_opacity, smooth_shading=True)

    plotter.add_mesh(nonermcs_surf, color=nonermcs_color, opacity=nonermcs_opacity, smooth_shading=True)
    plotter.add_mesh(ermcs_surf, color=ermcs_color, opacity=ermcs_opacity, smooth_shading=True)

    if ribo_surf is not None:
        plotter.add_mesh(ribo_surf, color=ribo_color, opacity=ribo_opacity, smooth_shading=True)

    plotter.add_light(
        pv.Light(position=(1, 1, 1), light_type="headlight", intensity=0.4)
    )

    # Orientation uses Control 1-only meshes.
    orientation_omm_surf = orientation_omm_surf if orientation_omm_surf is not None else omm_surf
    orientation_ibm_surf = orientation_ibm_surf if orientation_ibm_surf is not None else ibm_surf
    orientation_cm_surf = orientation_cm_surf if orientation_cm_surf is not None else cm_surf

    orientation_points = get_points_for_orientation(
        omm_surf=orientation_omm_surf,
        ibm_surf=orientation_ibm_surf,
        cm_surf=orientation_cm_surf,
        use_combined=use_combined_orientation,
    )

    # Centering still uses the combined rendered scene.
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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter.render()
    image = plotter.screenshot(return_img=True, transparent_background=True)

    save_tif(out_path, image)

def render_control_1_and_2(job, args):
    row = {
        "Label": job["label"],
        "ConditionRoot": job["condition_root"],
        "SampleFolder": "; ".join([str(p) for p in job["sample_dirs"]]),
        "MeshesFolder": "; ".join([str(p) for p in job["meshes_dirs"]]),
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
        "OrientationSource": "Control 1 only",
        "nonERMCSSource": "Control 1 only",
    }

    dates = []
    mitos = []

    for sample_dir in job["sample_dirs"]:
        try:
            date_str, condition, mito = parse_metadata_from_folder(sample_dir)
            dates.append(date_str)
            mitos.append(str(mito))
        except Exception:
            pass

    row["Date"] = "+".join(dates)
    row["Condition"] = "Control"
    row["Mito"] = "+".join(mitos)

    if job["status"] == "missing":
        row["Status"] = "missing_files"
        return row

    if job["status"] == "skip_existing":
        row["Status"] = "skipped_existing"
        return row

    try:
        (
            omm_surf,
            ibm_surf,
            cm_surf,
            ermcs_surf,
            nonermcs_surf,
            ribo_surf,
            orientation_omm_surf,
            orientation_ibm_surf,
            orientation_cm_surf,
        ) = load_control_1_and_2_inputs(job["meshes_dirs"])

        row["RiboRendered"] = ribo_surf is not None

        plotter, camera_info = make_ermcs_scene(
            omm_surf=omm_surf,
            ibm_surf=ibm_surf,
            cm_surf=cm_surf,
            ermcs_surf=ermcs_surf,
            nonermcs_surf=nonermcs_surf,
            ribo_surf=ribo_surf,
            orientation_omm_surf=orientation_omm_surf,
            orientation_ibm_surf=orientation_ibm_surf,
            orientation_cm_surf=orientation_cm_surf,
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

        del omm_surf, ibm_surf, cm_surf, ermcs_surf, nonermcs_surf, ribo_surf
        del orientation_omm_surf, orientation_ibm_surf, orientation_cm_surf, plotter
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
            f"Screenshot must be 4:3. Got width/height={args.width}/{args.height} "
            f"= {aspect:.4f}. Use 1920x1440, 2560x1920, or 3840x2880."
        )

    if args.horizontal_fov_um <= 0:
        raise ValueError("--horizontal-fov-um must be > 0.")

    vertical_fov_um = args.horizontal_fov_um / aspect

    job = discover_control_1_and_2(
        root=root,
        out_dir=out_dir,
        overwrite=args.overwrite,
    )

    print("")
    print("Root:", root)
    print("Output folder:", out_dir)
    print("Output TIF:", job["out_path"])
    print("Control 1 folder:", job["sample_dirs"][0])
    print("Control 2 folder:", job["sample_dirs"][1])
    print("Control 1 meshes:", job["meshes_dirs"][0])
    print("Control 2 meshes:", job["meshes_dirs"][1])
    print("Plane:", args.plane.upper())
    print("Camera sign:", args.camera_sign)
    print("Window:", f"{args.width} x {args.height}")
    print("Horizontal FOV:", f"{args.horizontal_fov_um:.4f} um")
    print("Vertical FOV:", f"{vertical_fov_um:.4f} um")
    print("Center to:", args.center_to)
    print("Cristae color RGB:", CRISTAE_COLOR_RGB)
    print("Rendered OMM/IBM/CM/ERMCS/Ribo source: Control 1 + Control 2")
    print("nonERMCS source: Control 1 only")
    print("Orientation source: Control 1 only")
    print("Use combined orientation within Control 1:", args.use_combined_orientation)
    print("Overwrite:", args.overwrite)
    print("")

    row = render_control_1_and_2(job, args)

    manifest_df = pd.DataFrame([row])
    manifest_df.to_csv(manifest_path, index=False)

    if row["Status"] in ["failed", "missing_files"]:
        manifest_df.to_csv(failures_path, index=False)

    if row["Status"] == "saved":
        print(f"Saved {row['Label']} -> {row['OutputTIF']}")
    elif row["Status"] == "skipped_existing":
        print(f"Skipped existing {row['Label']} -> {row['OutputTIF']}")
    else:
        print(f"{row['Status']}: {row['Label']}")
        print("Error:", row.get("Error", ""))
        print("Missing files:", row.get("MissingFiles", ""))

    print("")
    print("Done.")
    print("Status:", row["Status"])
    print("Manifest:", manifest_path)

    if row["Status"] in ["failed", "missing_files"]:
        print("Failures:", failures_path)

if __name__ == "__main__":
    main()
