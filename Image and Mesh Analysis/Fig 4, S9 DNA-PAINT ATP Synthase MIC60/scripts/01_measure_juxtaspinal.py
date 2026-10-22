from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import traceback
import zipfile
import struct
import re
import csv

import numpy as np

# ============================================================
# SETTINGS
# ============================================================
DRY_RUN = False
OVERWRITE = True

USE_PARALLEL = True
MAX_WORKERS = 4

# Keep True for testing first file. Set False to run all.
TEST_FIRST_FILE = False

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_DIR = SCRIPT_DIR.parent

CENTERS_DIR = PROJECT_DIR / "Undrift Masked SMLM Centers"
SYNAPSE_DIR = PROJECT_DIR / "Synapse Localizations"
DENDRITE_DIR = PROJECT_DIR / "Dendrites"
MITO_MASK_DIR = PROJECT_DIR / "Mito Masks"

CSV_OUTPUT_DIR = PROJECT_DIR / "Measurements" / "juxtaspinal"
VIS_OUTPUT_DIR = PROJECT_DIR / "Visualizations" / "juxtaspinal"

PROBE00 = "probe00"
PROBE01 = "probe01"
SYNAPSE_PROBE = "probe02"

# Output CSV columns exactly as requested.
# probe01 column = counts from input probe00
# probe02 column = counts from input probe01
CSV_COLUMNS = ["Date", "Dish", "Cell", "dendrite", "spine", "probe01", "probe02", "mito_area_um2"]

# ============================================================
# PIXEL / BOX GEOMETRY
# ============================================================
PIXEL_SIZE_X_NM = 98.44999694824219
PIXEL_SIZE_Y_NM = 97.9000015258789
PIXEL_SIZE_NM = (PIXEL_SIZE_X_NM + PIXEL_SIZE_Y_NM) / 2.0
PIXEL_SIZE_UM = PIXEL_SIZE_NM / 1000.0

# Box centered on nearest interpolated dendrite point.
# 2 um along dendrite, 4 um perpendicular to dendrite.
BOX_ALONG_DENDRITE_UM = 2.0
BOX_PERP_DENDRITE_UM = 4.0

BOX_HALF_ALONG_PX = (BOX_ALONG_DENDRITE_UM / PIXEL_SIZE_UM) / 2.0
BOX_HALF_PERP_PX = (BOX_PERP_DENDRITE_UM / PIXEL_SIZE_UM) / 2.0

INTERP_STEP_UM = 0.05
INTERP_STEP_PX = INTERP_STEP_UM / PIXEL_SIZE_UM

MITO_MASK_THRESHOLD = 0

# ============================================================
# RENDER SETTINGS
# ============================================================
RENDER_SCALE = 10

OUTPUT_PIXEL_SIZE_UM = PIXEL_SIZE_UM / RENDER_SCALE
OUTPUT_PIXELS_PER_UM = 1.0 / OUTPUT_PIXEL_SIZE_UM

GAUSSIAN_SIGMA_RENDER_PX = 1.2
NORMALIZE_PERCENTILE = 98.5
MOLECULE_BRIGHTNESS_MULTIPLIER = 1.0
MOLECULE_GAMMA = 0.65
EPS = 1e-9

# Smooth mito mask for visualization outline only.
MASK_BLUR_SIGMA_RENDER_PX = float(RENDER_SCALE)
USE_OTSU_THRESHOLD_FOR_RENDERED_MASK = True
INTERPOLATED_MASK_FALLBACK_THRESHOLD = 0.5
MITO_CONTOUR_LEVEL = 0.5
MITO_CONTOUR_THICKNESS_RENDER_PX = 1

BOX_LINE_THICKNESS_RENDER_PX = 2

CHANNEL_NAMES = [
    "Synapse probe02",
    "Probe00 centers",
    "Probe01 centers",
    "Mito mask outline",
    "2x4 um boxes",
]

# ============================================================
# PARSING
# ============================================================
def normalize_name_text(text):
    text = str(text)
    text = Path(text).name
    text = re.sub(r"\.hdf5$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.tiff?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.zip$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_hdf5_filename(path: Path):
    """
    2026-01-08 Control Dish1 Cell1 probe02.hdf5
    2026-01-08 Homeostatic Scaling Dish1 Cell1 probe02.hdf5
    """
    stem = normalize_name_text(path.name)

    m = re.match(
        r"^(?P<Date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<Condition>.+?)\s+"
        r"(?P<Dish>Dish\d+)\s+"
        r"(?P<Cell>Cell\d+)\s+"
        r"(?P<Probe>probe\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if not m:
        return None

    d = m.groupdict()
    prefix = f"{d['Date']} {d['Condition']} {d['Dish']} {d['Cell']}"

    return {"Date": d["Date"], "Condition": d["Condition"], "Dish": d["Dish"], "Cell": d["Cell"], "Probe": d["Probe"], "prefix": prefix}

def parse_simple_cell_filename(path: Path):
    """
    2026-01-08 Control Dish1 Cell1.zip
    2026-01-08 Control Dish1 Cell1.tif
    """
    stem = normalize_name_text(path.name)

    m = re.match(
        r"^(?P<Date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<Condition>.+?)\s+"
        r"(?P<Dish>Dish\d+)\s+"
        r"(?P<Cell>Cell\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if not m:
        return None

    d = m.groupdict()
    prefix = f"{d['Date']} {d['Condition']} {d['Dish']} {d['Cell']}"

    return {"Date": d["Date"], "Condition": d["Condition"], "Dish": d["Dish"], "Cell": d["Cell"], "prefix": prefix}

def output_csv_path_for_prefix(prefix):
    return CSV_OUTPUT_DIR / f"{prefix}.csv"

def output_tif_path_for_prefix(prefix):
    return VIS_OUTPUT_DIR / f"{prefix}.tif"

# ============================================================
# FILE DISCOVERY
# ============================================================
def find_synapse_hdfs():
    files = sorted(SYNAPSE_DIR.glob(f"* {SYNAPSE_PROBE}.hdf5"))
    out = []
    skipped = []

    for p in files:
        meta = parse_hdf5_filename(p)

        if meta is None:
            skipped.append((p, "unrecognized_filename"))
            continue

        if meta["Probe"] != SYNAPSE_PROBE:
            skipped.append((p, f"wrong_probe_{meta['Probe']}"))
            continue

        out.append((p, meta))

    return out, skipped

def build_probe_index(folder: Path, probe: str):
    files = sorted(folder.glob(f"* {probe}.hdf5"))
    index = {}
    skipped = []

    for p in files:
        meta = parse_hdf5_filename(p)

        if meta is None:
            skipped.append((p, "unrecognized_filename"))
            continue

        if meta["Probe"] != probe:
            skipped.append((p, f"wrong_probe_{meta['Probe']}"))
            continue

        prefix = meta["prefix"]

        if prefix in index:
            skipped.append((p, "duplicate_prefix"))
            continue

        index[prefix] = p

    return index, skipped

def build_simple_file_index(folder: Path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(sorted(folder.glob(f"*{suffix}")))

    index = {}
    skipped = []

    for p in files:
        meta = parse_simple_cell_filename(p)

        if meta is None:
            skipped.append((p, "unrecognized_filename"))
            continue

        prefix = meta["prefix"]

        if prefix in index:
            skipped.append((p, "duplicate_prefix"))
            continue

        index[prefix] = p

    return index, skipped

# ============================================================
# PICASSO / HDF HELPERS
# ============================================================
def has_column(locs, colname: str) -> bool:
    if hasattr(locs, "columns"):
        return colname in locs.columns

    if hasattr(locs, "dtype") and locs.dtype.names is not None:
        return colname in locs.dtype.names

    return False

def get_column(locs, colname: str):
    if hasattr(locs, "columns"):
        return locs[colname].to_numpy()

    return locs[colname]

def get_column_names(locs):
    if hasattr(locs, "columns"):
        return list(locs.columns)

    if hasattr(locs, "dtype") and locs.dtype.names is not None:
        return list(locs.dtype.names)

    return []

def load_xy_from_hdf5(path: Path):
    from picasso import io

    locs, info = io.load_locs(str(path))

    if not has_column(locs, "x") or not has_column(locs, "y"):
        raise ValueError(f"Missing x/y columns in {path.name}. Available columns: {get_column_names(locs)}")

    x = get_column(locs, "x").astype(float)
    y = get_column(locs, "y").astype(float)

    finite = np.isfinite(x) & np.isfinite(y)

    x = x[finite]
    y = y[finite]

    return np.column_stack([x, y])

# ============================================================
# IMAGE HELPERS
# ============================================================
def load_binary_image(path: Path, threshold=0):
    import tifffile

    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {arr.shape} for {path}")

    return (arr > threshold).astype(bool)

def resize_mask_to_render_float(mask, out_shape_hw):
    from skimage.transform import resize

    out = resize(
        mask.astype(np.float32),
        output_shape=out_shape_hw,
        order=1,
        mode="constant",
        cval=0,
        clip=True,
        preserve_range=True,
        anti_aliasing=False,
    )

    return out.astype(np.float32)

def smooth_and_otsu_rendered_mask(mito_mask, out_shape_hw):
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    mask_render = resize_mask_to_render_float(mito_mask, out_shape_hw=out_shape_hw)

    if MASK_BLUR_SIGMA_RENDER_PX and MASK_BLUR_SIGMA_RENDER_PX > 0:
        mask_blur = gaussian_filter(mask_render.astype(np.float32), sigma=float(MASK_BLUR_SIGMA_RENDER_PX), mode="constant")
    else:
        mask_blur = mask_render.astype(np.float32)

    if not np.any(mask_blur > 0):
        return np.zeros(out_shape_hw, dtype=bool), mask_blur

    if USE_OTSU_THRESHOLD_FOR_RENDERED_MASK:
        try:
            otsu_thr = float(threshold_otsu(mask_blur[np.isfinite(mask_blur)]))
        except Exception:
            otsu_thr = float(INTERPOLATED_MASK_FALLBACK_THRESHOLD)
    else:
        otsu_thr = float(INTERPOLATED_MASK_FALLBACK_THRESHOLD)

    mask_final = mask_blur >= otsu_thr

    return mask_final.astype(bool), mask_blur.astype(np.float32)

# ============================================================
# IMAGEJ ROI READING FOR DENDRITES
# ============================================================
def read_u8(data, offset):
    if len(data) < offset + 1:
        return 0
    return int(data[offset])

def read_u16(data, offset):
    if len(data) < offset + 2:
        return 0
    return int(struct.unpack_from(">H", data, offset)[0])

def read_i16(data, offset):
    if len(data) < offset + 2:
        return 0
    return int(struct.unpack_from(">h", data, offset)[0])

def read_f32(data, offset):
    if len(data) < offset + 4:
        return 0.0
    return float(struct.unpack_from(">f", data, offset)[0])

def parse_imagej_roi_polyline(data: bytes):
    """
    Returns Nx2 x/y points from common ImageJ dendrite line/polyline ROIs.

    Supported:
        0 polygon
        3 straight line
        4 free line
        5 polyline / segmented line
        7 freehand
        8 traced
        9 angle
    """
    if len(data) < 64:
        raise ValueError("ROI byte data is shorter than 64-byte ImageJ header.")

    if data[0:4] != b"Iout":
        raise ValueError("ROI does not start with ImageJ magic 'Iout'.")

    roi_type = read_u8(data, 6)
    top = read_i16(data, 8)
    left = read_i16(data, 10)
    bottom = read_i16(data, 12)
    right = read_i16(data, 14)
    n_coords = read_u16(data, 16)

    if roi_type == 3:
        x1 = read_f32(data, 18)
        y1 = read_f32(data, 22)
        x2 = read_f32(data, 26)
        y2 = read_f32(data, 30)

        if np.isfinite([x1, y1, x2, y2]).all() and not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
            return np.array([[x1, y1], [x2, y2]], dtype=np.float64)

        return np.array([[left, top], [right, bottom]], dtype=np.float64)

    if roi_type in {0, 4, 5, 7, 8, 9} and n_coords > 0:
        x_start = 64
        y_start = 64 + 2 * n_coords

        if len(data) < y_start + 2 * n_coords:
            raise ValueError(f"ROI coordinate data shorter than expected for n_coords={n_coords}.")

        xs = np.array([read_u16(data, x_start + 2 * i) for i in range(n_coords)], dtype=np.float64) + float(left)
        ys = np.array([read_u16(data, y_start + 2 * i) for i in range(n_coords)], dtype=np.float64) + float(top)

        return np.column_stack([xs, ys])

    raise ValueError(f"Unsupported dendrite ROI type {roi_type}.")

def read_dendrite_zip(zip_path: Path):
    polylines = []

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        roi_names = sorted([name for name in zf.namelist() if name.lower().endswith(".roi")])

        for roi_name in roi_names:
            data = zf.read(roi_name)

            try:
                pts = parse_imagej_roi_polyline(data)
            except Exception as e:
                raise ValueError(f"Failed reading dendrite ROI {roi_name} in {zip_path}:\n{repr(e)}") from e

            if pts.shape[0] >= 2:
                polylines.append({"roi_name": roi_name, "points": pts})

    if len(polylines) == 0:
        raise ValueError(f"No usable dendrite line/polyline ROIs found in {zip_path}")

    return polylines

# ============================================================
# DENDRITE INTERPOLATION / GEOMETRY
# ============================================================
def interpolate_dendrites(polylines):
    all_points = []
    all_tangents = []
    all_dendrite_ids = []

    # Dendrites are numbered 1, 2, 3, ... in the sorted ROI-file order
    # returned by read_dendrite_zip().
    for dendrite_id, poly in enumerate(polylines, start=1):
        pts = np.asarray(poly["points"], dtype=np.float64)

        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            vec = p1 - p0
            length = float(np.linalg.norm(vec))

            if length <= 0:
                continue

            tangent = vec / length
            n_steps = max(2, int(np.ceil(length / INTERP_STEP_PX)) + 1)
            ts = np.linspace(0.0, 1.0, n_steps, endpoint=False)

            for t in ts:
                all_points.append(p0 + t * vec)
                all_tangents.append(tangent)
                all_dendrite_ids.append(dendrite_id)

        last_vec = pts[-1] - pts[-2]
        last_len = float(np.linalg.norm(last_vec))

        all_points.append(pts[-1])

        if last_len > 0:
            all_tangents.append(last_vec / last_len)
        else:
            all_tangents.append(np.array([1.0, 0.0], dtype=np.float64))

        all_dendrite_ids.append(dendrite_id)

    if len(all_points) == 0:
        raise ValueError("No interpolated dendrite points were generated.")

    return (
        np.asarray(all_points, dtype=np.float64),
        np.asarray(all_tangents, dtype=np.float64),
        np.asarray(all_dendrite_ids, dtype=np.int64),
    )

def closest_dendrite_points(synapse_xy, dendrite_points, dendrite_tangents, dendrite_ids):
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(dendrite_points)
        distances, indices = tree.query(synapse_xy, k=1)
    except Exception:
        diffs = synapse_xy[:, None, :] - dendrite_points[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        indices = np.argmin(d2, axis=1)
        distances = np.sqrt(np.min(d2, axis=1))

    centers = dendrite_points[indices]
    tangents = dendrite_tangents[indices]
    assigned_dendrite_ids = dendrite_ids[indices]

    norms = np.linalg.norm(tangents, axis=1)
    bad = norms <= 0
    norms[bad] = 1.0
    tangents = tangents / norms[:, None]
    tangents[bad] = np.array([1.0, 0.0], dtype=np.float64)

    return centers, tangents, distances, assigned_dendrite_ids

def box_axes_from_tangent(tangent_xy):
    t = np.asarray(tangent_xy, dtype=np.float64)
    t = t / max(float(np.linalg.norm(t)), 1e-12)

    n = np.array([-t[1], t[0]], dtype=np.float64)

    return t, n

def oriented_box_corners(center_xy, tangent_xy):
    center = np.asarray(center_xy, dtype=np.float64)
    t, n = box_axes_from_tangent(tangent_xy)

    corners = np.array(
        [
            center + (+BOX_HALF_ALONG_PX * t) + (+BOX_HALF_PERP_PX * n),
            center + (+BOX_HALF_ALONG_PX * t) + (-BOX_HALF_PERP_PX * n),
            center + (-BOX_HALF_ALONG_PX * t) + (-BOX_HALF_PERP_PX * n),
            center + (-BOX_HALF_ALONG_PX * t) + (+BOX_HALF_PERP_PX * n),
        ],
        dtype=np.float64,
    )

    return corners

def points_inside_oriented_box(points_xy, center_xy, tangent_xy):
    if points_xy is None or points_xy.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    t, n = box_axes_from_tangent(tangent_xy)

    rel = points_xy.astype(np.float64) - np.asarray(center_xy, dtype=np.float64)[None, :]

    along = rel @ t
    perp = rel @ n

    inside = (np.abs(along) <= BOX_HALF_ALONG_PX) & (np.abs(perp) <= BOX_HALF_PERP_PX)

    return inside

def make_box_records(synapse_xy, dendrite_points, dendrite_tangents, dendrite_ids):
    centers, tangents, distances, assigned_dendrite_ids = closest_dendrite_points(
        synapse_xy=synapse_xy,
        dendrite_points=dendrite_points,
        dendrite_tangents=dendrite_tangents,
        dendrite_ids=dendrite_ids,
    )

    records = []

    for i, syn in enumerate(synapse_xy, start=1):
        center_xy = centers[i - 1]
        tangent_xy = tangents[i - 1]
        corners_xy = oriented_box_corners(center_xy, tangent_xy)

        records.append(
            {
                "dendrite": int(assigned_dendrite_ids[i - 1]),
                "spine": i,
                "synapse_xy": syn.astype(np.float64),
                "center_xy": center_xy.astype(np.float64),
                "tangent_xy": tangent_xy.astype(np.float64),
                "corners_xy": corners_xy.astype(np.float64),
                "dist_to_dendrite_px": float(distances[i - 1]),
            }
        )

    return records

def mito_area_inside_oriented_box(mito_mask, center_xy, tangent_xy):
    h, w = mito_mask.shape

    corners = oriented_box_corners(center_xy, tangent_xy)

    xmin = max(0, int(np.floor(np.min(corners[:, 0]))) - 2)
    xmax = min(w - 1, int(np.ceil(np.max(corners[:, 0]))) + 2)
    ymin = max(0, int(np.floor(np.min(corners[:, 1]))) - 2)
    ymax = min(h - 1, int(np.ceil(np.max(corners[:, 1]))) + 2)

    if xmax < xmin or ymax < ymin:
        return 0.0

    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]

    pts = np.column_stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)])
    inside = points_inside_oriented_box(pts, center_xy, tangent_xy).reshape(yy.shape)

    mito_inside = mito_mask[ymin : ymax + 1, xmin : xmax + 1] & inside

    area_um2 = float(np.count_nonzero(mito_inside) * (PIXEL_SIZE_UM ** 2))

    return area_um2

def build_measurement_rows(meta, box_records, probe00_xy, probe01_xy, mito_mask):
    rows = []

    dish_int = int(re.search(r"\d+", str(meta["Dish"])).group())
    cell_int = int(re.search(r"\d+", str(meta["Cell"])).group())

    for rec in box_records:
        center_xy = rec["center_xy"]
        tangent_xy = rec["tangent_xy"]

        in_probe00 = points_inside_oriented_box(probe00_xy, center_xy, tangent_xy)
        in_probe01 = points_inside_oriented_box(probe01_xy, center_xy, tangent_xy)

        mito_area_um2 = mito_area_inside_oriented_box(mito_mask, center_xy, tangent_xy)

        rows.append(
            {
                "Date": meta["Date"],
                "Dish": dish_int,
                "Cell": cell_int,
                "dendrite": int(rec["dendrite"]),
                "spine": int(rec["spine"]),
                "probe01": int(np.count_nonzero(in_probe00)),
                "probe02": int(np.count_nonzero(in_probe01)),
                "mito_area_um2": mito_area_um2,
            }
        )

    return rows

# ============================================================
# RENDERING 16-BIT CHANNELS
# ============================================================
def normalize_render_channel_uint16(img, brightness_multiplier=1.0, gamma=1.0):
    img = img.astype(np.float32, copy=False)

    if not np.any(img > 0):
        return np.zeros_like(img, dtype=np.uint16)

    vmax = np.percentile(img[img > 0], NORMALIZE_PERCENTILE)
    vmax = max(float(vmax), EPS)

    out = np.clip(img / vmax, 0, 1)

    if gamma is not None and gamma > 0:
        out = out ** float(gamma)

    out = out * float(brightness_multiplier)
    out = np.clip(out, 0, 1)

    out = (out * 65535.0).astype(np.uint16)

    return out

def render_points_float(points_xy, out_shape_hw):
    from scipy.ndimage import gaussian_filter

    h, w = out_shape_hw
    img = np.zeros((h, w), dtype=np.float32)

    if points_xy is None or points_xy.shape[0] == 0:
        return img

    xs = np.round(points_xy[:, 0].astype(float) * RENDER_SCALE).astype(int)
    ys = np.round(points_xy[:, 1].astype(float) * RENDER_SCALE).astype(int)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    xs = xs[valid]
    ys = ys[valid]

    if xs.size == 0:
        return img

    np.add.at(img, (ys, xs), 1.0)

    if GAUSSIAN_SIGMA_RENDER_PX and GAUSSIAN_SIGMA_RENDER_PX > 0:
        img = gaussian_filter(img, sigma=float(GAUSSIAN_SIGMA_RENDER_PX), mode="constant")

    return img.astype(np.float32)

def draw_line_uint16(channel, p0_yx, p1_yx, value=65535, thickness=1):
    from skimage.draw import line
    from scipy.ndimage import binary_dilation

    h, w = channel.shape

    y0 = int(round(float(p0_yx[0])))
    x0 = int(round(float(p0_yx[1])))
    y1 = int(round(float(p1_yx[0])))
    x1 = int(round(float(p1_yx[1])))

    rr, cc = line(y0, x0, y1, x1)

    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)

    rr = rr[valid]
    cc = cc[valid]

    if rr.size == 0:
        return

    if int(thickness) <= 1:
        channel[rr, cc] = np.maximum(channel[rr, cc], np.uint16(value))
        return

    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True

    mask = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=int(thickness) - 1)

    channel[mask] = np.maximum(channel[mask], np.uint16(value))

def draw_contours_from_binary_mask_uint16(binary_mask):
    from skimage.measure import find_contours

    h, w = binary_mask.shape
    out = np.zeros((h, w), dtype=np.uint16)

    mask_float = binary_mask.astype(np.float32)
    contours = find_contours(mask_float, level=MITO_CONTOUR_LEVEL)

    for contour in contours:
        if contour.shape[0] < 2:
            continue

        contour_yx = contour.astype(float)

        for i in range(contour_yx.shape[0] - 1):
            draw_line_uint16(
                channel=out,
                p0_yx=contour_yx[i],
                p1_yx=contour_yx[i + 1],
                value=65535,
                thickness=MITO_CONTOUR_THICKNESS_RENDER_PX,
            )

        draw_line_uint16(
            channel=out,
            p0_yx=contour_yx[-1],
            p1_yx=contour_yx[0],
            value=65535,
            thickness=MITO_CONTOUR_THICKNESS_RENDER_PX,
        )

    return out

def draw_boxes_uint16(box_records, out_shape_hw):
    h, w = out_shape_hw
    out = np.zeros((h, w), dtype=np.uint16)

    for rec in box_records:
        corners = rec["corners_xy"].astype(np.float64) * float(RENDER_SCALE)

        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]

            draw_line_uint16(
                channel=out,
                p0_yx=np.array([p0[1], p0[0]], dtype=np.float64),
                p1_yx=np.array([p1[1], p1[0]], dtype=np.float64),
                value=65535,
                thickness=BOX_LINE_THICKNESS_RENDER_PX,
            )

    return out

def make_multichannel_render(mito_mask, synapse_xy, probe00_xy, probe01_xy, box_records):
    h0, w0 = mito_mask.shape
    h = int(h0 * RENDER_SCALE)
    w = int(w0 * RENDER_SCALE)
    out_shape_hw = (h, w)

    syn_img = render_points_float(synapse_xy, out_shape_hw)
    p00_img = render_points_float(probe00_xy, out_shape_hw)
    p01_img = render_points_float(probe01_xy, out_shape_hw)

    syn_u16 = normalize_render_channel_uint16(syn_img, brightness_multiplier=MOLECULE_BRIGHTNESS_MULTIPLIER, gamma=MOLECULE_GAMMA)
    p00_u16 = normalize_render_channel_uint16(p00_img, brightness_multiplier=MOLECULE_BRIGHTNESS_MULTIPLIER, gamma=MOLECULE_GAMMA)
    p01_u16 = normalize_render_channel_uint16(p01_img, brightness_multiplier=MOLECULE_BRIGHTNESS_MULTIPLIER, gamma=MOLECULE_GAMMA)

    mito_mask_final, mito_mask_blurred = smooth_and_otsu_rendered_mask(mito_mask=mito_mask, out_shape_hw=out_shape_hw)
    mito_outline_u16 = draw_contours_from_binary_mask_uint16(binary_mask=mito_mask_final)

    box_u16 = draw_boxes_uint16(box_records=box_records, out_shape_hw=out_shape_hw)

    out = np.stack([syn_u16, p00_u16, p01_u16, mito_outline_u16, box_u16], axis=0)

    return out

# ============================================================
# LUTS
# ============================================================
def make_rgb_lut(r_gain, g_gain, b_gain, hot=False):
    x = np.arange(256, dtype=np.float32) / 255.0

    if hot:
        r = np.clip(x * r_gain + np.clip((x - 0.55) / 0.45, 0, 1) * (1 if r_gain > 0 else 0), 0, 1)
        g = np.clip(x * g_gain + np.clip((x - 0.55) / 0.45, 0, 1) * (1 if g_gain > 0 else 0), 0, 1)
        b = np.clip(x * b_gain + np.clip((x - 0.55) / 0.45, 0, 1) * (1 if b_gain > 0 else 0), 0, 1)
    else:
        r = np.clip(x * r_gain, 0, 1)
        g = np.clip(x * g_gain, 0, 1)
        b = np.clip(x * b_gain, 0, 1)

    lut = np.stack([r, g, b], axis=1)

    return np.round(lut * 255).astype(np.uint8)

def make_imagej_luts():
    luts = np.stack(
        [
            make_rgb_lut(1.35, 1.35, 0.0, hot=True).T,   # synapse probe02 = yellow hot
            make_rgb_lut(0.0, 1.35, 1.35, hot=True).T,   # probe00 = cyan hot
            make_rgb_lut(1.35, 0.0, 1.35, hot=True).T,   # probe01 = magenta hot
            make_rgb_lut(0.0, 1.0, 0.0, hot=False).T,    # mito outline = green
            make_rgb_lut(1.0, 1.0, 1.0, hot=False).T,    # boxes = white
        ],
        axis=0,
    ).astype(np.uint8)

    return luts

# ============================================================
# SAVE HELPERS
# ============================================================
def save_measurement_csv(out_csv: Path, rows):
    if DRY_RUN:
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

def save_visualization_tif(out_tif: Path, multichannel):
    if DRY_RUN:
        return

    import tifffile

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(
        str(out_tif),
        multichannel,
        imagej=True,
        resolution=(OUTPUT_PIXELS_PER_UM, OUTPUT_PIXELS_PER_UM),
        metadata={
            "axes": "CYX",
            "mode": "composite",
            "unit": "um",
            "spacing": 1.0,
            "Labels": CHANNEL_NAMES,
            "LUTs": make_imagej_luts(),
            "Info": (
                "ImageJ 5-channel juxtaspinal DNA-PAINT visualization.\n"
                "C1=Synapse probe02 rendered localizations, LUT=yellow hot\n"
                "C2=Probe00 centers rendered localizations, LUT=cyan hot\n"
                "C3=Probe01 centers rendered localizations, LUT=magenta hot\n"
                "C4=Mito mask outline from resized + Gaussian blurred + Otsu thresholded mask, LUT=green\n"
                "C5=2x4 um dendrite-oriented box outlines, LUT=white\n"
                "Localization channels are rendered as Gaussian-blurred puncta.\n"
                "Measurements use original-pixel geometry and original binary mito mask.\n"
                "Each synapse is mapped to the closest point on the interpolated dendritic arbor.\n"
                f"PIXEL_SIZE_X_NM={PIXEL_SIZE_X_NM}\n"
                f"PIXEL_SIZE_Y_NM={PIXEL_SIZE_Y_NM}\n"
                f"PIXEL_SIZE_NM_AVG={PIXEL_SIZE_NM}\n"
                f"PIXEL_SIZE_UM={PIXEL_SIZE_UM}\n"
                f"RENDER_SCALE={RENDER_SCALE}\n"
                f"OUTPUT_PIXEL_SIZE_UM={OUTPUT_PIXEL_SIZE_UM}\n"
                f"BOX_ALONG_DENDRITE_UM={BOX_ALONG_DENDRITE_UM}\n"
                f"BOX_PERP_DENDRITE_UM={BOX_PERP_DENDRITE_UM}\n"
                f"INTERP_STEP_UM={INTERP_STEP_UM}\n"
                f"GAUSSIAN_SIGMA_RENDER_PX={GAUSSIAN_SIGMA_RENDER_PX}\n"
                f"NORMALIZE_PERCENTILE={NORMALIZE_PERCENTILE}\n"
                f"MOLECULE_GAMMA={MOLECULE_GAMMA}\n"
                f"MASK_BLUR_SIGMA_RENDER_PX={MASK_BLUR_SIGMA_RENDER_PX}\n"
            ),
        },
    )

# ============================================================
# PROCESS ONE CELL
# ============================================================
def process_one_cell(job):
    syn_hdf, meta, p00_hdf, p01_hdf, dendrite_zip, mito_mask_path = job

    prefix = meta["prefix"]
    out_csv = output_csv_path_for_prefix(prefix)
    out_tif = output_tif_path_for_prefix(prefix)

    result = {
        "status": None,
        "prefix": prefix,
        "syn_hdf": str(syn_hdf),
        "probe00_hdf": str(p00_hdf) if p00_hdf is not None else "",
        "probe01_hdf": str(p01_hdf) if p01_hdf is not None else "",
        "dendrite_zip": str(dendrite_zip) if dendrite_zip is not None else "",
        "mito_mask": str(mito_mask_path) if mito_mask_path is not None else "",
        "output_csv": str(out_csv),
        "output_tif": str(out_tif),
        "n_synapses": None,
        "n_probe00": None,
        "n_probe01": None,
        "n_rows": None,
        "error": None,
    }

    try:
        if (out_csv.exists() or out_tif.exists()) and not OVERWRITE:
            result["status"] = "skipped_exists"
            return result

        missing = []

        if p00_hdf is None or not Path(p00_hdf).exists():
            missing.append("probe00_hdf")

        if p01_hdf is None or not Path(p01_hdf).exists():
            missing.append("probe01_hdf")

        if dendrite_zip is None or not Path(dendrite_zip).exists():
            missing.append("dendrite_zip")

        if mito_mask_path is None or not Path(mito_mask_path).exists():
            missing.append("mito_mask")

        if missing:
            result["status"] = "missing_required_file"
            result["error"] = ";".join(missing)
            return result

        synapse_xy = load_xy_from_hdf5(syn_hdf)
        probe00_xy = load_xy_from_hdf5(p00_hdf)
        probe01_xy = load_xy_from_hdf5(p01_hdf)

        result["n_synapses"] = int(synapse_xy.shape[0])
        result["n_probe00"] = int(probe00_xy.shape[0])
        result["n_probe01"] = int(probe01_xy.shape[0])

        mito_mask = load_binary_image(mito_mask_path, threshold=MITO_MASK_THRESHOLD)

        dendrite_polylines = read_dendrite_zip(dendrite_zip)
        dendrite_points, dendrite_tangents, dendrite_ids = interpolate_dendrites(dendrite_polylines)

        box_records = make_box_records(
            synapse_xy=synapse_xy,
            dendrite_points=dendrite_points,
            dendrite_tangents=dendrite_tangents,
            dendrite_ids=dendrite_ids,
        )

        rows = build_measurement_rows(
            meta=meta,
            box_records=box_records,
            probe00_xy=probe00_xy,
            probe01_xy=probe01_xy,
            mito_mask=mito_mask,
        )

        result["n_rows"] = int(len(rows))

        multichannel = make_multichannel_render(
            mito_mask=mito_mask,
            synapse_xy=synapse_xy,
            probe00_xy=probe00_xy,
            probe01_xy=probe01_xy,
            box_records=box_records,
        )

        if DRY_RUN:
            result["status"] = "dry_run"
            return result

        save_measurement_csv(out_csv, rows)
        save_visualization_tif(out_tif, multichannel)

        result["status"] = "saved"

        return result

    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        return result

def print_result(i, total, result):
    prefix = result["prefix"]
    status = result["status"]

    if status == "saved":
        print(f"[{i}/{total}] saved: {prefix} | syn={result['n_synapses']} | probe00={result['n_probe00']} | probe01={result['n_probe01']} | rows={result['n_rows']}")
        print(f"    csv: {result['output_csv']}")
        print(f"    tif: {result['output_tif']}")
    elif status == "dry_run":
        print(f"[{i}/{total}] DRY RUN: {prefix} | syn={result['n_synapses']} | probe00={result['n_probe00']} | probe01={result['n_probe01']} | rows={result['n_rows']}")
    elif status == "skipped_exists":
        print(f"[{i}/{total}] SKIP exists: {prefix}")
    elif status == "missing_required_file":
        print(f"[{i}/{total}] missing_required_file: {prefix} | {result['error']}")
    elif status == "error":
        print(f"[{i}/{total}] ERROR: {prefix}")
    else:
        print(f"[{i}/{total}] {status}: {prefix}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("Juxtaspinal DNA-PAINT box measurements + separate-channel rendered visualizations")
    print("=" * 80)
    print(f"SCRIPT_DIR:                  {SCRIPT_DIR}")
    print(f"PROJECT_DIR:                 {PROJECT_DIR}")
    print(f"CENTERS_DIR:                 {CENTERS_DIR}")
    print(f"SYNAPSE_DIR:                 {SYNAPSE_DIR}")
    print(f"DENDRITE_DIR:                {DENDRITE_DIR}")
    print(f"MITO_MASK_DIR:               {MITO_MASK_DIR}")
    print(f"CSV_OUTPUT_DIR:              {CSV_OUTPUT_DIR}")
    print(f"VIS_OUTPUT_DIR:              {VIS_OUTPUT_DIR}")
    print(f"PROBE00:                     {PROBE00}")
    print(f"PROBE01:                     {PROBE01}")
    print(f"SYNAPSE_PROBE:               {SYNAPSE_PROBE}")
    print(f"PIXEL_SIZE_X_NM:             {PIXEL_SIZE_X_NM}")
    print(f"PIXEL_SIZE_Y_NM:             {PIXEL_SIZE_Y_NM}")
    print(f"PIXEL_SIZE_NM_AVG:           {PIXEL_SIZE_NM}")
    print(f"PIXEL_SIZE_UM:               {PIXEL_SIZE_UM}")
    print(f"BOX_ALONG_DENDRITE_UM:       {BOX_ALONG_DENDRITE_UM}")
    print(f"BOX_PERP_DENDRITE_UM:        {BOX_PERP_DENDRITE_UM}")
    print(f"BOX_HALF_ALONG_PX:           {BOX_HALF_ALONG_PX:.3f}")
    print(f"BOX_HALF_PERP_PX:            {BOX_HALF_PERP_PX:.3f}")
    print(f"INTERP_STEP_UM:              {INTERP_STEP_UM}")
    print(f"INTERP_STEP_PX:              {INTERP_STEP_PX:.3f}")
    print(f"RENDER_SCALE:                {RENDER_SCALE}")
    print(f"OUTPUT_PIXEL_SIZE_UM:        {OUTPUT_PIXEL_SIZE_UM}")
    print(f"OUTPUT_PIXELS_PER_UM:        {OUTPUT_PIXELS_PER_UM}")
    print(f"GAUSSIAN_SIGMA_RENDER_PX:    {GAUSSIAN_SIGMA_RENDER_PX}")
    print(f"NORMALIZE_PERCENTILE:        {NORMALIZE_PERCENTILE}")
    print(f"MOLECULE_GAMMA:              {MOLECULE_GAMMA}")
    print(f"CHANNEL_NAMES:               {CHANNEL_NAMES}")
    print(f"TEST_FIRST_FILE:             {TEST_FIRST_FILE}")
    print(f"DRY_RUN:                     {DRY_RUN}")
    print(f"OVERWRITE:                   {OVERWRITE}")
    print(f"USE_PARALLEL:                {USE_PARALLEL}")
    print(f"MAX_WORKERS:                 {MAX_WORKERS}")
    print("=" * 80)

    if not CENTERS_DIR.exists():
        raise FileNotFoundError(f"Centers folder does not exist: {CENTERS_DIR}")

    if not SYNAPSE_DIR.exists():
        raise FileNotFoundError(f"Synapse folder does not exist: {SYNAPSE_DIR}")

    if not DENDRITE_DIR.exists():
        raise FileNotFoundError(f"Dendrites folder does not exist: {DENDRITE_DIR}")

    if not MITO_MASK_DIR.exists():
        raise FileNotFoundError(f"Mito Masks folder does not exist: {MITO_MASK_DIR}")

    synapse_files, syn_skipped = find_synapse_hdfs()
    probe00_index, probe00_skipped = build_probe_index(CENTERS_DIR, PROBE00)
    probe01_index, probe01_skipped = build_probe_index(CENTERS_DIR, PROBE01)
    dendrite_index, dendrite_skipped = build_simple_file_index(DENDRITE_DIR, suffixes=[".zip"])
    mito_index, mito_skipped = build_simple_file_index(MITO_MASK_DIR, suffixes=[".tif", ".tiff"])

    print(f"\nSynapse probe02 files: {len(synapse_files)}")
    print(f"Probe00 files:         {len(probe00_index)}")
    print(f"Probe01 files:         {len(probe01_index)}")
    print(f"Dendrite zip files:    {len(dendrite_index)}")
    print(f"Mito mask files:       {len(mito_index)}")

    print(f"\nSkipped synapse files: {len(syn_skipped)}")
    print(f"Skipped probe00 files: {len(probe00_skipped)}")
    print(f"Skipped probe01 files: {len(probe01_skipped)}")
    print(f"Skipped dendrite zips: {len(dendrite_skipped)}")
    print(f"Skipped mito masks:    {len(mito_skipped)}")

    if len(synapse_files) == 0:
        print("\nNo synapse probe02 HDF5 files found.")
        return

    jobs = []

    print("\nFirst few jobs:")
    for syn_hdf, meta in synapse_files[:20]:
        prefix = meta["prefix"]

        p00_hdf = probe00_index.get(prefix)
        p01_hdf = probe01_index.get(prefix)
        dendrite_zip = dendrite_index.get(prefix)
        mito_mask_path = mito_index.get(prefix)

        print(f"  {prefix}")
        print(f"    synapse:  {syn_hdf.name}")
        print(f"    probe00:  {p00_hdf.name if p00_hdf else 'MISSING'}")
        print(f"    probe01:  {p01_hdf.name if p01_hdf else 'MISSING'}")
        print(f"    dendrite: {dendrite_zip.name if dendrite_zip else 'MISSING'}")
        print(f"    mito:     {mito_mask_path.name if mito_mask_path else 'MISSING'}")
        print(f"    csv:      {output_csv_path_for_prefix(prefix).name}")
        print(f"    tif:      {output_tif_path_for_prefix(prefix).name}")

    if len(synapse_files) > 20:
        print(f"  ... plus {len(synapse_files) - 20} more")

    for syn_hdf, meta in synapse_files:
        prefix = meta["prefix"]
        jobs.append(
            (
                syn_hdf,
                meta,
                probe00_index.get(prefix),
                probe01_index.get(prefix),
                dendrite_index.get(prefix),
                mito_index.get(prefix),
            )
        )

    if TEST_FIRST_FILE:
        jobs = jobs[:1]
        print("\nTEST_FIRST_FILE=True, processing only the first job.")

    print("\n" + "=" * 80)
    print("Processing measurements and visualizations")
    print("=" * 80)

    results = []

    if USE_PARALLEL and not DRY_RUN and not TEST_FIRST_FILE:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_one_cell, job) for job in jobs]

            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print_result(i, len(jobs), result)
    else:
        for i, job in enumerate(jobs, start=1):
            result = process_one_cell(job)
            results.append(result)
            print_result(i, len(jobs), result)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    counts = Counter(r["status"] for r in results)

    for status, n in counts.items():
        print(f"{status}: {n}")

    total_rows = sum(int(r["n_rows"]) for r in results if r.get("n_rows") is not None)

    print(f"\nTotal rows/spines measured: {total_rows}")
    print(f"CSV output folder:          {CSV_OUTPUT_DIR}")
    print(f"Visualization folder:       {VIS_OUTPUT_DIR}")

    errors = [r for r in results if r["status"] == "error"]

    if errors:
        print("\n" + "=" * 80)
        print("ERROR DETAILS")
        print("=" * 80)

        for r in errors:
            print(f"\nCELL: {r['prefix']}")
            print(r["error"])

    missing = [r for r in results if r["status"] == "missing_required_file"]

    if missing:
        print("\n" + "=" * 80)
        print("MISSING FILE DETAILS")
        print("=" * 80)

        for r in missing[:50]:
            print(f"{r['prefix']}: {r['error']}")

        if len(missing) > 50:
            print(f"... plus {len(missing) - 50} more")

    if DRY_RUN:
        print("\nDRY_RUN enabled. No files written.")

    print("\nDone.")

if __name__ == "__main__":
    main()