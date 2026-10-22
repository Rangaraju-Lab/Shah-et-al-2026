from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import zipfile
import struct
import re
import csv
import traceback

import numpy as np

# ============================================================
# SETTINGS
# ============================================================
DRY_RUN = False
OVERWRITE = True
USE_PARALLEL = True
MAX_WORKERS = 4

# Keep True to process only the first cell.
TEST_FIRST_FILE = False

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_DIR = SCRIPT_DIR.parent

CENTERS_DIR = PROJECT_DIR / "Undrift Masked SMLM Centers"
NONJUXTA_ROI_DIR = PROJECT_DIR / "NonJuxta ROI"
DENDRITE_DIR = PROJECT_DIR / "Dendrites"
MITO_MASK_DIR = PROJECT_DIR / "Mito Masks"

CSV_OUTPUT_DIR = PROJECT_DIR / "Measurements" / "nonjuxtaspinal"
VIS_OUTPUT_DIR = PROJECT_DIR / "Visualizations" / "nonjuxtaspinal"

PROBE00 = "probe00"
PROBE01 = "probe01"

# probe01 column = counts from input probe00
# probe02 column = counts from input probe01
CSV_COLUMNS = [
    "Date",
    "Dish",
    "Cell",
    "dendrite",
    "non_juxta",
    "probe01",
    "probe02",
    "mito_area_um2",
]

# ============================================================
# PIXEL / BOX GEOMETRY
# ============================================================
PIXEL_SIZE_X_NM = 98.44999694824219
PIXEL_SIZE_Y_NM = 97.9000015258789
PIXEL_SIZE_NM = (PIXEL_SIZE_X_NM + PIXEL_SIZE_Y_NM) / 2.0
PIXEL_SIZE_UM = PIXEL_SIZE_NM / 1000.0

# Box dimensions:
# 2 µm along the dendrite and 4 µm perpendicular to the dendrite.
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

MASK_BLUR_SIGMA_RENDER_PX = float(RENDER_SCALE)
USE_OTSU_THRESHOLD_FOR_RENDERED_MASK = True
INTERPOLATED_MASK_FALLBACK_THRESHOLD = 0.5
MITO_CONTOUR_LEVEL = 0.5
MITO_CONTOUR_THICKNESS_RENDER_PX = 1

BOX_LINE_THICKNESS_RENDER_PX = 2

CHANNEL_NAMES = [
    "Non-juxta ROI centroids",
    "Probe00 centers",
    "Probe01 centers",
    "Mito mask outline",
    "2x4 um boxes",
]

# ============================================================
# FILENAME PARSING
# ============================================================
def normalize_name_text(text):
    text = Path(str(text)).name
    text = re.sub(
        r"\.(hdf5|tiff?|zip|roi)$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()

def parse_hdf5_filename(path):
    """
    Examples:
        2026-01-08 Control Dish1 Cell1 probe00.hdf5
        2026-01-08 Control Dish1 Cell1 probe01.hdf5
    """
    stem = normalize_name_text(path.name)

    match = re.match(
        r"^(?P<Date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<Condition>.+?)\s+"
        r"(?P<Dish>Dish\d+)\s+"
        r"(?P<Cell>Cell\d+)\s+"
        r"(?P<Probe>probe\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    metadata = match.groupdict()
    metadata["Probe"] = metadata["Probe"].lower()
    metadata["prefix"] = (
        f"{metadata['Date']} {metadata['Condition']} "
        f"{metadata['Dish']} {metadata['Cell']}"
    )

    return metadata

def parse_simple_cell_filename(path):
    """
    Examples:
        2026-01-08 Control Dish1 Cell1.zip
        2026-01-08 Control Dish1 Cell1.tif
    """
    stem = normalize_name_text(path.name)

    match = re.match(
        r"^(?P<Date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<Condition>.+?)\s+"
        r"(?P<Dish>Dish\d+)\s+"
        r"(?P<Cell>Cell\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    metadata = match.groupdict()
    metadata["prefix"] = (
        f"{metadata['Date']} {metadata['Condition']} "
        f"{metadata['Dish']} {metadata['Cell']}"
    )

    return metadata

def natural_key(text):
    return [
        int(value) if value.isdigit() else value.lower()
        for value in re.split(r"(\d+)", str(text))
    ]

def parse_integer_label(value, label):
    match = re.search(r"\d+", str(value))

    if not match:
        raise ValueError(
            f"Could not extract an integer from {label}: {value}"
        )

    return int(match.group())

def parse_nonjuxta_number(roi_name, fallback):
    stem = normalize_name_text(roi_name)

    match = re.search(
        r"non[\s_-]*juxta[\s_-]*(\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if not match:
        match = re.search(r"(\d+)$", stem)

    if match:
        return int(match.group(1))

    return int(fallback)

# ============================================================
# FILE DISCOVERY
# ============================================================
def build_probe_index(probe):
    index = {}
    skipped = []

    for path in sorted(CENTERS_DIR.glob(f"* {probe}.hdf5")):
        metadata = parse_hdf5_filename(path)

        if metadata is None:
            skipped.append(path)
            continue

        if metadata["Probe"] != probe:
            skipped.append(path)
            continue

        prefix = metadata["prefix"]

        if prefix in index:
            skipped.append(path)
            continue

        index[prefix] = path

    return index, skipped

def build_simple_index(folder, suffixes):
    index = {}
    skipped = []
    paths = []

    for suffix in suffixes:
        paths.extend(folder.glob(f"*{suffix}"))

    for path in sorted(paths):
        metadata = parse_simple_cell_filename(path)

        if metadata is None:
            skipped.append(path)
            continue

        prefix = metadata["prefix"]

        if prefix in index:
            skipped.append(path)
            continue

        index[prefix] = path

    return index, skipped

def find_nonjuxta_zips():
    output = []
    skipped = []

    for path in sorted(NONJUXTA_ROI_DIR.glob("*.zip")):
        metadata = parse_simple_cell_filename(path)

        if metadata is None:
            skipped.append(path)
        else:
            output.append((path, metadata))

    return output, skipped

def output_csv_path(prefix):
    return CSV_OUTPUT_DIR / f"{prefix}.csv"

def output_tif_path(prefix):
    return VIS_OUTPUT_DIR / f"{prefix}.tif"

# ============================================================
# PICASSO / IMAGE HELPERS
# ============================================================
def load_xy_from_hdf5(path):
    from picasso import io

    locs, _ = io.load_locs(str(path))

    if hasattr(locs, "columns"):
        if "x" not in locs.columns or "y" not in locs.columns:
            raise ValueError(f"Missing x/y columns in {path}")

        x = locs["x"].to_numpy(dtype=float)
        y = locs["y"].to_numpy(dtype=float)

    else:
        names = locs.dtype.names or ()

        if "x" not in names or "y" not in names:
            raise ValueError(f"Missing x/y columns in {path}")

        x = np.asarray(locs["x"], dtype=float)
        y = np.asarray(locs["y"], dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)

    return np.column_stack([x[keep], y[keep]])

def load_binary_image(path, threshold=0):
    import tifffile

    image = tifffile.imread(str(path))
    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D mito mask, got {image.shape} for {path}"
        )

    return image > threshold

# ============================================================
# IMAGEJ ROI READING
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

def roi_header(data):
    if len(data) < 64 or data[:4] != b"Iout":
        raise ValueError("Invalid ImageJ ROI data.")

    return {
        "type": read_u8(data, 6),
        "top": read_i16(data, 8),
        "left": read_i16(data, 10),
        "bottom": read_i16(data, 12),
        "right": read_i16(data, 14),
        "n": read_u16(data, 16),
    }

def roi_coordinate_points(data):
    """
    Returns:
        points_xy, roi_type

    Supported ROI types:
        0  polygon
        1  rectangle
        2  oval
        3  straight line
        4  free line
        5  polyline
        7  freehand
        8  traced
        9  angle
        10 point
    """
    header = roi_header(data)
    roi_type = header["type"]

    if roi_type == 3:
        points = np.array(
            [
                [read_f32(data, 18), read_f32(data, 22)],
                [read_f32(data, 26), read_f32(data, 30)],
            ],
            dtype=np.float64,
        )

        if np.isfinite(points).all():
            return points, roi_type

        return (
            np.array(
                [
                    [header["left"], header["top"]],
                    [header["right"], header["bottom"]],
                ],
                dtype=np.float64,
            ),
            roi_type,
        )

    if roi_type in {0, 4, 5, 7, 8, 9, 10} and header["n"] > 0:
        x_start = 64
        y_start = 64 + 2 * header["n"]

        if len(data) < y_start + 2 * header["n"]:
            raise ValueError(
                f"ROI coordinate data is incomplete for n={header['n']}."
            )

        xs = (
            np.array(
                [
                    read_u16(data, x_start + 2 * i)
                    for i in range(header["n"])
                ],
                dtype=float,
            )
            + header["left"]
        )

        ys = (
            np.array(
                [
                    read_u16(data, y_start + 2 * i)
                    for i in range(header["n"])
                ],
                dtype=float,
            )
            + header["top"]
        )

        return np.column_stack([xs, ys]), roi_type

    # Rectangle, oval, or unsupported ROI:
    # use the center of its bounding box.
    return (
        np.array(
            [
                [
                    (header["left"] + header["right"]) / 2.0,
                    (header["top"] + header["bottom"]) / 2.0,
                ]
            ],
            dtype=np.float64,
        ),
        roi_type,
    )

def polygon_centroid(points):
    points = np.asarray(points, dtype=float)

    if len(points) < 3:
        return points.mean(axis=0)

    x = points[:, 0]
    y = points[:, 1]

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    area_twice = np.sum(cross)

    if (
        not np.isfinite(area_twice)
        or abs(area_twice) < 1e-12
    ):
        return points.mean(axis=0)

    centroid = np.array(
        [
            np.sum((x + x_next) * cross) / (3.0 * area_twice),
            np.sum((y + y_next) * cross) / (3.0 * area_twice),
        ],
        dtype=float,
    )

    if not np.isfinite(centroid).all():
        return points.mean(axis=0)

    return centroid

def read_nonjuxta_roi_zip(path):
    records = []

    with zipfile.ZipFile(path, "r") as zip_file:
        names = sorted(
            [
                name
                for name in zip_file.namelist()
                if name.lower().endswith(".roi")
            ],
            key=natural_key,
        )

        for fallback, name in enumerate(names, start=1):
            data = zip_file.read(name)
            points, roi_type = roi_coordinate_points(data)

            if roi_type in {0, 7, 8} and len(points) >= 3:
                centroid = polygon_centroid(points)
            else:
                centroid = points.mean(axis=0)

            records.append(
                {
                    "non_juxta": int(
                        parse_nonjuxta_number(name, fallback)
                    ),
                    "roi_name": Path(name).name,
                    "centroid_xy": np.asarray(
                        centroid,
                        dtype=float,
                    ),
                }
            )

    if not records:
        raise ValueError(f"No ROI files found in {path}")

    return sorted(
        records,
        key=lambda record: (
            record["non_juxta"],
            natural_key(record["roi_name"]),
        ),
    )

def read_dendrite_zip(path):
    """
    Reads dendrite ROIs in natural-sorted filename order.

    Dendrite IDs are assigned later as 1, 2, 3, ...
    in this exact order.
    """
    polylines = []

    with zipfile.ZipFile(path, "r") as zip_file:
        names = sorted(
            [
                name
                for name in zip_file.namelist()
                if name.lower().endswith(".roi")
            ],
            key=natural_key,
        )

        for name in names:
            points, roi_type = roi_coordinate_points(
                zip_file.read(name)
            )

            if roi_type not in {0, 3, 4, 5, 7, 8, 9}:
                raise ValueError(
                    f"Unsupported dendrite ROI type {roi_type} in {name}"
                )

            if len(points) >= 2:
                polylines.append(points)

    if not polylines:
        raise ValueError(
            f"No usable dendrite line ROIs found in {path}"
        )

    return polylines

# ============================================================
# DENDRITE INTERPOLATION
# ============================================================
def interpolate_dendrites(polylines):
    """
    Interpolates every dendrite and retains the integer ID of the
    dendrite that generated each interpolated point.

    Dendrites are numbered 1, 2, 3, ... in the natural-sorted
    ROI order returned by read_dendrite_zip().
    """
    all_points = []
    all_tangents = []
    all_dendrite_ids = []

    for dendrite_id, points in enumerate(polylines, start=1):
        points = np.asarray(points, dtype=float)

        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]

            vector = p1 - p0
            length = float(np.linalg.norm(vector))

            if length <= 0:
                continue

            tangent = vector / length
            n_steps = max(
                2,
                int(np.ceil(length / INTERP_STEP_PX)) + 1,
            )

            for t in np.linspace(
                0.0,
                1.0,
                n_steps,
                endpoint=False,
            ):
                all_points.append(p0 + t * vector)
                all_tangents.append(tangent)
                all_dendrite_ids.append(dendrite_id)

        last_vector = points[-1] - points[-2]
        last_length = float(np.linalg.norm(last_vector))

        all_points.append(points[-1])

        if last_length > 0:
            all_tangents.append(last_vector / last_length)
        else:
            all_tangents.append(
                np.array([1.0, 0.0], dtype=float)
            )

        all_dendrite_ids.append(dendrite_id)

    if not all_points:
        raise ValueError(
            "No interpolated dendrite points were generated."
        )

    return (
        np.asarray(all_points, dtype=float),
        np.asarray(all_tangents, dtype=float),
        np.asarray(all_dendrite_ids, dtype=np.int64),
    )

def closest_dendrite_points(
    query_xy,
    dendrite_points,
    dendrite_tangents,
    dendrite_ids,
):
    """
    Assigns each non-juxta centroid to its closest interpolated
    dendrite point and returns the associated integer dendrite ID.
    """
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(dendrite_points)
        distances, indices = tree.query(query_xy, k=1)

    except Exception:
        distance_squared = np.sum(
            (
                query_xy[:, None, :]
                - dendrite_points[None, :, :]
            )
            ** 2,
            axis=2,
        )

        indices = np.argmin(distance_squared, axis=1)
        distances = np.sqrt(
            np.min(distance_squared, axis=1)
        )

    centers = dendrite_points[indices]
    tangents = dendrite_tangents[indices]
    assigned_dendrite_ids = dendrite_ids[indices]

    norms = np.linalg.norm(tangents, axis=1)

    tangents = tangents / np.maximum(
        norms[:, None],
        1e-12,
    )

    tangents[norms <= 0] = np.array(
        [1.0, 0.0]
    )

    return (
        centers,
        tangents,
        distances,
        assigned_dendrite_ids,
    )

# ============================================================
# BOX GEOMETRY
# ============================================================
def box_axes(tangent):
    tangent = np.asarray(tangent, dtype=float)

    tangent = tangent / max(
        float(np.linalg.norm(tangent)),
        1e-12,
    )

    normal = np.array(
        [-tangent[1], tangent[0]],
        dtype=float,
    )

    return tangent, normal

def box_corners(center, tangent):
    tangent, normal = box_axes(tangent)
    center = np.asarray(center, dtype=float)

    return np.array(
        [
            center
            + BOX_HALF_ALONG_PX * tangent
            + BOX_HALF_PERP_PX * normal,

            center
            + BOX_HALF_ALONG_PX * tangent
            - BOX_HALF_PERP_PX * normal,

            center
            - BOX_HALF_ALONG_PX * tangent
            - BOX_HALF_PERP_PX * normal,

            center
            - BOX_HALF_ALONG_PX * tangent
            + BOX_HALF_PERP_PX * normal,
        ]
    )

def points_inside_box(points, center, tangent):
    if points is None or len(points) == 0:
        return np.zeros(0, dtype=bool)

    tangent, normal = box_axes(tangent)

    relative = (
        np.asarray(points, dtype=float)
        - np.asarray(center, dtype=float)
    )

    along = relative @ tangent
    perpendicular = relative @ normal

    return (
        (np.abs(along) <= BOX_HALF_ALONG_PX)
        & (np.abs(perpendicular) <= BOX_HALF_PERP_PX)
    )

def make_box_records(
    nonjuxta_records,
    dendrite_points,
    dendrite_tangents,
    dendrite_ids,
):
    centroids = np.asarray(
        [
            record["centroid_xy"]
            for record in nonjuxta_records
        ],
        dtype=float,
    )

    (
        centers,
        tangents,
        distances,
        assigned_dendrite_ids,
    ) = closest_dendrite_points(
        centroids,
        dendrite_points,
        dendrite_tangents,
        dendrite_ids,
    )

    records = []

    for i, record in enumerate(nonjuxta_records):
        records.append(
            {
                "dendrite": int(
                    assigned_dendrite_ids[i]
                ),
                "non_juxta": int(
                    record["non_juxta"]
                ),
                "roi_name": record["roi_name"],
                "roi_centroid_xy": np.asarray(
                    record["centroid_xy"],
                    dtype=float,
                ),
                "center_xy": centers[i],
                "tangent_xy": tangents[i],
                "corners_xy": box_corners(
                    centers[i],
                    tangents[i],
                ),
                "dist_to_dendrite_px": float(
                    distances[i]
                ),
            }
        )

    return records

def mito_area_inside_box(mito_mask, center, tangent):
    height, width = mito_mask.shape
    corners = box_corners(center, tangent)

    xmin = max(
        0,
        int(np.floor(corners[:, 0].min())) - 2,
    )

    xmax = min(
        width - 1,
        int(np.ceil(corners[:, 0].max())) + 2,
    )

    ymin = max(
        0,
        int(np.floor(corners[:, 1].min())) - 2,
    )

    ymax = min(
        height - 1,
        int(np.ceil(corners[:, 1].max())) + 2,
    )

    if xmax < xmin or ymax < ymin:
        return 0.0

    yy, xx = np.mgrid[
        ymin : ymax + 1,
        xmin : xmax + 1,
    ]

    points = np.column_stack(
        [xx.ravel(), yy.ravel()]
    )

    inside = points_inside_box(
        points,
        center,
        tangent,
    ).reshape(yy.shape)

    mito_inside = (
        mito_mask[
            ymin : ymax + 1,
            xmin : xmax + 1,
        ]
        & inside
    )

    return float(
        np.count_nonzero(mito_inside)
        * PIXEL_SIZE_UM**2
    )

# ============================================================
# MEASUREMENTS
# ============================================================
def build_measurement_rows(
    metadata,
    box_records,
    probe00_xy,
    probe01_xy,
    mito_mask,
):
    rows = []

    dish_int = parse_integer_label(
        metadata["Dish"],
        "Dish",
    )

    cell_int = parse_integer_label(
        metadata["Cell"],
        "Cell",
    )

    for record in box_records:
        probe00_inside = points_inside_box(
            probe00_xy,
            record["center_xy"],
            record["tangent_xy"],
        )

        probe01_inside = points_inside_box(
            probe01_xy,
            record["center_xy"],
            record["tangent_xy"],
        )

        rows.append(
            {
                "Date": metadata["Date"],
                "Dish": int(dish_int),
                "Cell": int(cell_int),
                "dendrite": int(
                    record["dendrite"]
                ),
                "non_juxta": int(
                    record["non_juxta"]
                ),
                "probe01": int(
                    np.count_nonzero(
                        probe00_inside
                    )
                ),
                "probe02": int(
                    np.count_nonzero(
                        probe01_inside
                    )
                ),
                "mito_area_um2": (
                    mito_area_inside_box(
                        mito_mask,
                        record["center_xy"],
                        record["tangent_xy"],
                    )
                ),
            }
        )

    return rows

# ============================================================
# RENDERING
# ============================================================
def normalize_render_channel_uint16(
    image,
    brightness=1.0,
    gamma=1.0,
):
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if not np.any(image > 0):
        return np.zeros_like(
            image,
            dtype=np.uint16,
        )

    vmax = max(
        float(
            np.percentile(
                image[image > 0],
                NORMALIZE_PERCENTILE,
            )
        ),
        EPS,
    )

    output = np.clip(image / vmax, 0, 1)

    if gamma > 0:
        output = output**gamma

    output = np.clip(
        output * brightness,
        0,
        1,
    )

    return (output * 65535).astype(np.uint16)

def render_points_float(points, shape):
    from scipy.ndimage import gaussian_filter

    image = np.zeros(
        shape,
        dtype=np.float32,
    )

    if points is None or len(points) == 0:
        return image

    xs = np.round(
        points[:, 0] * RENDER_SCALE
    ).astype(int)

    ys = np.round(
        points[:, 1] * RENDER_SCALE
    ).astype(int)

    keep = (
        (xs >= 0)
        & (xs < shape[1])
        & (ys >= 0)
        & (ys < shape[0])
    )

    np.add.at(
        image,
        (ys[keep], xs[keep]),
        1.0,
    )

    return gaussian_filter(
        image,
        GAUSSIAN_SIGMA_RENDER_PX,
        mode="constant",
    )

def rendered_mito_outline(mito_mask, shape):
    from skimage.transform import resize
    from skimage.filters import threshold_otsu
    from skimage.measure import find_contours
    from skimage.draw import line
    from scipy.ndimage import gaussian_filter, binary_dilation

    mask = resize(
        mito_mask.astype(np.float32),
        shape,
        order=1,
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )

    mask = gaussian_filter(
        mask,
        MASK_BLUR_SIGMA_RENDER_PX,
        mode="constant",
    )

    if (
        USE_OTSU_THRESHOLD_FOR_RENDERED_MASK
        and np.any(mask > 0)
    ):
        threshold = threshold_otsu(
            mask[np.isfinite(mask)]
        )
    else:
        threshold = (
            INTERPOLATED_MASK_FALLBACK_THRESHOLD
        )

    binary = mask >= threshold
    output = np.zeros(shape, dtype=np.uint16)

    for contour in find_contours(
        binary.astype(np.float32),
        MITO_CONTOUR_LEVEL,
    ):
        draw = np.zeros(shape, dtype=bool)

        for i in range(len(contour)):
            p0 = contour[i - 1]
            p1 = contour[i]

            rr, cc = line(
                round(p0[0]),
                round(p0[1]),
                round(p1[0]),
                round(p1[1]),
            )

            keep = (
                (rr >= 0)
                & (rr < shape[0])
                & (cc >= 0)
                & (cc < shape[1])
            )

            draw[rr[keep], cc[keep]] = True

        if MITO_CONTOUR_THICKNESS_RENDER_PX > 1:
            draw = binary_dilation(
                draw,
                iterations=(
                    MITO_CONTOUR_THICKNESS_RENDER_PX
                    - 1
                ),
            )

        output[draw] = 65535

    return output

def rendered_boxes(box_records, shape):
    from skimage.draw import line
    from scipy.ndimage import binary_dilation

    output = np.zeros(shape, dtype=np.uint16)

    for record in box_records:
        corners = (
            record["corners_xy"]
            * RENDER_SCALE
        )

        draw = np.zeros(shape, dtype=bool)

        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]

            rr, cc = line(
                round(p0[1]),
                round(p0[0]),
                round(p1[1]),
                round(p1[0]),
            )

            keep = (
                (rr >= 0)
                & (rr < shape[0])
                & (cc >= 0)
                & (cc < shape[1])
            )

            draw[rr[keep], cc[keep]] = True

        if BOX_LINE_THICKNESS_RENDER_PX > 1:
            draw = binary_dilation(
                draw,
                iterations=(
                    BOX_LINE_THICKNESS_RENDER_PX
                    - 1
                ),
            )

        output[draw] = 65535

    return output

def make_multichannel_render(
    mito_mask,
    nonjuxta_xy,
    probe00_xy,
    probe01_xy,
    box_records,
):
    shape = (
        mito_mask.shape[0] * RENDER_SCALE,
        mito_mask.shape[1] * RENDER_SCALE,
    )

    nonjuxta_channel = (
        normalize_render_channel_uint16(
            render_points_float(
                nonjuxta_xy,
                shape,
            ),
            MOLECULE_BRIGHTNESS_MULTIPLIER,
            MOLECULE_GAMMA,
        )
    )

    probe00_channel = (
        normalize_render_channel_uint16(
            render_points_float(
                probe00_xy,
                shape,
            ),
            MOLECULE_BRIGHTNESS_MULTIPLIER,
            MOLECULE_GAMMA,
        )
    )

    probe01_channel = (
        normalize_render_channel_uint16(
            render_points_float(
                probe01_xy,
                shape,
            ),
            MOLECULE_BRIGHTNESS_MULTIPLIER,
            MOLECULE_GAMMA,
        )
    )

    mito_channel = rendered_mito_outline(
        mito_mask,
        shape,
    )

    box_channel = rendered_boxes(
        box_records,
        shape,
    )

    return np.stack(
        [
            nonjuxta_channel,
            probe00_channel,
            probe01_channel,
            mito_channel,
            box_channel,
        ],
        axis=0,
    )

# ============================================================
# IMAGEJ LUTS
# ============================================================
def make_rgb_lut(
    r_gain,
    g_gain,
    b_gain,
    hot=False,
):
    x = np.arange(
        256,
        dtype=np.float32,
    ) / 255.0

    if hot:
        extra = np.clip(
            (x - 0.55) / 0.45,
            0,
            1,
        )
    else:
        extra = np.zeros_like(x)

    red = np.clip(
        x * r_gain + extra * (r_gain > 0),
        0,
        1,
    )

    green = np.clip(
        x * g_gain + extra * (g_gain > 0),
        0,
        1,
    )

    blue = np.clip(
        x * b_gain + extra * (b_gain > 0),
        0,
        1,
    )

    return np.round(
        np.stack(
            [red, green, blue],
            axis=1,
        )
        * 255
    ).astype(np.uint8)

def make_imagej_luts():
    return np.stack(
        [
            # Non-juxta ROI centroids: yellow
            make_rgb_lut(
                1.35,
                1.35,
                0.0,
                hot=True,
            ).T,

            # Probe00: cyan
            make_rgb_lut(
                0.0,
                1.35,
                1.35,
                hot=True,
            ).T,

            # Probe01: magenta
            make_rgb_lut(
                1.35,
                0.0,
                1.35,
                hot=True,
            ).T,

            # Mito outline: green
            make_rgb_lut(
                0.0,
                1.0,
                0.0,
            ).T,

            # Boxes: white
            make_rgb_lut(
                1.0,
                1.0,
                1.0,
            ).T,
        ],
        axis=0,
    )

# ============================================================
# SAVE HELPERS
# ============================================================
def save_csv(path, rows):
    if DRY_RUN:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)

def save_tif(path, image):
    if DRY_RUN:
        return

    import tifffile

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tifffile.imwrite(
        str(path),
        image,
        imagej=True,
        resolution=(
            OUTPUT_PIXELS_PER_UM,
            OUTPUT_PIXELS_PER_UM,
        ),
        metadata={
            "axes": "CYX",
            "mode": "composite",
            "unit": "um",
            "Labels": CHANNEL_NAMES,
            "LUTs": make_imagej_luts(),
            "Info": (
                "Each non-juxta ROI centroid was mapped "
                "to the closest point on the interpolated "
                "dendritic arbor.\n"
                "The associated integer dendrite ID is "
                "saved in the CSV.\n"
                "Measurements use the original localization "
                "coordinates and original binary mito mask.\n"
                f"BOX_ALONG_DENDRITE_UM="
                f"{BOX_ALONG_DENDRITE_UM}\n"
                f"BOX_PERP_DENDRITE_UM="
                f"{BOX_PERP_DENDRITE_UM}\n"
                f"PIXEL_SIZE_UM={PIXEL_SIZE_UM}\n"
                f"RENDER_SCALE={RENDER_SCALE}\n"
            ),
        },
    )

# ============================================================
# PROCESS ONE CELL
# ============================================================
def process_one_cell(job):
    (
        nonjuxta_zip,
        metadata,
        probe00_hdf,
        probe01_hdf,
        dendrite_zip,
        mito_path,
    ) = job

    prefix = metadata["prefix"]

    out_csv = output_csv_path(prefix)
    out_tif = output_tif_path(prefix)

    result = {
        "prefix": prefix,
        "status": None,
        "n_nonjuxta": None,
        "n_probe00": None,
        "n_probe01": None,
        "n_rows": None,
        "error": None,
    }

    try:
        if (
            (out_csv.exists() or out_tif.exists())
            and not OVERWRITE
        ):
            result["status"] = "skipped_exists"
            return result

        missing = []

        required_files = [
            ("probe00", probe00_hdf),
            ("probe01", probe01_hdf),
            ("dendrite", dendrite_zip),
            ("mito", mito_path),
        ]

        for name, path in required_files:
            if path is None or not Path(path).exists():
                missing.append(name)

        if missing:
            result["status"] = "missing_required_file"
            result["error"] = ";".join(missing)
            return result

        nonjuxta_records = read_nonjuxta_roi_zip(
            nonjuxta_zip
        )

        nonjuxta_xy = np.asarray(
            [
                record["centroid_xy"]
                for record in nonjuxta_records
            ],
            dtype=float,
        )

        probe00_xy = load_xy_from_hdf5(
            probe00_hdf
        )

        probe01_xy = load_xy_from_hdf5(
            probe01_hdf
        )

        mito_mask = load_binary_image(
            mito_path,
            MITO_MASK_THRESHOLD,
        )

        dendrite_polylines = read_dendrite_zip(
            dendrite_zip
        )

        (
            dendrite_points,
            dendrite_tangents,
            dendrite_ids,
        ) = interpolate_dendrites(
            dendrite_polylines
        )

        boxes = make_box_records(
            nonjuxta_records,
            dendrite_points,
            dendrite_tangents,
            dendrite_ids,
        )

        rows = build_measurement_rows(
            metadata,
            boxes,
            probe00_xy,
            probe01_xy,
            mito_mask,
        )

        render = make_multichannel_render(
            mito_mask,
            nonjuxta_xy,
            probe00_xy,
            probe01_xy,
            boxes,
        )

        result["n_nonjuxta"] = int(
            len(nonjuxta_records)
        )

        result["n_probe00"] = int(
            len(probe00_xy)
        )

        result["n_probe01"] = int(
            len(probe01_xy)
        )

        result["n_rows"] = int(len(rows))

        if DRY_RUN:
            result["status"] = "dry_run"
            return result

        save_csv(out_csv, rows)
        save_tif(out_tif, render)

        result["status"] = "saved"
        return result

    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        return result

def print_result(index, total, result):
    print(
        f"[{index}/{total}] "
        f"{result['status']}: "
        f"{result['prefix']} | "
        f"regions={result['n_nonjuxta']} | "
        f"rows={result['n_rows']}"
    )

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("Nonjuxtaspinal DNA-PAINT measurements")
    print("=" * 80)
    print(f"SCRIPT_DIR:         {SCRIPT_DIR}")
    print(f"PROJECT_DIR:        {PROJECT_DIR}")
    print(f"CENTERS_DIR:        {CENTERS_DIR}")
    print(f"NONJUXTA_ROI_DIR:   {NONJUXTA_ROI_DIR}")
    print(f"DENDRITE_DIR:       {DENDRITE_DIR}")
    print(f"MITO_MASK_DIR:      {MITO_MASK_DIR}")
    print(f"CSV_OUTPUT_DIR:     {CSV_OUTPUT_DIR}")
    print(f"VIS_OUTPUT_DIR:     {VIS_OUTPUT_DIR}")
    print(f"CSV_COLUMNS:        {CSV_COLUMNS}")
    print(
        f"BOX:                "
        f"{BOX_ALONG_DENDRITE_UM} µm along × "
        f"{BOX_PERP_DENDRITE_UM} µm perpendicular"
    )
    print(f"DRY_RUN:            {DRY_RUN}")
    print(f"OVERWRITE:          {OVERWRITE}")
    print(f"USE_PARALLEL:       {USE_PARALLEL}")
    print(f"MAX_WORKERS:        {MAX_WORKERS}")
    print(f"TEST_FIRST_FILE:    {TEST_FIRST_FILE}")
    print("=" * 80)

    required_folders = [
        CENTERS_DIR,
        NONJUXTA_ROI_DIR,
        DENDRITE_DIR,
        MITO_MASK_DIR,
    ]

    for folder in required_folders:
        if not folder.exists():
            raise FileNotFoundError(
                f"Folder does not exist: {folder}"
            )

    (
        nonjuxta_files,
        nonjuxta_skipped,
    ) = find_nonjuxta_zips()

    (
        probe00_index,
        probe00_skipped,
    ) = build_probe_index(PROBE00)

    (
        probe01_index,
        probe01_skipped,
    ) = build_probe_index(PROBE01)

    (
        dendrite_index,
        dendrite_skipped,
    ) = build_simple_index(
        DENDRITE_DIR,
        [".zip"],
    )

    (
        mito_index,
        mito_skipped,
    ) = build_simple_index(
        MITO_MASK_DIR,
        [".tif", ".tiff"],
    )

    print(
        f"\nNon-juxta ROI zips: {len(nonjuxta_files)}"
    )
    print(
        f"Probe00 HDF5s:      {len(probe00_index)}"
    )
    print(
        f"Probe01 HDF5s:      {len(probe01_index)}"
    )
    print(
        f"Dendrite zips:      {len(dendrite_index)}"
    )
    print(
        f"Mito masks:         {len(mito_index)}"
    )

    total_skipped = (
        len(nonjuxta_skipped)
        + len(probe00_skipped)
        + len(probe01_skipped)
        + len(dendrite_skipped)
        + len(mito_skipped)
    )

    print(
        f"Skipped filenames:  {total_skipped}"
    )

    if not nonjuxta_files:
        print("\nNo non-juxta ROI ZIP files found.")
        return

    jobs = []

    print("\nFirst few jobs:")

    for nonjuxta_zip, metadata in nonjuxta_files[:20]:
        prefix = metadata["prefix"]

        probe00_hdf = probe00_index.get(prefix)
        probe01_hdf = probe01_index.get(prefix)
        dendrite_zip = dendrite_index.get(prefix)
        mito_path = mito_index.get(prefix)

        print(f"  {prefix}")
        print(f"    non-juxta: {nonjuxta_zip.name}")
        print(
            f"    probe00:   "
            f"{probe00_hdf.name if probe00_hdf else 'MISSING'}"
        )
        print(
            f"    probe01:   "
            f"{probe01_hdf.name if probe01_hdf else 'MISSING'}"
        )
        print(
            f"    dendrite:  "
            f"{dendrite_zip.name if dendrite_zip else 'MISSING'}"
        )
        print(
            f"    mito:      "
            f"{mito_path.name if mito_path else 'MISSING'}"
        )
        print(
            f"    csv:       "
            f"{output_csv_path(prefix).name}"
        )
        print(
            f"    tif:       "
            f"{output_tif_path(prefix).name}"
        )

    if len(nonjuxta_files) > 20:
        print(
            f"  ... plus "
            f"{len(nonjuxta_files) - 20} more"
        )

    for nonjuxta_zip, metadata in nonjuxta_files:
        prefix = metadata["prefix"]

        jobs.append(
            (
                nonjuxta_zip,
                metadata,
                probe00_index.get(prefix),
                probe01_index.get(prefix),
                dendrite_index.get(prefix),
                mito_index.get(prefix),
            )
        )

    if TEST_FIRST_FILE:
        jobs = jobs[:1]
        print(
            "\nTEST_FIRST_FILE=True: "
            "processing only the first cell."
        )

    results = []

    print("\n" + "=" * 80)
    print("PROCESSING")
    print("=" * 80)

    if (
        USE_PARALLEL
        and not DRY_RUN
        and not TEST_FIRST_FILE
    ):
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            future_to_job = {
                executor.submit(
                    process_one_cell,
                    job,
                ): job
                for job in jobs
            }

            for index, future in enumerate(
                as_completed(future_to_job),
                start=1,
            ):
                result = future.result()
                results.append(result)

                print_result(
                    index,
                    len(jobs),
                    result,
                )

    else:
        for index, job in enumerate(
            jobs,
            start=1,
        ):
            result = process_one_cell(job)
            results.append(result)

            print_result(
                index,
                len(jobs),
                result,
            )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    status_counts = Counter(
        result["status"]
        for result in results
    )

    for status, count in status_counts.items():
        print(f"{status}: {count}")

    total_regions = sum(
        int(result["n_rows"] or 0)
        for result in results
    )

    print(
        f"\nTotal regions measured: {total_regions}"
    )
    print(
        f"CSV output:             {CSV_OUTPUT_DIR}"
    )
    print(
        f"Visualization output:   {VIS_OUTPUT_DIR}"
    )

    errors = [
        result
        for result in results
        if result["status"]
        in {
            "error",
            "missing_required_file",
        }
    ]

    if errors:
        print("\n" + "=" * 80)
        print("ERROR / MISSING DETAILS")
        print("=" * 80)

        for result in errors:
            print(
                f"\n{result['prefix']} | "
                f"{result['status']}"
            )
            print(result["error"])

    if DRY_RUN:
        print(
            "\nDRY_RUN enabled. No files written."
        )

    print("\nDone.")

if __name__ == "__main__":
    main()