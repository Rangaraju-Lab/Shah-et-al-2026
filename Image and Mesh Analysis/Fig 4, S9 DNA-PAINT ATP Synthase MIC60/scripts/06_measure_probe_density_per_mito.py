from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import traceback
import zipfile
import struct
import re

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
DRY_RUN = False
OVERWRITE = True

USE_PARALLEL = True
MAX_WORKERS = 4

# True = process only the first matched image set.
TEST_FIRST_FILE = False

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_DIR = SCRIPT_DIR.parent

CENTERS_DIR = PROJECT_DIR / "Undrift Masked SMLM Centers"
MITO_MASK_DIR = PROJECT_DIR / "Mito Masks"
MITO_SKEL_DIR = PROJECT_DIR / "Mito Skels"
DENDRITE_DIR = PROJECT_DIR / "Dendrites"

OUTPUT_DIR = PROJECT_DIR / "Measurements" / "molecules_per_mito"

PROBE00 = "probe00"
PROBE01 = "probe01"

# ============================================================
# PIXEL CALIBRATION
# ============================================================
PIXEL_SIZE_X_NM = 98.44999694824219
PIXEL_SIZE_Y_NM = 97.9000015258789

PIXEL_SIZE_X_UM = PIXEL_SIZE_X_NM / 1000.0
PIXEL_SIZE_Y_UM = PIXEL_SIZE_Y_NM / 1000.0

PIXEL_SIZE_UM = (
    PIXEL_SIZE_X_UM
    + PIXEL_SIZE_Y_UM
) / 2.0

PIXEL_AREA_UM2 = (
    PIXEL_SIZE_X_UM
    * PIXEL_SIZE_Y_UM
)

DIAGONAL_PIXEL_LENGTH_UM = float(
    np.hypot(
        PIXEL_SIZE_X_UM,
        PIXEL_SIZE_Y_UM,
    )
)

# ============================================================
# MASK / SKELETON / DENDRITE SETTINGS
# ============================================================
MITO_MASK_THRESHOLD = 0
MITO_SKEL_THRESHOLD = 0

# 1 = 4-connected components
# 2 = 8-connected components
MITO_CONNECTIVITY = 2

# Remove connected components smaller than this number of pixels.
# Set to 1 to retain every component.
MIN_MITO_AREA_PIXELS = 1

# Picasso x/y coordinates are converted to image pixels.
# Options: "round" or "floor"
MOLECULE_PIXEL_MAPPING = "round"

# Dendrite interpolation spacing.
INTERP_STEP_UM = 0.05
INTERP_STEP_PX = INTERP_STEP_UM / PIXEL_SIZE_UM

# ============================================================
# OUTPUT COLUMNS
# ============================================================
OUTPUT_COLUMNS = [
    "Date",
    "Condition",
    "Dish",
    "Cell",
    "dendrite",
    "mito",
    "mito_area_um2",
    "skeleton_length_um",
    "probe00",
    "probe01",
]

# ============================================================
# FILENAME PARSING
# ============================================================
def normalize_name_text(text):
    text = Path(str(text)).name

    text = re.sub(
        r"\.(hdf5|tiff?|png|zip|roi)$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

def natural_key(text):
    return [
        int(value)
        if value.isdigit()
        else value.lower()
        for value in re.split(
            r"(\d+)",
            str(text),
        )
    ]

def parse_hdf5_filename(path):
    """
    Expected examples:

        2026-01-08 Control Dish1 Cell1 probe00.hdf5
        2026-01-08 Control Dish1 Cell1 probe01.hdf5
        2026-01-08 Homeostatic Scaling Dish1 Cell1 probe00.hdf5
    """
    match = re.match(
        r"^(?P<Date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<Condition>.+?)\s+"
        r"(?P<Dish>Dish\d+)\s+"
        r"(?P<Cell>Cell\d+)\s+"
        r"(?P<Probe>probe\d+)$",
        normalize_name_text(path.name),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    metadata = match.groupdict()
    metadata["Probe"] = metadata["Probe"].lower()

    metadata["prefix"] = (
        f"{metadata['Date']} "
        f"{metadata['Condition']} "
        f"{metadata['Dish']} "
        f"{metadata['Cell']}"
    )

    return metadata

def parse_simple_cell_filename(path):
    """
    Expected examples:

        2026-01-08 Control Dish1 Cell1.tif
        2026-01-08 Control Dish1 Cell1.zip
        2026-01-08 Homeostatic Scaling Dish1 Cell1.tif
    """
    match = re.match(
        r"^(?P<Date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<Condition>.+?)\s+"
        r"(?P<Dish>Dish\d+)\s+"
        r"(?P<Cell>Cell\d+)$",
        normalize_name_text(path.name),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    metadata = match.groupdict()

    metadata["prefix"] = (
        f"{metadata['Date']} "
        f"{metadata['Condition']} "
        f"{metadata['Dish']} "
        f"{metadata['Cell']}"
    )

    return metadata

def parse_integer_label(value, label):
    match = re.search(
        r"\d+",
        str(value),
    )

    if not match:
        raise ValueError(
            f"Could not extract an integer "
            f"from {label}: {value}"
        )

    return int(
        match.group()
    )

def output_csv_path(prefix):
    return OUTPUT_DIR / f"{prefix}.csv"

# ============================================================
# FILE DISCOVERY
# ============================================================
def find_image_files(folder):
    files = []

    for suffix in [
        ".tif",
        ".tiff",
        ".TIF",
        ".TIFF",
    ]:
        files.extend(
            folder.glob(
                f"*{suffix}"
            )
        )

    return sorted(
        set(files),
        key=lambda path: natural_key(
            path.name
        ),
    )

def build_probe_index(probe):
    index = {}
    skipped = []

    files = sorted(
        CENTERS_DIR.glob(
            f"* {probe}.hdf5"
        ),
        key=lambda path: natural_key(
            path.name
        ),
    )

    for path in files:
        metadata = parse_hdf5_filename(
            path
        )

        if metadata is None:
            skipped.append(
                (
                    path,
                    "unrecognized filename",
                )
            )
            continue

        if metadata["Probe"] != probe:
            skipped.append(
                (
                    path,
                    f"wrong probe: "
                    f"{metadata['Probe']}",
                )
            )
            continue

        prefix = metadata["prefix"]

        if prefix in index:
            skipped.append(
                (
                    path,
                    "duplicate prefix",
                )
            )
            continue

        index[prefix] = path

    return index, skipped

def build_image_index(folder):
    index = {}
    skipped = []

    for path in find_image_files(
        folder
    ):
        metadata = parse_simple_cell_filename(
            path
        )

        if metadata is None:
            skipped.append(
                (
                    path,
                    "unrecognized filename",
                )
            )
            continue

        prefix = metadata["prefix"]

        if prefix in index:
            skipped.append(
                (
                    path,
                    "duplicate prefix",
                )
            )
            continue

        index[prefix] = path

    return index, skipped

def build_dendrite_index():
    index = {}
    skipped = []

    files = sorted(
        DENDRITE_DIR.glob("*.zip"),
        key=lambda path: natural_key(
            path.name
        ),
    )

    for path in files:
        metadata = parse_simple_cell_filename(
            path
        )

        if metadata is None:
            skipped.append(
                (
                    path,
                    "unrecognized filename",
                )
            )
            continue

        prefix = metadata["prefix"]

        if prefix in index:
            skipped.append(
                (
                    path,
                    "duplicate prefix",
                )
            )
            continue

        index[prefix] = path

    return index, skipped

def find_mito_mask_files():
    found = []
    skipped = []

    for path in find_image_files(
        MITO_MASK_DIR
    ):
        metadata = parse_simple_cell_filename(
            path
        )

        if metadata is None:
            skipped.append(
                (
                    path,
                    "unrecognized filename",
                )
            )
            continue

        found.append(
            (
                path,
                metadata,
            )
        )

    return found, skipped

# ============================================================
# PICASSO HDF5 READING
# ============================================================
def load_xy_from_hdf5(path):
    from picasso import io

    localizations, _ = io.load_locs(
        str(path)
    )

    if hasattr(
        localizations,
        "columns",
    ):
        if (
            "x" not in localizations.columns
            or "y" not in localizations.columns
        ):
            raise ValueError(
                f"Missing x/y columns in "
                f"{path.name}. Available columns: "
                f"{list(localizations.columns)}"
            )

        x = localizations[
            "x"
        ].to_numpy(
            dtype=float
        )

        y = localizations[
            "y"
        ].to_numpy(
            dtype=float
        )

    else:
        column_names = (
            localizations.dtype.names
            or ()
        )

        if (
            "x" not in column_names
            or "y" not in column_names
        ):
            raise ValueError(
                f"Missing x/y columns in "
                f"{path.name}. Available columns: "
                f"{list(column_names)}"
            )

        x = np.asarray(
            localizations["x"],
            dtype=float,
        )

        y = np.asarray(
            localizations["y"],
            dtype=float,
        )

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    return np.column_stack(
        [
            x[finite],
            y[finite],
        ]
    )

# ============================================================
# IMAGE READING
# ============================================================
def load_2d_image(path):
    import tifffile

    image = tifffile.imread(
        str(path)
    )

    image = np.asarray(
        image
    )

    image = np.squeeze(
        image
    )

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, but got "
            f"shape={image.shape} for:\n"
            f"{path}"
        )

    return image

def load_binary_image(
    path,
    threshold,
):
    image = load_2d_image(
        path
    )

    return image > threshold

# ============================================================
# IMAGEJ DENDRITE ROI READING
# ============================================================
def read_u8(data, offset):
    if len(data) < offset + 1:
        return 0

    return int(
        data[offset]
    )

def read_u16(data, offset):
    if len(data) < offset + 2:
        return 0

    return int(
        struct.unpack_from(
            ">H",
            data,
            offset,
        )[0]
    )

def read_i16(data, offset):
    if len(data) < offset + 2:
        return 0

    return int(
        struct.unpack_from(
            ">h",
            data,
            offset,
        )[0]
    )

def read_f32(data, offset):
    if len(data) < offset + 4:
        return 0.0

    return float(
        struct.unpack_from(
            ">f",
            data,
            offset,
        )[0]
    )

def parse_imagej_roi_polyline(data):
    """
    Returns an Nx2 array of x/y points.

    Supported dendrite ROI types:
        0 = polygon
        3 = straight line
        4 = free line
        5 = polyline
        7 = freehand
        8 = traced
        9 = angle
    """
    if (
        len(data) < 64
        or data[:4] != b"Iout"
    ):
        raise ValueError(
            "Invalid ImageJ ROI data."
        )

    roi_type = read_u8(
        data,
        6,
    )

    top = read_i16(
        data,
        8,
    )

    left = read_i16(
        data,
        10,
    )

    bottom = read_i16(
        data,
        12,
    )

    right = read_i16(
        data,
        14,
    )

    number_coordinates = read_u16(
        data,
        16,
    )

    if roi_type == 3:
        points = np.array(
            [
                [
                    read_f32(data, 18),
                    read_f32(data, 22),
                ],
                [
                    read_f32(data, 26),
                    read_f32(data, 30),
                ],
            ],
            dtype=float,
        )

        if (
            np.isfinite(points).all()
            and not np.all(points == 0)
        ):
            return points

        return np.array(
            [
                [left, top],
                [right, bottom],
            ],
            dtype=float,
        )

    if (
        roi_type in {
            0,
            4,
            5,
            7,
            8,
            9,
        }
        and number_coordinates > 0
    ):
        x_start = 64
        y_start = (
            64
            + 2 * number_coordinates
        )

        if (
            len(data)
            < y_start
            + 2 * number_coordinates
        ):
            raise ValueError(
                "ROI coordinate data is "
                f"incomplete for "
                f"n={number_coordinates}."
            )

        xs = np.array(
            [
                read_u16(
                    data,
                    x_start + 2 * i,
                )
                for i in range(
                    number_coordinates
                )
            ],
            dtype=float,
        ) + left

        ys = np.array(
            [
                read_u16(
                    data,
                    y_start + 2 * i,
                )
                for i in range(
                    number_coordinates
                )
            ],
            dtype=float,
        ) + top

        return np.column_stack(
            [
                xs,
                ys,
            ]
        )

    raise ValueError(
        f"Unsupported dendrite ROI "
        f"type {roi_type}."
    )

def read_dendrite_zip(path):
    """
    Dendrite ROIs are naturally sorted by ROI filename.

    Their integer dendrite IDs are assigned as:
        first ROI  = dendrite 1
        second ROI = dendrite 2
        third ROI  = dendrite 3
        etc.
    """
    polylines = []

    with zipfile.ZipFile(
        path,
        "r",
    ) as zip_file:
        names = sorted(
            [
                name
                for name in zip_file.namelist()
                if name.lower().endswith(
                    ".roi"
                )
            ],
            key=natural_key,
        )

        for name in names:
            try:
                points = parse_imagej_roi_polyline(
                    zip_file.read(name)
                )

            except Exception as error:
                raise ValueError(
                    f"Failed reading dendrite ROI "
                    f"{name} in {path}: {error}"
                ) from error

            if len(points) >= 2:
                polylines.append(
                    {
                        "roi_name": Path(name).name,
                        "points": points,
                    }
                )

    if not polylines:
        raise ValueError(
            f"No usable dendrite line ROIs "
            f"found in {path}"
        )

    return polylines

# ============================================================
# DENDRITE INTERPOLATION
# ============================================================
def interpolate_dendrites(
    polylines,
):
    all_points = []
    all_dendrite_ids = []

    # Dendrites are numbered 1, 2, 3, ...
    # in natural-sorted ROI filename order.
    for dendrite_id, polyline in enumerate(
        polylines,
        start=1,
    ):
        points = np.asarray(
            polyline["points"],
            dtype=float,
        )

        for i in range(
            len(points) - 1
        ):
            point_0 = points[i]
            point_1 = points[i + 1]

            vector = (
                point_1
                - point_0
            )

            length = float(
                np.linalg.norm(
                    vector
                )
            )

            if length <= 0:
                continue

            number_steps = max(
                2,
                int(
                    np.ceil(
                        length
                        / INTERP_STEP_PX
                    )
                )
                + 1,
            )

            for fraction in np.linspace(
                0.0,
                1.0,
                number_steps,
                endpoint=False,
            ):
                all_points.append(
                    point_0
                    + fraction * vector
                )

                all_dendrite_ids.append(
                    dendrite_id
                )

        all_points.append(
            points[-1]
        )

        all_dendrite_ids.append(
            dendrite_id
        )

    if not all_points:
        raise ValueError(
            "No interpolated dendrite "
            "points were generated."
        )

    return (
        np.asarray(
            all_points,
            dtype=float,
        ),
        np.asarray(
            all_dendrite_ids,
            dtype=np.int64,
        ),
    )

# ============================================================
# MITOCHONDRIAL COMPONENTS
# ============================================================
def label_mitochondria(
    mito_mask,
):
    from skimage.measure import label

    labels = label(
        mito_mask,
        connectivity=MITO_CONNECTIVITY,
    ).astype(
        np.int32
    )

    if MIN_MITO_AREA_PIXELS <= 1:
        return labels

    component_sizes = np.bincount(
        labels.ravel()
    )

    keep_labels = np.flatnonzero(
        component_sizes
        >= MIN_MITO_AREA_PIXELS
    )

    keep_labels = keep_labels[
        keep_labels != 0
    ]

    keep_mask = np.isin(
        labels,
        keep_labels,
    )

    labels = label(
        keep_mask,
        connectivity=MITO_CONNECTIVITY,
    ).astype(
        np.int32
    )

    return labels

# ============================================================
# MAP EACH MITOCHONDRION TO A DENDRITE
# ============================================================
def assign_mitochondria_to_dendrites(
    mito_labels,
    skeleton,
    dendrite_points,
    dendrite_ids,
):
    """
    Assign each mitochondrial component to the dendrite along
    which it is located.

    For each mitochondrion:

    1. Use its skeleton pixels inside the mitochondrial mask.
    2. If it has no skeleton pixels, use all mask pixels belonging
       to that mitochondrion.
    3. Map every selected pixel to its closest interpolated
       dendrite point.
    4. Assign the mitochondrion to the dendrite supported by the
       largest number of selected pixels.
    5. If multiple dendrites receive the same number of pixels,
       choose the one with the smaller mean distance.
    """
    from scipy.spatial import cKDTree

    number_mito = int(
        mito_labels.max()
    )

    assignments = np.zeros(
        number_mito + 1,
        dtype=np.int64,
    )

    tree = cKDTree(
        dendrite_points
    )

    for mito_label in range(
        1,
        number_mito + 1,
    ):
        component_mask = (
            mito_labels
            == mito_label
        )

        sample_mask = (
            skeleton
            & component_mask
        )

        # Fall back to all mitochondrial mask pixels
        # when this component has no skeleton pixels.
        if not np.any(
            sample_mask
        ):
            sample_mask = component_mask

        y_pixels, x_pixels = np.nonzero(
            sample_mask
        )

        if len(x_pixels) == 0:
            raise ValueError(
                f"Mitochondrion {mito_label} "
                f"contains no usable pixels."
            )

        xy = np.column_stack(
            [
                x_pixels.astype(float),
                y_pixels.astype(float),
            ]
        )

        (
            distances,
            nearest_indices,
        ) = tree.query(
            xy,
            k=1,
        )

        nearest_dendrites = dendrite_ids[
            nearest_indices
        ]

        (
            unique_dendrites,
            dendrite_counts,
        ) = np.unique(
            nearest_dendrites,
            return_counts=True,
        )

        maximum_count = dendrite_counts.max()

        candidates = unique_dendrites[
            dendrite_counts
            == maximum_count
        ]

        if len(candidates) == 1:
            chosen_dendrite = int(
                candidates[0]
            )

        else:
            candidate_scores = []

            for candidate in candidates:
                candidate_distances = distances[
                    nearest_dendrites
                    == candidate
                ]

                mean_distance = float(
                    np.mean(
                        candidate_distances
                    )
                )

                candidate_scores.append(
                    (
                        mean_distance,
                        int(candidate),
                    )
                )

            # Sort first by mean distance and then by ID.
            chosen_dendrite = min(
                candidate_scores
            )[1]

        assignments[
            mito_label
        ] = chosen_dendrite

    return assignments

# ============================================================
# MOLECULE ASSIGNMENT
# ============================================================
def localization_pixel_indices(
    xy,
    image_shape,
):
    """
    Converts Picasso x/y coordinates into integer image indices.

    Returns:
        x_pixels
        y_pixels
        valid
    """
    xy = np.asarray(
        xy,
        dtype=float,
    )

    if len(xy) == 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=bool,
            ),
        )

    if MOLECULE_PIXEL_MAPPING == "round":
        x_pixels = np.rint(
            xy[:, 0]
        ).astype(
            np.int64
        )

        y_pixels = np.rint(
            xy[:, 1]
        ).astype(
            np.int64
        )

    elif MOLECULE_PIXEL_MAPPING == "floor":
        x_pixels = np.floor(
            xy[:, 0]
        ).astype(
            np.int64
        )

        y_pixels = np.floor(
            xy[:, 1]
        ).astype(
            np.int64
        )

    else:
        raise ValueError(
            "MOLECULE_PIXEL_MAPPING must be "
            "'round' or 'floor'."
        )

    height, width = image_shape

    valid = (
        (x_pixels >= 0)
        & (x_pixels < width)
        & (y_pixels >= 0)
        & (y_pixels < height)
    )

    return (
        x_pixels,
        y_pixels,
        valid,
    )

def count_molecules_per_mito(
    xy,
    mito_labels,
):
    number_mito = int(
        mito_labels.max()
    )

    counts = np.zeros(
        number_mito + 1,
        dtype=np.int64,
    )

    (
        x_pixels,
        y_pixels,
        valid,
    ) = localization_pixel_indices(
        xy=xy,
        image_shape=mito_labels.shape,
    )

    if not np.any(valid):
        return (
            counts,
            0,
            int(len(xy)),
        )

    valid_x = x_pixels[
        valid
    ]

    valid_y = y_pixels[
        valid
    ]

    assigned_labels = mito_labels[
        valid_y,
        valid_x,
    ]

    counts = np.bincount(
        assigned_labels,
        minlength=number_mito + 1,
    ).astype(
        np.int64
    )

    number_inside_mito = int(
        np.count_nonzero(
            assigned_labels > 0
        )
    )

    number_outside_image = int(
        len(xy)
        - np.count_nonzero(valid)
    )

    return (
        counts,
        number_inside_mito,
        number_outside_image,
    )

# ============================================================
# SKELETON LENGTH
# ============================================================
def skeleton_graph_length_um(
    skeleton,
):
    """
    Calculate total skeleton length by summing each connection
    between neighboring skeleton pixels once.

    Right-neighbor connections:
        PIXEL_SIZE_X_UM

    Down-neighbor connections:
        PIXEL_SIZE_Y_UM

    Diagonal connections:
        sqrt(PIXEL_SIZE_X_UM^2 + PIXEL_SIZE_Y_UM^2)
    """
    skeleton = np.asarray(
        skeleton,
        dtype=bool,
    )

    if not np.any(
        skeleton
    ):
        return 0.0

    horizontal_connections = np.count_nonzero(
        skeleton[:, :-1]
        & skeleton[:, 1:]
    )

    vertical_connections = np.count_nonzero(
        skeleton[:-1, :]
        & skeleton[1:, :]
    )

    diagonal_down_right = np.count_nonzero(
        skeleton[:-1, :-1]
        & skeleton[1:, 1:]
    )

    diagonal_down_left = np.count_nonzero(
        skeleton[:-1, 1:]
        & skeleton[1:, :-1]
    )

    length_um = (
        horizontal_connections
        * PIXEL_SIZE_X_UM
        + vertical_connections
        * PIXEL_SIZE_Y_UM
        + (
            diagonal_down_right
            + diagonal_down_left
        )
        * DIAGONAL_PIXEL_LENGTH_UM
    )

    return float(
        length_um
    )

def skeleton_lengths_per_mito(
    mito_labels,
    skeleton,
):
    number_mito = int(
        mito_labels.max()
    )

    lengths = np.zeros(
        number_mito + 1,
        dtype=float,
    )

    skeleton_pixels_inside_mask = (
        skeleton
        & (
            mito_labels > 0
        )
    )

    unassigned_skeleton_pixels = int(
        np.count_nonzero(
            skeleton
            & (
                mito_labels == 0
            )
        )
    )

    for mito_label in range(
        1,
        number_mito + 1,
    ):
        component_skeleton = (
            skeleton_pixels_inside_mask
            & (
                mito_labels
                == mito_label
            )
        )

        lengths[
            mito_label
        ] = skeleton_graph_length_um(
            component_skeleton
        )

    return (
        lengths,
        unassigned_skeleton_pixels,
    )

# ============================================================
# BUILD OUTPUT ROWS
# ============================================================
def build_measurement_rows(
    metadata,
    mito_labels,
    skeleton,
    probe00_xy,
    probe01_xy,
    dendrite_assignments,
):
    number_mito = int(
        mito_labels.max()
    )

    component_pixel_counts = np.bincount(
        mito_labels.ravel(),
        minlength=number_mito + 1,
    )

    (
        skeleton_lengths,
        unassigned_skeleton_pixels,
    ) = skeleton_lengths_per_mito(
        mito_labels=mito_labels,
        skeleton=skeleton,
    )

    (
        probe00_counts,
        probe00_inside,
        probe00_outside_image,
    ) = count_molecules_per_mito(
        xy=probe00_xy,
        mito_labels=mito_labels,
    )

    (
        probe01_counts,
        probe01_inside,
        probe01_outside_image,
    ) = count_molecules_per_mito(
        xy=probe01_xy,
        mito_labels=mito_labels,
    )

    dish_integer = parse_integer_label(
        metadata["Dish"],
        "Dish",
    )

    cell_integer = parse_integer_label(
        metadata["Cell"],
        "Cell",
    )

    rows = []

    for mito_label in range(
        1,
        number_mito + 1,
    ):
        area_pixels = int(
            component_pixel_counts[
                mito_label
            ]
        )

        rows.append(
            {
                "Date": metadata["Date"],
                "Condition": metadata[
                    "Condition"
                ],
                "Dish": int(
                    dish_integer
                ),
                "Cell": int(
                    cell_integer
                ),
                "dendrite": int(
                    dendrite_assignments[
                        mito_label
                    ]
                ),
                "mito": int(
                    mito_label
                ),
                "mito_area_um2": float(
                    area_pixels
                    * PIXEL_AREA_UM2
                ),
                "skeleton_length_um": float(
                    skeleton_lengths[
                        mito_label
                    ]
                ),
                "probe00": int(
                    probe00_counts[
                        mito_label
                    ]
                ),
                "probe01": int(
                    probe01_counts[
                        mito_label
                    ]
                ),
            }
        )

    qc = {
        "number_mito": number_mito,
        "unassigned_skeleton_pixels": (
            unassigned_skeleton_pixels
        ),
        "probe00_total": int(
            len(probe00_xy)
        ),
        "probe00_inside_mito": (
            probe00_inside
        ),
        "probe00_outside_image": (
            probe00_outside_image
        ),
        "probe01_total": int(
            len(probe01_xy)
        ),
        "probe01_inside_mito": (
            probe01_inside
        ),
        "probe01_outside_image": (
            probe01_outside_image
        ),
    }

    return rows, qc

# ============================================================
# SAVE
# ============================================================
def save_csv(
    path,
    rows,
):
    if DRY_RUN:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if len(dataframe) > 0:
        dataframe = dataframe.sort_values(
            [
                "Date",
                "Condition",
                "Dish",
                "Cell",
                "dendrite",
                "mito",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

    dataframe.to_csv(
        path,
        index=False,
        float_format="%.6f",
    )

# ============================================================
# PROCESS ONE IMAGE SET
# ============================================================
def process_one_image_set(
    job,
):
    (
        mito_mask_path,
        metadata,
        mito_skel_path,
        dendrite_zip_path,
        probe00_path,
        probe01_path,
    ) = job

    prefix = metadata[
        "prefix"
    ]

    output_path = output_csv_path(
        prefix
    )

    result = {
        "status": None,
        "prefix": prefix,
        "output": str(
            output_path
        ),
        "number_mito": None,
        "unassigned_skeleton_pixels": None,
        "probe00_total": None,
        "probe00_inside_mito": None,
        "probe00_outside_image": None,
        "probe01_total": None,
        "probe01_inside_mito": None,
        "probe01_outside_image": None,
        "error": None,
    }

    try:
        if (
            output_path.exists()
            and not OVERWRITE
        ):
            result["status"] = (
                "skipped exists"
            )

            return result

        required_files = [
            (
                "mito mask",
                mito_mask_path,
            ),
            (
                "mito skeleton",
                mito_skel_path,
            ),
            (
                "dendrite ZIP",
                dendrite_zip_path,
            ),
            (
                "probe00",
                probe00_path,
            ),
            (
                "probe01",
                probe01_path,
            ),
        ]

        missing = [
            name
            for name, path in required_files
            if (
                path is None
                or not Path(path).exists()
            )
        ]

        if missing:
            result["status"] = (
                "missing required file"
            )

            result["error"] = (
                "; ".join(
                    missing
                )
            )

            return result

        mito_mask = load_binary_image(
            path=mito_mask_path,
            threshold=MITO_MASK_THRESHOLD,
        )

        skeleton = load_binary_image(
            path=mito_skel_path,
            threshold=MITO_SKEL_THRESHOLD,
        )

        if (
            mito_mask.shape
            != skeleton.shape
        ):
            raise ValueError(
                "Mito mask and skeleton shapes "
                "do not match:\n"
                f"mask: {mito_mask.shape} | "
                f"{mito_mask_path}\n"
                f"skel: {skeleton.shape} | "
                f"{mito_skel_path}"
            )

        mito_labels = label_mitochondria(
            mito_mask
        )

        probe00_xy = load_xy_from_hdf5(
            probe00_path
        )

        probe01_xy = load_xy_from_hdf5(
            probe01_path
        )

        dendrite_polylines = read_dendrite_zip(
            dendrite_zip_path
        )

        (
            dendrite_points,
            dendrite_ids,
        ) = interpolate_dendrites(
            dendrite_polylines
        )

        dendrite_assignments = (
            assign_mitochondria_to_dendrites(
                mito_labels=mito_labels,
                skeleton=skeleton,
                dendrite_points=dendrite_points,
                dendrite_ids=dendrite_ids,
            )
        )

        rows, qc = build_measurement_rows(
            metadata=metadata,
            mito_labels=mito_labels,
            skeleton=skeleton,
            probe00_xy=probe00_xy,
            probe01_xy=probe01_xy,
            dendrite_assignments=(
                dendrite_assignments
            ),
        )

        for key, value in qc.items():
            result[key] = value

        save_csv(
            path=output_path,
            rows=rows,
        )

        result["status"] = (
            "dry run"
            if DRY_RUN
            else "saved"
        )

        return result

    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()

        return result

def print_result(
    index,
    total,
    result,
):
    status = result[
        "status"
    ]

    prefix = result[
        "prefix"
    ]

    if status in {
        "saved",
        "dry run",
    }:
        print(
            f"[{index}/{total}] "
            f"{status}: {prefix} | "
            f"mito={result['number_mito']} | "
            f"probe00 inside="
            f"{result['probe00_inside_mito']}/"
            f"{result['probe00_total']} | "
            f"probe01 inside="
            f"{result['probe01_inside_mito']}/"
            f"{result['probe01_total']} | "
            f"unassigned skel pixels="
            f"{result['unassigned_skeleton_pixels']}"
        )

        print(
            f"    output: "
            f"{result['output']}"
        )

    elif status == "missing required file":
        print(
            f"[{index}/{total}] "
            f"missing required file: "
            f"{prefix} | "
            f"{result['error']}"
        )

    elif status == "skipped exists":
        print(
            f"[{index}/{total}] "
            f"skipped exists: "
            f"{prefix}"
        )

    elif status == "error":
        print(
            f"[{index}/{total}] "
            f"ERROR: {prefix}"
        )

    else:
        print(
            f"[{index}/{total}] "
            f"{status}: {prefix}"
        )

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print(
        "Molecules per mitochondrion "
        "with dendrite assignment"
    )
    print("=" * 100)

    print(
        f"CENTERS_DIR:                 "
        f"{CENTERS_DIR}"
    )

    print(
        f"MITO_MASK_DIR:               "
        f"{MITO_MASK_DIR}"
    )

    print(
        f"MITO_SKEL_DIR:               "
        f"{MITO_SKEL_DIR}"
    )

    print(
        f"DENDRITE_DIR:                "
        f"{DENDRITE_DIR}"
    )

    print(
        f"OUTPUT_DIR:                  "
        f"{OUTPUT_DIR}"
    )

    print(
        f"OUTPUT_COLUMNS:              "
        f"{OUTPUT_COLUMNS}"
    )

    print(
        f"PIXEL_SIZE_X_UM:             "
        f"{PIXEL_SIZE_X_UM}"
    )

    print(
        f"PIXEL_SIZE_Y_UM:             "
        f"{PIXEL_SIZE_Y_UM}"
    )

    print(
        f"PIXEL_AREA_UM2:              "
        f"{PIXEL_AREA_UM2}"
    )

    print(
        f"MITO_CONNECTIVITY:           "
        f"{MITO_CONNECTIVITY}"
    )

    print(
        f"MIN_MITO_AREA_PIXELS:        "
        f"{MIN_MITO_AREA_PIXELS}"
    )

    print(
        f"MOLECULE_PIXEL_MAPPING:      "
        f"{MOLECULE_PIXEL_MAPPING}"
    )

    print(
        f"INTERP_STEP_UM:              "
        f"{INTERP_STEP_UM}"
    )

    print(
        f"TEST_FIRST_FILE:             "
        f"{TEST_FIRST_FILE}"
    )

    print(
        f"DRY_RUN:                     "
        f"{DRY_RUN}"
    )

    print(
        f"OVERWRITE:                   "
        f"{OVERWRITE}"
    )

    print(
        f"USE_PARALLEL:                "
        f"{USE_PARALLEL}"
    )

    print(
        f"MAX_WORKERS:                 "
        f"{MAX_WORKERS}"
    )

    print("=" * 100)

    if MOLECULE_PIXEL_MAPPING not in {
        "round",
        "floor",
    }:
        raise ValueError(
            "MOLECULE_PIXEL_MAPPING must be "
            "'round' or 'floor'."
        )

    if MITO_CONNECTIVITY not in {
        1,
        2,
    }:
        raise ValueError(
            "MITO_CONNECTIVITY must be 1 or 2."
        )

    required_folders = [
        CENTERS_DIR,
        MITO_MASK_DIR,
        MITO_SKEL_DIR,
        DENDRITE_DIR,
    ]

    for folder in required_folders:
        if not folder.exists():
            raise FileNotFoundError(
                f"Required folder does not exist:\n"
                f"{folder}"
            )

    (
        mito_mask_files,
        mask_skipped,
    ) = find_mito_mask_files()

    (
        mito_skel_index,
        skel_skipped,
    ) = build_image_index(
        MITO_SKEL_DIR
    )

    (
        dendrite_index,
        dendrite_skipped,
    ) = build_dendrite_index()

    (
        probe00_index,
        probe00_skipped,
    ) = build_probe_index(
        PROBE00
    )

    (
        probe01_index,
        probe01_skipped,
    ) = build_probe_index(
        PROBE01
    )

    print(
        f"\nMito mask image sets: "
        f"{len(mito_mask_files)}"
    )

    print(
        f"Mito skeleton files:  "
        f"{len(mito_skel_index)}"
    )

    print(
        f"Dendrite ZIP files:   "
        f"{len(dendrite_index)}"
    )

    print(
        f"Probe00 files:         "
        f"{len(probe00_index)}"
    )

    print(
        f"Probe01 files:         "
        f"{len(probe01_index)}"
    )

    total_skipped = (
        len(mask_skipped)
        + len(skel_skipped)
        + len(dendrite_skipped)
        + len(probe00_skipped)
        + len(probe01_skipped)
    )

    print(
        f"Skipped filenames:     "
        f"{total_skipped}"
    )

    if len(mito_mask_files) == 0:
        print(
            "\nNo mito mask files were found."
        )

        return

    jobs = []

    print("\nFirst few jobs:")

    for (
        mito_mask_path,
        metadata,
    ) in mito_mask_files[:20]:
        prefix = metadata[
            "prefix"
        ]

        mito_skel_path = (
            mito_skel_index.get(
                prefix
            )
        )

        dendrite_zip_path = (
            dendrite_index.get(
                prefix
            )
        )

        probe00_path = (
            probe00_index.get(
                prefix
            )
        )

        probe01_path = (
            probe01_index.get(
                prefix
            )
        )

        print(
            f"  {prefix}"
        )

        print(
            f"    mask:      "
            f"{mito_mask_path.name}"
        )

        print(
            f"    skel:      "
            f"{mito_skel_path.name if mito_skel_path else 'MISSING'}"
        )

        print(
            f"    dendrites: "
            f"{dendrite_zip_path.name if dendrite_zip_path else 'MISSING'}"
        )

        print(
            f"    probe00:   "
            f"{probe00_path.name if probe00_path else 'MISSING'}"
        )

        print(
            f"    probe01:   "
            f"{probe01_path.name if probe01_path else 'MISSING'}"
        )

        print(
            f"    output:    "
            f"{output_csv_path(prefix).name}"
        )

    if len(mito_mask_files) > 20:
        print(
            f"  ... plus "
            f"{len(mito_mask_files) - 20} more"
        )

    for (
        mito_mask_path,
        metadata,
    ) in mito_mask_files:
        prefix = metadata[
            "prefix"
        ]

        jobs.append(
            (
                mito_mask_path,
                metadata,
                mito_skel_index.get(
                    prefix
                ),
                dendrite_index.get(
                    prefix
                ),
                probe00_index.get(
                    prefix
                ),
                probe01_index.get(
                    prefix
                ),
            )
        )

    if TEST_FIRST_FILE:
        jobs = jobs[:1]

        print(
            "\nTEST_FIRST_FILE=True: "
            "processing only the first image set."
        )

    print("\n" + "=" * 100)
    print("PROCESSING")
    print("=" * 100)

    results = []

    if (
        USE_PARALLEL
        and not TEST_FIRST_FILE
    ):
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            futures = [
                executor.submit(
                    process_one_image_set,
                    job,
                )
                for job in jobs
            ]

            for index, future in enumerate(
                as_completed(
                    futures
                ),
                start=1,
            ):
                result = future.result()

                results.append(
                    result
                )

                print_result(
                    index=index,
                    total=len(jobs),
                    result=result,
                )

    else:
        for index, job in enumerate(
            jobs,
            start=1,
        ):
            result = process_one_image_set(
                job
            )

            results.append(
                result
            )

            print_result(
                index=index,
                total=len(jobs),
                result=result,
            )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    status_counts = Counter(
        result["status"]
        for result in results
    )

    for status, count in status_counts.items():
        print(
            f"{status}: {count}"
        )

    total_mito = sum(
        result["number_mito"] or 0
        for result in results
    )

    total_probe00_inside = sum(
        result["probe00_inside_mito"] or 0
        for result in results
    )

    total_probe01_inside = sum(
        result["probe01_inside_mito"] or 0
        for result in results
    )

    total_unassigned_skeleton = sum(
        result["unassigned_skeleton_pixels"] or 0
        for result in results
    )

    print(
        f"\nTotal mitochondria:             "
        f"{total_mito}"
    )

    print(
        f"Total probe00 inside mito:      "
        f"{total_probe00_inside}"
    )

    print(
        f"Total probe01 inside mito:      "
        f"{total_probe01_inside}"
    )

    print(
        f"Unassigned skeleton pixels:     "
        f"{total_unassigned_skeleton}"
    )

    print(
        f"Output folder:                  "
        f"{OUTPUT_DIR}"
    )

    errors = [
        result
        for result in results
        if result["status"] == "error"
    ]

    if errors:
        print("\n" + "=" * 100)
        print("ERROR DETAILS")
        print("=" * 100)

        for result in errors:
            print(
                f"\nIMAGE SET: "
                f"{result['prefix']}"
            )

            print(
                result["error"]
            )

    missing = [
        result
        for result in results
        if result["status"]
        == "missing required file"
    ]

    if missing:
        print("\n" + "=" * 100)
        print("MISSING FILE DETAILS")
        print("=" * 100)

        for result in missing:
            print(
                f"{result['prefix']}: "
                f"{result['error']}"
            )

    if DRY_RUN:
        print(
            "\nDRY_RUN enabled. "
            "No files were written."
        )

    print("\nDone.")

if __name__ == "__main__":
    main()