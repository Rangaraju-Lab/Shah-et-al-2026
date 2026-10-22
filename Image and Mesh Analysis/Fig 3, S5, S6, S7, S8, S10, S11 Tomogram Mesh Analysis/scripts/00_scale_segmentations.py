# ============================================================
# Faster parallel OMM-centered Z-scaling and XY cropping
# Works only on direct subfolders under ../Control and ../TTX
# ============================================================

import os

# Limit internal thread storms when using multiple processes
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
import tifffile as tiff
from scipy import ndimage as ndi
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
TARGET_Z = 60
VOXEL_SIZE_NM = 2.0
RADIUS_UM = 1.0
RADIUS_NM = RADIUS_UM * 1000.0
RADIUS_PX = RADIUS_NM / VOXEL_SIZE_NM  # 500 px

ROOT_NAMES = ["Control", "TTX"]

# On a network drive, too many workers can slow things down.
# Try 4, 6, 8 and compare.
MAX_WORKERS = WORKERS

SKIP_EXISTING = False

SCALED_FOLDER_NAME = "scaled"
FAILURE_LOG_NAME = "scale_crop_failures.csv"

# If False, OMM is used to compute the centroid but is not saved into scaled/.
# This avoids processing OMM twice.
SAVE_OMM_OUTPUT = True

# Fast default: do not compress output TIFFs.
# Compression can be much slower on network drives.
TIFF_COMPRESSION = None

# Filename-based segmentation detection.
# Add or remove keywords as needed.
SEGMENTATION_NAME_KEYWORDS = [
    "ER",
    "OMM",
    "IBM",
    "IMS",
    "Ribo",

]

# If filename does not match segmentation keywords:
# False = assume image, fastest.
# True = fall back to checking unique values, safer but slower.
USE_UNIQUE_FALLBACK_FOR_SEGMENTATION = False

# ============================================================
# PATH HELPERS
# ============================================================
def get_script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def find_omm_file(folder):
    candidates = [
        folder / "OMM.tiff",
        folder / "OMM.tif",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None

def discover_direct_subfolders_with_omm(root_name, root):
    jobs = []

    if not root.exists():
        return jobs

    direct_subfolders = sorted([
        p for p in root.iterdir()
        if p.is_dir()
    ])

    for folder in direct_subfolders:
        omm_file = find_omm_file(folder)

        if omm_file is not None:
            jobs.append({
                "root_name": root_name,
                "folder": str(folder),
                "omm_path": str(omm_file),
            })

    return jobs

# ============================================================
# IMAGE HELPERS
# ============================================================
def load_tiff_stack(path):
    arr = tiff.imread(path)
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"{Path(path).name}: expected 3D stack, got shape {arr.shape}")

    return arr

def get_largest_component(mask):
    lab, n = ndi.label(mask)

    if n == 0:
        return mask

    counts = np.bincount(lab.ravel())
    counts[0] = 0

    largest = np.argmax(counts)
    return lab == largest

def compute_centroid_zyx(mask):
    mask = get_largest_component(mask)

    if not np.any(mask):
        raise ValueError("OMM mask is empty after processing.")

    centroid = ndi.center_of_mass(mask)
    return np.array(centroid, dtype=np.float64)

def make_xy_radius_mask_2d(shape_zyx, centroid_zyx, radius_px):
    _, y, x = shape_zyx
    _, cy, cx = centroid_zyx

    yy, xx = np.ogrid[:y, :x]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist2 <= radius_px ** 2

def is_segmentation_by_name(path):
    name = Path(path).stem.lower()
    return any(k in name for k in SEGMENTATION_NAME_KEYWORDS)

def is_segmentation_array_slow(arr):
    """
    Slow fallback. Avoid unless needed.
    """
    if arr.dtype == bool:
        return True

    if np.issubdtype(arr.dtype, np.integer):
        uniq = np.unique(arr)
        if uniq.size <= 16:
            return True

    return False

def decide_segmentation(path, arr):
    """
    Faster segmentation detection:
    1. Filename keywords first.
    2. Optional slow unique fallback.
    """
    if is_segmentation_by_name(path):
        return True

    if USE_UNIQUE_FALLBACK_FOR_SEGMENTATION:
        return is_segmentation_array_slow(arr)

    return False

def resize_z_binary(mask, target_z=60):
    z, y, x = mask.shape

    if z == target_z:
        return mask.copy()

    if z > target_z:
        start = (z - target_z) // 2
        end = start + target_z
        return mask[start:end].copy()

    zoom_factor = target_z / z

    resized = ndi.zoom(
        mask.astype(np.float32),
        zoom=(zoom_factor, 1.0, 1.0),
        order=1,
        mode="nearest",
        prefilter=False,
    )

    if resized.shape[0] != target_z:
        if resized.shape[0] > target_z:
            start = (resized.shape[0] - target_z) // 2
            resized = resized[start:start + target_z]
        else:
            pad_before = (target_z - resized.shape[0]) // 2
            pad_after = target_z - resized.shape[0] - pad_before
            resized = np.pad(
                resized,
                ((pad_before, pad_after), (0, 0), (0, 0)),
                mode="edge",
            )

    return resized >= 0.5

def resize_z_image(arr, target_z=60):
    z, y, x = arr.shape

    if z == target_z:
        return arr.copy()

    if z > target_z:
        start = (z - target_z) // 2
        end = start + target_z
        return arr[start:end].copy()

    zoom_factor = target_z / z

    resized = ndi.zoom(
        arr.astype(np.float32),
        zoom=(zoom_factor, 1.0, 1.0),
        order=1,
        mode="nearest",
        prefilter=True,
    )

    if resized.shape[0] != target_z:
        if resized.shape[0] > target_z:
            start = (resized.shape[0] - target_z) // 2
            resized = resized[start:start + target_z]
        else:
            pad_before = (target_z - resized.shape[0]) // 2
            pad_after = target_z - resized.shape[0] - pad_before
            resized = np.pad(
                resized,
                ((pad_before, pad_after), (0, 0), (0, 0)),
                mode="edge",
            )

    return resized

def apply_xy_crop_binary_fast(mask, xy_mask_2d):
    return mask & xy_mask_2d[None, :, :]

def apply_xy_crop_image_fast(arr, xy_mask_2d):
    out = arr.copy()
    out[:, ~xy_mask_2d] = 0
    return out

def cast_like_original(arr, original_dtype, was_segmentation=False):
    if was_segmentation:
        if original_dtype == bool:
            return arr.astype(bool, copy=False)

        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            return arr.astype(original_dtype, copy=False) * info.max

        if np.issubdtype(original_dtype, np.floating):
            return arr.astype(original_dtype, copy=False)

        return arr.astype(original_dtype, copy=False)

    if np.issubdtype(original_dtype, np.floating):
        return arr.astype(original_dtype, copy=False)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        arr = np.clip(arr, info.min, info.max)
        return np.rint(arr).astype(original_dtype, copy=False)

    return arr.astype(original_dtype, copy=False)

def save_tiff(path, arr):
    tiff.imwrite(
        path,
        arr,
        imagej=True,
        compression=TIFF_COMPRESSION,
        photometric="minisblack",
    )

# ============================================================
# FOLDER PROCESSING WORKER
# ============================================================
def process_single_folder(job):
    root_name = job["root_name"]
    folder = Path(job["folder"])
    omm_path = Path(job["omm_path"])

    saved_count = 0
    skipped_count = 0
    failed_files = []

    try:
        scaled_dir = folder / SCALED_FOLDER_NAME
        scaled_dir.mkdir(exist_ok=True)

        # --------------------------------------------------------
        # 1. Process OMM once to define centroid
        # --------------------------------------------------------
        omm_raw = load_tiff_stack(omm_path)
        omm_original_dtype = omm_raw.dtype

        omm_bin = omm_raw > 0
        omm_z = resize_z_binary(omm_bin, target_z=TARGET_Z)
        omm_centroid = compute_centroid_zyx(omm_z)

        # Precompute 2D XY mask once per folder.
        xy_mask_2d = make_xy_radius_mask_2d(omm_z.shape, omm_centroid, RADIUS_PX)

        # --------------------------------------------------------
        # 2. Process all TIFF files in this folder
        # --------------------------------------------------------
        tif_files = sorted([
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in [".tif", ".tiff"]
        ])

        for tif_path in tif_files:
            try:
                out_path = scaled_dir / tif_path.name

                if SKIP_EXISTING and out_path.exists():
                    skipped_count += 1
                    continue

                # Avoid rereading and reprocessing OMM if possible.
                is_omm = tif_path.resolve() == omm_path.resolve()

                if is_omm:
                    if not SAVE_OMM_OUTPUT:
                        skipped_count += 1
                        continue

                    cropped = apply_xy_crop_binary_fast(omm_z, xy_mask_2d)
                    out = cast_like_original(
                        cropped,
                        omm_original_dtype,
                        was_segmentation=True,
                    )

                    save_tiff(out_path, out)
                    saved_count += 1
                    continue

                arr = load_tiff_stack(tif_path)
                original_dtype = arr.dtype
                seg_like = decide_segmentation(tif_path, arr)

                if seg_like:
                    mask = arr > 0
                    resized = resize_z_binary(mask, target_z=TARGET_Z)
                    cropped = apply_xy_crop_binary_fast(resized, xy_mask_2d)
                    out = cast_like_original(
                        cropped,
                        original_dtype,
                        was_segmentation=True,
                    )
                else:
                    resized = resize_z_image(arr, target_z=TARGET_Z)
                    cropped = apply_xy_crop_image_fast(resized, xy_mask_2d)
                    out = cast_like_original(
                        cropped,
                        original_dtype,
                        was_segmentation=False,
                    )

                save_tiff(out_path, out)
                saved_count += 1

            except Exception as e:
                failed_files.append({
                    "root_name": root_name,
                    "folder": str(folder),
                    "file": str(tif_path),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        return {
            "status": "ok",
            "root_name": root_name,
            "folder": str(folder),
            "saved_count": saved_count,
            "skipped_count": skipped_count,
            "failed_count": len(failed_files),
            "failed_files": failed_files,
        }

    except Exception as e:
        failed_files.append({
            "root_name": root_name,
            "folder": str(folder),
            "file": str(omm_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

        return {
            "status": "failed",
            "root_name": root_name,
            "folder": str(folder),
            "saved_count": saved_count,
            "skipped_count": skipped_count,
            "failed_count": len(failed_files),
            "failed_files": failed_files,
        }

# ============================================================
# FAILURE LOG
# ============================================================
def write_failure_log(path, failed_files):
    if not failed_files:
        return

    fieldnames = [
        "root_name",
        "folder",
        "file",
        "error",
        "traceback",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in failed_files:
            writer.writerow(row)

# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = get_script_dir()

    roots = {
        root_name: (script_dir.parent / root_name).resolve()
        for root_name in ROOT_NAMES
    }

    jobs = []

    for root_name, root in roots.items():
        root_jobs = discover_direct_subfolders_with_omm(root_name, root)
        jobs.extend(root_jobs)

    print("Script dir:", script_dir)
    for root_name, root in roots.items():
        print(f"{root_name} root:", root)

    print("")
    print(f"Total direct subfolders to process: {len(jobs)}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"SKIP_EXISTING: {SKIP_EXISTING}")
    print(f"SAVE_OMM_OUTPUT: {SAVE_OMM_OUTPUT}")
    print(f"USE_UNIQUE_FALLBACK_FOR_SEGMENTATION: {USE_UNIQUE_FALLBACK_FOR_SEGMENTATION}")
    print("")

    if len(jobs) == 0:
        print("No direct subfolders containing OMM.tif/f were found.")
        return

    n_ok = 0
    n_failed_folders = 0
    total_saved = 0
    total_skipped = 0
    total_failed_files = 0
    all_failed_files = []

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=ctx,
    ) as executor:

        futures = [executor.submit(process_single_folder, job) for job in jobs]

        with tqdm(
            total=len(futures),
            desc="Processing folders",
            unit="folder",
            dynamic_ncols=True,
        ) as pbar:

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "status": "failed",
                        "root_name": "unknown",
                        "folder": "unknown",
                        "saved_count": 0,
                        "skipped_count": 0,
                        "failed_count": 1,
                        "failed_files": [{
                            "root_name": "unknown",
                            "folder": "unknown",
                            "file": "unknown",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }],
                    }

                if result["status"] == "ok":
                    n_ok += 1
                else:
                    n_failed_folders += 1

                total_saved += result["saved_count"]
                total_skipped += result["skipped_count"]
                total_failed_files += result["failed_count"]
                all_failed_files.extend(result["failed_files"])

                pbar.update(1)

    failure_log_path = script_dir / FAILURE_LOG_NAME
    write_failure_log(failure_log_path, all_failed_files)

    print("")
    print("Done.")
    print(f"Folders processed successfully: {n_ok}")
    print(f"Folders failed:                 {n_failed_folders}")
    print(f"TIFF files saved:               {total_saved}")
    print(f"TIFF files skipped:             {total_skipped}")
    print(f"TIFF/file failures:             {total_failed_files}")

    if all_failed_files:
        print(f"Failure log saved to: {failure_log_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()