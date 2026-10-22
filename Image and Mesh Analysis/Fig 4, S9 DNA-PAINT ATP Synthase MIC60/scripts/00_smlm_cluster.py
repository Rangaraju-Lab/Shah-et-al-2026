from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import traceback
import warnings
import inspect

# ============================================================
# SETTINGS
# ============================================================
DRY_RUN = False
OVERWRITE = True

# Terminal .py execution is best for process pool.
USE_PARALLEL = True
MAX_WORKERS = 8

# Folder structure:
# Project/
#   scripts/
#   Undrift Masked/
#   Undrift Masked SMLM/
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_DIR = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_DIR / "Undrift Masked"
OUTPUT_DIR = PROJECT_DIR / "Undrift Masked SMLM"

# Only process clean probe files:
#   2026-01-08 Control Dish1 Cell1 probe00.hdf5
#   2026-01-08 Control Dish1 Cell1 probe01.hdf5
#
# Reject:
#   probe00_smlm_nena.hdf5
#   probe01_smlm_nena.hdf5
ALLOWED_PROBES = {"probe00", "probe01"}

# Picasso SMLM clusterer parameters.
# Picasso recommendation: radius_xy = 2 * NeNA.
NENA_RADIUS_MULTIPLIER = 2.0

# z radius derived from xy radius.
# Example:
#   xy radius = 20 nm -> z radius = 40 nm
#   xy radius = 30 nm -> z radius = 60 nm
Z_RADIUS_FACTOR = 2.0

MIN_LOCS = 10
FRAME_ANALYSIS = True
REQUIRE_3D = True

# Output:
#   input:  file probe00.hdf5
#   output: file probe00 smlm.hdf5
OUTPUT_SUFFIX = ""

# Optional cluster centers
SAVE_CLUSTER_CENTERS = True
CENTERS_OUTPUT_DIR = PROJECT_DIR / "Undrift Masked SMLM Centers"
CENTERS_SUFFIX = ""

# ============================================================
# FILE HELPERS
# ============================================================
def is_allowed_hdf5(path: Path) -> bool:
    if path.suffix.lower() != ".hdf5":
        return False

    last_token = path.stem.split()[-1]
    return last_token in ALLOWED_PROBES

def output_path_for_input(hdf5_path: Path) -> Path:
    return OUTPUT_DIR / f"{hdf5_path.stem}{OUTPUT_SUFFIX}.hdf5"

def centers_output_path_for_input(hdf5_path: Path) -> Path:
    return CENTERS_OUTPUT_DIR / f"{hdf5_path.stem}{CENTERS_SUFFIX}.hdf5"

def find_input_hdf5_files():
    files = sorted(INPUT_DIR.glob("*.hdf5"))
    return [p for p in files if is_allowed_hdf5(p)]

# ============================================================
# TABLE COMPATIBILITY HELPERS
# ============================================================
def has_column(locs, colname: str) -> bool:
    """
    Works for both pandas DataFrame locs and numpy recarray/structured-array locs.
    """
    if hasattr(locs, "columns"):
        return colname in locs.columns

    if hasattr(locs, "dtype") and locs.dtype.names is not None:
        return colname in locs.dtype.names

    return False

def get_column(locs, colname: str):
    """
    Works for both pandas DataFrame locs and numpy recarray/structured-array locs.
    """
    if hasattr(locs, "columns"):
        return locs[colname].to_numpy()

    return locs[colname]

def get_column_names(locs):
    """
    For debug printing.
    """
    if hasattr(locs, "columns"):
        return list(locs.columns)

    if hasattr(locs, "dtype") and locs.dtype.names is not None:
        return list(locs.dtype.names)

    return []

# ============================================================
# PICASSO / METADATA HELPERS
# ============================================================
def get_pixelsize_nm(info):
    """
    Try Picasso's metadata helper first. Fall back to a manual metadata search.
    """
    try:
        from picasso import lib

        px = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
        return float(px)
    except Exception:
        pass

    possible_keys = [
        "Pixelsize",
        "Pixel size",
        "Pixel size (nm)",
        "PixelSize",
        "pixelsize",
    ]

    for entry in info:
        if isinstance(entry, dict):
            for key in possible_keys:
                if key in entry:
                    return float(entry[key])

    raise ValueError(
        "Could not find pixel size in Picasso YAML metadata. "
        "Expected key like 'Pixelsize'."
    )

def compute_nena_px(locs, info):
    """
    Picasso NeNA returns s in camera pixels.
    """
    from picasso import postprocess

    nena_result, nena_px = postprocess.nena(locs, info)
    return float(nena_px)

def call_clusterer_compatible(
    locs,
    radius_xy_px,
    radius_z_px,
    min_locs,
    frame_analysis,
    pixelsize_nm,
):
    """
    Call picasso.clusterer.cluster in a way that works across Picasso versions.

    Some versions have:
        cluster(locs, radius_xy, radius_z, min_locs, frame_analysis, pixelsize)

    Some newer versions may support:
        return_info=True
    """
    from picasso import clusterer

    sig = inspect.signature(clusterer.cluster)
    params = sig.parameters

    kwargs = {}

    if "locs" in params:
        kwargs["locs"] = locs
    if "radius_xy" in params:
        kwargs["radius_xy"] = float(radius_xy_px)
    if "radius_z" in params:
        kwargs["radius_z"] = float(radius_z_px)
    if "min_locs" in params:
        kwargs["min_locs"] = int(min_locs)
    if "frame_analysis" in params:
        kwargs["frame_analysis"] = bool(frame_analysis)
    if "pixelsize" in params:
        kwargs["pixelsize"] = float(pixelsize_nm)
    if "return_info" in params:
        kwargs["return_info"] = True

    if kwargs:
        try:
            return clusterer.cluster(**kwargs)
        except TypeError:
            pass

    # Fallback for older/simple positional signatures
    return clusterer.cluster(
        locs,
        float(radius_xy_px),
        float(radius_z_px),
        int(min_locs),
        bool(frame_analysis),
        float(pixelsize_nm),
    )

def run_smlm_clusterer(
    locs,
    info,
    radius_xy_px,
    radius_z_px,
    min_locs,
    frame_analysis,
):
    """
    Run Picasso's built-in SMLM clusterer.

    Compatible with Picasso versions where clusterer.cluster()
    does or does not return cluster_info.
    """
    pixelsize_nm = get_pixelsize_nm(info)

    result = call_clusterer_compatible(
        locs=locs,
        radius_xy_px=radius_xy_px,
        radius_z_px=radius_z_px,
        min_locs=min_locs,
        frame_analysis=frame_analysis,
        pixelsize_nm=pixelsize_nm,
    )

    if isinstance(result, tuple) and len(result) == 2:
        clustered_locs, cluster_info = result
    else:
        clustered_locs = result
        cluster_info = {
            "Generated by": "Picasso clusterer.cluster",
            "Note": "This Picasso version did not return cluster_info.",
            "Clustering radius xy px": float(radius_xy_px),
            "Clustering radius z px": float(radius_z_px),
            "Min localizations per cluster": int(min_locs),
            "Basic frame analysis": bool(frame_analysis),
            "Pixelsize nm": float(pixelsize_nm),
        }

    return clustered_locs, cluster_info

def save_cluster_centers_if_requested(clustered_locs, info, hdf5_path, pixelsize_nm):
    if not SAVE_CLUSTER_CENTERS:
        return None, None

    from picasso import io, clusterer

    centers_out_path = centers_output_path_for_input(hdf5_path)
    centers_out_path.parent.mkdir(parents=True, exist_ok=True)

    centers = clusterer.find_cluster_centers(
        clustered_locs,
        pixelsize=float(pixelsize_nm),
    )

    centers_info = list(info)
    centers_info.append(
        {
            "Generated by": "Picasso SMLM clusterer center extraction",
            "Source clustered file": str(hdf5_path),
            "Pixelsize nm": float(pixelsize_nm),
        }
    )

    io.save_locs(str(centers_out_path), centers, centers_info)

    return str(centers_out_path), len(centers)

# ============================================================
# WORKER
# ============================================================
def process_one_file(job):
    input_path_str, output_path_str = job

    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    try:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if output_path.exists() and not OVERWRITE:
            return {
                "status": "skipped_exists",
                "input": str(input_path),
                "output": str(output_path),
                "n_input": None,
                "n_clustered": None,
                "n_clusters": None,
                "nena_px": None,
                "nena_nm": None,
                "radius_xy_px": None,
                "radius_xy_nm": None,
                "radius_z_px": None,
                "radius_z_nm": None,
                "centers_output": None,
                "n_centers": None,
                "columns": None,
                "error": None,
            }

        if DRY_RUN:
            return {
                "status": "dry_run",
                "input": str(input_path),
                "output": str(output_path),
                "n_input": None,
                "n_clustered": None,
                "n_clusters": None,
                "nena_px": None,
                "nena_nm": None,
                "radius_xy_px": None,
                "radius_xy_nm": None,
                "radius_z_px": None,
                "radius_z_nm": None,
                "centers_output": None,
                "n_centers": None,
                "columns": None,
                "error": None,
            }

        from picasso import io
        import numpy as np

        output_path.parent.mkdir(parents=True, exist_ok=True)

        locs, info = io.load_locs(str(input_path))
        n_input = len(locs)
        columns = get_column_names(locs)

        if REQUIRE_3D and not has_column(locs, "z"):
            raise ValueError(
                f"Input does not contain a z column, but REQUIRE_3D=True: {input_path}\n"
                f"Available columns: {columns}"
            )

        pixelsize_nm = get_pixelsize_nm(info)

        # Compute NeNA from this file.
        # NeNA returned by Picasso is in camera pixels.
        nena_px = compute_nena_px(locs, info)
        nena_nm = nena_px * pixelsize_nm

        # Picasso recommendation: radius_xy = 2 * NeNA
        radius_xy_px = NENA_RADIUS_MULTIPLIER * nena_px
        radius_xy_nm = radius_xy_px * pixelsize_nm

        # Derived 3D z radius
        radius_z_nm = Z_RADIUS_FACTOR * radius_xy_nm
        radius_z_px = radius_z_nm / pixelsize_nm

        clustered_locs, cluster_info = run_smlm_clusterer(
            locs=locs,
            info=info,
            radius_xy_px=radius_xy_px,
            radius_z_px=radius_z_px,
            min_locs=MIN_LOCS,
            frame_analysis=FRAME_ANALYSIS,
        )

        n_clustered = len(clustered_locs)

        if len(clustered_locs) > 0 and has_column(clustered_locs, "group"):
            group_values = get_column(clustered_locs, "group")
            n_clusters = int(len(np.unique(group_values)))
        else:
            n_clusters = 0

        out_info = list(info)
        out_info.append(
            {
                "Generated by": "Picasso SMLM clusterer via script",
                "Input file": str(input_path),
                "Input columns": columns,
                "NeNA xy px": float(nena_px),
                "NeNA xy nm": float(nena_nm),
                "NeNA radius multiplier": float(NENA_RADIUS_MULTIPLIER),
                "Clustering radius xy px": float(radius_xy_px),
                "Clustering radius xy nm": float(radius_xy_nm),
                "Z radius factor": float(Z_RADIUS_FACTOR),
                "Clustering radius z px": float(radius_z_px),
                "Clustering radius z nm": float(radius_z_nm),
                "Min localizations per cluster": int(MIN_LOCS),
                "Basic frame analysis": bool(FRAME_ANALYSIS),
                "Pixelsize nm": float(pixelsize_nm),
            }
        )
        out_info.append(cluster_info)

        # save_locs writes both .hdf5 and .yaml
        io.save_locs(str(output_path), clustered_locs, out_info)

        centers_output, n_centers = save_cluster_centers_if_requested(
            clustered_locs=clustered_locs,
            info=out_info,
            hdf5_path=input_path,
            pixelsize_nm=pixelsize_nm,
        )

        return {
            "status": "clustered",
            "input": str(input_path),
            "output": str(output_path),
            "n_input": int(n_input),
            "n_clustered": int(n_clustered),
            "n_clusters": int(n_clusters),
            "nena_px": float(nena_px),
            "nena_nm": float(nena_nm),
            "radius_xy_px": float(radius_xy_px),
            "radius_xy_nm": float(radius_xy_nm),
            "radius_z_px": float(radius_z_px),
            "radius_z_nm": float(radius_z_nm),
            "centers_output": centers_output,
            "n_centers": n_centers,
            "columns": columns,
            "error": None,
        }

    except Exception:
        return {
            "status": "error",
            "input": str(input_path),
            "output": str(output_path),
            "n_input": None,
            "n_clustered": None,
            "n_clusters": None,
            "nena_px": None,
            "nena_nm": None,
            "radius_xy_px": None,
            "radius_xy_nm": None,
            "radius_z_px": None,
            "radius_z_nm": None,
            "centers_output": None,
            "n_centers": None,
            "columns": None,
            "error": traceback.format_exc(),
        }

# ============================================================
# PRINTING
# ============================================================
def print_result(i, total, result):
    in_name = Path(result["input"]).name
    out_name = Path(result["output"]).name
    status = result["status"]

    if status == "clustered":
        print(
            f"[{i}/{total}] clustered: {in_name} -> {out_name} | "
            f"locs {result['n_input']} -> {result['n_clustered']} | "
            f"clusters {result['n_clusters']} | "
            f"NeNA {result['nena_nm']:.2f} nm | "
            f"r_xy {result['radius_xy_nm']:.2f} nm | "
            f"r_z {result['radius_z_nm']:.2f} nm"
        )

        if result["centers_output"] is not None:
            print(
                f"    centers: {Path(result['centers_output']).name} "
                f"n={result['n_centers']}"
            )

    elif status == "dry_run":
        print(f"[{i}/{total}] DRY RUN: {in_name} -> {out_name}")

    elif status == "skipped_exists":
        print(f"[{i}/{total}] SKIP exists: {out_name}")

    elif status == "error":
        print(f"[{i}/{total}] ERROR: {in_name}")

    else:
        print(f"[{i}/{total}] {status}: {in_name}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("Picasso SMLM clusterer script")
    print("=" * 80)
    print(f"SCRIPT_DIR:              {SCRIPT_DIR}")
    print(f"PROJECT_DIR:             {PROJECT_DIR}")
    print(f"INPUT_DIR:               {INPUT_DIR}")
    print(f"OUTPUT_DIR:              {OUTPUT_DIR}")
    print(f"DRY_RUN:                 {DRY_RUN}")
    print(f"OVERWRITE:               {OVERWRITE}")
    print(f"USE_PARALLEL:            {USE_PARALLEL}")
    print(f"MAX_WORKERS:             {MAX_WORKERS}")
    print(f"NENA_RADIUS_MULTIPLIER:  {NENA_RADIUS_MULTIPLIER}")
    print(f"Z_RADIUS_FACTOR:         {Z_RADIUS_FACTOR}")
    print(f"MIN_LOCS:                {MIN_LOCS}")
    print(f"FRAME_ANALYSIS:          {FRAME_ANALYSIS}")
    print(f"REQUIRE_3D:              {REQUIRE_3D}")
    print(f"SAVE_CLUSTER_CENTERS:    {SAVE_CLUSTER_CENTERS}")
    print("=" * 80)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_DIR}")

    hdf5_files = find_input_hdf5_files()

    print(f"\nFound allowed HDF5 files: {len(hdf5_files)}")

    if len(hdf5_files) == 0:
        print("No HDF5 files found. Check folder name and filename pattern.")
        return

    jobs = []

    for input_path in hdf5_files:
        output_path = output_path_for_input(input_path)
        jobs.append((str(input_path), str(output_path)))

    print("\nFirst few jobs:")
    for src, dst in jobs[:10]:
        print(f"  FROM: {src}")
        print(f"  TO:   {dst}")

    if len(jobs) > 10:
        print(f"  ... plus {len(jobs) - 10} more")

    # Print installed Picasso cluster signature once for debugging
    try:
        from picasso import clusterer

        print("\nInstalled Picasso clusterer.cluster signature:")
        print(inspect.signature(clusterer.cluster))
    except Exception as e:
        print("\nCould not inspect Picasso clusterer.cluster signature:")
        print(repr(e))

    print("\n" + "=" * 80)
    print("Running SMLM clustering")
    print("=" * 80)

    results = []

    if USE_PARALLEL and not DRY_RUN:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_one_file, job) for job in jobs]

            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print_result(i, len(jobs), result)

    else:
        for i, job in enumerate(jobs, start=1):
            result = process_one_file(job)
            results.append(result)
            print_result(i, len(jobs), result)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    counts = Counter(r["status"] for r in results)

    for status, n in counts.items():
        print(f"{status}: {n}")

    clustered = [r for r in results if r["status"] == "clustered"]

    if clustered:
        print("\nClustered file parameter summary:")
        for r in clustered:
            print(
                f"{Path(r['input']).name}: "
                f"NeNA={r['nena_nm']:.2f} nm, "
                f"r_xy={r['radius_xy_nm']:.2f} nm, "
                f"r_z={r['radius_z_nm']:.2f} nm, "
                f"clusters={r['n_clusters']}, "
                f"locs_kept={r['n_clustered']}/{r['n_input']}"
            )

    errors = [r for r in results if r["status"] == "error"]

    if errors:
        print("\n" + "=" * 80)
        print("ERROR DETAILS")
        print("=" * 80)

        for r in errors:
            print(f"\nInput:  {r['input']}")
            print(f"Output: {r['output']}")
            print(r["error"])

    if DRY_RUN:
        print("\nDRY RUN was enabled. No files were written.")
        print("Set DRY_RUN = False to actually run SMLM clustering.")

if __name__ == "__main__":
    main()