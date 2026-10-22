from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import traceback
import warnings
import re

import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
DRY_RUN = False
OVERWRITE = True

USE_PARALLEL = True
MAX_WORKERS = 8

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_DIR = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_DIR / "Undrift Masked"
OUTPUT_DIR = PROJECT_DIR / "Measurements"
OUTPUT_CSV = OUTPUT_DIR / f"{INPUT_DIR.name} NeNA.csv"

ALLOWED_PROBES = {"probe00", "probe01"}

CSV_COLUMNS = [
    "Date",
    "Condition",
    "Dish",
    "Cell",
    "Probe",
    "NeNA_nm",
]

# ============================================================
# FILE HELPERS
# ============================================================
def is_allowed_hdf5(path):
    return path.suffix.lower() == ".hdf5" and path.stem.split()[-1] in ALLOWED_PROBES

def find_input_hdf5_files():
    return [path for path in sorted(INPUT_DIR.glob("*.hdf5")) if is_allowed_hdf5(path)]

def parse_filename(path):
    """
    Expected:
        2026-01-08 Control Dish1 Cell1 probe00.hdf5
    """
    match = re.match(
        r"^(?P<Date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<Condition>.+?)\s+"
        r"(?P<Dish>Dish\d+)\s+"
        r"(?P<Cell>Cell\d+)\s+"
        r"(?P<Probe>probe\d+)$",
        path.stem,
    )

    if not match:
        raise ValueError(
            f"Could not parse filename:\n  {path.name}\n"
            f"Expected format:\n  2026-01-08 Control Dish1 Cell1 probe00.hdf5"
        )

    metadata = match.groupdict()
    metadata["Dish"] = int(re.search(r"\d+", metadata["Dish"]).group())
    metadata["Cell"] = int(re.search(r"\d+", metadata["Cell"]).group())

    return metadata

# ============================================================
# PICASSO HELPERS
# ============================================================
def get_pixelsize_nm(info):
    """Read the pixel size in nm from the Picasso metadata."""
    try:
        from picasso import lib

        return float(lib.get_from_metadata(info, "Pixelsize", raise_error=True))
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
        "Expected a key such as 'Pixelsize'."
    )

def compute_nena_nm(hdf5_path):
    """
    Picasso returns NeNA in pixels. Convert it to nm using the
    pixel size stored in the Picasso metadata.
    """
    from picasso import io, postprocess

    locs, info = io.load_locs(str(hdf5_path))
    pixelsize_nm = get_pixelsize_nm(info)
    _, nena_px = postprocess.nena(locs, info)

    return float(nena_px) * pixelsize_nm

# ============================================================
# WORKER
# ============================================================
def process_one_file(path_str):
    path = Path(path_str)

    try:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        metadata = parse_filename(path)
        nena_nm = compute_nena_nm(path)

        return {
            "status": "ok",
            "path": str(path),
            "Date": metadata["Date"],
            "Condition": metadata["Condition"],
            "Dish": metadata["Dish"],
            "Cell": metadata["Cell"],
            "Probe": metadata["Probe"],
            "NeNA_nm": round(float(nena_nm), 3),
            "error": None,
        }

    except Exception:
        return {
            "status": "error",
            "path": str(path),
            "Date": None,
            "Condition": None,
            "Dish": None,
            "Cell": None,
            "Probe": None,
            "NeNA_nm": None,
            "error": traceback.format_exc(),
        }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("PICASSO NeNA BATCH CALCULATION")
    print("=" * 80)
    print(f"INPUT_DIR:    {INPUT_DIR}")
    print(f"OUTPUT_CSV:   {OUTPUT_CSV}")
    print(f"DRY_RUN:      {DRY_RUN}")
    print(f"OVERWRITE:    {OVERWRITE}")
    print(f"USE_PARALLEL: {USE_PARALLEL}")
    print(f"MAX_WORKERS:  {MAX_WORKERS}")
    print("=" * 80)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_DIR}")

    hdf5_files = find_input_hdf5_files()

    print(f"\nFound allowed HDF5 files: {len(hdf5_files)}")

    if not hdf5_files:
        print("No HDF5 files found. Check the folder and filename pattern.")
        return

    print("\nFirst few files:")

    for path in hdf5_files[:10]:
        print(f"  {path}")

    if len(hdf5_files) > 10:
        print(f"  ... plus {len(hdf5_files) - 10} more")

    if DRY_RUN:
        print("\nDRY_RUN enabled. No CSV will be written.")
        return

    print("\n" + "=" * 80)
    print("COMPUTING NeNA")
    print("=" * 80)

    results = []

    if USE_PARALLEL:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_one_file, str(path)) for path in hdf5_files]

            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                name = Path(result["path"]).name

                if result["status"] == "ok":
                    print(
                        f"[{index}/{len(hdf5_files)}] "
                        f"NeNA: {name} = {result['NeNA_nm']:.3f} nm"
                    )
                else:
                    print(f"[{index}/{len(hdf5_files)}] ERROR: {name}")

    else:
        for index, path in enumerate(hdf5_files, start=1):
            result = process_one_file(str(path))
            results.append(result)

            if result["status"] == "ok":
                print(
                    f"[{index}/{len(hdf5_files)}] "
                    f"NeNA: {path.name} = {result['NeNA_nm']:.3f} nm"
                )
            else:
                print(f"[{index}/{len(hdf5_files)}] ERROR: {path.name}")

    counts = Counter(result["status"] for result in results)

    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    for status, count in counts.items():
        print(f"{status}: {count}")

    error_results = [result for result in results if result["status"] == "error"]

    if error_results:
        print("\n" + "=" * 80)
        print("ERROR DETAILS")
        print("=" * 80)

        for result in error_results:
            print(f"\nInput: {result['path']}")
            print(result["error"])

    ok_results = [result for result in results if result["status"] == "ok"]

    if not ok_results:
        print("\nNo successful NeNA results. Exiting without saving a CSV.")
        return

    output_df = pd.DataFrame(ok_results)[CSV_COLUMNS].copy()

    output_df["Dish"] = output_df["Dish"].astype("int64")
    output_df["Cell"] = output_df["Cell"].astype("int64")
    output_df["NeNA_nm"] = output_df["NeNA_nm"].round(3)

    output_df = output_df.sort_values(
        ["Date", "Condition", "Dish", "Cell", "Probe"],
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() and not OVERWRITE:
        raise FileExistsError(f"CSV already exists and OVERWRITE=False: {OUTPUT_CSV}")

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.3f",
    )

    print("\nSaved CSV:")
    print(f"  {OUTPUT_CSV}")
    print("\nDone.")

if __name__ == "__main__":
    main()