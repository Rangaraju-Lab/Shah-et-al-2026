from pathlib import Path
import re

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE = True
FRAME_BIN_SIZE = 1200

PROBE00 = "probe00"  # ATP synthase
PROBE01 = "probe01"  # Mic60

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_DIR = SCRIPT_DIR.parent
CENTERS_DIR = PROJECT_DIR / "Undrift Masked SMLM Centers"
OUTPUT_DIR = PROJECT_DIR / "Measurements" / "molecules_through_frames"
OUTPUT_CSV = OUTPUT_DIR / "molecules_through_frames.csv"

CSV_COLUMNS = ["Date", "Condition", "Dish", "Cell", "frame_bin", "ATP_synthase", "Mic60"]

# ============================================================
# FILENAME PARSING
# ============================================================
def normalize_name_text(text):
    text = Path(str(text)).name
    text = re.sub(r"\.hdf5$", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]

def parse_hdf5_filename(path):
    """
    Expected filenames:

        2026-01-08 Control Dish1 Cell1 probe00.hdf5
        2026-01-08 Homeostatic Scaling Dish1 Cell1 probe01.hdf5
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

    if match is None:
        return None

    metadata = match.groupdict()
    metadata["Probe"] = metadata["Probe"].lower()
    metadata["prefix"] = f"{metadata['Date']} {metadata['Condition']} {metadata['Dish']} {metadata['Cell']}"
    return metadata

def parse_integer_label(value, label):
    match = re.search(r"\d+", str(value))

    if not match:
        raise ValueError(f"Could not extract an integer from {label}: {value}")

    return int(match.group())

# ============================================================
# PICASSO HDF5 HELPERS
# ============================================================
def has_column(localizations, column):
    if hasattr(localizations, "columns"):
        return column in localizations.columns

    return (
        hasattr(localizations, "dtype")
        and localizations.dtype.names is not None
        and column in localizations.dtype.names
    )

def get_column(localizations, column):
    if hasattr(localizations, "columns"):
        return localizations[column].to_numpy()

    return localizations[column]

def get_column_names(localizations):
    if hasattr(localizations, "columns"):
        return list(localizations.columns)

    if hasattr(localizations, "dtype") and localizations.dtype.names is not None:
        return list(localizations.dtype.names)

    return []

def get_metadata_frame_count(info):
    """Read the total acquisition frame count from the Picasso YAML information."""
    frame_counts = []

    if info is None:
        return None

    for record in info:
        if not hasattr(record, "items"):
            continue

        for key, value in record.items():
            if str(key).strip().lower() != "frames":
                continue

            try:
                value = int(float(value))
            except (TypeError, ValueError):
                continue

            if value > 0:
                frame_counts.append(value)

    return max(frame_counts) if frame_counts else None

def load_center_frames(path):
    """
    Load one cluster-center HDF5.

    Each valid row is one molecule. The frame column assigns that
    molecule to exactly one frame and therefore exactly one frame bin.
    """
    from picasso import io

    centers, info = io.load_locs(str(path))

    required_columns = ["x", "y", "frame"]
    missing_columns = [column for column in required_columns if not has_column(centers, column)]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path.name}: {missing_columns}\n"
            f"Available columns: {get_column_names(centers)}"
        )

    x = np.asarray(get_column(centers, "x"), dtype=np.float64)
    y = np.asarray(get_column(centers, "y"), dtype=np.float64)
    frame = np.asarray(get_column(centers, "frame"), dtype=np.float64)

    if not len(x) == len(y) == len(frame):
        raise RuntimeError(f"x, y, and frame column lengths do not match in {path.name}.")

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(frame) & (frame >= 0)
    valid_frames = frame[valid]
    metadata_frames = get_metadata_frame_count(info)

    observed_frame_count = int(np.floor(valid_frames.max())) + 1 if valid_frames.size > 0 else 0
    total_frames = observed_frame_count if metadata_frames is None else max(metadata_frames, observed_frame_count)

    return {
        "frames": valid_frames,
        "total_frames": total_frames,
        "n_centers": int(valid_frames.size),
        "n_invalid": int(len(frame) - valid_frames.size),
    }

# ============================================================
# FILE DISCOVERY
# ============================================================
def build_probe_index():
    """
    Returns:

        index[prefix]["probe00"] = path
        index[prefix]["probe01"] = path
    """
    index = {}
    metadata_index = {}
    skipped = []

    files = sorted(CENTERS_DIR.glob("*.hdf5"), key=lambda path: natural_key(path.name))

    for path in files:
        metadata = parse_hdf5_filename(path)

        if metadata is None:
            skipped.append((path, "unrecognized filename"))
            continue

        probe = metadata["Probe"]

        if probe not in {PROBE00, PROBE01}:
            skipped.append((path, f"unused probe: {probe}"))
            continue

        prefix = metadata["prefix"]

        if prefix not in index:
            index[prefix] = {}
            metadata_index[prefix] = metadata

        if probe in index[prefix]:
            raise RuntimeError(
                f"Duplicate {probe} file for {prefix}:\n"
                f"  Existing: {index[prefix][probe]}\n"
                f"  Duplicate: {path}"
            )

        index[prefix][probe] = path

    return index, metadata_index, skipped

# ============================================================
# COUNT CENTERS IN ONE FRAME BIN
# ============================================================
def count_centers_in_bin(center_frames, frame_start, frame_end):
    """
    Uses a non-overlapping, half-open frame interval:

        frame_start <= frame < frame_end
    """
    in_bin = (center_frames >= frame_start) & (center_frames < frame_end)
    return int(np.count_nonzero(in_bin))

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("MOLECULES THROUGH FRAMES")
    print("=" * 80)
    print(f"Centers directory: {CENTERS_DIR}")
    print(f"Frame bin size:    {FRAME_BIN_SIZE}")
    print(f"Output CSV:        {OUTPUT_CSV}")
    print("=" * 80)

    if not CENTERS_DIR.exists():
        raise FileNotFoundError(f"Cluster-center directory was not found:\n{CENTERS_DIR}")

    probe_index, metadata_index, skipped = build_probe_index()

    if skipped:
        print("\nSkipped files:")
        for path, reason in skipped:
            print(f"  {path.name}: {reason}")

    if not probe_index:
        raise FileNotFoundError(
            f"No valid probe00 or probe01 center files were found in:\n{CENTERS_DIR}"
        )

    incomplete_sets = []

    for prefix, probe_files in probe_index.items():
        missing = [probe for probe in (PROBE00, PROBE01) if probe not in probe_files]

        if missing:
            incomplete_sets.append(f"{prefix}: missing {', '.join(missing)}")

    if incomplete_sets:
        raise RuntimeError("Incomplete probe pairs were found:\n  " + "\n  ".join(incomplete_sets))

    rows = []
    prefixes = sorted(probe_index, key=natural_key)

    for file_number, prefix in enumerate(prefixes, start=1):
        metadata = metadata_index[prefix]
        dish_integer = parse_integer_label(metadata["Dish"], "Dish")
        cell_integer = parse_integer_label(metadata["Cell"], "Cell")

        probe00_path = probe_index[prefix][PROBE00]
        probe01_path = probe_index[prefix][PROBE01]

        print("\n" + "-" * 80)
        print(f"[{file_number}/{len(prefixes)}] {prefix}")
        print(f"ATP synthase: {probe00_path.name}")
        print(f"Mic60:        {probe01_path.name}")

        atp_data = load_center_frames(probe00_path)
        mic60_data = load_center_frames(probe01_path)
        total_frames = max(atp_data["total_frames"], mic60_data["total_frames"])

        if total_frames <= 0:
            raise ValueError(f"Could not determine the acquisition frame count for {prefix}.")

        if atp_data["total_frames"] != mic60_data["total_frames"]:
            print(
                f"WARNING: Probe frame counts differ. "
                f"ATP synthase={atp_data['total_frames']}, "
                f"Mic60={mic60_data['total_frames']}. Using {total_frames}."
            )

        print(f"ATP synthase molecules: {atp_data['n_centers']}")
        print(f"Mic60 molecules:        {mic60_data['n_centers']}")

        if atp_data["n_invalid"] > 0:
            print(f"WARNING: ATP synthase excluded {atp_data['n_invalid']} invalid center rows.")

        if mic60_data["n_invalid"] > 0:
            print(f"WARNING: Mic60 excluded {mic60_data['n_invalid']} invalid center rows.")

        saved_atp_total = 0
        saved_mic60_total = 0

        for frame_start in range(0, total_frames, FRAME_BIN_SIZE):
            frame_end = frame_start + FRAME_BIN_SIZE
            atp_count = count_centers_in_bin(atp_data["frames"], frame_start, frame_end)
            mic60_count = count_centers_in_bin(mic60_data["frames"], frame_start, frame_end)

            saved_atp_total += atp_count
            saved_mic60_total += mic60_count

            rows.append(
                {
                    "Date": metadata["Date"],
                    "Condition": metadata["Condition"],
                    "Dish": int(dish_integer),
                    "Cell": int(cell_integer),
                    "frame_bin": f"{frame_start}-{frame_end}",
                    "ATP_synthase": atp_count,
                    "Mic60": mic60_count,
                    "_frame_start": frame_start,
                }
            )

            print(
                f"  {frame_start}-{frame_end}: "
                f"ATP synthase={atp_count}, Mic60={mic60_count}"
            )

        if saved_atp_total != atp_data["n_centers"]:
            raise RuntimeError(
                f"ATP synthase bin total does not match the number of centers for {prefix}.\n"
                f"Centers: {atp_data['n_centers']}\n"
                f"Binned:  {saved_atp_total}"
            )

        if saved_mic60_total != mic60_data["n_centers"]:
            raise RuntimeError(
                f"Mic60 bin total does not match the number of centers for {prefix}.\n"
                f"Centers: {mic60_data['n_centers']}\n"
                f"Binned:  {saved_mic60_total}"
            )

    output_df = pd.DataFrame(rows)
    output_df = output_df.sort_values(
        ["Date", "Condition", "Dish", "Cell", "_frame_start"]
    ).reset_index(drop=True)
    output_df = output_df[CSV_COLUMNS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() and not OVERWRITE:
        raise FileExistsError(f"Output already exists and OVERWRITE=False:\n{OUTPUT_CSV}")

    output_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Datasets processed: {len(prefixes)}")
    print(f"Rows saved:         {len(output_df)}")
    print(f"Saved:              {OUTPUT_CSV}")

if __name__ == "__main__":
    main()