from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import traceback
import re

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE = True

USE_PARALLEL = True
MAX_WORKERS = 4

# Set True to process only the first Date/Condition/Dish/Cell set.
TEST_FIRST_FILE = False

PROBE00 = "probe00"  # ATP synthase
PROBE01 = "probe01"  # Mic60

# Picasso SMLM cluster assignments are stored here.
CLUSTER_COLUMN = "group"

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_DIR = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_DIR / "Undrift Masked SMLM"

OUTPUT_DIR = PROJECT_DIR / "Measurements" / "locs_per_cluster"
OUTPUT_CSV = OUTPUT_DIR / "locs_per_cluster.csv"

CSV_COLUMNS = [
    "Date",
    "Condition",
    "Dish",
    "Cell",
    "cluster_id",
    "ATP_synthase_locs",
    "Mic60_locs",
]

# ============================================================
# FILENAME PARSING
# ============================================================
def normalize_name_text(text):
    text = Path(str(text)).name
    text = re.sub(r"\.hdf5$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]

def parse_hdf5_filename(path):
    """
    Expected filenames:

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

    if match is None:
        return None

    metadata = match.groupdict()
    metadata["Probe"] = metadata["Probe"].lower()
    metadata["prefix"] = (
        f"{metadata['Date']} {metadata['Condition']} "
        f"{metadata['Dish']} {metadata['Cell']}"
    )

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

# ============================================================
# COUNT LOCALIZATIONS PER CLUSTER
# ============================================================
def load_cluster_counts(path):
    """
    Read a clustered Picasso HDF5 file and count localization
    rows belonging to each cluster/group.

    Returns:
        {
            cluster_id: number_of_localizations,
            ...
        }
    """
    from picasso import io

    localizations, _ = io.load_locs(str(path))

    if not has_column(localizations, CLUSTER_COLUMN):
        raise ValueError(
            f"Missing cluster column {CLUSTER_COLUMN!r} in {path.name}.\n"
            f"Available columns: {get_column_names(localizations)}"
        )

    raw_groups = np.asarray(get_column(localizations, CLUSTER_COLUMN))

    if not np.issubdtype(raw_groups.dtype, np.number):
        raise ValueError(
            f"Cluster column {CLUSTER_COLUMN!r} must be numeric in {path.name}.\n"
            f"Detected dtype: {raw_groups.dtype}"
        )

    group_values = raw_groups.astype(np.float64, copy=False)
    valid = np.isfinite(group_values) & (group_values >= 0)
    group_values = group_values[valid]

    if group_values.size == 0:
        return {
            "counts": {},
            "n_localizations_total": int(len(localizations)),
            "n_localizations_valid": 0,
            "n_localizations_excluded": int(len(localizations)),
            "n_clusters": 0,
        }

    # Cluster IDs should be integers.
    rounded_groups = np.rint(group_values)

    if np.any(np.abs(group_values - rounded_groups) > 1e-6):
        bad_values = np.unique(group_values[np.abs(group_values - rounded_groups) > 1e-6])

        raise ValueError(
            f"Non-integer cluster IDs were found in {path.name}.\n"
            f"Examples: {bad_values[:10]}"
        )

    group_ids = rounded_groups.astype(np.int64)

    unique_groups, localization_counts = np.unique(
        group_ids,
        return_counts=True,
    )

    counts = {
        int(cluster_id): int(n_localizations)
        for cluster_id, n_localizations in zip(unique_groups, localization_counts)
    }

    return {
        "counts": counts,
        "n_localizations_total": int(len(localizations)),
        "n_localizations_valid": int(group_ids.size),
        "n_localizations_excluded": int(len(localizations) - group_ids.size),
        "n_clusters": int(len(unique_groups)),
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

    files = sorted(INPUT_DIR.glob("*.hdf5"), key=lambda path: natural_key(path.name))

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
# PROCESS ONE DATASET
# ============================================================
def process_dataset(job):
    prefix = job["prefix"]
    metadata = job["metadata"]
    probe00_path = job["probe00_path"]
    probe01_path = job["probe01_path"]

    result = {
        "prefix": prefix,
        "status": "started",
        "rows": [],
        "error": "",
        "traceback": "",
        "probe00_path": str(probe00_path),
        "probe01_path": str(probe01_path),
        "n_probe00_clusters": 0,
        "n_probe01_clusters": 0,
        "n_probe00_locs": 0,
        "n_probe01_locs": 0,
        "n_rows": 0,
    }

    try:
        dish_integer = parse_integer_label(metadata["Dish"], "Dish")
        cell_integer = parse_integer_label(metadata["Cell"], "Cell")

        probe00_data = load_cluster_counts(probe00_path)
        probe01_data = load_cluster_counts(probe01_path)

        atp_counts = probe00_data["counts"]
        mic60_counts = probe01_data["counts"]

        result["n_probe00_clusters"] = probe00_data["n_clusters"]
        result["n_probe01_clusters"] = probe01_data["n_clusters"]
        result["n_probe00_locs"] = probe00_data["n_localizations_valid"]
        result["n_probe01_locs"] = probe01_data["n_localizations_valid"]

        # Cluster IDs are generated independently for each probe.
        # This outer union retains every cluster ID occurring in
        # either probe. A missing cluster ID receives zero.
        cluster_ids = sorted(set(atp_counts) | set(mic60_counts))

        rows = []

        for cluster_id in cluster_ids:
            rows.append(
                {
                    "Date": metadata["Date"],
                    "Condition": metadata["Condition"],
                    "Dish": int(dish_integer),
                    "Cell": int(cell_integer),
                    "cluster_id": int(cluster_id),
                    "ATP_synthase_locs": int(atp_counts.get(cluster_id, 0)),
                    "Mic60_locs": int(mic60_counts.get(cluster_id, 0)),
                }
            )

        result["rows"] = rows
        result["n_rows"] = len(rows)
        result["status"] = "processed"

        return result

    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()

        return result

# ============================================================
# PRINT RESULT
# ============================================================
def print_result(current, total, result):
    prefix = result["prefix"]
    status = result["status"]

    print(f"[{current}/{total}] {status.upper()}: {prefix}")

    if status == "processed":
        print(
            f"  ATP synthase: {result['n_probe00_clusters']} clusters, "
            f"{result['n_probe00_locs']} localizations"
        )
        print(
            f"  Mic60:        {result['n_probe01_clusters']} clusters, "
            f"{result['n_probe01_locs']} localizations"
        )
        print(f"  Output rows:  {result['n_rows']}")

    else:
        print(f"  Error: {result['error']}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("LOCALIZATIONS PER CLUSTER")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Cluster column:  {CLUSTER_COLUMN}")
    print(f"Output CSV:      {OUTPUT_CSV}")
    print(f"Parallel:        {USE_PARALLEL}")
    print(f"Workers:         {MAX_WORKERS}")
    print("=" * 80)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory was not found:\n"
            f"{INPUT_DIR}"
        )

    probe_index, metadata_index, skipped = build_probe_index()

    if skipped:
        print("\nSkipped files:")

        for path, reason in skipped:
            print(f"  {path.name}: {reason}")

    if not probe_index:
        raise FileNotFoundError(
            f"No valid probe00 or probe01 HDF5 files were found in:\n"
            f"{INPUT_DIR}"
        )

    incomplete_sets = []

    for prefix, probe_files in probe_index.items():
        missing = [
            probe
            for probe in (PROBE00, PROBE01)
            if probe not in probe_files
        ]

        if missing:
            incomplete_sets.append(
                f"{prefix}: missing {', '.join(missing)}"
            )

    if incomplete_sets:
        raise RuntimeError(
            "Incomplete probe pairs were found:\n  "
            + "\n  ".join(incomplete_sets)
        )

    prefixes = sorted(probe_index, key=natural_key)

    if TEST_FIRST_FILE:
        prefixes = prefixes[:1]

    jobs = []

    for prefix in prefixes:
        jobs.append(
            {
                "prefix": prefix,
                "metadata": metadata_index[prefix],
                "probe00_path": probe_index[prefix][PROBE00],
                "probe01_path": probe_index[prefix][PROBE01],
            }
        )

    print(f"\nDatasets to process: {len(jobs)}")

    results = []

    if USE_PARALLEL and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_job = {
                executor.submit(process_dataset, job): job
                for job in jobs
            }

            completed = 0

            for future in as_completed(future_to_job):
                completed += 1

                try:
                    result = future.result()

                except Exception as error:
                    job = future_to_job[future]

                    result = {
                        "prefix": job["prefix"],
                        "status": "failed",
                        "rows": [],
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }

                results.append(result)
                print_result(completed, len(jobs), result)

    else:
        for index, job in enumerate(jobs, start=1):
            result = process_dataset(job)
            results.append(result)
            print_result(index, len(jobs), result)

    status_counts = Counter(result["status"] for result in results)

    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    failed_results = [
        result
        for result in results
        if result["status"] == "failed"
    ]

    if failed_results:
        print("\nFailed datasets:")

        for result in failed_results:
            print(
                f"\n{result['prefix']}\n"
                f"{result['error']}"
            )

            if result.get("traceback"):
                print(result["traceback"])

        raise RuntimeError(
            f"{len(failed_results)} dataset(s) failed. "
            f"The combined CSV was not saved."
        )

    all_rows = []

    for result in results:
        all_rows.extend(result["rows"])

    output_df = pd.DataFrame(
        all_rows,
        columns=CSV_COLUMNS,
    )

    if not output_df.empty:
        output_df = (
            output_df
            .sort_values(
                [
                    "Date",
                    "Condition",
                    "Dish",
                    "Cell",
                    "cluster_id",
                ]
            )
            .reset_index(drop=True)
        )

        integer_columns = [
            "Dish",
            "Cell",
            "cluster_id",
            "ATP_synthase_locs",
            "Mic60_locs",
        ]

        output_df[integer_columns] = output_df[
            integer_columns
        ].astype(np.int64)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_CSV.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output already exists and OVERWRITE=False:\n"
            f"{OUTPUT_CSV}"
        )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Datasets processed: {len(results)}")
    print(f"Rows saved:         {len(output_df)}")
    print(f"Saved CSV:          {OUTPUT_CSV}")

if __name__ == "__main__":
    main()