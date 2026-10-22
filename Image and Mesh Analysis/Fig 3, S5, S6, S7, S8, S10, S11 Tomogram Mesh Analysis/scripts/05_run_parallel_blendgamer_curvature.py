# ============================================================
# Parallel launcher for Blender/GAMer smooth K1 computation
#
# Run with normal Python:
#   python run_parallel_blendgamer_curvature.py
#
# This launches multiple Blender background processes in parallel.
# Compatible with macOS Blender .app bundles and Windows blender.exe.
# ============================================================

from project_paths import WORKERS, BLENDER_WORKERS
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import csv
import os
import platform
from tqdm import tqdm

# ============================================================
# SETTINGS
# ============================================================
ROOT_NAMES = ["Control", "TTX"]

SCALED_FOLDER_NAME = "scaled"
MESHES_FOLDER_NAME = "meshes"

INPUT_NAME = "Matrix_vol.stl"
OUTPUT_NAME = "smooth_k1"

WORKER_SCRIPT_NAME = "05b_blendgamer_curvature_worker.py"

# ============================================================
# BLENDER PATH
# ============================================================
# Leave as None to auto-detect Blender on Mac.
BLENDER_EXE = None

# Manual override examples:

# Start with 2. If memory is fine, try 3.
# GAMer curvature can be memory-heavy, so avoid going too high if it crashes.
MAX_PARALLEL_BLENDER = BLENDER_WORKERS

SKIP_EXISTING = False

FAILURE_LOG_NAME = "parallel_blendgamer_curvature_failures.csv"
SUMMARY_LOG_NAME = "parallel_blendgamer_curvature_summary.csv"

# ============================================================
# PATH HELPERS
# ============================================================
def get_script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def app_bundle_to_executable(app_path):
    app_path = Path(app_path)

    candidates = [
        app_path / "Contents" / "MacOS" / "Blender",
        app_path / "Contents" / "MacOS" / "blender",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None

def resolve_blender_exe():
    from project_paths import resolve_blender
    return resolve_blender(BLENDER_EXE)

# ============================================================
# JOB DISCOVERY
# ============================================================
def discover_jobs(script_dir):
    roots = {
        root_name: (script_dir.parent / root_name).resolve()
        for root_name in ROOT_NAMES
    }

    jobs = []

    for root_name, root in roots.items():
        if not root.exists():
            continue

        direct_subfolders = sorted([
            p for p in root.iterdir()
            if p.is_dir()
        ])

        for folder in direct_subfolders:
            meshes_dir = folder / SCALED_FOLDER_NAME / MESHES_FOLDER_NAME

            if not meshes_dir.exists():
                continue

            stl_path = meshes_dir / INPUT_NAME
            out_path = meshes_dir / OUTPUT_NAME

            if not stl_path.exists():
                continue

            if SKIP_EXISTING and out_path.exists():
                continue

            jobs.append({
                "root_name": root_name,
                "folder": str(folder),
                "meshes_dir": str(meshes_dir),
                "stl_path": str(stl_path),
                "out_path": str(out_path),
            })

    return jobs, roots

# ============================================================
# JOB RUNNER
# ============================================================
def run_one_job(job, blender_exe, worker_script):
    stl_path = job["stl_path"]
    out_path = job["out_path"]

    cmd = [
        str(blender_exe),
        "--background",
        "--python-exit-code", "1",
        "--python",
        str(worker_script),
        "--",
        stl_path,
        out_path,
    ]

    env = os.environ.copy()

    # Keep each Blender process from over-threading.
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    row = dict(job)
    row["returncode"] = completed.returncode
    row["stdout"] = completed.stdout
    row["stderr"] = completed.stderr

    if completed.returncode != 0:
        row["status"] = "failed"
        return row

    if not Path(out_path).exists():
        row["status"] = "failed"
        row["stderr"] = row["stderr"] + "\nOutput file was not created."
        return row

    row["status"] = "ok"
    return row

# ============================================================
# CSV LOGGING
# ============================================================
def write_csv(path, rows):
    if not rows:
        return

    fieldnames = [
        "status",
        "root_name",
        "folder",
        "meshes_dir",
        "stl_path",
        "out_path",
        "returncode",
        "stdout",
        "stderr",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = get_script_dir()
    worker_script = script_dir / WORKER_SCRIPT_NAME

    blender_exe, checked_paths = resolve_blender_exe()

    if blender_exe is None:
        raise FileNotFoundError("Blender not found. Install the CellBlender bundle and run setup_env.py, or pass --blender to setup_env.py.")

    if not worker_script.exists():
        raise FileNotFoundError(f"Worker script not found: {worker_script}")

    jobs, roots = discover_jobs(script_dir)

    print("Script dir:", script_dir)

    for root_name, root in roots.items():
        print(f"{root_name} root:", root)

    print("")
    print("Blender executable:")
    print(blender_exe)
    print("")
    print(f"Found Matrix_vol.stl jobs: {len(jobs)}")
    print(f"MAX_PARALLEL_BLENDER: {MAX_PARALLEL_BLENDER}")
    print(f"SKIP_EXISTING: {SKIP_EXISTING}")
    print("")

    if len(jobs) == 0:
        print("No jobs found.")
        return

    results = []
    ok_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BLENDER) as executor:
        futures = [
            executor.submit(run_one_job, job, blender_exe, worker_script)
            for job in jobs
        ]

        with tqdm(
            total=len(futures),
            desc="Computing smooth_k1",
            unit="mesh",
            dynamic_ncols=True,
        ) as pbar:

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result["status"] == "ok":
                    ok_count += 1
                else:
                    failed_count += 1

                pbar.update(1)

    summary_path = script_dir / SUMMARY_LOG_NAME
    write_csv(summary_path, results)

    failure_rows = [
        row for row in results
        if row["status"] != "ok"
    ]

    failure_path = script_dir / FAILURE_LOG_NAME
    write_csv(failure_path, failure_rows)

    print("")
    print("Done.")
    print(f"Successful smooth_k1 files: {ok_count}")
    print(f"Failed jobs:                {failed_count}")
    print(f"Summary log saved to:       {summary_path}")

    if failure_rows:
        print(f"Failure log saved to:       {failure_path}")

if __name__ == "__main__":
    main()
