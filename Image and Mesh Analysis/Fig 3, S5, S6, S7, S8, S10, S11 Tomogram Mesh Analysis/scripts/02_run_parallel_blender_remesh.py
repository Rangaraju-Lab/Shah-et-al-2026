# ============================================================
# Parallel launcher for Blender remesh jobs
#
# Run with normal Python:
# python run_parallel_blender_remesh.py
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

TARGET_STEMS = [
    "OMM",
    "IMS",
    "IBM",
    "ER",
    "Ribo",
]

DEFAULT_VOXEL_SIZE = 0.004

VOXEL_SIZE_BY_STEM = {
    "OMM": 0.004,
    "IMS": 0.004,
    "IBM": 0.004,
    "ER": 0.004,
    "Ribo": 0.002,
}

SKIP_EXISTING = False

# ============================================================
# BLENDER PATH
# ============================================================
# On macOS, Finder may show this as:
#
# But the real path is usually:
#
# You can leave BLENDER_EXE as None to auto-detect.
BLENDER_EXE = None

# Optional manual override examples:

WORKER_SCRIPT_NAME = "02b_blender_remesh_worker.py"

# Number of Blender instances running at once.
# On Mac, 8 may be aggressive depending on mesh size.
# Start with 2-4 if Blender crashes or memory pressure is high.
MAX_PARALLEL_BLENDER = BLENDER_WORKERS

FAILURE_LOG_NAME = "parallel_blender_remesh_failures.csv"

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

            for stem in TARGET_STEMS:
                in_path = meshes_dir / f"{stem}.stl"

                if not in_path.exists():
                    continue

                out_path = meshes_dir / f"{stem}_remeshed.stl"

                if SKIP_EXISTING and out_path.exists():
                    continue

                voxel_size = VOXEL_SIZE_BY_STEM.get(stem, DEFAULT_VOXEL_SIZE)

                jobs.append({
                    "root_name": root_name,
                    "folder": str(folder),
                    "meshes_dir": str(meshes_dir),
                    "stem": stem,
                    "voxel_size": voxel_size,
                    "in_path": str(in_path),
                    "out_path": str(out_path),
                })

    return jobs, roots

# ============================================================
# BLENDER JOB RUNNER
# ============================================================
def run_one_blender_job(job, blender_exe, worker_script):
    in_path = job["in_path"]
    out_path = job["out_path"]
    voxel_size = str(job["voxel_size"])

    cmd = [
        str(blender_exe),
        "--background",
        "--python-exit-code", "1",
        "--python",
        str(worker_script),
        "--",
        in_path,
        out_path,
        voxel_size,
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

    if completed.returncode != 0:
        failed = dict(job)
        failed["returncode"] = completed.returncode
        failed["stdout"] = completed.stdout
        failed["stderr"] = completed.stderr
        return {
            "status": "failed",
            "job": failed,
        }

    return {
        "status": "ok",
        "job": job,
    }

# ============================================================
# FAILURE LOG
# ============================================================
def write_failure_log(path, failed_jobs):
    if not failed_jobs:
        return

    fieldnames = [
        "root_name",
        "folder",
        "meshes_dir",
        "stem",
        "voxel_size",
        "in_path",
        "out_path",
        "returncode",
        "stdout",
        "stderr",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in failed_jobs:
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
    print(f"Found jobs: {len(jobs)}")
    print(f"MAX_PARALLEL_BLENDER: {MAX_PARALLEL_BLENDER}")
    print(f"SKIP_EXISTING: {SKIP_EXISTING}")
    print("Voxel sizes:")
    for stem in TARGET_STEMS:
        print(f"  {stem}: {VOXEL_SIZE_BY_STEM.get(stem, DEFAULT_VOXEL_SIZE)}")
    print("")

    if len(jobs) == 0:
        print("No jobs found.")
        return

    ok_count = 0
    failed_jobs = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BLENDER) as executor:
        futures = [
            executor.submit(run_one_blender_job, job, blender_exe, worker_script)
            for job in jobs
        ]

        with tqdm(
            total=len(futures),
            desc="Blender remesh jobs",
            unit="mesh",
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(futures):
                result = future.result()

                if result["status"] == "ok":
                    ok_count += 1
                else:
                    failed_jobs.append(result["job"])

                pbar.update(1)

    failure_log_path = script_dir / FAILURE_LOG_NAME
    write_failure_log(failure_log_path, failed_jobs)

    print("")
    print("Done.")
    print(f"Successful remesh jobs: {ok_count}")
    print(f"Failed remesh jobs:     {len(failed_jobs)}")

    if failed_jobs:
        print(f"Failure log saved to: {failure_log_path}")

if __name__ == "__main__":
    main()
