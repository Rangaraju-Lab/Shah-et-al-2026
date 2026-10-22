from pathlib import Path
import argparse
import csv
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
PRIMARY = [f"{i:02d}" for i in range(14)]
OPTIONAL = ["08b", "13b"]
FAILURES = {
    "00": "scale_crop_failures.csv", "01": "mesh_reconstruction_failures.csv",
    "02": "parallel_blender_remesh_failures.csv", "03": "make_vol_mesh_failures.csv",
    "04": "make_matrix_vol_failures.csv", "05": "parallel_blendgamer_curvature_failures.csv",
    "06": "make_surfs_and_highk1_failures.csv", "07": "CJ_failures.csv",
    "08": "mesh_summary_failures.csv", "08b": "mesh_summary_failures.csv",
    "09": "ERMCS_mesh_summary_failures.csv", "10": "Ribo_mesh_summary_failures.csv",
    "11": "cristae snaps/cristae_snaps_failures.csv", "12": "curvature snaps/curvature_snaps_failures.csv",
    "13": "ERMCS snaps/ERMCS_snaps_failures.csv", "13b": "ERMCS snaps/Control_1_and_2_ERMCS_failures.csv",
}

def stage_script(stage):
    matches = list(SCRIPTS.glob(stage + "_*.py"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one script for stage {stage}; found {len(matches)}.")
    return matches[0]

def failure_path(stage):
    return (ROOT if stage in ("11", "12", "13", "13b") else SCRIPTS) / FAILURES[stage]

def signature(path):
    return (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None

def main():
    parser = argparse.ArgumentParser(description="Run the ordered tomogram mesh analysis stages.")
    parser.add_argument("--from-stage", choices=PRIMARY, default="00")
    parser.add_argument("--to-stage", choices=PRIMARY, default="13")
    parser.add_argument("--only", choices=PRIMARY + OPTIONAL, help="Run one primary or optional stage.")
    args = parser.parse_args()
    if int(args.from_stage) > int(args.to_stage):
        parser.error("--from-stage must not be later than --to-stage.")
    stages = [args.only] if args.only else PRIMARY[int(args.from_stage):int(args.to_stage) + 1]
    samples = [p for condition in ("Control", "TTX") for p in (ROOT / condition).glob("*") if p.is_dir()]
    if not samples:
        parser.error("No tomogram folders found. Merge sample data into Control and/or TTX first.")
    logs = ROOT / "pipeline logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for stage in stages:
        script = stage_script(stage)
        failed = failure_path(stage)
        before = signature(failed)
        log = logs / (script.stem + ".log")
        print(f"\nStage {stage}: {script.name}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen([sys.executable, str(script)], cwd=SCRIPTS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        if code:
            raise SystemExit(f"Stage {stage} failed with exit code {code}. See {log}.")
        if signature(failed) != before and failed.exists():
            with failed.open(newline="", encoding="utf-8-sig") as handle:
                errors = list(csv.DictReader(handle))
            if errors:
                raise SystemExit(f"Stage {stage} reported {len(errors)} failed or missing jobs. See {failed} and {log}.")
    print("\nRequested stages completed.")

if __name__ == "__main__":
    main()
