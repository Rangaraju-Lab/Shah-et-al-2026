from pathlib import Path
import argparse
import fnmatch
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    "cluster": ("00_smlm_cluster.py", [("Undrift Masked", "*probe0[01].hdf5")]),
    "juxta": ("01_measure_juxtaspinal.py", [("Undrift Masked SMLM Centers", "*.hdf5"), ("Synapse Localizations", "* probe02.hdf5"), ("Dendrites", "*.zip"), ("Mito Masks", "*.tif*")]),
    "nonjuxta": ("02_measure_nonjuxtaspinal.py", [("Undrift Masked SMLM Centers", "*.hdf5"), ("NonJuxta ROI", "*.zip"), ("Dendrites", "*.zip"), ("Mito Masks", "*.tif*")]),
    "nena": ("03_measure_NeNA.py", [("Undrift Masked", "*probe0[01].hdf5")]),
    "locs-per-cluster": ("04_QC_locs_per_cluster.py", [("Undrift Masked SMLM", "*.hdf5")]),
    "frames": ("05_QC_cluster_density_frames.py", [("Undrift Masked SMLM Centers", "*.hdf5")]),
    "per-mito": ("06_measure_probe_density_per_mito.py", [("Undrift Masked SMLM Centers", "*.hdf5"), ("Mito Masks", "*.tif*"), ("Mito Skels", "*.tif*"), ("Dendrites", "*.zip")]),
}
FIG4 = ["cluster", "juxta", "nonjuxta"]
S9 = ["nena", "locs-per-cluster", "frames", "per-mito"]

def main():
    parser = argparse.ArgumentParser(description="Run Fig 4 DNA-PAINT measurements or the S9 analyses.")
    parser.add_argument("--section", choices=["fig4", "s9", "all"], default="fig4")
    parser.add_argument("--only", choices=STAGES, help="Run one selected stage.")
    args = parser.parse_args()
    stages = [args.only] if args.only else (FIG4 if args.section == "fig4" else S9 if args.section == "s9" else FIG4 + S9)
    logs = ROOT / "Run logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for stage in stages:
        filename, inputs = STAGES[stage]
        for folder, pattern in inputs:
            if not (ROOT / folder).is_dir():
                raise SystemExit(f"Missing input folder: {ROOT / folder}. Run setup_env.py and merge the data folders.")
            if not any(p.is_file() and fnmatch.fnmatchcase(p.name.lower(), pattern.lower()) for p in (ROOT / folder).iterdir()):
                raise SystemExit(f"Stage {stage} needs {pattern} in {ROOT / folder}. Merge the matching data or run clustering first where applicable.")
        script = ROOT / "scripts" / filename
        log = logs / (script.stem + ".log")
        reported_errors = []
        print(f"\nRunning {stage}: {filename}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen([sys.executable, str(script)], cwd=ROOT / "scripts", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                    if re.fullmatch(r"(?:error|failed|missing_required_file|missing required file):\s*[1-9]\d*", line.strip()):
                        reported_errors.append(line.strip())
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        if code or reported_errors:
            detail = "; ".join(reported_errors) if reported_errors else f"exit code {code}"
            raise SystemExit(f"Stage {stage} failed ({detail}). See {log}.")
    print("\nRequested analyses completed.")

if __name__ == "__main__":
    main()
