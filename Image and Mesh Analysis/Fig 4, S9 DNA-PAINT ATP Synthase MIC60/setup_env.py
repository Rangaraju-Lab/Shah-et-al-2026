from pathlib import Path
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
FOLDERS = ["Undrift Masked", "Undrift Masked SMLM", "Undrift Masked SMLM Centers", "Synapse Localizations", "Dendrites", "Mito Masks", "NonJuxta ROI", "Mito Skels", "Measurements/juxtaspinal", "Measurements/nonjuxtaspinal", "Measurements/locs_per_cluster", "Measurements/molecules_through_frames", "Measurements/molecules_per_mito", "Visualizations/juxtaspinal", "Visualizations/nonjuxtaspinal", "Run logs"]

def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Use Python 3.11 or newer; Python 3.12 is recommended.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(env)
    for folder in FOLDERS:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    print("Setup complete. Merge your data folders, then run:")
    print(f'"{python}" "{ROOT / "run_analysis.py"}"')

if __name__ == "__main__":
    main()
