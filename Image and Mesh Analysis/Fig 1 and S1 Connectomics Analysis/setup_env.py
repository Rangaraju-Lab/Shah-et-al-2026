from pathlib import Path
import argparse
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Create the local environment and sample input folder.")
    parser.add_argument("--skip-install", action="store_true", help="Create the environment and folders without installing packages.")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required; Python 3.11 is recommended.")
    env = ROOT / ".venv"
    if not env.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        parser.error(f"Existing environment is incomplete: {env}. Remove it and rerun setup.")
    (ROOT / "sample data").mkdir(exist_ok=True)
    if not args.skip_install:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
    print(f"Environment ready: {python}")
    print("Merge your sample data folder, then run run_sample.py using this Python.")

if __name__ == "__main__":
    main()
