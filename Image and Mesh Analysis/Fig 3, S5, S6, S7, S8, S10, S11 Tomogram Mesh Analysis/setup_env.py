from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Install Python dependencies and configure Blender for tomogram mesh analysis.")
    parser.add_argument("--blender", type=Path, help="Blender executable, CellBlender launcher, or macOS .app bundle.")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required; Python 3.11 or 3.12 is recommended.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(env)
    for folder in ("Control", "TTX"):
        (ROOT / folder).mkdir(exist_ok=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from project_paths import resolve_blender
    blender, checked = resolve_blender(args.blender)
    if blender is None:
        raise SystemExit("Python environment installed. Install the CellBlender bundle from https://mcell.org/download.html and rerun setup_env.py; use --blender if installed elsewhere.")
    subprocess.run([str(blender), "--background", "--python-exit-code", "1", "--python", str(ROOT / "scripts/configure_blender.py")], check=True)
    (ROOT / ".runtime.json").write_text(json.dumps({"blender": str(blender)}, indent=2) + "\n")
    print("Setup complete. Merge the matching Control/TTX data folders, then run:")
    print(f'"{python}" "{ROOT / "run_pipeline.py"}"')

if __name__ == "__main__":
    main()
