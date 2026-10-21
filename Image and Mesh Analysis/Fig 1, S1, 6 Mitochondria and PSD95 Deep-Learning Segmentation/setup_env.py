from pathlib import Path
import argparse
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Set up both segmentation workflows.")
    parser.add_argument("--torch-index-url", help="Optional official PyTorch wheel index for CPU or CUDA builds.")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise SystemExit("Use Python 3.10 or newer; Python 3.11 or 3.12 is recommended.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    for target in ("Mitochondria", "PSD95"):
        for folder in ("TrainingTifs", "InferenceTifs/InferenceTifMasks", "Outputs/checkpoints", "Outputs/cache", "Outputs/visualizations"):
            (ROOT / target / folder).mkdir(parents=True, exist_ok=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    if args.torch_index_url:
        subprocess.run([str(python), "-m", "pip", "install", "torch>=2.2,<3", "--index-url", args.torch_index_url], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    print("Setup complete. Merge your TIFF data, then run:")
    for target in ("Mitochondria", "PSD95"):
        for script in ("train.py", "infer_all.py"):
            print(f'"{python}" "{ROOT / target / script}"')

if __name__ == "__main__":
    main()
