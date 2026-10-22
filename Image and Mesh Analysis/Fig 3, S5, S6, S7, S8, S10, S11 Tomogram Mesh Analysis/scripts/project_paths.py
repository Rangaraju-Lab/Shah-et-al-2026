from pathlib import Path
import json
import os
import platform
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from settings import WORKERS, BLENDER_WORKERS, BLENDER_PATH

def resolve_blender(override=None):
    candidates = []
    for value in (override, BLENDER_PATH, os.environ.get("BLENDER_EXE")):
        if value:
            candidates.append(Path(value).expanduser())
    runtime = PROJECT_ROOT / ".runtime.json"
    if runtime.exists():
        data = json.loads(runtime.read_text())
        if data.get("blender"):
            candidates.append(Path(data["blender"]))
    tools = PROJECT_ROOT / "tools"
    for parent in (tools, Path.home() / "Downloads", Path("/Applications")):
        if parent.exists():
            for directory in sorted(parent.glob("*Blender*CellBlender*"), reverse=True):
                candidates.extend([directory / "my_blender", directory / "blender.exe", directory / "Blender.app"])
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if base:
            candidates.extend(sorted((Path(base) / "Blender Foundation").glob("Blender*/blender.exe"), reverse=True))
    found = shutil.which("blender")
    if found:
        candidates.append(Path(found))
    candidates.extend([Path("/Applications/Blender.app"), Path("/usr/bin/blender"), Path("/usr/local/bin/blender")])
    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate.resolve(), checked
        bundles = [candidate, Path(str(candidate) + ".app")]
        for bundle in bundles:
            if bundle.is_dir():
                for name in ("Blender", "blender"):
                    exe = bundle / "Contents" / "MacOS" / name
                    if exe.is_file():
                        return exe.resolve(), checked
    return None, checked
