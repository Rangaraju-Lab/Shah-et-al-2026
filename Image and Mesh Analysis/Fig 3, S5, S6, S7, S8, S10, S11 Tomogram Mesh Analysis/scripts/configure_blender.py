from pathlib import Path
import importlib.util
import bpy

worker = Path(__file__).resolve().parent / "05b_blendgamer_curvature_worker.py"
spec = importlib.util.spec_from_file_location("curvature_worker", worker)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
addon = module.import_blendgamer()
if not hasattr(addon, "colormap") or not hasattr(addon.colormap, "curveToData"):
    raise RuntimeError("BlendGAMer must provide colormap.curveToData.")
if not hasattr(bpy.ops, "gamer"):
    raise RuntimeError("BlendGAMer curvature operators are unavailable.")
print("Blender", bpy.app.version_string, "with BlendGAMer is ready.")
