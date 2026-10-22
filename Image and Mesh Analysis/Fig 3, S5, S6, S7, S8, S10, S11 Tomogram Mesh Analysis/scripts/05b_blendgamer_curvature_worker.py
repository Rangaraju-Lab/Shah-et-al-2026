# ============================================================
# Blender/GAMer worker: compute smooth K1 for one Matrix_vol.stl
#
# Called by launcher:
# blender --background --python blendgamer_curvature_worker.py -- input.stl output_path
#
# IMPORTANT:
# This saves BOTH:
#   1) Blender vertex coordinates
#   2) smooth_k1 values
#
# This fixes downstream vertex-order problems caused by STL import/export.
# ============================================================

import sys
import os
import pickle
import importlib
import types
import traceback
from pathlib import Path

import bpy

# ============================================================
# SETTINGS
# ============================================================
ALGORITHM = "MDSB"
CURVATURE_TYPE = "K1"
CURVE_ITER = 1

# ============================================================
# ARGUMENTS
# ============================================================
def get_args_after_double_dash():
    if "--" not in sys.argv:
        raise RuntimeError("Expected arguments after -- : input.stl output_path")

    idx = sys.argv.index("--")
    return sys.argv[idx + 1:]

def parse_args():
    args = get_args_after_double_dash()

    if len(args) != 2:
        raise RuntimeError(
            "Usage: blender --background --python blendgamer_curvature_worker.py -- input.stl output_path"
        )

    stl_path = Path(args[0])
    out_path = Path(args[1])

    return stl_path, out_path

# ============================================================
# BLENDGAMER IMPORT HELPERS
# ============================================================
def enable_addon_safely(module_name):
    try:
        bpy.ops.preferences.addon_enable(module=module_name)
        return True
    except Exception:
        pass

    try:
        bpy.ops.wm.addon_enable(module=module_name)
        return True
    except Exception:
        pass

    return False

def make_blendgamer_handle_from_colormap(colormap_module):
    handle = types.SimpleNamespace()
    handle.colormap = colormap_module
    return handle

def find_gamer_module_names():
    names = [
        "blendgamer",
        "bl_ext.system.blendgamer",
        "bl_ext.user_default.blendgamer",
    ]

    try:
        import addon_utils

        for mod in addon_utils.modules():
            mod_name = getattr(mod, "__name__", "")
            mod_file = getattr(mod, "__file__", "")

            if "gamer" in mod_name.lower() or "gamer" in str(mod_file).lower():
                names.append(mod_name)

    except Exception:
        pass

    for name in list(sys.modules.keys()):
        if "gamer" in name.lower():
            names.append(name)

    seen = set()
    unique = []

    for name in names:
        if not name:
            continue

        if name not in seen:
            unique.append(name)
            seen.add(name)

    return unique

def import_blendgamer():
    module_names = find_gamer_module_names()

    for module_name in module_names:
        enable_addon_safely(module_name)

    errors = []

    for module_name in module_names:
        if module_name.endswith(".colormap"):
            continue

        try:
            module = importlib.import_module(module_name)

            if hasattr(module, "colormap"):
                return module

            try:
                colormap_module = importlib.import_module(module_name + ".colormap")
                module.colormap = colormap_module
                return module
            except Exception as e:
                errors.append(f"{module_name}.colormap: {repr(e)}")

        except Exception as e:
            errors.append(f"{module_name}: {repr(e)}")

    colormap_names = []

    for module_name in module_names:
        if module_name.endswith(".colormap"):
            colormap_names.append(module_name)
        else:
            colormap_names.append(module_name + ".colormap")

    colormap_names.extend([
        "blendgamer.colormap",
        "bl_ext.system.blendgamer.colormap",
        "bl_ext.user_default.blendgamer.colormap",
    ])

    seen = set()
    colormap_names = [
        name for name in colormap_names
        if name and not (name in seen or seen.add(name))
    ]

    for colormap_name in colormap_names:
        try:
            colormap_module = importlib.import_module(colormap_name)
            return make_blendgamer_handle_from_colormap(colormap_module)
        except Exception as e:
            errors.append(f"{colormap_name}: {repr(e)}")

    for name, module in list(sys.modules.items()):
        lname = name.lower()

        if "gamer" not in lname:
            continue

        if hasattr(module, "colormap"):
            return module

        if lname.endswith(".colormap"):
            return make_blendgamer_handle_from_colormap(module)

    print("")
    print("Could not import BlendGAMer.")
    print("")
    print("GAMer/BlendGAMer-like module names found:")
    for name in module_names:
        print("  ", name)

    print("")
    print("Import errors:")
    for err in errors:
        print("  ", err)

    print("")
    print("Already-loaded modules containing 'gamer':")
    for name in sorted(sys.modules.keys()):
        if "gamer" in name.lower():
            print("  ", name)

    raise ModuleNotFoundError("Could not import BlendGAMer or BlendGAMer colormap module.")

# ============================================================
# HELPERS
# ============================================================
class Curvature:
    def __init__(self, algorithm, curvatureType, curveIter):
        self.algorithm = algorithm
        self.curvatureType = curvatureType
        self.curveIter = curveIter

def clear_scene():
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def import_stl(filepath):
    filepath = str(filepath)

    before_objects = set(bpy.context.scene.objects)

    # Blender 4/5 STL importer
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=filepath)

    # Older Blender fallback
    elif hasattr(bpy.ops, "import_mesh") and hasattr(bpy.ops.import_mesh, "stl"):
        bpy.ops.import_mesh.stl(filepath=filepath)

    else:
        raise RuntimeError(
            "No STL import operator found. Tried bpy.ops.wm.stl_import and bpy.ops.import_mesh.stl."
        )

    after_objects = set(bpy.context.scene.objects)
    new_objects = list(after_objects - before_objects)

    imported = [
        obj for obj in new_objects
        if obj.type == "MESH"
    ]

    if not imported:
        imported = [
            obj for obj in bpy.context.selected_objects
            if obj.type == "MESH"
        ]

    if not imported:
        imported = [
            obj for obj in bpy.context.scene.objects
            if obj.type == "MESH"
        ]

    if not imported:
        raise RuntimeError(f"Failed to import STL: {filepath}")

    obj = imported[0]

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj

def blender_world_vertices(obj):
    """
    Return vertices in the exact Blender mesh vertex order used by BlendGAMer.
    Stored as plain Python tuples so pickle can be read by normal Python.
    """
    verts = []

    for v in obj.data.vertices:
        co = obj.matrix_world @ v.co
        verts.append((float(co.x), float(co.y), float(co.z)))

    return verts

def compute_smooth_k1(obj, blendgamer):
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if not hasattr(bpy.ops, "gamer"):
        raise RuntimeError("bpy.ops.gamer does not exist. BlendGAMer add-on is not registered.")

    bpy.ops.gamer.compute_curvatures()

    crv = Curvature(ALGORITHM, CURVATURE_TYPE, CURVE_ITER)
    smooth_data = blendgamer.colormap.curveToData(crv, bpy.context)

    # Convert to plain float list for safe pickle use outside Blender.
    smooth_data = [float(x) for x in smooth_data]

    return smooth_data

# ============================================================
# MAIN
# ============================================================
def main():
    stl_path, out_path = parse_args()

    if not stl_path.exists():
        raise FileNotFoundError(f"Missing input STL: {stl_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    blendgamer = import_blendgamer()

    clear_scene()

    obj = import_stl(stl_path)
    print(f"Imported STL: {stl_path}")

    smooth_k1 = compute_smooth_k1(obj, blendgamer)
    vertices = blender_world_vertices(obj)

    print(f"Blender vertices: {len(vertices)}")
    print(f"smooth_k1 values: {len(smooth_k1)}")

    if len(vertices) != len(smooth_k1):
        raise RuntimeError(
            f"Curvature length does not match Blender vertex count. "
            f"vertices={len(vertices)}, smooth_k1={len(smooth_k1)}"
        )

    payload = {
        "format": "blendgamer_smooth_k1_with_vertices_v1",
        "source_stl": str(stl_path),
        "algorithm": ALGORITHM,
        "curvature_type": CURVATURE_TYPE,
        "curve_iter": CURVE_ITER,
        "vertices": vertices,
        "smooth_k1": smooth_k1,
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved curvature payload: {out_path}")

    clear_scene()

if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()

        # Avoid harmless BlendGAMer unregister errors on Blender quit.
        os._exit(0)

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
