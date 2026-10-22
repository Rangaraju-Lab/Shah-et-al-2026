# ============================================================
# Blender worker: remesh one STL file and export one STL file
#
# This script is called by Blender in background mode:
# blender --background --python blender_remesh_worker.py -- input.stl output.stl voxel_size
# ============================================================

import sys
from pathlib import Path
import bpy

# ============================================================
# ARGUMENTS
# ============================================================
def get_args_after_double_dash():
    if "--" not in sys.argv:
        raise RuntimeError(
            "Expected arguments after -- : input.stl output.stl voxel_size"
        )

    idx = sys.argv.index("--")
    return sys.argv[idx + 1:]

def parse_args():
    args = get_args_after_double_dash()

    if len(args) != 3:
        raise RuntimeError(
            "Usage: blender --background --python blender_remesh_worker.py -- input.stl output.stl voxel_size"
        )

    in_path = Path(args[0])
    out_path = Path(args[1])
    voxel_size = float(args[2])

    return in_path, out_path, voxel_size

# ============================================================
# BLENDER HELPERS
# ============================================================
def enable_stl_addon():
    try:
        bpy.ops.preferences.addon_enable(module="io_mesh_stl")
        return
    except Exception:
        pass

    try:
        bpy.ops.wm.addon_enable(module="io_mesh_stl")
        return
    except Exception:
        pass

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def import_stl(path):
    path = str(path)
    enable_stl_addon()

    # Older Blender / CellBlender
    try:
        bpy.ops.import_mesh.stl(filepath=path)
        return
    except Exception:
        pass

    # Newer Blender
    try:
        bpy.ops.wm.stl_import(filepath=path)
        return
    except Exception as e:
        raise RuntimeError(f"Could not import STL: {path}\n{e}")

def export_stl(path):
    path = str(path)
    enable_stl_addon()

    # Older Blender / CellBlender
    try:
        bpy.ops.export_mesh.stl(filepath=path)
        return
    except Exception:
        pass

    # Newer Blender
    try:
        bpy.ops.wm.stl_export(filepath=path)
        return
    except Exception as e:
        raise RuntimeError(f"Could not export STL: {path}\n{e}")

def get_active_mesh_object():
    mesh_objects = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH"
    ]

    if len(mesh_objects) == 0:
        raise RuntimeError("No mesh object found after import.")

    if len(mesh_objects) > 1:
        bpy.ops.object.select_all(action="DESELECT")

        for obj in mesh_objects:
            obj.select_set(True)

        bpy.context.view_layer.objects.active = mesh_objects[0]
        bpy.ops.object.join()

        return bpy.context.view_layer.objects.active

    obj = mesh_objects[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj

def apply_remesh_and_triangulate(obj, voxel_size):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    except Exception:
        pass

    remesh_mod = obj.modifiers.new(name="Remesh", type="REMESH")
    remesh_mod.mode = "VOXEL"
    remesh_mod.voxel_size = voxel_size

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

    tri_mod = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=tri_mod.name)

def main():
    in_path, out_path, voxel_size = parse_args()

    if not in_path.exists():
        raise FileNotFoundError(f"Input STL not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    enable_stl_addon()
    clear_scene()

    import_stl(in_path)

    obj = get_active_mesh_object()
    apply_remesh_and_triangulate(obj, voxel_size)

    export_stl(out_path)

    clear_scene()

if __name__ == "__main__":
    main()