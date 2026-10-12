from pathlib import Path
import argparse
import json
import os
import platform
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np
import pandas as pd
from tqdm import tqdm
import settings as cfg

def configure(module, values):
    for name, value in values.items():
        setattr(module, name, value)

def prepare_synapses(spine):
    df = pd.read_csv(cfg.SYNAPSES_CSV)
    source = [cfg.SYNAPSE_ID_COLUMN, cfg.X_COL, cfg.Y_COL, cfg.Z_COL]
    missing = set(source) - set(df.columns)
    if missing:
        raise ValueError(f"Missing synapse columns: {sorted(missing)}")
    ids = pd.to_numeric(df[cfg.SYNAPSE_ID_COLUMN], errors="raise")
    if ids.isna().any() or np.any(ids != np.floor(ids)) or ids.duplicated().any():
        raise ValueError("Synapse IDs must be unique integers for the extended SWC format.")
    df[spine.SYNAPSE_ID_COLUMN] = ids.astype(np.int64)
    coords = df[[cfg.X_COL, cfg.Y_COL, cfg.Z_COL]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Synapse coordinates must be finite.")
    for axis, name in enumerate((spine.X_COL, spine.Y_COL, spine.Z_COL)):
        df[name] = coords[:, axis]
    if cfg.COORDS_ARE_VOXELS:
        coords = coords * np.asarray(cfg.VOXEL_SIZE_NM)[None]
    df[["_orig_x_nm", "_orig_y_nm", "_orig_z_nm"]] = coords
    df["_contour_norm"] = df.get("contour", pd.Series("no_pair", index=df.index)).fillna("no_pair").astype(str).str.strip()
    df["aggloIds_2"] = cfg.SAMPLE_ID
    if "contour" not in df:
        df["contour"] = "no_pair"
    return df

def run_spine(output):
    from algorithms import spine_base as spine
    configure(spine, cfg.SPINE)
    mesh = spine.load_mesh_npz(cfg.NEURON_MESH)
    df = prepare_synapses(spine)
    df = spine.snap_synapses_to_mesh_vertices(df, mesh)
    graph = spine.build_mesh_vertex_graph_csr(mesh)
    success, failures = [], []
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Spine bases"):
        try:
            result = spine.process_one_synapse(row, mesh, graph)
            success.append(result)
            records.append({"synapse_id": result["synapse_id"], "status": "ok",
                            "base_x_nm": result["base_pt"][0], "base_y_nm": result["base_pt"][1],
                            "base_z_nm": result["base_pt"][2], "syn_base_dist_nm": result["syn_base_dist_nm"],
                            "syn_base_dist_raw_nm": result["syn_base_dist_raw_nm"],
                            "base_dend_dist_nm": result["base_dend_dist_nm"], "snap_dist_nm": result["snap_dist_nm"], "error": ""})
        except Exception as exc:
            failed = {"synapse_id": row[spine.SYNAPSE_ID_COLUMN], "status": "failed", "error": repr(exc)}
            failures.append(failed)
            records.append(failed)
    pd.DataFrame(records).to_csv(output / "spine_base_results.csv", index=False)
    pd.DataFrame(failures, columns=["synapse_id", "status", "error"]).to_csv(output / "failed_synapses.csv", index=False)
    if not success:
        raise RuntimeError("No spine bases were found. See results/failed_synapses.csv and the documented geometry requirements.")
    spine.write_extended_swc(output / "spines" / f"{cfg.SAMPLE_ID}.swc", cfg.SAMPLE_ID, success, failures)
    np.savez_compressed(output / "neuron_mesh" / f"{cfg.SAMPLE_ID}.npz", vertices=mesh.vertices, faces=mesh.faces)
    df.to_csv(output / "synapses_metadata.csv", index=False)
    print(f"Spine bases: {len(success)} succeeded, {len(failures)} failed")

def run_width(output):
    from algorithms import mitochondrial_width as mito
    configure(mito, cfg.MITO)
    mesh = mito.load_npz_mesh(cfg.MITO_MESH)
    components = mito.split_mito_components(mesh)
    skeleton = mito.skeletonize_mito_mesh_once(mesh)
    retained = mito.compute_component_skeletons(skeleton, components)
    diameter = mito.assign_diameter_to_mito_vertices_global(components, retained)
    values, vmin, vmax = mito.build_display_values(diameter["diameter_values"], diameter["all_mito_vertex_components"])
    mito.save_width_npz(output / "mito_width" / f"{cfg.SAMPLE_ID}.npz", cfg.SAMPLE_ID, cfg.MITO_MESH, diameter, values, vmin, vmax)
    mito.save_skeleton_npz(output / "mito_skeleton" / f"{cfg.SAMPLE_ID}.npz", cfg.SAMPLE_ID, cfg.MITO_MESH, skeleton, retained, diameter)
    summary = [{"component_id": r["mito_comp_i"], "main_length_nm": r["main_length_nm"],
                "n_retained_branches": r["n_retained_branches"], "n_relaxed_segments": r["n_relaxed_segments"]} for r in retained]
    pd.DataFrame(summary).to_csv(output / "mito_component_summary.csv", index=False)
    print(f"Mitochondria: {len(components)} components; {np.isfinite(diameter['diameter_values']).sum()} vertices with width")

def run_mapping(output):
    from algorithms import spine_mito_mapping as mapping
    configure(mapping, cfg.MAPPING)
    width = mapping.load_width_file(output / "mito_width" / f"{cfg.SAMPLE_ID}.npz")
    base_df, points = mapping.load_spine_bases(output / "spines" / f"{cfg.SAMPLE_ID}.swc")
    table = mapping.measure_spine_bases(cfg.SAMPLE_ID, width, base_df, points)
    mapping.save_measurements_atomic(table, output / "spine_base_mito_width_synapse_size.csv")
    print(f"Mapped {len(table)} spine bases and synapses to the mitochondrial surface")

def configure_morphology(output):
    from algorithms import morphology
    configure(morphology, cfg.MORPHOLOGY)
    morphology.output_dir = output / "morphology_renders"
    morphology.width_mesh_dir = output / "mito_width"
    morphology.neuron_mesh_dir = output / "neuron_mesh"
    morphology.spine_swc_dir = output / "spines"
    morphology.synapse_table_path = output / "synapses_metadata.csv"
    return morphology

def run_classification(output):
    morphology = configure_morphology(output)
    table = pd.read_csv(output / "spine_base_mito_width_synapse_size.csv")
    selected = morphology.select_measurements(table.copy(), max_snaps=None)
    selected.to_csv(output / "morphology_categories.csv", index=False)
    memberships = selected.groupby(["_agglo_key", "_synapse_key"])["_category"].agg(lambda x: ";".join(x)).to_dict() if len(selected) else {}
    table["morphology_categories"] = [memberships.get((morphology.normalize_id(a), morphology.normalize_id(s)), "none") for a, s in zip(table["agglo_id"], table["synapse_id"])]
    table.to_csv(output / "measurements_with_categories.csv", index=False)
    print("Category memberships:", selected["_category"].value_counts().to_dict() if len(selected) else {})
    return selected

def run_render(output):
    morphology = configure_morphology(output)
    selected = morphology.select_measurements(pd.read_csv(output / "spine_base_mito_width_synapse_size.csv"), max_snaps=morphology.MAX_SNAPS)
    if selected.empty:
        print("No morphology rules matched; no category screenshots are expected.")
        return
    for category in morphology.CATEGORIES:
        (morphology.output_dir / category).mkdir(parents=True, exist_ok=True)
    metadata = morphology.load_synapse_table()
    with tqdm(total=len(selected), desc="Category screenshots") as progress:
        result = morphology.render_agglo_group(cfg.SAMPLE_ID, selected, metadata, overwrite=True, progress_bar=progress)
    pd.DataFrame(result["failed"]).to_csv(output / "render_failures.csv", index=False)
    print(f"Screenshots: {result['saved']} saved, {len(result['failed'])} failed")

def main():
    parser = argparse.ArgumentParser(description="Analyze one neuron mesh, its synapses, and the corresponding mitochondria.")
    parser.add_argument("--stage", choices=("all", "spine", "width", "map", "classify", "render"), default="all")
    parser.add_argument("--render", action="store_true", help="Also render qualifying morphology examples after the full pipeline.")
    args = parser.parse_args()
    output = cfg.OUTPUT_DIR
    for folder in (output, output / "spines", output / "mito_width", output / "mito_skeleton", output / "neuron_mesh"):
        folder.mkdir(parents=True, exist_ok=True)
    effective = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items() if k.isupper()}
    start = time.time()
    for name, action in (("spine", run_spine), ("width", run_width), ("map", run_mapping), ("classify", run_classification)):
        if args.stage in ("all", name):
            action(output)
    if args.stage == "render" or (args.stage == "all" and args.render):
        run_render(output)
    effective.update({"stage": args.stage, "render": args.render, "seconds": time.time() - start, "python": platform.python_version()})
    (output / "run_settings.json").write_text(json.dumps(effective, indent=2) + "\n")
    print(f"Results: {output}")

if __name__ == "__main__":
    main()
