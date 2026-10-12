import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
import trimesh
from algorithms import spine_base as spine, mitochondrial_width as width, spine_mito_mapping as mapping, morphology

def main():
    x = np.arange(20, dtype=float)
    y = 80 + 25 * np.maximum(0, x - 8)
    fit = spine._fit_relu_ramp_x(x, y)
    assert abs(fit["ramp_up_point_fit"] - 8) < 1e-4
    assert fit["relu_r2"] > 0.99999
    path = np.array([[0,0,0],[1,2,0],[2,0,0]], dtype=float)
    smoothed = spine.relax_polyline_lst_keep_idx_fixed(path, 2)
    assert np.allclose(smoothed[[0,-1]], path[[0,-1]])
    assert abs(smoothed[1,1]) < abs(path[1,1])
    vertices = np.array([[-3000,0,0],[0,0,0],[3000,0,0]] + [[0,y,0] for y in range(100,1100,100)], dtype=float)
    edges = np.array([[0,1],[1,2],[1,3]] + [[j,j+1] for j in range(3,12)])
    radii = np.r_[400.,400.,400., 400.,240.,100.,80.,70.,70.,70.,70.,70.,70.]
    skeleton = SimpleNamespace(vertices=vertices, edges=edges, radius=radii)
    with patch.object(spine.sk.skeletonize, "by_wavefront_exact", return_value=skeleton):
        result = spine.find_spine_base_on_local_mesh(trimesh.creation.box(), 0, vertices[-1], refine=True)
    assert np.isfinite(result["base_pt"]).all()
    assert result["relu_result"] is not None
    assert result["base_pt"][1] <= 400
    # Known-radius surface rings assigned to a two-node centerline.
    mesh = trimesh.Trimesh(vertices=[[0,10,0],[0,-10,0],[100,20,0],[100,-20,0]], faces=[[0,1,2],[1,2,3]], process=False)
    retained = [{"mito_comp_i":0,"relaxed_segments":[{"relaxed_path_points":np.array([[0,0,0],[100,0,0]]),"length_nm":100.}]}]
    measured = width.assign_diameter_to_mito_vertices_global([mesh], retained)
    assert np.allclose(measured["path_node_diameters"], [20,40])
    z,_,_ = width.build_display_values(measured["diameter_values"], measured["all_mito_vertex_components"])
    assert np.allclose(z, [-1,-1,1,1])
    box = trimesh.creation.box(extents=[1000,1000,1000])
    face_components = np.zeros(len(box.faces), dtype=int)
    volume = mapping.component_mesh_volume_um3(box.vertices, box.faces, face_components, 0)
    assert np.isclose(volume, 1.0)
    surface = mapping.make_mito_polydata(box.vertices, box.faces)
    points = np.array([[0,0,0],[800,0,0]], dtype=float)
    inside = mapping.classify_points_inside_surface(points, surface)
    _,_,distance = mapping.query_closest_surface_points(points, surface)
    assert inside.tolist() == [True,False]
    assert np.allclose(distance, [500,300])
    # Exact production morphology rules, including strict threshold edges and overlapping budding.
    rows = []
    for index,(z,signed,distance,syn_distance,base_volume,syn_volume) in enumerate([(2.,100.,100.,1000.,1.,1.),(1.2,300.,300.,1000.,1.,1.),(1.,100.,100.,1000.,1.,1.),(2.,100.,100.,100.,1.,0.02)]):
        row = dict(agglo_id="sample",synapse_id=index+1,branch_id=index+1,base_x_nm=0,base_y_nm=0,base_z_nm=0,
                   nearest_surface_x_nm=0,nearest_surface_y_nm=0,nearest_surface_z_nm=0,mito_component_id=0,
                   mito_width_normalized_z=z,signed_distance_nm=signed,absolute_distance_nm=distance,
                   synapse_to_mito_distance_nm=syn_distance,base_nearest_mito_volume_um3=base_volume,synapse_nearest_mito_volume_um3=syn_volume)
        rows.append(row)
    with tempfile.TemporaryDirectory() as temp:
        morphology.output_dir = Path(temp)
        classified = morphology.select_measurements(pd.DataFrame(rows), max_snaps=None)
    category = classified.groupby("_synapse_key")["_category"].agg(set).to_dict()
    assert category["1"] == {"protrusion"}
    assert category["2"] == {"recruitment"}
    assert "3" not in category
    assert category["4"] == {"protrusion","budding"}
    print("PASS: ReLU knee, fixed endpoints, spine-base graph logic, diameter/z-score mapping, signed distances, volume, and exact morphology rules.")
    print("The spine graph fixture isolates logic; it does not validate wavefront extraction on experimental meshes.")

if __name__ == "__main__":
    main()
