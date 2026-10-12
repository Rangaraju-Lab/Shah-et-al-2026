"""Numerical and rendering routines extracted from the supplied analysis; sample orchestration is in run_sample.py."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

os.environ.setdefault("MKL_NUM_THREADS", "1")

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import heapq

import traceback

import gc

import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np

import pandas as pd

import networkx as nx

import trimesh

import skeletor as sk

from scipy.spatial import cKDTree

from scipy.optimize import curve_fit

from scipy.sparse import coo_matrix

from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

from tqdm import tqdm

TARGET_AGGLO_IDS = None

SKIP_EXISTING = True

OVERWRITE_SUMMARY_CSV = True

WRITE_EMPTY_SWCS = False

SAVE_SUMMARY_INCREMENTALLY = True

SMALL_AGGLO_MAX_GB = 0.35

MAX_SMALL_AGGLO_WORKERS = 2

PROCESS_START_METHOD = "spawn"

TASK_PRIORITY_MODE = "mesh_size_synapses"

MAX_SYN_WORKERS = 4

TQDM_MININTERVAL_SEC = 1.0

AGGLO_COLUMN = "aggloIds_2"

SYNAPSE_ID_COLUMN = "id"

CONTOUR_COLUMN = "contour"

X_COL = "pos_1"

Y_COL = "pos_2"

Z_COL = "pos_3"

COORDS_ARE_VOXELS = True

VOXEL_SIZE_NM = (11.24, 11.24, 28.0)

LOCAL_GEODESIC_RADIUS_NM = 7000.0

MAX_LOCAL_VERTICES = 250_000

MIN_LOCAL_VERTICES = 300

MIN_LOCAL_FACES = 300

WAVEFRONT_STEP_SIZE_NM = 30

MIN_DEND_RADIUS_NM = 100.0

USE_RELU_BASE_REFINE = True

LST_N_ITERS = 100

LST_STEP_SIZE = 0.5

LST_TANGENT_DAMPING = 0.5

SWC_RADIUS_NM = 20.0

SWC_TYPE_SYNAPSE = 5

SWC_TYPE_INTERIOR = 7

SWC_TYPE_BASE = 6

ROLE_SYNAPSE = 1

ROLE_INTERIOR = 2

ROLE_BASE = 3

DEDUP_EPS_NM = 1e-6

def load_mesh_npz(path):
    data = np.load(path)

    mesh = trimesh.Trimesh(
        vertices=data["vertices"],
        faces=data["faces"],
        process=False,
    )

    data.close()
    return mesh

def snap_synapses_to_mesh_vertices(df, mesh):
    mesh_vertices = np.asarray(mesh.vertices, dtype=float)
    tree = cKDTree(mesh_vertices)

    orig_pts = df[["_orig_x_nm", "_orig_y_nm", "_orig_z_nm"]].to_numpy(dtype=float)

    snap_dist_nm, snap_vertex_idx = tree.query(
        orig_pts,
        k=1,
        workers=-1,
    )

    snapped_pts = mesh_vertices[snap_vertex_idx]

    df = df.copy()

    df["_snap_vertex_idx"] = snap_vertex_idx.astype(int)
    df["_snap_dist_nm"] = snap_dist_nm.astype(float)

    df["_snap_x_nm"] = snapped_pts[:, 0]
    df["_snap_y_nm"] = snapped_pts[:, 1]
    df["_snap_z_nm"] = snapped_pts[:, 2]

    return df

def build_mesh_vertex_graph_csr(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)

    n = len(vertices)

    edge_vec = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    edge_len = np.linalg.norm(edge_vec, axis=1).astype(np.float64)

    row = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    col = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    data = np.concatenate([edge_len, edge_len]).astype(np.float64)

    graph = coo_matrix(
        (data, (row, col)),
        shape=(n, n),
    ).tocsr()

    return graph

def geodesic_vertex_neighborhood(graph_csr, start_vertex, radius_nm, max_vertices=None):
    start_vertex = int(start_vertex)
    radius_nm = float(radius_nm)

    dist = sparse_dijkstra(
        csgraph=graph_csr,
        directed=False,
        indices=start_vertex,
        limit=radius_nm,
        return_predecessors=False,
    )

    selected = np.flatnonzero(np.isfinite(dist))

    if max_vertices is not None and len(selected) > int(max_vertices):
        selected = selected[np.argsort(dist[selected])[:int(max_vertices)]]

    return set(selected.astype(int).tolist())

def extract_local_submesh_from_vertex_set(full_mesh, selected_vertices, full_origin_vertex):
    selected_vertices = set(int(v) for v in selected_vertices)
    full_origin_vertex = int(full_origin_vertex)

    if full_origin_vertex not in selected_vertices:
        selected_vertices.add(full_origin_vertex)

    faces = np.asarray(full_mesh.faces, dtype=int)
    vertices = np.asarray(full_mesh.vertices, dtype=float)

    selected_mask = np.zeros(len(vertices), dtype=bool)
    selected_mask[list(selected_vertices)] = True

    face_mask = np.all(selected_mask[faces], axis=1)
    kept_faces_full = faces[face_mask]

    if len(kept_faces_full) == 0:
        raise ValueError("Local crop produced zero faces.")

    used_full_vertices = np.unique(kept_faces_full.ravel())

    if full_origin_vertex not in set(used_full_vertices.tolist()):
        raise ValueError(
            "The snapped origin vertex is not part of any kept local face. "
            "Increase LOCAL_GEODESIC_RADIUS_NM."
        )

    full_to_local = {
        int(full_idx): int(local_idx)
        for local_idx, full_idx in enumerate(used_full_vertices)
    }

    local_vertices = vertices[used_full_vertices]

    local_faces = np.array(
        [
            [full_to_local[int(a)], full_to_local[int(b)], full_to_local[int(c)]]
            for a, b, c in kept_faces_full
        ],
        dtype=int,
    )

    local_origin_vertex = full_to_local[full_origin_vertex]

    local_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=local_faces,
        process=False,
    )

    if len(local_mesh.vertices) < MIN_LOCAL_VERTICES or len(local_mesh.faces) < MIN_LOCAL_FACES:
        raise ValueError(
            "Local mesh too small: "
            f"{len(local_mesh.vertices):,} vertices, "
            f"{len(local_mesh.faces):,} faces."
        )

    return local_mesh, local_origin_vertex

def _dijkstra_to_any_target(adj, start_node, target_nodes):
    target_nodes = set(int(x) for x in target_nodes)
    start_node = int(start_node)

    if start_node in target_nodes:
        return [start_node], 0.0, start_node

    heap = [(0.0, start_node)]
    dist = {start_node: 0.0}
    parent = {start_node: None}
    visited = set()

    while heap:
        cur_dist, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)

        if u in target_nodes:
            path = []
            cur = u

            while cur is not None:
                path.append(cur)
                cur = parent[cur]

            path.reverse()
            return path, float(cur_dist), u

        for v, w in adj.get(u, []):
            if v in visited:
                continue

            new_dist = cur_dist + float(w)

            if (v not in dist) or (new_dist < dist[v]):
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(heap, (new_dist, v))

    raise ValueError("No target node is reachable from the start node.")

def _point_on_polyline_at_straight_distance_from_end(polyline, end_point, target_distance):
    polyline = np.asarray(polyline, dtype=float)
    end_point = np.asarray(end_point, dtype=float)

    if polyline.ndim != 2 or polyline.shape[1] != 3:
        raise ValueError("polyline must be an (N, 3) array.")

    if len(polyline) == 0:
        raise ValueError("polyline is empty.")

    dists = np.linalg.norm(polyline - end_point[None, :], axis=1)
    idx = int(np.argmin(np.abs(dists - float(target_distance))))

    return polyline[idx].copy(), float(dists[idx]), idx

def _relu_model(x, baseline, slope, x0):
    x = np.asarray(x, dtype=float)
    return baseline + slope * np.maximum(0.0, x - x0)

def _fit_relu_ramp_x(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 4:
        raise ValueError("Need at least 4 points to fit ReLU.")

    baseline0 = float(np.median(y[:max(2, len(y) // 4)]))
    slope0 = float((y[-1] - y[0]) / max(x[-1] - x[0], 1.0))
    x0_0 = float(np.median(x))

    lower_bounds = [-np.inf, 0.0, float(x.min())]
    upper_bounds = [np.inf, np.inf, float(x.max())]

    popt, _ = curve_fit(
        _relu_model,
        x,
        y,
        p0=[baseline0, max(slope0, 1e-6), x0_0],
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

    baseline, slope, x0 = popt
    yhat = _relu_model(x, *popt)

    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if sst == 0 else 1.0 - (sse / sst)

    nearest_idx = int(np.argmin(np.abs(x - x0)))

    return {
        "baseline_nm": float(baseline),
        "ramp_slope_nm_per_node": float(slope),
        "ramp_up_point_fit": float(x0),
        "nearest_observed_ramp_idx": nearest_idx,
        "relu_sse": sse,
        "relu_r2": r2,
    }

def _polyline_length_nm(points):
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return 0.0

    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def find_spine_base_on_local_mesh(
    local_mesh,
    local_origin_vertex,
    syn_pt_mesh,
    step_size=30,
    min_radius_nm=100.0,
    refine=True,
):
    local_origin_vertex = int(local_origin_vertex)
    syn_pt_mesh = np.asarray(syn_pt_mesh, dtype=float).ravel()

    raw_skel = sk.skeletonize.by_wavefront_exact(
        local_mesh,
        step_size=step_size,
        origins=local_origin_vertex,
        progress=False,
    )

    raw_vertices = np.asarray(raw_skel.vertices, dtype=float)
    raw_edges = np.asarray(raw_skel.edges, dtype=int)
    raw_radii = np.asarray(raw_skel.radius, dtype=float)

    if len(raw_vertices) < 2 or len(raw_edges) < 1:
        raise ValueError("Local wavefront skeletonization produced too small a skeleton.")

    G_skel = nx.Graph()

    for i, j in raw_edges:
        i = int(i)
        j = int(j)
        w = float(np.linalg.norm(raw_vertices[i] - raw_vertices[j]))
        G_skel.add_edge(i, j, weight=w)

    low_radius_nodes = np.where(raw_radii < float(min_radius_nm))[0]
    G_skel.remove_nodes_from(low_radius_nodes.tolist())

    if G_skel.number_of_nodes() < 2 or G_skel.number_of_edges() == 0:
        raise ValueError("No valid skeleton graph remains after removing low-radius vertices.")

    best_path = None
    best_length = -np.inf

    for comp_nodes in nx.connected_components(G_skel):
        H = G_skel.subgraph(comp_nodes)
        leaves = [n for n in H.nodes if H.degree[n] == 1]

        if len(leaves) < 2:
            continue

        for a in leaves:
            lengths = nx.single_source_dijkstra_path_length(H, a, weight="weight")

            for b in leaves:
                if b <= a or b not in lengths:
                    continue

                if lengths[b] > best_length:
                    best_length = lengths[b]
                    best_path = nx.shortest_path(H, a, b, weight="weight")

    if best_path is None or len(best_path) < 2:
        raise ValueError("Could not determine longest high-radius dendrite path.")

    dend_vertices = raw_vertices[best_path]
    dend_radii = raw_radii[best_path]

    swc = pd.DataFrame({
        "node_id": np.arange(len(best_path), dtype=int),
        "parent_id": np.r_[-1, np.arange(len(best_path) - 1, dtype=int)],
        "x": dend_vertices[:, 0],
        "y": dend_vertices[:, 1],
        "z": dend_vertices[:, 2],
        "radius": dend_radii,
    })

    dend_skel = sk.Skeleton(swc)
    dend_skel.mesh = local_mesh

    raw_tree = cKDTree(raw_vertices)
    _, start_raw_idx = raw_tree.query(syn_pt_mesh, k=1)
    start_raw_idx = int(start_raw_idx)

    dend_tree = cKDTree(dend_vertices)
    dist_raw_to_dend, _ = dend_tree.query(raw_vertices, k=1)

    meet_tol_nm = 1e-6
    meeting_nodes_set = set(
        np.where(dist_raw_to_dend <= meet_tol_nm)[0].astype(int).tolist()
    )

    if len(meeting_nodes_set) == 0:
        meeting_nodes_set = {int(np.argmin(dist_raw_to_dend))}

    adj = {}

    for i, j in raw_edges:
        i = int(i)
        j = int(j)
        w = float(np.linalg.norm(raw_vertices[i] - raw_vertices[j]))
        adj.setdefault(i, []).append((j, w))
        adj.setdefault(j, []).append((i, w))

    if start_raw_idx not in adj:
        raise ValueError("Start node is not present in raw skeleton graph.")

    path_node_ids, trace_length_nm_graph, meet_node = _dijkstra_to_any_target(
        adj=adj,
        start_node=start_raw_idx,
        target_nodes=meeting_nodes_set,
    )

    syn_pt_skel = raw_vertices[start_raw_idx].copy()
    graph_path_points_nm = raw_vertices[path_node_ids]
    dend_pt = raw_vertices[meet_node].copy()

    if len(graph_path_points_nm) == 0:
        full_path_points_nm = np.asarray([syn_pt_mesh, syn_pt_skel], dtype=float)
    else:
        if np.linalg.norm(graph_path_points_nm[0] - syn_pt_skel) < 1e-9:
            full_path_points_nm = np.vstack([
                syn_pt_mesh,
                syn_pt_skel,
                graph_path_points_nm[1:],
            ])
        else:
            full_path_points_nm = np.vstack([
                syn_pt_mesh,
                syn_pt_skel,
                graph_path_points_nm,
            ])

    _, nearest_dend_idx = dend_tree.query(dend_pt, k=1)
    nearest_dend_idx = int(nearest_dend_idx)

    meet_radius_nm = float(dend_skel.radius[nearest_dend_idx])

    base_pt, _, base_idx = _point_on_polyline_at_straight_distance_from_end(
        full_path_points_nm,
        dend_pt,
        meet_radius_nm,
    )

    relu_result = None

    if refine and len(graph_path_points_nm) >= 4:
        yellow_start_idx = int(
            np.argmin(
                np.linalg.norm(
                    graph_path_points_nm - np.asarray(base_pt, dtype=float)[None, :],
                    axis=1,
                )
            )
        )

        sub_node_ids = np.asarray(path_node_ids[yellow_start_idx:], dtype=int)
        sub_radii_nm = raw_radii[sub_node_ids]
        sub_x = np.arange(len(sub_node_ids), dtype=float)

        if len(sub_node_ids) >= 4:
            relu_result = _fit_relu_ramp_x(sub_x, sub_radii_nm)

            ramp_sub_idx = int(relu_result["nearest_observed_ramp_idx"])
            ramp_global_graph_idx = yellow_start_idx + ramp_sub_idx

            base_pt = graph_path_points_nm[ramp_global_graph_idx].copy()

            if ramp_global_graph_idx == 0:
                base_idx = 1
            else:
                base_idx = 1 + ramp_global_graph_idx
        else:
            relu_result = {
                "warning": "Not enough yellow-to-dendrite points for ReLU refine."
            }

    traversed_path_to_dend = full_path_points_nm[:base_idx + 1].copy()

    if np.linalg.norm(traversed_path_to_dend[-1] - base_pt) > 1e-9:
        traversed_path_to_dend = np.vstack([traversed_path_to_dend, base_pt])

    if np.linalg.norm(traversed_path_to_dend[-1] - dend_pt) > 1e-9:
        traversed_path_to_dend = np.vstack([traversed_path_to_dend, dend_pt])

    syn_base_dist_nm = _polyline_length_nm(traversed_path_to_dend[:-1])
    base_dend_dist_nm = float(np.linalg.norm(dend_pt - base_pt))

    return {
        "traversed_path_to_dend": traversed_path_to_dend,
        "syn_base_dist_nm": syn_base_dist_nm,
        "base_dend_dist_nm": base_dend_dist_nm,
        "base_pt": base_pt,
        "dend_pt": dend_pt,
        "syn_pt_mesh": syn_pt_mesh,
        "syn_pt_skel": syn_pt_skel,
        "relu_result": relu_result,
    }

def _polyline_tangents(path):
    path = np.asarray(path, dtype=float)
    n = len(path)
    T = np.zeros_like(path)

    if n < 2:
        return T

    if n == 2:
        d = path[1] - path[0]
        dn = np.linalg.norm(d)

        if dn > 0:
            T[:] = d / dn

        return T

    T[0] = path[1] - path[0]
    T[-1] = path[-1] - path[-2]
    T[1:-1] = path[2:] - path[:-2]

    norms = np.linalg.norm(T, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    T[valid] /= norms[valid]

    return T

def relax_polyline_lst_keep_idx_fixed(
    path,
    fixed_idx,
    n_iters=100,
    step_size=0.5,
    tangent_damping=0.5,
    smooth_endpoints=False,
):
    path = np.asarray(path, dtype=float).copy()
    n = len(path)

    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path must be (N, 3).")

    if n < 3:
        return path

    if fixed_idx < 0 or fixed_idx >= n:
        raise ValueError("fixed_idx out of bounds.")

    fixed_pt = path[fixed_idx].copy()

    for _ in range(int(n_iters)):
        old = path.copy()
        tangents = _polyline_tangents(old)
        updated = old.copy()

        if n > 2:
            nbr_mean = 0.5 * (old[:-2] + old[2:])
            disp = nbr_mean - old[1:-1]

            t = tangents[1:-1]
            disp_t = np.sum(disp * t, axis=1, keepdims=True) * t
            disp_n = disp - disp_t

            anisotropic_disp = disp_n + float(tangent_damping) * disp_t
            updated[1:-1] = old[1:-1] + float(step_size) * anisotropic_disp

        if not smooth_endpoints:
            updated[0] = old[0]
            updated[-1] = old[-1]
        else:
            t0 = tangents[0]
            disp0 = old[1] - old[0]
            disp0_t = np.sum(disp0 * t0) * t0
            disp0_n = disp0 - disp0_t
            updated[0] = old[0] + float(step_size) * (
                disp0_n + float(tangent_damping) * disp0_t
            )

            tN = tangents[-1]
            dispN = old[-2] - old[-1]
            dispN_t = np.sum(dispN * tN) * tN
            dispN_n = dispN - dispN_t
            updated[-1] = old[-1] + float(step_size) * (
                dispN_n + float(tangent_damping) * dispN_t
            )

        updated[fixed_idx] = fixed_pt
        path = updated

    return path

def get_synapse_to_base_path_from_traversed_path(traversed_path_to_dend, base_pt):
    path = np.asarray(traversed_path_to_dend, dtype=float)
    base_pt = np.asarray(base_pt, dtype=float).ravel()

    base_idx = int(np.argmin(np.linalg.norm(path - base_pt[None, :], axis=1)))

    syn_to_base = path[:base_idx + 1].copy()

    if np.linalg.norm(syn_to_base[-1] - base_pt) > 1e-9:
        syn_to_base = np.vstack([syn_to_base, base_pt])
    else:
        syn_to_base[-1] = base_pt

    return syn_to_base

def refine_synapse_to_base_path(local_mesh, syn_to_base_path, base_pt):
    syn_to_base_path = np.asarray(syn_to_base_path, dtype=float)
    base_pt = np.asarray(base_pt, dtype=float).ravel()

    if len(syn_to_base_path) < 3:
        return syn_to_base_path.copy()

    base_idx = len(syn_to_base_path) - 1

    relaxed = relax_polyline_lst_keep_idx_fixed(
        path=syn_to_base_path,
        fixed_idx=base_idx,
        n_iters=LST_N_ITERS,
        step_size=LST_STEP_SIZE,
        tangent_damping=LST_TANGENT_DAMPING,
        smooth_endpoints=False,
    )

    relaxed[-1] = base_pt

    return relaxed

def remove_consecutive_duplicate_points(path, eps=1e-6):
    path = np.asarray(path, dtype=float)

    if len(path) <= 1:
        return path.copy()

    keep = [0]

    for i in range(1, len(path)):
        if np.linalg.norm(path[i] - path[keep[-1]]) > float(eps):
            keep.append(i)

    return path[keep].copy()

def process_one_synapse(row, full_mesh, full_mesh_graph):
    synapse_id = row[SYNAPSE_ID_COLUMN]

    full_origin_vertex = int(row["_snap_vertex_idx"])
    syn_pt_mesh = row[["_snap_x_nm", "_snap_y_nm", "_snap_z_nm"]].to_numpy(dtype=float)

    selected_vertices = geodesic_vertex_neighborhood(
        graph_csr=full_mesh_graph,
        start_vertex=full_origin_vertex,
        radius_nm=LOCAL_GEODESIC_RADIUS_NM,
        max_vertices=MAX_LOCAL_VERTICES,
    )

    local_mesh, local_origin_vertex = extract_local_submesh_from_vertex_set(
        full_mesh=full_mesh,
        selected_vertices=selected_vertices,
        full_origin_vertex=full_origin_vertex,
    )

    base_result = find_spine_base_on_local_mesh(
        local_mesh=local_mesh,
        local_origin_vertex=local_origin_vertex,
        syn_pt_mesh=syn_pt_mesh,
        step_size=WAVEFRONT_STEP_SIZE_NM,
        min_radius_nm=MIN_DEND_RADIUS_NM,
        refine=USE_RELU_BASE_REFINE,
    )

    syn_to_base_path = get_synapse_to_base_path_from_traversed_path(
        traversed_path_to_dend=base_result["traversed_path_to_dend"],
        base_pt=base_result["base_pt"],
    )

    relaxed_syn_to_base_path = refine_synapse_to_base_path(
        local_mesh=local_mesh,
        syn_to_base_path=syn_to_base_path,
        base_pt=base_result["base_pt"],
    )

    relaxed_syn_to_base_path = remove_consecutive_duplicate_points(
        relaxed_syn_to_base_path,
        eps=DEDUP_EPS_NM,
    )

    syn_base_dist_raw = _polyline_length_nm(syn_to_base_path)
    syn_base_dist_relaxed = _polyline_length_nm(relaxed_syn_to_base_path)

    return {
        "status": "ok",
        "synapse_id": synapse_id,
        "contour": row.get("_contour_norm", "NA"),
        "syn_pt": syn_pt_mesh,
        "base_pt": base_result["base_pt"],
        "path": relaxed_syn_to_base_path,
        "syn_base_dist_nm": float(syn_base_dist_relaxed),
        "syn_base_dist_raw_nm": float(syn_base_dist_raw),
        "base_dend_dist_nm": float(base_result["base_dend_dist_nm"]),
        "snap_dist_nm": float(row.get("_snap_dist_nm", np.nan)),
        "n_path_nodes": int(len(relaxed_syn_to_base_path)),
        "error": "",
    }

def safe_numeric_synapse_id(synapse_id):
    try:
        return int(synapse_id)
    except Exception:
        try:
            return int(float(synapse_id))
        except Exception:
            return -1

def build_swc_rows_from_results(success_results):
    rows = []
    node_id = 1

    for result in success_results:
        path = np.asarray(result["path"], dtype=float)

        if len(path) < 2:
            continue

        syn_id_numeric = safe_numeric_synapse_id(result["synapse_id"])
        syn_base_dist_nm = float(result["syn_base_dist_nm"])
        base_dend_dist_nm = float(result["base_dend_dist_nm"])

        prev_node_id = -1

        for i, pt in enumerate(path):
            if i == 0:
                swc_type = SWC_TYPE_SYNAPSE
                role_code = ROLE_SYNAPSE
                parent_id = -1
            elif i == len(path) - 1:
                swc_type = SWC_TYPE_BASE
                role_code = ROLE_BASE
                parent_id = prev_node_id
            else:
                swc_type = SWC_TYPE_INTERIOR
                role_code = ROLE_INTERIOR
                parent_id = prev_node_id

            rows.append({
                "node_id": int(node_id),
                "type": int(swc_type),
                "x": float(pt[0]),
                "y": float(pt[1]),
                "z": float(pt[2]),
                "radius": float(SWC_RADIUS_NM),
                "parent_id": int(parent_id),
                "synapse_id": int(syn_id_numeric),
                "role_code": int(role_code),
                "path_order": int(i),
                "syn_base_dist_nm": syn_base_dist_nm,
                "base_dend_dist_nm": base_dend_dist_nm,
            })

            prev_node_id = node_id
            node_id += 1

    return rows

def write_extended_swc(path, agglo_id, success_results, failed_results):
    rows = build_swc_rows_from_results(success_results)

    if len(rows) == 0 and not WRITE_EMPTY_SWCS:
        return False, 0

    tmp_path = Path(str(path) + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("# Extended SWC generated by Batch_local_spine_base_swc_hybrid.py\n")
        f.write(f"# aggloIds_2 {agglo_id}\n")
        f.write(f"# successful_synapses {len(success_results)}\n")
        f.write(f"# failed_synapses {len(failed_results)}\n")
        f.write("# columns: node_id type x y z radius parent_id synapse_id role_code path_order syn_base_dist_nm base_dend_dist_nm\n")
        f.write("# role_code: 1=synapse_node 2=path_interior 3=base_node\n")
        f.write("# swc_type: 5=synapse_node 7=path_interior 6=base_node\n")
        f.write("# units: nm\n")

        if len(failed_results) > 0:
            failed_ids = ",".join(str(r.get("synapse_id", "NA")) for r in failed_results)
            f.write(f"# failed_synapse_ids {failed_ids}\n")

        for r in rows:
            f.write(
                f"{r['node_id']} "
                f"{r['type']} "
                f"{r['x']:.6f} "
                f"{r['y']:.6f} "
                f"{r['z']:.6f} "
                f"{r['radius']:.6f} "
                f"{r['parent_id']} "
                f"{r['synapse_id']} "
                f"{r['role_code']} "
                f"{r['path_order']} "
                f"{r['syn_base_dist_nm']:.6f} "
                f"{r['base_dend_dist_nm']:.6f}\n"
            )

    tmp_path.replace(path)

    return True, len(rows)
