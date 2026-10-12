"""Numerical and rendering routines extracted from the supplied analysis; sample orchestration is in run_sample.py."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

os.environ.setdefault("MKL_NUM_THREADS", "1")

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing as mp

import contextlib

import heapq

import traceback

import time

import gc

import numpy as np

import pandas as pd

import trimesh

import skeletor as sk

from scipy.spatial import cKDTree

from tqdm import tqdm

TARGET_AGGLO_IDS = None

SKIP_EXISTING = True

OVERWRITE_SUMMARY_CSV = True

SAVE_SUMMARY_INCREMENTALLY = True

SMALL_MESH_MAX_GB = 0.35

MAX_SMALL_WORKERS = 2

PROCESS_START_METHOD = "spawn"

TASK_PRIORITY_MODE = "mesh_size_id"

SAVE_RAW_WAVEFRONT_SKEL = True

SUPPRESS_INTERNAL_OUTPUT = True

SKELETOR_PROGRESS = False

SKELETON_STEP_SIZE_NM = 30

USE_WAVEFRONT_EXACT = True

BRANCH_LENGTH_THRESHOLD_NM = 1000.0

KEEP_CYCLE_FALLBACK_LONGEST_PATH = True

DO_LST_RELAX_COMPONENT_SKELETON = True

LST_N_ITERS = 100

LST_STEP_SIZE = 0.5

LST_TANGENT_DAMPING = 0.5

LST_SMOOTH_ENDPOINTS = False

CHECK_RELAXED_PATH_INSIDE_COMPONENT = False

DIAMETER_SCALE_FACTOR = 2.0

NORMALIZE_PER_COMPONENT_DISPLAY = True

DIAMETER_VMIN_NM = 200

DIAMETER_VMAX_NM = 400

def load_npz_mesh(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh: {path}")

    data = np.load(path)

    mesh = trimesh.Trimesh(
        vertices=data["vertices"],
        faces=data["faces"],
        process=False,
    )

    data.close()

    return mesh

def _empty_int_edges():
    return np.empty((0, 2), dtype=np.int64)

def _as_float32_nan_safe(x):
    return np.asarray(x, dtype=np.float32)

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

def relax_polyline_lst_keep_endpoints_fixed(path, n_iters=100, step_size=0.5, tangent_damping=0.5, smooth_endpoints=False):
    path = np.asarray(path, dtype=float).copy()

    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path must be an (N, 3) array.")

    n = len(path)

    if n < 3:
        return path

    first_pt = path[0].copy()
    last_pt = path[-1].copy()

    for _ in range(int(n_iters)):
        old = path.copy()
        tangents = _polyline_tangents(old)
        updated = old.copy()

        nbr_mean = 0.5 * (old[:-2] + old[2:])
        disp = nbr_mean - old[1:-1]

        t = tangents[1:-1]
        disp_t = np.sum(disp * t, axis=1, keepdims=True) * t
        disp_n = disp - disp_t

        anisotropic_disp = disp_n + float(tangent_damping) * disp_t
        updated[1:-1] = old[1:-1] + float(step_size) * anisotropic_disp

        if smooth_endpoints:
            t0 = tangents[0]
            disp0 = old[1] - old[0]
            disp0_t = np.sum(disp0 * t0) * t0
            disp0_n = disp0 - disp0_t
            updated[0] = old[0] + float(step_size) * (disp0_n + float(tangent_damping) * disp0_t)

            tN = tangents[-1]
            dispN = old[-2] - old[-1]
            dispN_t = np.sum(dispN * tN) * tN
            dispN_n = dispN - dispN_t
            updated[-1] = old[-1] + float(step_size) * (dispN_n + float(tangent_damping) * dispN_t)
        else:
            updated[0] = first_pt
            updated[-1] = last_pt

        path = updated

    path[0] = first_pt
    path[-1] = last_pt

    return path

def _all_points_inside_mesh(mesh, points):
    points = np.asarray(points, dtype=float)

    try:
        inside = np.asarray(mesh.contains(points), dtype=bool)
    except Exception:
        return False, np.zeros(len(points), dtype=bool)

    return bool(np.all(inside)), inside

def maybe_relax_polyline(path_points, component_mesh):
    path_points = np.asarray(path_points, dtype=float)

    if not DO_LST_RELAX_COMPONENT_SKELETON:
        return path_points.copy()

    if len(path_points) < 3:
        return path_points.copy()

    relaxed = relax_polyline_lst_keep_endpoints_fixed(
        path=path_points,
        n_iters=LST_N_ITERS,
        step_size=LST_STEP_SIZE,
        tangent_damping=LST_TANGENT_DAMPING,
        smooth_endpoints=LST_SMOOTH_ENDPOINTS,
    )

    if CHECK_RELAXED_PATH_INSIDE_COMPONENT:
        all_inside, _inside = _all_points_inside_mesh(component_mesh, relaxed)

        if not all_inside:
            return path_points.copy()

    return relaxed

def skeletonize_mito_mesh_once(mito_mesh):
    if USE_WAVEFRONT_EXACT:
        try:
            return sk.skeletonize.by_wavefront_exact(
                mito_mesh,
                step_size=SKELETON_STEP_SIZE_NM,
                progress=SKELETOR_PROGRESS,
            )
        except Exception:
            pass

    return sk.skeletonize.by_wavefront(
        mito_mesh,
        step_size=SKELETON_STEP_SIZE_NM,
        progress=SKELETOR_PROGRESS,
    )

def connected_components_from_edges(n_nodes, edges):
    edges = np.asarray(edges, dtype=int)
    neighbors = [[] for _ in range(n_nodes)]

    for a, b in edges:
        a = int(a)
        b = int(b)
        neighbors[a].append(b)
        neighbors[b].append(a)

    seen = np.zeros(n_nodes, dtype=bool)
    comps = []

    for start in range(n_nodes):
        if seen[start]:
            continue

        stack = [start]
        seen[start] = True
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)

            for v in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

        comps.append(np.asarray(comp, dtype=int))

    return comps

def build_induced_adjacency(vertices, edges, node_ids):
    vertices = np.asarray(vertices, dtype=float)
    edges = np.asarray(edges, dtype=int)
    node_ids = np.asarray(node_ids, dtype=int)

    node_set = set(node_ids.tolist())
    adj = {int(i): [] for i in node_ids}

    for a, b in edges:
        a = int(a)
        b = int(b)

        if a not in node_set or b not in node_set:
            continue

        w = float(np.linalg.norm(vertices[a] - vertices[b]))
        adj[a].append((b, w))
        adj[b].append((a, w))

    return adj

def edge_key(a, b):
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)

def path_length_from_nodes(vertices, path_nodes):
    vertices = np.asarray(vertices, dtype=float)
    path_nodes = np.asarray(path_nodes, dtype=int)

    if len(path_nodes) < 2:
        return 0.0

    pts = vertices[path_nodes]

    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

def dijkstra_all_within_adj(adj, start):
    start = int(start)

    heap = [(0.0, start)]
    dist = {start: 0.0}
    parent = {start: None}
    visited = set()

    while heap:
        cur_dist, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)

        for v, w in adj.get(u, []):
            if v in visited:
                continue

            new_dist = cur_dist + float(w)

            if (v not in dist) or (new_dist < dist[v]):
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, parent

def reconstruct_path(parent, target):
    path = []
    cur = int(target)

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()

    return np.asarray(path, dtype=int)

def longest_path_within_nodes(vertices, edges, node_ids):
    vertices = np.asarray(vertices, dtype=float)
    node_ids = np.asarray(node_ids, dtype=int)

    if len(node_ids) < 2:
        return None, 0.0

    adj = build_induced_adjacency(vertices, edges, node_ids)

    degrees = {int(n): len(adj[int(n)]) for n in node_ids}
    terminals = [int(n) for n in node_ids if degrees[int(n)] == 1]

    if len(terminals) >= 2:
        starts = terminals
        targets_all = set(terminals)
    else:
        if not KEEP_CYCLE_FALLBACK_LONGEST_PATH:
            return None, 0.0

        starts = [int(n) for n in node_ids]
        targets_all = set(int(n) for n in node_ids)

    best_path = None
    best_len = -np.inf

    for start in starts:
        dist, parent = dijkstra_all_within_adj(adj, start)

        for target in targets_all:
            target = int(target)

            if target == start or target not in dist:
                continue

            if dist[target] > best_len:
                best_path = reconstruct_path(parent, target)
                best_len = float(dist[target])

    if best_path is None:
        return None, 0.0

    return best_path, best_len

def collect_subtree_nodes_excluding_main(adj, start_attach, first_off_node, main_nodes):
    start_attach = int(start_attach)
    first_off_node = int(first_off_node)
    main_nodes = set(int(n) for n in main_nodes)

    visited = set([start_attach])
    stack = [first_off_node]
    subtree_nodes = set([start_attach])

    while stack:
        u = int(stack.pop())

        if u in visited:
            continue

        if u in main_nodes:
            continue

        visited.add(u)
        subtree_nodes.add(u)

        for v, _w in adj.get(u, []):
            v = int(v)

            if v in visited:
                continue

            if v in main_nodes and v != start_attach:
                continue

            stack.append(v)

    return np.asarray(sorted(subtree_nodes), dtype=int)

def longest_branch_from_attachment(vertices, edges, node_ids, attach_node):
    vertices = np.asarray(vertices, dtype=float)
    node_ids = np.asarray(node_ids, dtype=int)
    attach_node = int(attach_node)

    if len(node_ids) < 2 or attach_node not in set(node_ids.tolist()):
        return None, 0.0

    adj = build_induced_adjacency(vertices, edges, node_ids)

    dist, parent = dijkstra_all_within_adj(adj, attach_node)

    degrees = {int(n): len(adj[int(n)]) for n in node_ids}
    candidates = [
        int(n)
        for n in node_ids
        if int(n) != attach_node and degrees[int(n)] == 1
    ]

    if len(candidates) == 0:
        candidates = [
            int(n)
            for n in node_ids
            if int(n) != attach_node and int(n) in dist
        ]

    if len(candidates) == 0:
        return None, 0.0

    best_target = None
    best_len = -np.inf

    for target in candidates:
        if target not in dist:
            continue

        if dist[target] > best_len:
            best_target = int(target)
            best_len = float(dist[target])

    if best_target is None:
        return None, 0.0

    path = reconstruct_path(parent, best_target)

    return path, best_len

def extract_component_longest_path_plus_branches(vertices, edges, node_ids):
    vertices = np.asarray(vertices, dtype=float)
    edges = np.asarray(edges, dtype=int)
    node_ids = np.asarray(node_ids, dtype=int)

    if len(node_ids) < 2:
        return None

    adj = build_induced_adjacency(vertices, edges, node_ids)

    main_path, main_len = longest_path_within_nodes(
        vertices=vertices,
        edges=edges,
        node_ids=node_ids,
    )

    if main_path is None or len(main_path) < 2:
        return None

    main_nodes = set(int(n) for n in main_path)
    main_edges = set(
        edge_key(main_path[i], main_path[i + 1])
        for i in range(len(main_path) - 1)
    )

    retained_edges = set(main_edges)
    branch_paths = []

    for attach in main_path:
        attach = int(attach)

        for nbr, _w in adj.get(attach, []):
            nbr = int(nbr)

            if edge_key(attach, nbr) in main_edges:
                continue

            if nbr in main_nodes:
                continue

            subtree_nodes = collect_subtree_nodes_excluding_main(
                adj=adj,
                start_attach=attach,
                first_off_node=nbr,
                main_nodes=main_nodes,
            )

            branch_path, branch_len = longest_branch_from_attachment(
                vertices=vertices,
                edges=edges,
                node_ids=subtree_nodes,
                attach_node=attach,
            )

            if branch_path is None or len(branch_path) < 2:
                continue

            if branch_len < float(BRANCH_LENGTH_THRESHOLD_NM):
                continue

            branch_paths.append({
                "attach_node": attach,
                "path_idx": branch_path,
                "length_nm": float(branch_len),
            })

            for i in range(len(branch_path) - 1):
                retained_edges.add(edge_key(branch_path[i], branch_path[i + 1]))

    retained_nodes = sorted(set([n for e in retained_edges for n in e]))

    return {
        "main_path_idx": np.asarray(main_path, dtype=int),
        "main_length_nm": float(main_len),
        "branch_paths": branch_paths,
        "retained_nodes": np.asarray(retained_nodes, dtype=int),
        "retained_edges": sorted(retained_edges),
    }

def decompose_retained_graph_into_segments(retained_nodes, retained_edges):
    retained_nodes = np.asarray(retained_nodes, dtype=int)
    retained_edges = [edge_key(a, b) for a, b in retained_edges]

    adj = {int(n): [] for n in retained_nodes}

    for a, b in retained_edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    degrees = {int(n): len(adj[int(n)]) for n in retained_nodes}
    key_nodes = set(int(n) for n in retained_nodes if degrees[int(n)] != 2)

    visited_edges = set()
    segments = []

    if len(key_nodes) == 0:
        return []

    for start in sorted(key_nodes):
        for nbr in adj[start]:
            e0 = edge_key(start, nbr)

            if e0 in visited_edges:
                continue

            path = [int(start)]
            prev = int(start)
            cur = int(nbr)

            while True:
                path.append(cur)
                visited_edges.add(edge_key(prev, cur))

                if cur in key_nodes and cur != start:
                    break

                next_nodes = [
                    int(v)
                    for v in adj.get(cur, [])
                    if int(v) != prev
                ]

                if len(next_nodes) == 0:
                    break

                chosen = None

                for candidate in next_nodes:
                    if edge_key(cur, candidate) not in visited_edges:
                        chosen = candidate
                        break

                if chosen is None:
                    break

                prev, cur = cur, chosen

            if len(path) >= 2:
                segments.append(np.asarray(path, dtype=int))

    return segments

def relax_component_skeleton_graph(vertices, retained_graph, component_mesh):
    vertices = np.asarray(vertices, dtype=float)

    if retained_graph is None:
        return []

    retained_nodes = retained_graph["retained_nodes"]
    retained_edges = retained_graph["retained_edges"]

    segment_node_paths = decompose_retained_graph_into_segments(
        retained_nodes=retained_nodes,
        retained_edges=retained_edges,
    )

    relaxed_segments = []

    for seg_idx in segment_node_paths:
        pts = vertices[seg_idx]

        if len(pts) < 2:
            continue

        relaxed_pts = maybe_relax_polyline(
            path_points=pts,
            component_mesh=component_mesh,
        )

        relaxed_segments.append({
            "path_idx": np.asarray(seg_idx, dtype=int),
            "path_points": pts,
            "relaxed_path_points": relaxed_pts,
            "length_nm": path_length_from_nodes(vertices, seg_idx),
        })

    return relaxed_segments

def split_mito_components(mito_mesh):
    comps = mito_mesh.split(only_watertight=False)

    comps = [
        trimesh.Trimesh(
            vertices=np.asarray(c.vertices, dtype=float),
            faces=np.asarray(c.faces, dtype=int),
            process=False,
        )
        for c in comps
        if len(c.vertices) > 0 and len(c.faces) > 0
    ]

    return comps

def build_component_vertex_kdtree(mito_components):
    all_vertices = []
    all_labels = []

    for comp_i, comp in enumerate(mito_components):
        v = np.asarray(comp.vertices, dtype=float)

        if len(v) == 0:
            continue

        all_vertices.append(v)
        all_labels.append(np.full(len(v), comp_i, dtype=int))

    if len(all_vertices) == 0:
        raise ValueError("No vertices found in mito components.")

    all_vertices = np.vstack(all_vertices)
    all_labels = np.concatenate(all_labels)

    tree = cKDTree(all_vertices)

    return tree, all_labels

def assign_skeleton_components_to_mito_components(mito_skel, mito_components):
    skel_vertices = np.asarray(mito_skel.vertices, dtype=float)
    skel_edges = np.asarray(mito_skel.edges, dtype=int)

    skel_comps = connected_components_from_edges(
        n_nodes=len(skel_vertices),
        edges=skel_edges,
    )

    comp_tree, comp_vertex_labels = build_component_vertex_kdtree(mito_components)

    skel_comp_to_mito_comp = {}

    for skel_comp_i, node_ids in enumerate(skel_comps):
        pts = skel_vertices[node_ids]

        if len(pts) == 0:
            continue

        _, nearest_vertex_idx = comp_tree.query(pts, k=1, workers=-1)
        nearest_comp_labels = comp_vertex_labels[nearest_vertex_idx]

        labels, counts = np.unique(nearest_comp_labels, return_counts=True)
        mito_comp_i = int(labels[np.argmax(counts)])

        skel_comp_to_mito_comp[skel_comp_i] = mito_comp_i

    return skel_comps, skel_comp_to_mito_comp

def compute_component_skeletons(mito_skel, mito_components):
    skel_vertices = np.asarray(mito_skel.vertices, dtype=float)
    skel_edges = np.asarray(mito_skel.edges, dtype=int)

    skel_comps, skel_comp_to_mito_comp = assign_skeleton_components_to_mito_components(
        mito_skel=mito_skel,
        mito_components=mito_components,
    )

    mito_to_skel_comps = {i: [] for i in range(len(mito_components))}

    for skel_comp_i, mito_comp_i in skel_comp_to_mito_comp.items():
        mito_to_skel_comps[mito_comp_i].append(skel_comp_i)

    results = []

    for mito_comp_i, assigned_skel_comp_ids in mito_to_skel_comps.items():
        best_graph = None
        best_skel_comp_i = None
        best_main_length = 0.0

        for skel_comp_i in assigned_skel_comp_ids:
            node_ids = skel_comps[skel_comp_i]

            candidate_graph = extract_component_longest_path_plus_branches(
                vertices=skel_vertices,
                edges=skel_edges,
                node_ids=node_ids,
            )

            if candidate_graph is None:
                continue

            if candidate_graph["main_length_nm"] > best_main_length:
                best_graph = candidate_graph
                best_skel_comp_i = int(skel_comp_i)
                best_main_length = float(candidate_graph["main_length_nm"])

        if best_graph is None:
            relaxed_segments = []
            n_branches = 0
        else:
            n_branches = len(best_graph["branch_paths"])

            relaxed_segments = relax_component_skeleton_graph(
                vertices=skel_vertices,
                retained_graph=best_graph,
                component_mesh=mito_components[mito_comp_i],
            )

        results.append({
            "mito_comp_i": int(mito_comp_i),
            "assigned_skel_comp_ids": assigned_skel_comp_ids,
            "best_skel_comp_i": best_skel_comp_i,
            "main_length_nm": float(best_main_length),
            "n_retained_branches": int(n_branches),
            "retained_graph": best_graph,
            "relaxed_segments": relaxed_segments,
            "n_relaxed_segments": int(len(relaxed_segments)),
        })

    return results

def combine_mito_components(mito_components):
    vertices_all = []
    faces_all = []
    component_vertex_ranges = []

    v_offset = 0

    for comp in mito_components:
        v = np.asarray(comp.vertices, dtype=float)
        f = np.asarray(comp.faces, dtype=int)

        start = v_offset
        stop = v_offset + len(v)

        vertices_all.append(v)
        faces_all.append(f + v_offset)

        component_vertex_ranges.append((start, stop))

        v_offset = stop

    if len(vertices_all) == 0:
        raise ValueError("No mito component vertices found.")

    vertices_all = np.vstack(vertices_all)
    faces_all = np.vstack(faces_all)

    combined_mesh = trimesh.Trimesh(
        vertices=vertices_all,
        faces=faces_all,
        process=False,
    )

    return combined_mesh, np.asarray(component_vertex_ranges, dtype=np.int64)

def flatten_component_relaxed_skeletons(component_skeleton_results):
    flat_segments = []

    for comp_result in component_skeleton_results:
        mito_comp_i = int(comp_result["mito_comp_i"])

        for seg in comp_result["relaxed_segments"]:
            pts = np.asarray(seg["relaxed_path_points"], dtype=float)

            if len(pts) < 2:
                continue

            flat_segments.append({
                "mito_comp_i": mito_comp_i,
                "length_nm": float(seg["length_nm"]),
                "relaxed_path_points": pts,
            })

    return flat_segments

def assign_diameter_to_mito_vertices_global(mito_components, component_skeleton_results):
    combined_mito_mesh, component_vertex_ranges = combine_mito_components(mito_components)

    all_mito_vertices = np.asarray(combined_mito_mesh.vertices, dtype=float)
    all_mito_vertex_components = np.empty(len(all_mito_vertices), dtype=int)

    for comp_i, (start, stop) in enumerate(component_vertex_ranges):
        all_mito_vertex_components[start:stop] = int(comp_i)

    flat_segments = flatten_component_relaxed_skeletons(component_skeleton_results)

    path_nodes_all = []
    path_node_component = []
    path_node_segment = []

    for seg_i, seg in enumerate(flat_segments):
        comp_i = int(seg["mito_comp_i"])
        pts = np.asarray(seg["relaxed_path_points"], dtype=float)

        if len(pts) < 2:
            continue

        path_nodes_all.append(pts)
        path_node_component.append(np.full(len(pts), comp_i, dtype=int))
        path_node_segment.append(np.full(len(pts), seg_i, dtype=int))

    if len(path_nodes_all) == 0:
        diameter_values = np.full(len(all_mito_vertices), np.nan, dtype=float)

        return {
            "combined_mito_mesh": combined_mito_mesh,
            "component_vertex_ranges": component_vertex_ranges,
            "all_mito_vertex_components": all_mito_vertex_components,
            "diameter_values": diameter_values,
            "path_nodes_all": np.empty((0, 3), dtype=float),
            "path_node_component": np.empty((0,), dtype=int),
            "path_node_segment": np.empty((0,), dtype=int),
            "path_node_diameters": np.empty((0,), dtype=float),
            "path_node_vertex_count": np.empty((0,), dtype=int),
            "nearest_path_node_idx": np.full(len(all_mito_vertices), -1, dtype=int),
            "vertex_to_path_dist_nm": np.full(len(all_mito_vertices), np.nan, dtype=float),
            "flat_segments": flat_segments,
        }

    path_nodes_all = np.vstack(path_nodes_all)
    path_node_component = np.concatenate(path_node_component).astype(int)
    path_node_segment = np.concatenate(path_node_segment).astype(int)

    bbox_diag = float(np.linalg.norm(
        np.nanmax(all_mito_vertices, axis=0) - np.nanmin(all_mito_vertices, axis=0)
    ))

    component_separation_nm = max(1e6, 10.0 * bbox_diag)

    path_nodes_aug = np.column_stack([
        path_nodes_all,
        path_node_component.astype(float) * component_separation_nm,
    ])

    mito_vertices_aug = np.column_stack([
        all_mito_vertices,
        all_mito_vertex_components.astype(float) * component_separation_nm,
    ])

    path_tree = cKDTree(path_nodes_aug)

    _aug_dist, nearest_path_node_idx = path_tree.query(
        mito_vertices_aug,
        k=1,
        workers=-1,
    )

    nearest_path_node_idx = nearest_path_node_idx.astype(int)

    nearest_xyz = path_nodes_all[nearest_path_node_idx]
    vertex_to_path_dist_nm = np.linalg.norm(
        all_mito_vertices - nearest_xyz,
        axis=1,
    )

    nearest_path_components = path_node_component[nearest_path_node_idx]
    same_component = nearest_path_components == all_mito_vertex_components

    vertex_to_path_dist_nm[~same_component] = np.nan
    nearest_path_node_idx[~same_component] = -1

    n_path_nodes = len(path_nodes_all)
    valid_vertex_mapping = nearest_path_node_idx >= 0

    counts = np.bincount(
        nearest_path_node_idx[valid_vertex_mapping],
        minlength=n_path_nodes,
    ).astype(float)

    dist_sums = np.bincount(
        nearest_path_node_idx[valid_vertex_mapping],
        weights=vertex_to_path_dist_nm[valid_vertex_mapping],
        minlength=n_path_nodes,
    ).astype(float)

    mean_radius_by_node = np.full(n_path_nodes, np.nan, dtype=float)

    valid_nodes = counts > 0
    mean_radius_by_node[valid_nodes] = dist_sums[valid_nodes] / counts[valid_nodes]

    path_node_diameters = DIAMETER_SCALE_FACTOR * mean_radius_by_node

    diameter_values = np.full(len(all_mito_vertices), np.nan, dtype=float)
    diameter_values[valid_vertex_mapping] = path_node_diameters[
        nearest_path_node_idx[valid_vertex_mapping]
    ]

    return {
        "combined_mito_mesh": combined_mito_mesh,
        "component_vertex_ranges": component_vertex_ranges,
        "all_mito_vertex_components": all_mito_vertex_components,
        "diameter_values": diameter_values,
        "path_nodes_all": path_nodes_all,
        "path_node_component": path_node_component,
        "path_node_segment": path_node_segment,
        "path_node_diameters": path_node_diameters,
        "path_node_vertex_count": counts.astype(np.int64),
        "nearest_path_node_idx": nearest_path_node_idx,
        "vertex_to_path_dist_nm": vertex_to_path_dist_nm,
        "flat_segments": flat_segments,
    }

def build_display_values(diameter_values, all_mito_vertex_components):
    diameter_values = np.asarray(diameter_values, dtype=float)
    all_mito_vertex_components = np.asarray(all_mito_vertex_components, dtype=int)

    if not NORMALIZE_PER_COMPONENT_DISPLAY:
        return diameter_values.copy(), DIAMETER_VMIN_NM, DIAMETER_VMAX_NM

    display_values = np.full_like(diameter_values, np.nan, dtype=float)
    unique_components = np.unique(all_mito_vertex_components)

    for comp_i in unique_components:
        comp_mask = all_mito_vertex_components == int(comp_i)
        vals = diameter_values[comp_mask]
        finite = np.isfinite(vals)

        if not np.any(finite):
            continue

        comp_mean = float(np.nanmean(vals))
        comp_std = float(np.nanstd(vals))

        if not np.isfinite(comp_std) or comp_std <= 0:
            local_display = np.full_like(vals, np.nan, dtype=float)
            local_display[finite] = 0.0
        else:
            local_display = np.full_like(vals, np.nan, dtype=float)
            local_display[finite] = (vals[finite] - comp_mean) / comp_std

        display_values[comp_mask] = local_display

    return display_values, 0.0, 1.0

def build_skeleton_save_arrays(diameter_result):
    flat_segments = diameter_result["flat_segments"]

    vertices_list = []
    edges_list = []
    node_component_id = []
    node_segment_id = []

    segment_id = []
    segment_component_id = []
    segment_node_start = []
    segment_node_stop = []
    segment_length_nm = []

    node_offset = 0

    for seg_i, seg in enumerate(flat_segments):
        pts = np.asarray(seg["relaxed_path_points"], dtype=float)

        if len(pts) < 2:
            continue

        n = len(pts)
        comp_i = int(seg["mito_comp_i"])

        vertices_list.append(pts)
        node_component_id.append(np.full(n, comp_i, dtype=np.int64))
        node_segment_id.append(np.full(n, seg_i, dtype=np.int64))

        e = np.column_stack([
            np.arange(node_offset, node_offset + n - 1, dtype=np.int64),
            np.arange(node_offset + 1, node_offset + n, dtype=np.int64),
        ])

        edges_list.append(e)

        segment_id.append(seg_i)
        segment_component_id.append(comp_i)
        segment_node_start.append(node_offset)
        segment_node_stop.append(node_offset + n)
        segment_length_nm.append(float(seg["length_nm"]))

        node_offset += n

    if len(vertices_list) == 0:
        vertices = np.empty((0, 3), dtype=np.float64)
        edges = _empty_int_edges()
        node_component_id_arr = np.empty((0,), dtype=np.int64)
        node_segment_id_arr = np.empty((0,), dtype=np.int64)
    else:
        vertices = np.vstack(vertices_list)
        edges = np.vstack(edges_list) if len(edges_list) > 0 else _empty_int_edges()
        node_component_id_arr = np.concatenate(node_component_id).astype(np.int64)
        node_segment_id_arr = np.concatenate(node_segment_id).astype(np.int64)

    return {
        "vertices": vertices,
        "edges": edges,
        "node_component_id": node_component_id_arr,
        "node_segment_id": node_segment_id_arr,
        "segment_id": np.asarray(segment_id, dtype=np.int64),
        "segment_component_id": np.asarray(segment_component_id, dtype=np.int64),
        "segment_node_start": np.asarray(segment_node_start, dtype=np.int64),
        "segment_node_stop": np.asarray(segment_node_stop, dtype=np.int64),
        "segment_length_nm": np.asarray(segment_length_nm, dtype=np.float64),
    }

def component_summary_arrays(component_skeleton_results):
    return {
        "component_id": np.asarray([r["mito_comp_i"] for r in component_skeleton_results], dtype=np.int64),
        "component_best_skel_comp_i": np.asarray([
            -1 if r["best_skel_comp_i"] is None else int(r["best_skel_comp_i"])
            for r in component_skeleton_results
        ], dtype=np.int64),
        "component_main_length_nm": np.asarray([r["main_length_nm"] for r in component_skeleton_results], dtype=np.float64),
        "component_n_retained_branches": np.asarray([r["n_retained_branches"] for r in component_skeleton_results], dtype=np.int64),
        "component_n_relaxed_segments": np.asarray([r["n_relaxed_segments"] for r in component_skeleton_results], dtype=np.int64),
    }

def save_width_npz(path, agglo_id, mito_path, diameter_result, display_values, display_vmin, display_vmax):
    combined_mito_mesh = diameter_result["combined_mito_mesh"]

    np.savez_compressed(
        path,
        agglo_id=np.asarray([str(agglo_id)]),
        source_mito_path=np.asarray([str(mito_path)]),
        vertices=np.asarray(combined_mito_mesh.vertices, dtype=np.float64),
        faces=np.asarray(combined_mito_mesh.faces, dtype=np.int64),
        diameter_nm=_as_float32_nan_safe(diameter_result["diameter_values"]),
        display_value=_as_float32_nan_safe(display_values),
        display_vmin=np.asarray(float(display_vmin), dtype=np.float32),
        display_vmax=np.asarray(float(display_vmax), dtype=np.float32),
        component_id=np.asarray(diameter_result["all_mito_vertex_components"], dtype=np.int64),
        component_vertex_ranges=np.asarray(diameter_result["component_vertex_ranges"], dtype=np.int64),
        nearest_skel_node_idx=np.asarray(diameter_result["nearest_path_node_idx"], dtype=np.int64),
        vertex_to_skel_dist_nm=_as_float32_nan_safe(diameter_result["vertex_to_path_dist_nm"]),
        diameter_scale_factor=np.asarray(float(DIAMETER_SCALE_FACTOR), dtype=np.float32),
        normalize_per_component_display=np.asarray(int(NORMALIZE_PER_COMPONENT_DISPLAY), dtype=np.int8),
        skeleton_step_size_nm=np.asarray(float(SKELETON_STEP_SIZE_NM), dtype=np.float32),
        branch_length_threshold_nm=np.asarray(float(BRANCH_LENGTH_THRESHOLD_NM), dtype=np.float32),
        lst_n_iters=np.asarray(int(LST_N_ITERS), dtype=np.int32),
        lst_step_size=np.asarray(float(LST_STEP_SIZE), dtype=np.float32),
        lst_tangent_damping=np.asarray(float(LST_TANGENT_DAMPING), dtype=np.float32),
    )

def save_skeleton_npz(path, agglo_id, mito_path, mito_skel, component_skeleton_results, diameter_result):
    skel_arrays = build_skeleton_save_arrays(diameter_result)
    comp_arrays = component_summary_arrays(component_skeleton_results)

    saved_vertices = skel_arrays["vertices"]
    path_nodes_all = np.asarray(diameter_result["path_nodes_all"], dtype=float)

    if len(saved_vertices) != len(path_nodes_all):
        raise ValueError(
            f"Saved skeleton node count ({len(saved_vertices)}) does not match "
            f"diameter mapping node count ({len(path_nodes_all)})."
        )

    if len(saved_vertices) > 0 and not np.allclose(saved_vertices, path_nodes_all):
        raise ValueError("Saved skeleton node coordinates do not match diameter mapping nodes.")

    raw_kwargs = {}

    if SAVE_RAW_WAVEFRONT_SKEL:
        raw_kwargs = {
            "raw_wavefront_vertices": np.asarray(mito_skel.vertices, dtype=np.float64),
            "raw_wavefront_edges": np.asarray(mito_skel.edges, dtype=np.int64),
        }

    np.savez_compressed(
        path,
        agglo_id=np.asarray([str(agglo_id)]),
        source_mito_path=np.asarray([str(mito_path)]),
        vertices=np.asarray(saved_vertices, dtype=np.float64),
        edges=np.asarray(skel_arrays["edges"], dtype=np.int64),
        node_component_id=np.asarray(skel_arrays["node_component_id"], dtype=np.int64),
        node_segment_id=np.asarray(skel_arrays["node_segment_id"], dtype=np.int64),
        node_diameter_nm=_as_float32_nan_safe(diameter_result["path_node_diameters"]),
        node_vertex_count=np.asarray(diameter_result["path_node_vertex_count"], dtype=np.int64),
        segment_id=np.asarray(skel_arrays["segment_id"], dtype=np.int64),
        segment_component_id=np.asarray(skel_arrays["segment_component_id"], dtype=np.int64),
        segment_node_start=np.asarray(skel_arrays["segment_node_start"], dtype=np.int64),
        segment_node_stop=np.asarray(skel_arrays["segment_node_stop"], dtype=np.int64),
        segment_length_nm=np.asarray(skel_arrays["segment_length_nm"], dtype=np.float32),
        component_id=np.asarray(comp_arrays["component_id"], dtype=np.int64),
        component_best_skel_comp_i=np.asarray(comp_arrays["component_best_skel_comp_i"], dtype=np.int64),
        component_main_length_nm=np.asarray(comp_arrays["component_main_length_nm"], dtype=np.float32),
        component_n_retained_branches=np.asarray(comp_arrays["component_n_retained_branches"], dtype=np.int64),
        component_n_relaxed_segments=np.asarray(comp_arrays["component_n_relaxed_segments"], dtype=np.int64),
        skeleton_step_size_nm=np.asarray(float(SKELETON_STEP_SIZE_NM), dtype=np.float32),
        branch_length_threshold_nm=np.asarray(float(BRANCH_LENGTH_THRESHOLD_NM), dtype=np.float32),
        diameter_scale_factor=np.asarray(float(DIAMETER_SCALE_FACTOR), dtype=np.float32),
        **raw_kwargs,
    )
