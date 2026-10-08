"""安全的 CPU 通用网格诊断与修复工具。

默认只读输入 GLB，生成：
  1. JSON 几何诊断报告；
  2. 带颜色标记的诊断 GLB；
  3. 可选的低风险自动修复或人工确认删除版本。

它不假设零件类型，不包含风扇或行星轮专用几何规则。疑似异常面不会被
自动删除。safe 模式只尝试可明确识别的重复数据、退化面和法向问题；任一
操作导致水密、边界、非流形、连通性、尺寸或体积变差时，该操作自动回滚。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import trimesh


AREA_EPSILON = 1e-12
SAFE_VOLUME_RELATIVE_TOLERANCE = 1e-6
SAFE_EXTENT_RELATIVE_TOLERANCE = 1e-9
STANDARD_MAX_HOLE_EDGES = 4
STANDARD_MAX_HOLE_PERIMETER_RATIO = 0.15
STANDARD_MAX_HOLE_AREA_SHARE = 0.002
ADVANCED_MAX_AREA_RELATIVE_CHANGE = 0.05
DEGENERATE_COLOR = [255, 0, 0, 255]
SUSPECT_COLOR = [255, 165, 0, 255]
NORMAL_COLOR = [180, 180, 180, 255]
SURFACE_SHARP_ANGLE_DEGREES = 35.0
SURFACE_REMESH_FEATURE_ANGLE_DEGREES = 45.0
SURFACE_REMESH_MAX_EXTENT_RELATIVE_CHANGE = 0.001
SURFACE_REMESH_MAX_AREA_RELATIVE_CHANGE = 0.005
SURFACE_REMESH_MAX_VOLUME_RELATIVE_CHANGE = 0.001
SURFACE_REMESH_MAX_P99_DISTANCE_RATIO = 0.001
SURFACE_REMESH_MAX_DISTANCE_RATIO = 0.002
RECONSTRUCTION_MIN_MAIN_AREA_SHARE = 0.95
RECONSTRUCTION_MAX_EXTENT_RELATIVE_CHANGE = 0.05
RECONSTRUCTION_MAX_CENTER_SHIFT_RATIO = 0.02
RECONSTRUCTION_MAX_P99_DISTANCE_RATIO = 0.02
RECONSTRUCTION_MAX_DISTANCE_RATIO = 0.04
ARTIFACT_CLEANUP_MIN_MAIN_AREA_SHARE = 0.995
ARTIFACT_CLEANUP_MAX_DISCARDED_AREA_SHARE = 0.005
ARTIFACT_CLEANUP_MAX_DISCARDED_VOLUME_SHARE = 0.005
ARTIFACT_CLEANUP_MAX_FRAGMENT_EXTENT_RATIO = 0.05
ARTIFACT_CLEANUP_MAX_COMPONENTS = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    result = np.percentile(values, [50, 90, 95, 99, 100])
    return {
        "p50": float(result[0]),
        "p90": float(result[1]),
        "p95": float(result[2]),
        "p99": float(result[3]),
        "max": float(result[4]),
    }


def surface_quality_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Measure surface sampling and normal changes without altering geometry.

    These are diagnostic signals, not automatic defect labels. A gear tooth
    boundary is intentionally sharp, so high normal changes can be legitimate.
    """
    if not len(mesh.faces):
        return {
            "edge_length": _percentiles(np.empty(0)),
            "triangle_aspect_ratio": _percentiles(np.empty(0)),
            "adjacent_normal_angle_degrees": _percentiles(np.empty(0)),
            "sharp_adjacent_edge_count": 0,
            "sharp_angle_threshold_degrees": SURFACE_SHARP_ANGLE_DEGREES,
        }

    edge_lengths = np.asarray(mesh.edges_unique_length, dtype=float)
    aspect = triangle_aspect_ratios(mesh)
    angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    angles = np.degrees(angles[np.isfinite(angles)])
    return {
        "edge_length": _percentiles(edge_lengths),
        "triangle_aspect_ratio": _percentiles(aspect),
        "adjacent_normal_angle_degrees": _percentiles(angles),
        "sharp_adjacent_edge_count": int(
            np.count_nonzero(angles >= SURFACE_SHARP_ANGLE_DEGREES)
        ),
        "sharp_angle_threshold_degrees": SURFACE_SHARP_ANGLE_DEGREES,
        "interpretation": (
            "高法向夹角可能是齿尖、齿根、中心孔等合法锐边；"
            "该指标用于定位曲面质量候选，不能单独判定应平滑或删除。"
        ),
    }


def smooth_region_normal_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Measure adjacent-normal changes below the protected sharp-edge cutoff."""
    angles = np.degrees(np.asarray(mesh.face_adjacency_angles, dtype=float))
    angles = angles[np.isfinite(angles)]
    smooth = angles[angles < SURFACE_SHARP_ANGLE_DEGREES]
    return {
        "below_sharp_threshold_degrees": _percentiles(smooth),
        "smooth_adjacency_count": int(len(smooth)),
        "sharp_adjacency_count": int(np.count_nonzero(
            angles >= SURFACE_SHARP_ANGLE_DEGREES
        )),
        "sharp_threshold_degrees": SURFACE_SHARP_ANGLE_DEGREES,
    }


def bidirectional_surface_distance_metrics(
    source: trimesh.Trimesh,
    candidate: trimesh.Trimesh,
    *,
    samples: int = 20000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Compute sampled point-to-triangle distance in both directions."""
    if samples < 1000:
        raise ValueError("曲面距离采样数不得少于 1000。")
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        source_points, _ = trimesh.sample.sample_surface(source, samples)
        candidate_points, _ = trimesh.sample.sample_surface(candidate, samples)
    finally:
        np.random.set_state(state)
    _, source_to_candidate, _ = trimesh.proximity.closest_point(
        candidate, source_points
    )
    _, candidate_to_source, _ = trimesh.proximity.closest_point(
        source, candidate_points
    )
    diagonal = max(float(np.linalg.norm(source.extents)), AREA_EPSILON)

    def summarize(values: np.ndarray) -> dict[str, float]:
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(values.mean()),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max_sampled": float(values.max(initial=0.0)),
        }

    source_stats = summarize(source_to_candidate)
    candidate_stats = summarize(candidate_to_source)
    p99 = max(source_stats["p99"], candidate_stats["p99"])
    maximum = max(
        source_stats["max_sampled"], candidate_stats["max_sampled"]
    )
    return {
        "samples_per_direction": int(samples),
        "seed": int(seed),
        "source_to_candidate": source_stats,
        "candidate_to_source": candidate_stats,
        "symmetric_mean": float(
            (source_stats["mean"] + candidate_stats["mean"]) / 2.0
        ),
        "p99_bidirectional": float(p99),
        "max_bidirectional_sampled": float(maximum),
        "source_bbox_diagonal": diagonal,
        "p99_to_diagonal_ratio": float(p99 / diagonal),
        "max_to_diagonal_ratio": float(maximum / diagonal),
    }


def metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    parts = sorted(mesh.split(only_watertight=False), key=lambda item: item.area, reverse=True)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "degenerate_faces": int((np.asarray(mesh.area_faces) < AREA_EPSILON).sum()),
        "duplicate_vertices_exact": int(
            len(mesh.vertices) - len(np.unique(np.asarray(mesh.vertices), axis=0))
        ),
        "duplicate_faces": int(len(mesh.faces) - int(mesh.unique_faces().sum())),
        "volume": float(mesh.volume),
        "area": float(mesh.area),
        "bounds": np.asarray(mesh.bounds).tolist(),
        "extents": np.asarray(mesh.extents).tolist(),
        "component_area_shares": [
            float(part.area / mesh.area) if mesh.area else 0.0 for part in parts
        ],
        "surface_quality": surface_quality_metrics(mesh),
    }


def assess_repairability(values: dict[str, Any]) -> dict[str, Any]:
    """Route a mesh to the least invasive reasonable next stage.

    This is geometry-only triage. It cannot decide whether a gear tooth count,
    hole diameter or other semantic feature is correct.
    """
    reasons: list[str] = []
    if values["components"] > 32 or values["boundary_edges"] > 1000:
        route = "RECONSTRUCTION_RECOMMENDED"
        reasons.append("存在大量连通部件或开放边界，局部自动修复风险过高。")
    elif (
        1 < values["components"] <= ARTIFACT_CLEANUP_MAX_COMPONENTS
        and values.get("component_area_shares")
        and values["component_area_shares"][0]
        >= ARTIFACT_CLEANUP_MIN_MAIN_AREA_SHARE
        and values["nonmanifold_edges"] == 0
    ):
        route = "DETACHED_ARTIFACT_CLEANUP_CANDIDATE"
        reasons.append(
            "主体表面积占比至少 99.5%，其余为独立小部件；可生成受控伪影清理候选。"
        )
    elif values["components"] > 1 or values["nonmanifold_edges"] > 0:
        route = "ADVANCED_TOPOLOGY_REPAIR_CANDIDATE"
        reasons.append("存在多部件或非流形结构，需要高级拓扑候选和人工复核。")
    elif values["boundary_edges"] > 0:
        route = "STANDARD_HOLE_REPAIR_CANDIDATE"
        reasons.append("存在开放边界，只允许受限小孔修复；复杂孔洞应转重建。")
    elif (
        values["degenerate_faces"] > 0
        or values["duplicate_faces"] > 0
        or values["duplicate_vertices_exact"] > 0
        or not values["winding_consistent"]
    ):
        route = "SAFE_REPAIR_CANDIDATE"
        reasons.append("问题属于退化、重复数据或法向，可优先尝试低风险修复。")
    else:
        quality = values.get("surface_quality", {})
        aspect = quality.get("triangle_aspect_ratio", {})
        normal = quality.get("adjacent_normal_angle_degrees", {})
        if aspect.get("p99", 0.0) > 20.0 or aspect.get("max", 0.0) > 1000.0:
            route = "SURFACE_QUALITY_VISUAL_REVIEW_REQUIRED"
            reasons.append("基础拓扑健康，但存在显著细长三角形或曲面采样异常。")
        elif normal.get("p99", 0.0) > 45.0:
            route = "SURFACE_QUALITY_VISUAL_REVIEW_REQUIRED"
            reasons.append("基础拓扑健康，但局部法向变化较大，需要区分合法锐边与表面噪声。")
        else:
            route = "BASE_GEOMETRY_CHECK_PASSED"
            reasons.append("基础拓扑与通用曲面指标未发现明确自动修复目标。")
    return {
        "route": route,
        "reasons": reasons,
        "requires_visual_review": route not in {
            "BASE_GEOMETRY_CHECK_PASSED",
            "SAFE_REPAIR_CANDIDATE",
        },
        "semantic_limit": (
            "该分流不识别齿数、孔径、公差、配合面或渐开线；"
            "视觉复核失败时应转向结构/CAD 重建。"
        ),
    }


def safe_gate(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    allow_oriented_volume_change: bool = False,
) -> tuple[bool, list[str]]:
    """Reject a trial operation if it makes core geometry invariants worse."""
    failures: list[str] = []
    if before["watertight"] and not after["watertight"]:
        failures.append("破坏了输入网格的水密性")
    if after["boundary_edges"] > before["boundary_edges"]:
        failures.append("边界边数量增加")
    if after["nonmanifold_edges"] > before["nonmanifold_edges"]:
        failures.append("非流形边数量增加")
    if after["components"] > before["components"]:
        failures.append("连通部件数量增加")
    if before["winding_consistent"] and not after["winding_consistent"]:
        failures.append("破坏了原有法向一致性")

    before_extents = np.asarray(before["extents"], dtype=float)
    after_extents = np.asarray(after["extents"], dtype=float)
    extent_atol = max(float(before_extents.max(initial=0.0)) * 1e-12, 1e-12)
    if not np.allclose(
        before_extents,
        after_extents,
        rtol=SAFE_EXTENT_RELATIVE_TOLERANCE,
        atol=extent_atol,
    ):
        failures.append("包围盒尺寸发生超出 safe 阈值的变化")

    before_volume = abs(float(before["volume"]))
    after_volume = abs(float(after["volume"]))
    if (
        before["watertight"]
        and before_volume > AREA_EPSILON
        and not allow_oriented_volume_change
    ):
        relative_change = abs(after_volume - before_volume) / before_volume
        if relative_change > SAFE_VOLUME_RELATIVE_TOLERANCE:
            failures.append("有效体积变化超过 safe 阈值")
    return not failures, failures


def export_reload_gate(
    before_export: dict[str, Any],
    after_reload: dict[str, Any],
    *,
    geometry_relative_tolerance: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Verify that serialization did not materially alter an accepted candidate."""
    failures: list[str] = []
    for key, label in (
        ("components", "连通部件"),
        ("boundary_edges", "边界边"),
        ("nonmanifold_edges", "非流形边"),
        ("degenerate_faces", "退化面"),
    ):
        if after_reload[key] != before_export[key]:
            failures.append(f"导出重载后{label}数量变化")
    if after_reload["watertight"] != before_export["watertight"]:
        failures.append("导出重载后水密性变化")
    if after_reload["winding_consistent"] != before_export["winding_consistent"]:
        failures.append("导出重载后法向一致性变化")

    before_extents = np.asarray(before_export["extents"], dtype=float)
    after_extents = np.asarray(after_reload["extents"], dtype=float)
    if not np.allclose(
        before_extents,
        after_extents,
        rtol=geometry_relative_tolerance,
        atol=max(float(before_extents.max(initial=0.0)) * 1e-9, 1e-12),
    ):
        failures.append("导出重载后包围盒尺寸变化超过浮点容差")
    for key, label in (("area", "表面积"), ("volume", "体积")):
        before_value = abs(float(before_export[key]))
        after_value = abs(float(after_reload[key]))
        relative = abs(after_value - before_value) / max(before_value, AREA_EPSILON)
        if relative > geometry_relative_tolerance:
            failures.append(f"导出重载后{label}变化超过浮点容差")
    return not failures, failures


def _attempt_safe_operation(
    mesh: trimesh.Trimesh,
    name: str,
    operation: Any,
    *,
    allow_oriented_volume_change: bool = False,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    before = metrics(mesh)
    trial = mesh.copy()
    operation(trial)
    after = metrics(trial)
    accepted, failures = safe_gate(
        before,
        after,
        allow_oriented_volume_change=allow_oriented_volume_change,
    )
    changed = (
        before["vertices"] != after["vertices"]
        or before["faces"] != after["faces"]
        or before["winding_consistent"] != after["winding_consistent"]
        or before["duplicate_vertices_exact"] != after["duplicate_vertices_exact"]
        or before["duplicate_faces"] != after["duplicate_faces"]
        or before["degenerate_faces"] != after["degenerate_faces"]
    )
    if not changed:
        accepted = False
        failures = ["未发现该操作可修复的问题"]
    return (trial if accepted else mesh), {
        "name": name,
        "accepted": accepted,
        "changed": changed,
        "rejection_reasons": failures,
        "before": before,
        "after_trial": after,
    }


def safe_repair(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[dict[str, Any]]]:
    """Apply low-risk operations independently, rolling back unsafe trials."""
    current = mesh.copy()
    operations: list[dict[str, Any]] = []

    def remove_nonfinite(value: trimesh.Trimesh) -> None:
        value.remove_infinite_values()

    def remove_unreferenced(value: trimesh.Trimesh) -> None:
        value.remove_unreferenced_vertices()

    def merge_exact_vertices(value: trimesh.Trimesh) -> None:
        # 15 位小数只用于合并数值相同的顶点，避免把相邻齿面焊接在一起。
        value.merge_vertices(digits_vertex=15)

    def remove_duplicate_faces(value: trimesh.Trimesh) -> None:
        value.update_faces(value.unique_faces())
        value.remove_unreferenced_vertices()

    def remove_degenerate_faces(value: trimesh.Trimesh) -> None:
        value.update_faces(np.asarray(value.area_faces) >= AREA_EPSILON)
        value.remove_unreferenced_vertices()

    def fix_normals(value: trimesh.Trimesh) -> None:
        trimesh.repair.fix_normals(value, multibody=True)

    for name, operation, allow_volume_change in (
        ("remove_nonfinite_values", remove_nonfinite, False),
        ("remove_unreferenced_vertices", remove_unreferenced, False),
        ("merge_exact_duplicate_vertices", merge_exact_vertices, False),
        ("remove_duplicate_faces", remove_duplicate_faces, False),
        ("remove_near_zero_area_faces", remove_degenerate_faces, False),
        ("fix_winding_and_normals", fix_normals, True),
    ):
        current, record = _attempt_safe_operation(
            current,
            name,
            operation,
            allow_oriented_volume_change=allow_volume_change,
        )
        operations.append(record)
    return current, operations


def boundary_loops(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    """Return simple boundary cycles and reject branched/open boundary graphs."""
    if not len(mesh.faces):
        return []
    boundary_groups = trimesh.grouping.group_rows(
        np.asarray(mesh.edges_sorted), require_count=1
    )
    if len(boundary_groups) < 3:
        return []
    oriented_edges = np.asarray(mesh.edges)[boundary_groups]
    graph = nx.Graph()
    for occurrence, edge in zip(boundary_groups, oriented_edges):
        graph.add_edge(int(edge[0]), int(edge[1]), occurrence=int(occurrence))

    records: list[dict[str, Any]] = []
    for component_nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(component_nodes).copy()
        simple = bool(
            len(subgraph) >= 3
            and all(degree == 2 for _, degree in subgraph.degree())
        )
        cycles = nx.cycle_basis(subgraph) if simple else []
        if len(cycles) != 1:
            records.append(
                {
                    "vertices": sorted(int(value) for value in component_nodes),
                    "edge_count": int(subgraph.number_of_edges()),
                    "simple_cycle": False,
                    "reason": "边界图存在分叉、断链或多个环，拒绝自动填补。",
                }
            )
            continue
        loop = [int(value) for value in cycles[0]]
        points = np.asarray(mesh.vertices)[loop]
        perimeter = float(
            np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum()
        )
        records.append(
            {
                "vertices": loop,
                "edge_count": len(loop),
                "simple_cycle": True,
                "perimeter": perimeter,
            }
        )
    return records


def _faces_for_boundary_loop(
    mesh: trimesh.Trimesh, loop: list[int]
) -> np.ndarray:
    """Triangulate a 3- or 4-edge loop and orient it against adjacent faces."""
    vertices = np.asarray(loop, dtype=int)
    if len(vertices) == 3:
        faces = np.asarray([vertices], dtype=int)
    elif len(vertices) == 4:
        faces = np.asarray(
            [vertices[[0, 1, 2]], vertices[[2, 3, 0]]], dtype=int
        )
    else:
        return np.empty((0, 3), dtype=int)

    boundary_groups = trimesh.grouping.group_rows(
        np.asarray(mesh.edges_sorted), require_count=1
    )
    edge_lookup: dict[tuple[int, int], np.ndarray] = {}
    for occurrence in boundary_groups:
        edge = np.asarray(mesh.edges)[occurrence]
        edge_lookup[tuple(sorted((int(edge[0]), int(edge[1]))))] = edge
    edge_test = faces[0, :2]
    adjacent = edge_lookup.get(tuple(sorted((int(edge_test[0]), int(edge_test[1])))))
    if adjacent is None:
        return np.empty((0, 3), dtype=int)
    if int(edge_test[0]) != int(adjacent[1]):
        faces = faces[:, ::-1]
    return faces


def fill_small_boundary_holes(
    mesh: trimesh.Trimesh,
    *,
    max_edges: int = STANDARD_MAX_HOLE_EDGES,
    max_perimeter_ratio: float = STANDARD_MAX_HOLE_PERIMETER_RATIO,
    max_area_share: float = STANDARD_MAX_HOLE_AREA_SHARE,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Fill only simple, tiny triangle/quad boundary loops with rollback gates."""
    current = mesh.copy()
    before = metrics(current)
    diagonal = float(np.linalg.norm(np.asarray(current.extents, dtype=float)))
    total_area = float(current.area)
    records = boundary_loops(current)
    eligible_faces: list[np.ndarray] = []

    for record in records:
        record["eligible"] = False
        if not record.get("simple_cycle"):
            continue
        if record["edge_count"] > max_edges:
            record["reason"] = "边界环边数超过 standard 限制。"
            continue
        perimeter_ratio = record["perimeter"] / diagonal if diagonal else float("inf")
        record["perimeter_ratio"] = perimeter_ratio
        if perimeter_ratio > max_perimeter_ratio:
            record["reason"] = "边界环周长相对模型尺寸过大。"
            continue
        faces = _faces_for_boundary_loop(current, record["vertices"])
        if not len(faces):
            record["reason"] = "边界环无法安全三角化或确定绕序。"
            continue
        patch_area = float(
            trimesh.triangles.area(np.asarray(current.vertices)[faces]).sum()
        )
        area_share = patch_area / total_area if total_area else float("inf")
        record["patch_area"] = patch_area
        record["patch_area_share"] = area_share
        if area_share > max_area_share:
            record["reason"] = "补片面积占模型总面积比例过大。"
            continue
        record["eligible"] = True
        record["reason"] = "满足小型简单边界环限制。"
        eligible_faces.append(faces)

    def compact_records() -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for source in records:
            record = dict(source)
            vertices = record.pop("vertices", [])
            record["vertex_count"] = len(vertices)
            record["vertex_preview"] = vertices[:16]
            compact.append(record)
        return compact

    if not eligible_faces:
        return current, {
            "name": "fill_small_boundary_holes",
            "accepted": False,
            "changed": False,
            "rejection_reasons": ["没有符合 standard 限制的小型简单边界环。"],
            "boundary_loops": compact_records(),
            "before": before,
            "after_trial": before,
        }

    trial = current.copy()
    added = np.vstack(eligible_faces)
    trial.faces = np.vstack((np.asarray(trial.faces), added))
    trimesh.repair.fix_normals(trial, multibody=True)
    after = metrics(trial)
    accepted, failures = safe_gate(
        before,
        after,
        allow_oriented_volume_change=True,
    )
    if after["boundary_edges"] >= before["boundary_edges"]:
        accepted = False
        failures.append("边界边数量没有减少。")
    if after["nonmanifold_edges"] > before["nonmanifold_edges"]:
        accepted = False
        failures.append("补洞产生了新的非流形边。")
    return (trial if accepted else current), {
        "name": "fill_small_boundary_holes",
        "accepted": accepted,
        "changed": True,
        "faces_added": int(len(added)),
        "rejection_reasons": failures,
        "boundary_loops": compact_records(),
        "before": before,
        "after_trial": after,
    }


def standard_repair(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, list[dict[str, Any]]]:
    """Run safe repair, then guarded filling of tiny triangle/quad holes."""
    repaired, operations = safe_repair(mesh)
    repaired, hole_record = fill_small_boundary_holes(repaired)
    operations.append(hole_record)
    return repaired, operations


def repair_degenerate_faces_by_edge_flip(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Retriangulate tiny manifold face pairs without moving any vertex."""
    before = metrics(mesh)
    current = mesh.copy()
    flips: list[dict[str, Any]] = []
    initial_degenerate = int(before["degenerate_faces"])
    if initial_degenerate == 0:
        return current, {
            "name": "local_degenerate_face_edge_flip",
            "accepted": False,
            "changed": False,
            "flips": [],
            "rejection_reasons": ["主体没有需要局部重划分的退化面。"],
            "before": before,
            "after_trial": before,
        }

    for _ in range(initial_degenerate):
        degenerate = np.flatnonzero(np.asarray(current.area_faces) < AREA_EPSILON)
        if not len(degenerate):
            break
        face_index = int(degenerate[0])
        adjacency = np.asarray(current.face_adjacency)
        pairs = adjacency[np.any(adjacency == face_index, axis=1)]
        best: tuple[float, trimesh.Trimesh, dict[str, Any]] | None = None
        for pair in pairs:
            neighbor = int(pair[0] if pair[1] == face_index else pair[1])
            face = np.asarray(current.faces[face_index], dtype=int)
            other = np.asarray(current.faces[neighbor], dtype=int)
            shared = sorted(set(face.tolist()) & set(other.tolist()))
            unique_face = list(set(face.tolist()) - set(shared))
            unique_other = list(set(other.tolist()) - set(shared))
            if len(shared) != 2 or len(unique_face) != 1 or len(unique_other) != 1:
                continue
            u, v = shared
            c, d = unique_face[0], unique_other[0]
            for orientation, replacement in enumerate(
                (
                    np.asarray([[c, d, u], [d, c, v]], dtype=int),
                    np.asarray([[c, u, d], [d, v, c]], dtype=int),
                )
            ):
                faces = np.asarray(current.faces).copy()
                faces[[face_index, neighbor]] = replacement
                trial = trimesh.Trimesh(
                    vertices=np.asarray(current.vertices).copy(),
                    faces=faces,
                    process=False,
                )
                trial_metrics = metrics(trial)
                if not (
                    trial_metrics["watertight"]
                    and trial_metrics["winding_consistent"]
                    and trial_metrics["components"] == before["components"]
                    and trial_metrics["boundary_edges"] == before["boundary_edges"]
                    and trial_metrics["nonmanifold_edges"] == before["nonmanifold_edges"]
                    and trial_metrics["degenerate_faces"]
                    < metrics(current)["degenerate_faces"]
                ):
                    continue
                pair_areas = np.asarray(trial.area_faces)[[face_index, neighbor]]
                score = float(pair_areas.min())
                detail = {
                    "face_index": face_index,
                    "neighbor_face_index": neighbor,
                    "orientation": orientation,
                    "replacement_faces": replacement.tolist(),
                    "pair_areas_after": pair_areas.tolist(),
                    "minimum_pair_area_after": score,
                }
                if best is None or score > best[0]:
                    best = (score, trial, detail)
        if best is None:
            break
        current = best[1]
        flips.append(best[2])

    after = metrics(current)
    failures: list[str] = []
    if after["degenerate_faces"] >= before["degenerate_faces"]:
        failures.append("局部换边没有减少退化面。")
    if not after["watertight"] or not after["winding_consistent"]:
        failures.append("局部换边破坏水密性或法向一致性。")
    if after["boundary_edges"] != before["boundary_edges"]:
        failures.append("局部换边改变边界边数量。")
    if after["nonmanifold_edges"] != before["nonmanifold_edges"]:
        failures.append("局部换边改变非流形边数量。")
    if not np.array_equal(np.asarray(current.vertices), np.asarray(mesh.vertices)):
        failures.append("局部换边意外移动了顶点。")
    area_change = abs(float(after["area"]) - float(before["area"])) / max(
        abs(float(before["area"])), AREA_EPSILON
    )
    volume_change = abs(abs(float(after["volume"])) - abs(float(before["volume"]))) / max(
        abs(float(before["volume"])), AREA_EPSILON
    )
    if area_change > 1e-6:
        failures.append("局部换边引起的总表面积变化超过 0.0001%。")
    if volume_change > 1e-6:
        failures.append("局部换边引起的总体积变化超过 0.0001%。")
    accepted = bool(flips) and not failures
    return (current if accepted else mesh), {
        "name": "local_degenerate_face_edge_flip",
        "accepted": accepted,
        "changed": bool(flips),
        "flips": flips,
        "area_relative_change": area_change,
        "volume_relative_change": volume_change,
        "rejection_reasons": failures,
        "before": before,
        "after_trial": after,
    }


def detached_artifact_cleanup_candidate(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Keep one dominant body and remove only tiny detached solid artifacts.

    This mode is intentionally limited to single-object reconstruction output.
    It does not infer assembly semantics, so every accepted result remains a
    candidate requiring a human before/after review.
    """
    before = metrics(mesh)
    parts = sorted(
        mesh.split(only_watertight=False), key=lambda item: item.area, reverse=True
    )
    record: dict[str, Any] = {
        "name": "remove_tiny_detached_artifacts_candidate",
        "accepted": False,
        "changed": False,
        "before": before,
        "component_count_before": int(len(parts)),
        "thresholds": {
            "minimum_main_area_share": ARTIFACT_CLEANUP_MIN_MAIN_AREA_SHARE,
            "maximum_discarded_area_share": ARTIFACT_CLEANUP_MAX_DISCARDED_AREA_SHARE,
            "maximum_discarded_volume_share": ARTIFACT_CLEANUP_MAX_DISCARDED_VOLUME_SHARE,
            "maximum_fragment_extent_to_main_ratio": ARTIFACT_CLEANUP_MAX_FRAGMENT_EXTENT_RATIO,
            "maximum_component_count": ARTIFACT_CLEANUP_MAX_COMPONENTS,
        },
        "rejection_reasons": [],
        "semantic_warning": (
            "仅适用于预期为单一零件的模型；独立小部件也可能是合法装配件，必须人工复核。"
        ),
    }
    failures: list[str] = record["rejection_reasons"]
    if len(parts) < 2:
        failures.append("输入只有一个连通部件，没有独立伪影可清理。")
    if len(parts) > ARTIFACT_CLEANUP_MAX_COMPONENTS:
        failures.append("连通部件过多，应转入严重模型重建分流。")
    if failures:
        record["after_trial"] = before
        return mesh, record

    total_area = max(float(mesh.area), AREA_EPSILON)
    total_volume = max(sum(abs(float(part.volume)) for part in parts), AREA_EPSILON)
    main = parts[0].copy()
    fragments = parts[1:]
    main_area_share = float(main.area / total_area)
    discarded_area_share = float(sum(part.area for part in fragments) / total_area)
    discarded_volume_share = float(
        sum(abs(float(part.volume)) for part in fragments) / total_volume
    )
    main_longest = max(float(np.max(main.extents)), AREA_EPSILON)
    fragment_extent_ratios = [
        float(np.max(part.extents) / main_longest) for part in fragments
    ]
    record.update(
        {
            "main_area_share": main_area_share,
            "discarded_component_count": int(len(fragments)),
            "discarded_area_share": discarded_area_share,
            "discarded_volume_share": discarded_volume_share,
            "fragment_extent_to_main_ratios": fragment_extent_ratios,
            "discarded_components": [
                {
                    "rank": index + 2,
                    "vertices": int(len(part.vertices)),
                    "faces": int(len(part.faces)),
                    "area": float(part.area),
                    "area_share": float(part.area / total_area),
                    "volume": float(part.volume),
                    "extent_to_main_ratio": fragment_extent_ratios[index],
                    "bounds": np.asarray(part.bounds).tolist(),
                }
                for index, part in enumerate(fragments)
            ],
        }
    )
    if main_area_share < ARTIFACT_CLEANUP_MIN_MAIN_AREA_SHARE:
        failures.append("最大主体表面积占比低于 99.5%，无法确认其余部件是伪影。")
    if discarded_area_share > ARTIFACT_CLEANUP_MAX_DISCARDED_AREA_SHARE:
        failures.append("待丢弃部件总表面积占比超过 0.5%。")
    if discarded_volume_share > ARTIFACT_CLEANUP_MAX_DISCARDED_VOLUME_SHARE:
        failures.append("待丢弃部件总体积占比超过 0.5%。")
    if max(fragment_extent_ratios, default=0.0) > ARTIFACT_CLEANUP_MAX_FRAGMENT_EXTENT_RATIO:
        failures.append("至少一个独立部件的尺寸超过主体最大尺寸的 5%。")

    main, degenerate_operation = repair_degenerate_faces_by_edge_flip(main)
    record["local_degenerate_retriangulation"] = degenerate_operation
    after = metrics(main)
    if after["components"] != 1:
        failures.append("候选不是单一连通主体。")
    if not after["watertight"] or not after["winding_consistent"]:
        failures.append("保留主体未通过水密或法向一致门禁。")
    if after["boundary_edges"] or after["nonmanifold_edges"] or after["degenerate_faces"]:
        failures.append("保留主体仍含边界、非流形或退化面。")
    if abs(float(after["volume"])) <= AREA_EPSILON:
        failures.append("保留主体没有有效正体积。")

    record.update(
        {
            "accepted": not failures,
            "changed": True,
            "after_trial": after,
        }
    )
    return (main if not failures else mesh), record


def advanced_repair_candidate(
    mesh: trimesh.Trimesh,
    input_path: Path,
    *,
    max_hole_size: int = 4,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Run PyMeshLab's guarded topology candidate filters.

    Self-intersecting faces are measured but never deleted automatically. The
    candidate is accepted only if topology does not regress and area/extent
    changes stay bounded.
    """
    try:
        import pymeshlab
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "advanced 模式需要 Hunyuan 环境中的 pymeshlab。"
        ) from exc

    before = metrics(mesh)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(input_path))

    def selected_self_intersections() -> int:
        try:
            mesh_set.apply_filter("compute_selection_by_self_intersections_per_face")
            return int(mesh_set.current_mesh().selected_face_number())
        except Exception:
            return -1

    self_intersections_before = selected_self_intersections()
    filters = [
        ("remove_duplicate_vertices", "meshing_remove_duplicate_vertices", {}),
        ("remove_duplicate_faces", "meshing_remove_duplicate_faces", {}),
        ("remove_null_faces", "meshing_remove_null_faces", {}),
        (
            "repair_non_manifold_edges",
            "meshing_repair_non_manifold_edges",
            {"method": 0},
        ),
        (
            "repair_non_manifold_vertices",
            "meshing_repair_non_manifold_vertices",
            {"vertdispratio": 0.0},
        ),
        (
            "close_small_holes_candidate",
            "meshing_close_holes",
            {
                "maxholesize": int(max_hole_size),
                "selected": False,
                "newfaceselected": True,
                "selfintersection": False,
            },
        ),
    ]
    filter_records: list[dict[str, Any]] = []
    for name, filter_name, parameters in filters:
        try:
            mesh_set.apply_filter(filter_name, **parameters)
            filter_records.append(
                {"name": name, "filter": filter_name, "parameters": parameters, "applied": True}
            )
        except Exception as exc:
            filter_records.append(
                {
                    "name": name,
                    "filter": filter_name,
                    "parameters": parameters,
                    "applied": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    self_intersections_after = selected_self_intersections()
    with tempfile.TemporaryDirectory(prefix="industrial_ai_advanced_mesh_") as temp_dir:
        temporary_mesh = Path(temp_dir) / "candidate.ply"
        mesh_set.save_current_mesh(
            str(temporary_mesh),
            save_vertex_color=False,
            save_wedge_color=False,
        )
        candidate = trimesh.load(temporary_mesh, force="mesh", process=False)
    if not isinstance(candidate, trimesh.Trimesh):
        raise RuntimeError("PyMeshLab 高级候选无法重新加载为 Trimesh。")

    after = metrics(candidate)
    accepted, failures = safe_gate(
        before,
        after,
        allow_oriented_volume_change=True,
    )
    area_change = (after["area"] - before["area"]) / max(
        abs(before["area"]), AREA_EPSILON
    )
    if area_change > ADVANCED_MAX_AREA_RELATIVE_CHANGE:
        accepted = False
        failures.append("表面积变化超过 advanced 5% 门禁")
    if (
        self_intersections_before >= 0
        and self_intersections_after > self_intersections_before
    ):
        accepted = False
        failures.append("自交面数量增加")
    changed = (
        before["vertices"] != after["vertices"]
        or before["faces"] != after["faces"]
        or before["components"] != after["components"]
        or before["boundary_edges"] != after["boundary_edges"]
        or before["nonmanifold_edges"] != after["nonmanifold_edges"]
        or before["degenerate_faces"] != after["degenerate_faces"]
    )
    if not changed:
        accepted = False
        failures.append("高级过滤器没有改变网格")
    record = {
        "name": "pymeshlab_advanced_topology_candidate",
        "accepted": accepted,
        "changed": changed,
        "max_hole_size": int(max_hole_size),
        "filters": filter_records,
        "self_intersections_before": self_intersections_before,
        "self_intersections_after": self_intersections_after,
        "area_relative_change": area_change,
        "rejection_reasons": failures,
        "before": before,
        "after_trial": after,
    }
    return (candidate if accepted else mesh), record


def triangle_aspect_ratios(mesh: trimesh.Trimesh) -> np.ndarray:
    triangles = np.asarray(mesh.triangles)
    edges = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    shortest = np.maximum(edges.min(axis=1), np.finfo(float).eps)
    return edges.max(axis=1) / shortest


def surface_smooth_candidate(
    mesh: trimesh.Trimesh,
    input_path: Path,
    *,
    aspect_threshold: float = 50.0,
    iterations: int = 1,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Try a tiny, feature-protected smoothing candidate.

    Only vertices incident to very stretched triangles are selected. Vertices
    touching a sharp adjacent-face angle are protected, so tooth tips, roots,
    hole rims and other feature edges are not intentionally smoothed. This is
    an experimental candidate, never the default repair path.
    """
    if iterations < 1 or iterations > 3:
        raise ValueError("surface 平滑迭代次数必须在 1 到 3 之间。")
    if aspect_threshold < 10.0:
        raise ValueError("surface 长宽比阈值过低，拒绝选择过大的区域。")
    try:
        import pymeshlab
    except ModuleNotFoundError as exc:
        raise RuntimeError("surface 平滑候选需要 Hunyuan 环境中的 pymeshlab。") from exc

    before = metrics(mesh)
    aspects = triangle_aspect_ratios(mesh)
    candidate_faces = np.flatnonzero(aspects >= aspect_threshold)
    candidate_vertices = np.unique(np.asarray(mesh.faces)[candidate_faces].ravel())

    protected = np.zeros(len(mesh.vertices), dtype=bool)
    adjacency_angles = np.degrees(np.asarray(mesh.face_adjacency_angles, dtype=float))
    adjacency_edges = np.asarray(mesh.face_adjacency_edges, dtype=int)
    sharp = np.isfinite(adjacency_angles) & (
        adjacency_angles >= SURFACE_SHARP_ANGLE_DEGREES
    )
    if np.any(sharp):
        protected[np.unique(adjacency_edges[sharp].ravel())] = True
    selected_vertices = candidate_vertices[~protected[candidate_vertices]]

    record: dict[str, Any] = {
        "name": "feature_protected_surface_smoothing_candidate",
        "accepted": False,
        "changed": False,
        "aspect_threshold": float(aspect_threshold),
        "iterations": int(iterations),
        "candidate_face_count": int(len(candidate_faces)),
        "candidate_vertex_count": int(len(candidate_vertices)),
        "protected_vertex_count": int(np.count_nonzero(protected)),
        "selected_vertex_count": int(len(selected_vertices)),
        "before": before,
        "rejection_reasons": [],
    }
    if not len(selected_vertices):
        record["rejection_reasons"] = [
            "所有疑似细长三角形顶点都位于保护锐边，未执行平滑。"
        ]
        record["after_trial"] = before
        return mesh, record

    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(input_path))
    current = mesh_set.current_mesh()
    selection = np.zeros(current.vertex_number(), dtype=bool)
    valid_selected = selected_vertices[selected_vertices < len(selection)]
    selection[valid_selected] = True
    # PyMeshLab exposes selection as a filter operation; a temporary scalar
    # attribute lets us select an exact vertex subset without global smoothing.
    scalar = selection.astype(float)
    current.add_vertex_custom_scalar_attribute(scalar, "industrial_ai_surface_candidate")
    mesh_set.apply_filter(
        "compute_selection_by_condition_per_vertex",
        condselect="(industrial_ai_surface_candidate > 0.5)",
    )
    mesh_set.apply_filter(
        "apply_coord_laplacian_smoothing",
        stepsmoothnum=int(iterations),
        boundary=False,
        cotangentweight=True,
        selected=True,
    )

    with tempfile.TemporaryDirectory(prefix="industrial_ai_surface_smooth_") as temp_dir:
        temporary_mesh = Path(temp_dir) / "candidate.ply"
        mesh_set.save_current_mesh(
            str(temporary_mesh), save_vertex_color=False, save_wedge_color=False
        )
        candidate = trimesh.load(temporary_mesh, force="mesh", process=False)
    if not isinstance(candidate, trimesh.Trimesh):
        raise RuntimeError("surface 平滑候选无法重新加载为 Trimesh。")

    after = metrics(candidate)
    accepted, failures = safe_gate(
        before, after, allow_oriented_volume_change=True
    )
    before_extents = np.asarray(before["extents"], dtype=float)
    displacement = 0.0
    if len(candidate.vertices) == len(mesh.vertices):
        displacement = float(
            np.linalg.norm(
                np.asarray(candidate.vertices) - np.asarray(mesh.vertices), axis=1
            ).max(initial=0.0)
        )
    max_extent = max(float(before_extents.max(initial=0.0)), AREA_EPSILON)
    area_change = abs(after["area"] - before["area"]) / max(
        abs(before["area"]), AREA_EPSILON
    )
    volume_change = abs(abs(after["volume"]) - abs(before["volume"])) / max(
        abs(before["volume"]), AREA_EPSILON
    )
    if displacement > max_extent * 0.01:
        accepted = False
        failures.append("单次顶点位移超过包围盒最大尺寸的 1%")
    if area_change > 0.02:
        accepted = False
        failures.append("表面积变化超过 2% 门禁")
    if volume_change > 0.01:
        accepted = False
        failures.append("体积变化超过 1% 门禁")
    after_quality = surface_quality_metrics(candidate)
    before_quality = before["surface_quality"]
    improved = (
        after_quality["triangle_aspect_ratio"]["p99"]
        <= before_quality["triangle_aspect_ratio"]["p99"]
        and after_quality["adjacent_normal_angle_degrees"]["p99"]
        <= before_quality["adjacent_normal_angle_degrees"]["p99"]
    )
    if not improved:
        accepted = False
        failures.append("曲面质量 P99 指标没有同时改善")
    changed = displacement > 0.0
    if not changed:
        accepted = False
        failures.append("平滑没有改变顶点")
    record.update(
        {
            "accepted": accepted,
            "changed": changed,
            "displacement_max": displacement,
            "area_relative_change": area_change,
            "volume_relative_change": volume_change,
            "surface_quality_before": before_quality,
            "surface_quality_after": after_quality,
            "rejection_reasons": failures,
            "after_trial": after,
        }
    )
    return (candidate if accepted else mesh), record


def surface_remesh_candidate(
    mesh: trimesh.Trimesh,
    input_path: Path,
    *,
    target_length_ratio: float = 1.0,
    feature_angle_degrees: float = SURFACE_REMESH_FEATURE_ANGLE_DEGREES,
    iterations: int = 1,
    max_surface_distance_ratio: float = 0.0005,
    distance_samples: int = 20000,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Generate a feature-aware isotropic remesh candidate with hard gates.

    This is intentionally a whole-surface candidate. Trials that remeshed only
    isolated stretched triangles did not improve the current gear and could
    create irregular patch boundaries. The global candidate preserves edges
    above ``feature_angle_degrees`` and reprojects vertices to the source.
    """
    if not 0.5 <= target_length_ratio <= 2.0:
        raise ValueError("surface remesh 目标边长比例必须在 0.5 到 2.0 之间。")
    if not 20.0 <= feature_angle_degrees <= 80.0:
        raise ValueError("surface remesh 特征角必须在 20 到 80 度之间。")
    if iterations < 1 or iterations > 3:
        raise ValueError("surface remesh 迭代次数必须在 1 到 3 之间。")
    if not 0.0001 <= max_surface_distance_ratio <= 0.002:
        raise ValueError("surface remesh 最大回投距离比例必须在 0.0001 到 0.002 之间。")
    if getattr(mesh.visual, "kind", None) == "texture" or hasattr(mesh.visual, "uv"):
        return mesh, {
            "name": "feature_aware_isotropic_surface_remesh_candidate",
            "accepted": False,
            "changed": False,
            "rejection_reasons": [
                "输入包含纹理或 UV；当前 PyMeshLab 重网格链路不能保证材质映射保持。"
            ],
            "before": metrics(mesh),
            "after_trial": metrics(mesh),
        }
    try:
        import pymeshlab
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "surface remesh 候选需要 Hunyuan 环境中的 pymeshlab。"
        ) from exc

    before = metrics(mesh)
    before_quality = before["surface_quality"]
    before_smooth_normals = smooth_region_normal_metrics(mesh)
    edge_lengths = np.asarray(mesh.edges_unique_length, dtype=float)
    finite_edges = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0)]
    if not len(finite_edges):
        return mesh, {
            "name": "feature_aware_isotropic_surface_remesh_candidate",
            "accepted": False,
            "changed": False,
            "rejection_reasons": ["输入没有可用于确定目标长度的有效边。"],
            "before": before,
            "after_trial": before,
        }
    median_edge = float(np.median(finite_edges))
    target_length = median_edge * target_length_ratio
    diagonal = max(float(np.linalg.norm(mesh.extents)), AREA_EPSILON)
    max_surface_distance = diagonal * max_surface_distance_ratio

    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(input_path))
    mesh_set.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        iterations=int(iterations),
        adaptive=False,
        selectedonly=False,
        targetlen=pymeshlab.AbsoluteValue(target_length),
        featuredeg=float(feature_angle_degrees),
        checksurfdist=True,
        maxsurfdist=pymeshlab.AbsoluteValue(max_surface_distance),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True,
    )
    with tempfile.TemporaryDirectory(prefix="industrial_ai_surface_remesh_") as temp_dir:
        temporary_mesh = Path(temp_dir) / "candidate.ply"
        mesh_set.save_current_mesh(
            str(temporary_mesh), save_vertex_color=False, save_wedge_color=False
        )
        candidate = trimesh.load(temporary_mesh, force="mesh", process=False)
    if not isinstance(candidate, trimesh.Trimesh):
        raise RuntimeError("surface remesh 候选无法重新加载为 Trimesh。")

    after = metrics(candidate)
    after_quality = after["surface_quality"]
    after_smooth_normals = smooth_region_normal_metrics(candidate)
    failures: list[str] = []
    if before["watertight"] and not after["watertight"]:
        failures.append("重网格破坏了水密性")
    if after["components"] != before["components"]:
        failures.append("连通部件数量发生变化")
    if after["boundary_edges"] > before["boundary_edges"]:
        failures.append("边界边数量增加")
    if after["nonmanifold_edges"] > before["nonmanifold_edges"]:
        failures.append("非流形边数量增加")
    if before["winding_consistent"] and not after["winding_consistent"]:
        failures.append("法向一致性退化")
    if after["degenerate_faces"] > before["degenerate_faces"]:
        failures.append("退化面数量增加")

    before_extents = np.asarray(before["extents"], dtype=float)
    after_extents = np.asarray(after["extents"], dtype=float)
    extent_change = np.divide(
        np.abs(after_extents - before_extents),
        np.maximum(np.abs(before_extents), AREA_EPSILON),
    )
    max_extent_change = float(extent_change.max(initial=0.0))
    area_change = abs(after["area"] - before["area"]) / max(
        abs(before["area"]), AREA_EPSILON
    )
    volume_change = abs(abs(after["volume"]) - abs(before["volume"])) / max(
        abs(before["volume"]), AREA_EPSILON
    )
    if max_extent_change > SURFACE_REMESH_MAX_EXTENT_RELATIVE_CHANGE:
        failures.append("包围盒尺寸变化超过 0.1% 门禁")
    if area_change > SURFACE_REMESH_MAX_AREA_RELATIVE_CHANGE:
        failures.append("表面积变化超过 0.5% 门禁")
    if before["watertight"] and volume_change > SURFACE_REMESH_MAX_VOLUME_RELATIVE_CHANGE:
        failures.append("体积变化超过 0.1% 门禁")

    distance_metrics = bidirectional_surface_distance_metrics(
        mesh, candidate, samples=distance_samples
    )
    if (
        distance_metrics["p99_to_diagonal_ratio"]
        > SURFACE_REMESH_MAX_P99_DISTANCE_RATIO
    ):
        failures.append("双向表面距离 P99 超过包围盒对角线的 0.1%")
    if distance_metrics["max_to_diagonal_ratio"] > SURFACE_REMESH_MAX_DISTANCE_RATIO:
        failures.append("双向表面最大采样距离超过包围盒对角线的 0.2%")

    aspect_before = before_quality["triangle_aspect_ratio"]["p99"]
    aspect_after = after_quality["triangle_aspect_ratio"]["p99"]
    if aspect_after >= aspect_before * 0.8:
        failures.append("三角形长宽比 P99 未改善至少 20%")
    smooth_p99_before = before_smooth_normals["below_sharp_threshold_degrees"]["p99"]
    smooth_p99_after = after_smooth_normals["below_sharp_threshold_degrees"]["p99"]
    if smooth_p99_after > smooth_p99_before * 1.05:
        failures.append("非锐边区域法向夹角 P99 恶化超过 5%")
    sharp_before = before_smooth_normals["sharp_adjacency_count"]
    sharp_after = after_smooth_normals["sharp_adjacency_count"]
    if sharp_after > max(int(sharp_before * 1.05), sharp_before + 8):
        failures.append("锐边邻接数量增加超过 5%")

    changed = (
        before["vertices"] != after["vertices"]
        or before["faces"] != after["faces"]
    )
    if not changed:
        failures.append("重网格没有改变顶点或面数量")
    accepted = not failures
    record = {
        "name": "feature_aware_isotropic_surface_remesh_candidate",
        "accepted": accepted,
        "changed": changed,
        "parameters": {
            "target_length_ratio": float(target_length_ratio),
            "source_median_edge_length": median_edge,
            "target_edge_length": target_length,
            "feature_angle_degrees": float(feature_angle_degrees),
            "iterations": int(iterations),
            "max_surface_distance_ratio": float(max_surface_distance_ratio),
            "max_surface_distance": max_surface_distance,
            "distance_samples": int(distance_samples),
        },
        "extent_relative_change_per_axis": extent_change.tolist(),
        "max_extent_relative_change": max_extent_change,
        "area_relative_change": area_change,
        "volume_relative_change": volume_change,
        "surface_distance": distance_metrics,
        "surface_quality_before": before_quality,
        "surface_quality_after": after_quality,
        "smooth_region_normals_before": before_smooth_normals,
        "smooth_region_normals_after": after_smooth_normals,
        "rejection_reasons": failures,
        "before": before,
        "after_trial": after,
    }
    return (candidate if accepted else mesh), record


def voxel_reconstruction_candidate(
    mesh: trimesh.Trimesh,
    *,
    resolution: int = 192,
    closing_iterations: int = 2,
    distance_samples: int = 10000,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Reconstruct a severely open mesh as a guarded watertight candidate.

    Only a strongly dominant surface component may be reconstructed. Small
    fragments are deliberately excluded, then the dominant surface is
    voxelized, morphologically closed and converted back with marching cubes.
    This changes topology and can close semantic holes, so even a passing
    result always requires visual review.
    """
    if resolution < 96 or resolution > 384:
        raise ValueError("重建分辨率必须在 96 到 384 之间。")
    if closing_iterations < 0 or closing_iterations > 4:
        raise ValueError("体素闭运算迭代次数必须在 0 到 4 之间。")
    if distance_samples < 1000:
        raise ValueError("重建表面距离采样数不得少于 1000。")
    if getattr(mesh.visual, "kind", None) == "texture" or hasattr(mesh.visual, "uv"):
        return mesh, {
            "name": "dominant_component_voxel_reconstruction_candidate",
            "accepted": False,
            "changed": False,
            "rejection_reasons": ["输入包含纹理或 UV；体素重建不能保持材质映射。"],
            "before": metrics(mesh),
            "after_trial": metrics(mesh),
        }
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise RuntimeError("严重模型体素重建需要 Hunyuan 环境中的 scipy。") from exc

    before = metrics(mesh)
    parts = sorted(
        mesh.split(only_watertight=False), key=lambda item: item.area, reverse=True
    )
    if not parts or mesh.area <= AREA_EPSILON:
        return mesh, {
            "name": "dominant_component_voxel_reconstruction_candidate",
            "accepted": False,
            "changed": False,
            "rejection_reasons": ["输入没有可重建的有效表面。"],
            "before": before,
            "after_trial": before,
        }
    main = parts[0]
    main_area_share = float(main.area / mesh.area)
    if main_area_share < RECONSTRUCTION_MIN_MAIN_AREA_SHARE:
        return mesh, {
            "name": "dominant_component_voxel_reconstruction_candidate",
            "accepted": False,
            "changed": False,
            "main_component_area_share": main_area_share,
            "rejection_reasons": [
                "最大部件表面积占比低于 95%，无法安全判断其余部件都是碎片。"
            ],
            "before": before,
            "after_trial": before,
        }

    diagonal = float(np.linalg.norm(main.extents))
    if diagonal <= AREA_EPSILON:
        raise ValueError("主体包围盒退化，无法确定体素间距。")
    pitch = diagonal / float(resolution)
    voxel = main.voxelized(pitch)
    occupancy = np.asarray(voxel.matrix, dtype=bool)
    padding = max(closing_iterations + 2, 2)
    padded = np.pad(occupancy, padding, mode="constant", constant_values=False)
    if closing_iterations:
        padded = ndimage.binary_closing(
            padded,
            structure=ndimage.generate_binary_structure(3, 2),
            iterations=closing_iterations,
        )
    filled = ndimage.binary_fill_holes(padded)
    transform = np.asarray(voxel.transform).copy()
    transform = transform @ trimesh.transformations.translation_matrix(
        [-padding, -padding, -padding]
    )
    reconstructed_grid = trimesh.voxel.VoxelGrid(filled, transform=transform)
    candidate = reconstructed_grid.marching_cubes
    candidate.apply_transform(transform)
    candidate.remove_unreferenced_vertices()
    candidate.remove_infinite_values()
    trimesh.repair.fix_normals(candidate, multibody=True)

    candidate_parts = sorted(
        candidate.split(only_watertight=False), key=lambda item: item.area, reverse=True
    )
    output_discarded_parts = 0
    output_largest_share = 1.0
    if len(candidate_parts) > 1:
        output_largest_share = float(candidate_parts[0].area / candidate.area)
        if output_largest_share >= 0.99:
            output_discarded_parts = len(candidate_parts) - 1
            candidate = candidate_parts[0].copy()
            trimesh.repair.fix_normals(candidate, multibody=True)

    after = metrics(candidate)
    failures: list[str] = []
    if after["components"] != 1:
        failures.append("重建候选仍包含多个有意义部件")
    if not after["watertight"]:
        failures.append("重建候选不水密")
    if not after["winding_consistent"]:
        failures.append("重建候选法向不一致")
    if after["boundary_edges"] != 0:
        failures.append("重建候选仍有边界边")
    if after["nonmanifold_edges"] != 0:
        failures.append("重建候选仍有非流形边")
    if after["degenerate_faces"] != 0:
        failures.append("重建候选仍有退化面")
    if after["volume"] <= 0:
        failures.append("重建候选没有正体积")

    source_extents = np.asarray(main.extents, dtype=float)
    candidate_extents = np.asarray(candidate.extents, dtype=float)
    extent_change = np.divide(
        np.abs(candidate_extents - source_extents),
        np.maximum(np.abs(source_extents), AREA_EPSILON),
    )
    max_extent_change = float(extent_change.max(initial=0.0))
    if max_extent_change > RECONSTRUCTION_MAX_EXTENT_RELATIVE_CHANGE:
        failures.append("主体包围盒尺寸变化超过 5%")
    center_shift = float(np.linalg.norm(candidate.bounding_box.centroid - main.bounding_box.centroid))
    center_shift_ratio = center_shift / diagonal
    if center_shift_ratio > RECONSTRUCTION_MAX_CENTER_SHIFT_RATIO:
        failures.append("主体中心偏移超过包围盒对角线的 2%")

    distance = bidirectional_surface_distance_metrics(
        main, candidate, samples=distance_samples
    )
    if distance["p99_to_diagonal_ratio"] > RECONSTRUCTION_MAX_P99_DISTANCE_RATIO:
        failures.append("双向表面距离 P99 超过包围盒对角线的 2%")
    if distance["max_to_diagonal_ratio"] > RECONSTRUCTION_MAX_DISTANCE_RATIO:
        failures.append("双向最大采样距离超过包围盒对角线的 4%")
    bbox_volume = float(np.prod(np.maximum(candidate.extents, AREA_EPSILON)))
    solid_fraction = float(abs(candidate.volume) / bbox_volume)
    if not 0.005 <= solid_fraction <= 0.98:
        failures.append("重建候选实体体积分数异常")

    changed = bool(
        before["vertices"] != after["vertices"]
        or before["faces"] != after["faces"]
        or before["components"] != after["components"]
    )
    if not changed:
        failures.append("体素重建没有改变网格")
    accepted = not failures
    record = {
        "name": "dominant_component_voxel_reconstruction_candidate",
        "accepted": accepted,
        "changed": changed,
        "parameters": {
            "resolution_across_bbox_diagonal": int(resolution),
            "voxel_pitch": pitch,
            "closing_iterations": int(closing_iterations),
            "distance_samples": int(distance_samples),
        },
        "input_component_count": len(parts),
        "main_component_area_share": main_area_share,
        "discarded_input_component_count": max(len(parts) - 1, 0),
        "output_largest_component_area_share_before_cleanup": output_largest_share,
        "discarded_output_component_count": output_discarded_parts,
        "voxel_grid_shape": list(map(int, occupancy.shape)),
        "occupied_surface_voxels": int(occupancy.sum()),
        "occupied_reconstructed_voxels": int(filled.sum()),
        "extent_relative_change_per_axis": extent_change.tolist(),
        "max_extent_relative_change": max_extent_change,
        "center_shift": center_shift,
        "center_shift_to_diagonal_ratio": center_shift_ratio,
        "solid_fraction_of_candidate_bbox": solid_fraction,
        "surface_distance": distance,
        "rejection_reasons": failures,
        "semantic_warning": (
            "体素重建会改变拓扑，并可能封闭真实孔洞或开口；必须人工检查孔洞、"
            "薄壁、配合面和整体轮廓。"
        ),
        "before": before,
        "main_component_before": metrics(main),
        "after_trial": after,
    }
    return (candidate if accepted else mesh), record


def candidate_faces(mesh: trimesh.Trimesh) -> tuple[list[int], dict[str, Any]]:
    areas = np.asarray(mesh.area_faces, dtype=float)
    positive = areas[areas >= AREA_EPSILON]
    median_area = float(np.median(positive)) if len(positive) else 0.0
    aspect = triangle_aspect_ratios(mesh)
    degenerate = np.flatnonzero(areas < AREA_EPSILON)
    large_threshold = max(median_area * 100.0, float(mesh.area) * 0.005)
    large = np.flatnonzero(areas >= large_threshold) if large_threshold else np.array([], dtype=int)
    stretched = np.flatnonzero(aspect >= 50.0)
    # 细长三角形在齿轮齿面、孔边和高曲率区域很常见，单独作为候选会
    # 大量误伤合法几何。因此它只进入统计报告，不自动染色或删除。
    candidates = sorted({int(index) for index in np.concatenate((degenerate, large))})
    return candidates, {
        "median_positive_face_area": median_area,
        "large_face_area_threshold": large_threshold,
        "large_face_count": int(len(large)),
        "stretched_face_count": int(len(stretched)),
        "candidate_face_count": len(candidates),
        "candidate_face_indices": candidates,
        "interpretation": (
            "候选面只是几何异常线索；大平面可能是合法的齿轮端面，"
            "必须人工确认后才能删除。"
        ),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def annotate(mesh: trimesh.Trimesh, candidates: list[int]) -> trimesh.Trimesh:
    result = mesh.copy()
    colors = np.tile(np.asarray(NORMAL_COLOR, dtype=np.uint8), (len(result.faces), 1))
    areas = np.asarray(result.area_faces)
    colors[areas < AREA_EPSILON] = DEGENERATE_COLOR
    if candidates:
        colors[np.asarray(candidates, dtype=int)] = SUSPECT_COLOR
    result.visual.face_colors = colors
    return result


def load_face_indices(path: Path, face_count: int) -> list[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    indices = value.get("face_indices") if isinstance(value, dict) else None
    if not isinstance(indices, list) or not all(isinstance(item, int) for item in indices):
        raise ValueError("人工确认文件必须是 {\"face_indices\": [整数索引]}。")
    unique = sorted(set(indices))
    if any(index < 0 or index >= face_count for index in unique):
        raise ValueError("人工确认文件包含超出 GLB 面数量范围的索引。")
    return unique


def repair_explicit_faces(mesh: trimesh.Trimesh, indices: list[int]) -> trimesh.Trimesh:
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[np.asarray(indices, dtype=int)] = False
    result = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.faces)[keep].copy(),
        process=False,
    )
    result.remove_unreferenced_vertices()
    result.remove_infinite_values()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断并安全修复任意 GLB 网格")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--annotated-glb", required=True, type=Path)
    parser.add_argument(
        "--remove-face-indices",
        type=Path,
        help="人工确认的 JSON 文件，例如 {\"face_indices\": [12, 13]}；不提供则只诊断。",
    )
    parser.add_argument(
        "--safe-repair",
        action="store_true",
        help="尝试低风险自动修复；任一步破坏核心指标都会自动回滚。",
    )
    parser.add_argument(
        "--standard-repair",
        action="store_true",
        help="先执行 safe，再尝试填补严格受限的小型三角形/四边形边界孔。",
    )
    parser.add_argument(
        "--advanced-repair",
        action="store_true",
        help="使用 PyMeshLab 生成非流形/小孔高级候选；自交只检测不自动删除。",
    )
    parser.add_argument(
        "--artifact-cleanup-candidate",
        action="store_true",
        help="单零件模型：仅清除相对主体极小的独立有体积伪影，并保留人工复核门禁。",
    )
    parser.add_argument(
        "--surface-smooth-candidate",
        action="store_true",
        help="实验性局部曲面候选：只平滑细长三角形周围且避开锐边顶点。",
    )
    parser.add_argument(
        "--surface-remesh-candidate",
        action="store_true",
        help="实验性保特征等距重网格候选；通过距离、尺寸、拓扑和曲面质量门禁才输出。",
    )
    parser.add_argument(
        "--reconstruction-candidate",
        action="store_true",
        help="严重开放/碎裂模型：保留主部件并生成受门禁保护的水密体素重建候选。",
    )
    parser.add_argument(
        "--surface-aspect-threshold",
        type=float,
        default=50.0,
        help="surface 候选使用的三角形长宽比阈值，默认 50。",
    )
    parser.add_argument(
        "--surface-smooth-iterations",
        type=int,
        default=1,
        help="surface 候选的平滑迭代次数，范围 1-3，默认 1。",
    )
    parser.add_argument(
        "--surface-remesh-target-ratio",
        type=float,
        default=1.0,
        help="surface remesh 目标边长相对输入中位边长的比例，默认 1.0。",
    )
    parser.add_argument(
        "--surface-remesh-feature-angle",
        type=float,
        default=SURFACE_REMESH_FEATURE_ANGLE_DEGREES,
        help="surface remesh 保护特征的角度阈值，默认 45 度。",
    )
    parser.add_argument(
        "--surface-remesh-iterations",
        type=int,
        default=1,
        help="surface remesh 迭代次数，范围 1-3，默认 1。",
    )
    parser.add_argument(
        "--surface-remesh-max-distance-ratio",
        type=float,
        default=0.0005,
        help="重网格回投最大距离占输入包围盒对角线比例，默认 0.0005。",
    )
    parser.add_argument(
        "--surface-remesh-distance-samples",
        type=int,
        default=20000,
        help="双向点到三角面距离的每方向采样数，默认 20000。",
    )
    parser.add_argument(
        "--max-hole-size",
        type=int,
        default=4,
        help="advanced 模式允许 PyMeshLab 尝试封闭的最大边界边数。",
    )
    parser.add_argument(
        "--reconstruction-resolution",
        type=int,
        default=192,
        help="体素重建沿包围盒对角线的分辨率，范围 96-384，默认 192。",
    )
    parser.add_argument(
        "--reconstruction-closing-iterations",
        type=int,
        default=2,
        help="体素闭运算迭代次数，范围 0-4，默认 2。",
    )
    parser.add_argument(
        "--reconstruction-distance-samples",
        type=int,
        default=10000,
        help="重建双向表面距离的每方向采样数，默认 10000。",
    )
    parser.add_argument("--repaired-glb", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file() or input_path.suffix.lower() != ".glb":
        raise ValueError("输入必须是存在的 GLB 文件。")
    selected_modes = sum(
        bool(value)
        for value in (
            args.remove_face_indices,
            args.safe_repair,
            args.standard_repair,
            args.advanced_repair,
            args.artifact_cleanup_candidate,
            args.surface_smooth_candidate,
            args.surface_remesh_candidate,
            args.reconstruction_candidate,
        )
    )
    if selected_modes > 1:
        raise ValueError("每次只能选择一种诊断、修复或重建模式。")
    if selected_modes and not args.repaired_glb:
        raise ValueError("执行修复时必须同时指定 --repaired-glb。")
    for path in (args.report, args.annotated_glb, args.repaired_glb):
        if path is not None and path.resolve().exists():
            raise FileExistsError(f"拒绝覆盖已有输出：{path}")

    mesh = trimesh.load(input_path, force="mesh", process=False)
    candidates, diagnosis = candidate_faces(mesh)
    annotated = annotate(mesh, candidates)
    args.annotated_glb.parent.mkdir(parents=True, exist_ok=True)
    annotated.export(args.annotated_glb, file_type="glb")

    before_metrics = metrics(mesh)
    report: dict[str, Any] = {
        "schema_version": "1.1",
        "operation": "generic_mesh_diagnosis_and_guarded_repair",
        "input_path": str(input_path),
        "input_sha256": sha256(input_path),
        "diagnosis": diagnosis,
        "before": before_metrics,
        "automatic_repairability_before": assess_repairability(before_metrics),
        "annotated_glb": str(args.annotated_glb.resolve()),
        "repair_applied": False,
        "status": "DIAGNOSIS_ONLY_NOT_PRINT_APPROVAL",
        "limitations": [
            "safe 模式不自动删除疑似异常大平面或有体积部件。",
            "不把水密性通过误认为几何形状正确。",
            "孔洞填补和主体内部伪影需要单独人工确认。",
        ],
    }

    if (
        args.safe_repair
        or args.standard_repair
        or args.advanced_repair
        or args.artifact_cleanup_candidate
        or args.surface_smooth_candidate
        or args.surface_remesh_candidate
        or args.reconstruction_candidate
    ):
        repair_mode = (
            "advanced"
            if args.advanced_repair
            else "artifact_cleanup_candidate"
            if args.artifact_cleanup_candidate
            else "reconstruction_candidate"
            if args.reconstruction_candidate
            else "surface_remesh_candidate"
            if args.surface_remesh_candidate
            else "surface_smooth_candidate"
            if args.surface_smooth_candidate
            else "standard"
            if args.standard_repair
            else "safe"
        )
        if args.advanced_repair:
            repaired, advanced_operation = advanced_repair_candidate(
                mesh,
                input_path,
                max_hole_size=args.max_hole_size,
            )
            operations = [advanced_operation]
        elif args.artifact_cleanup_candidate:
            repaired, artifact_operation = detached_artifact_cleanup_candidate(mesh)
            operations = [artifact_operation]
        elif args.reconstruction_candidate:
            repaired, reconstruction_operation = voxel_reconstruction_candidate(
                mesh,
                resolution=args.reconstruction_resolution,
                closing_iterations=args.reconstruction_closing_iterations,
                distance_samples=args.reconstruction_distance_samples,
            )
            operations = [reconstruction_operation]
        elif args.surface_remesh_candidate:
            repaired, remesh_operation = surface_remesh_candidate(
                mesh,
                input_path,
                target_length_ratio=args.surface_remesh_target_ratio,
                feature_angle_degrees=args.surface_remesh_feature_angle,
                iterations=args.surface_remesh_iterations,
                max_surface_distance_ratio=args.surface_remesh_max_distance_ratio,
                distance_samples=args.surface_remesh_distance_samples,
            )
            operations = [remesh_operation]
        elif args.surface_smooth_candidate:
            repaired, surface_operation = surface_smooth_candidate(
                mesh,
                input_path,
                aspect_threshold=args.surface_aspect_threshold,
                iterations=args.surface_smooth_iterations,
            )
            operations = [surface_operation]
        elif args.standard_repair:
            repaired, operations = standard_repair(mesh)
        else:
            repaired, operations = safe_repair(mesh)
        after = metrics(repaired)
        accepted_operations = [item["name"] for item in operations if item["accepted"]]
        report.update(
            {
                "repair_mode": repair_mode,
                "repair_operations": operations,
                "accepted_operations": accepted_operations,
                "after_in_memory": after,
                "automatic_repairability_after_in_memory": assess_repairability(after),
                "unresolved": {
                    "degenerate_faces": after["degenerate_faces"],
                    "duplicate_faces": after["duplicate_faces"],
                    "duplicate_vertices_exact": after["duplicate_vertices_exact"],
                    "watertight": after["watertight"],
                    "boundary_edges": after["boundary_edges"],
                    "nonmanifold_edges": after["nonmanifold_edges"],
                },
            }
        )
        if accepted_operations:
            args.repaired_glb.parent.mkdir(parents=True, exist_ok=True)
            repaired.export(args.repaired_glb, file_type="glb")
            exported = trimesh.load(args.repaired_glb, force="mesh", process=False)
            verified = metrics(exported)
            if repair_mode in {
                "artifact_cleanup_candidate",
                "surface_remesh_candidate",
                "reconstruction_candidate",
            }:
                passed, failures = export_reload_gate(after, verified)
            else:
                passed, failures = safe_gate(
                    metrics(mesh),
                    verified,
                    allow_oriented_volume_change=(
                        not metrics(mesh)["winding_consistent"]
                        and verified["winding_consistent"]
                    ),
                )
            if not passed:
                args.repaired_glb.unlink(missing_ok=True)
                raise RuntimeError(
                    "safe 修复导出重载后门禁失败，已撤销输出：" + "；".join(failures)
                )
            report.update(
                {
                    "repair_applied": True,
                    "repaired_glb": str(args.repaired_glb.resolve()),
                    "repaired_sha256": sha256(args.repaired_glb.resolve()),
                    "after_export_reload": verified,
                    "automatic_repairability_after_export": assess_repairability(verified),
                    "status": (
                        "RECONSTRUCTION_CANDIDATE_REQUIRES_VISUAL_REVIEW"
                        if repair_mode == "reconstruction_candidate"
                        else "REPAIRED_CANDIDATE_REQUIRES_VISUAL_REVIEW"
                        if (
                            verified["watertight"]
                            and verified["winding_consistent"]
                            and verified["boundary_edges"] == 0
                            and verified["nonmanifold_edges"] == 0
                            and verified["degenerate_faces"] == 0
                        )
                        else "PARTIAL_REPAIR_NOT_PRINT_APPROVAL"
                    ),
                }
            )
        else:
            report["repair_result"] = "NO_GUARDED_CHANGE_AVAILABLE"
            report["status"] = "NO_SAFE_REPAIR_AVAILABLE"

    elif args.remove_face_indices:
        indices = load_face_indices(args.remove_face_indices.resolve(), len(mesh.faces))
        repaired = repair_explicit_faces(mesh, indices)
        after = metrics(repaired)
        if not (
            after["watertight"]
            and after["winding_consistent"]
            and after["components"] == 1
            and after["boundary_edges"] == 0
            and after["nonmanifold_edges"] == 0
            and after["volume"] > 0
        ):
            raise RuntimeError("删除指定面后未通过水密/法向/连通性门禁，拒绝输出修复 GLB。")
        args.repaired_glb.parent.mkdir(parents=True, exist_ok=True)
        repaired.export(args.repaired_glb, file_type="glb")
        report.update(
            {
                "repair_applied": True,
                "repair_mode": "manual_face_removal",
                "removed_face_indices": indices,
                "repaired_glb": str(args.repaired_glb.resolve()),
                "after": after,
                "status": "REPAIRED_CANDIDATE_REQUIRES_VISUAL_REVIEW",
            }
        )

    write_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
