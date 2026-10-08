"""Export a reviewed task GLB as a bed-aligned, versioned binary STL."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()


def inside(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    return resolved == parent.resolve() or parent.resolve() in resolved.parents


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出已检查网格的版本化打印 STL")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--target-longest-mm", type=float, default=60.0)
    parser.add_argument("--allow-supported-overhangs", action="store_true")
    return parser.parse_args()


def metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(mesh.split(only_watertight=False))),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "degenerate_faces_below_1e-12": int(
            (np.asarray(mesh.area_faces) < 1e-12).sum()
        ),
        "volume_mm3": float(mesh.volume),
        "area_mm2": float(mesh.area),
        "bounds_mm": np.asarray(mesh.bounds).tolist(),
        "extents_mm": np.asarray(mesh.extents).tolist(),
    }


def validate_topology(metrics_value: dict[str, Any]) -> None:
    if metrics_value["components"] != 1:
        raise RuntimeError("网格不是单一连通部件。")
    if not metrics_value["watertight"]:
        raise RuntimeError("网格不是水密实体。")
    if not metrics_value["winding_consistent"]:
        raise RuntimeError("网格法向不一致。")
    if metrics_value["boundary_edges"] or metrics_value["nonmanifold_edges"]:
        raise RuntimeError("网格存在边界边或非流形边。")
    if metrics_value["volume_mm3"] <= 0:
        raise RuntimeError("网格没有正体积。")


def validate_print_metrics(
    metrics_value: dict[str, Any], target_longest_mm: float
) -> None:
    validate_topology(metrics_value)
    if metrics_value["degenerate_faces_below_1e-12"]:
        raise RuntimeError("缩放到毫米后网格仍存在退化三角面。")
    longest = max(metrics_value["extents_mm"])
    if abs(longest - target_longest_mm) > 0.02:
        raise RuntimeError("模型最长尺寸不符合目标毫米尺寸。")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file() or not inside(input_path, RUNS_ROOT):
        raise ValueError("输入 GLB 必须位于 workspace/runs 内。")
    run_dir = next((path for path in input_path.parents if path.parent == RUNS_ROOT), None)
    if run_dir is None:
        raise ValueError("无法识别输入所属任务。")
    if input_path.parent != run_dir / "meshes":
        raise ValueError("输入 GLB 必须位于当前任务 meshes 目录。")
    expected_sha = args.expected_sha256.lower()
    if len(expected_sha) != 64 or sha256(input_path) != expected_sha:
        raise ValueError("输入 GLB 的 SHA256 与批准记录不一致。")

    output_path = args.output.resolve()
    report_path = args.report_output.resolve()
    if output_path.parent != run_dir / "stl" or output_path.suffix.lower() != ".stl":
        raise ValueError("STL 输出必须位于当前任务 stl 目录。")
    for output in (output_path, report_path):
        if not inside(output, run_dir):
            raise ValueError("输出必须位于当前任务目录。")
        if output.exists():
            raise FileExistsError(f"拒绝覆盖已有输出：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    target_longest = float(args.target_longest_mm)
    if not 20.0 <= target_longest <= 200.0:
        raise ValueError("目标尺寸必须在 20–200 mm。")
    source = trimesh.load(input_path, force="mesh", process=True)
    source_coordinate_metrics = metrics(source)
    validate_topology(source_coordinate_metrics)
    source_longest = max(source_coordinate_metrics["extents_mm"])
    if not np.isfinite(source_longest) or source_longest <= 0.0:
        raise RuntimeError("源网格最长尺寸无效。")

    print_mesh = source.copy()
    scale_factor_to_mm = target_longest / source_longest
    print_mesh.apply_scale(scale_factor_to_mm)
    scaled_metrics = metrics(print_mesh)
    validate_print_metrics(scaled_metrics, target_longest)
    translation_z = -float(print_mesh.bounds[0, 2])
    print_mesh.apply_translation([0.0, 0.0, translation_z])
    aligned_metrics = metrics(print_mesh)
    validate_print_metrics(aligned_metrics, target_longest)
    if abs(aligned_metrics["bounds_mm"][0][2]) > 1e-6:
        raise RuntimeError("模型最低点未正确对齐到 Z=0。")

    normals = np.asarray(print_mesh.face_normals)
    centers = np.asarray(print_mesh.triangles_center)
    areas = np.asarray(print_mesh.area_faces)
    above_bed = centers[:, 2] > 0.30
    severe_downward = (normals[:, 2] < -np.cos(np.deg2rad(45.0))) & above_bed
    all_downward = (normals[:, 2] < -0.1) & above_bed
    bed_contact = (normals[:, 2] < -0.99) & (centers[:, 2] < 0.08)
    orientation = {
        "bed_plane_z_mm": 0.0,
        "translation_z_mm": translation_z,
        "bed_contact_area_mm2": float(areas[bed_contact].sum()),
        "severe_downward_overhang_area_above_bed_mm2": float(
            areas[severe_downward].sum()
        ),
        "all_downward_facing_area_above_bed_mm2": float(areas[all_downward].sum()),
        "severe_downward_face_count": int(severe_downward.sum()),
        "downward_face_count": int(all_downward.sum()),
    }
    support_required = (
        orientation["severe_downward_overhang_area_above_bed_mm2"] > 1e-6
    )
    if support_required and not args.allow_supported_overhangs:
        raise RuntimeError("检测到离床的严重向下悬空面，拒绝导出。")

    output_path.write_bytes(trimesh.exchange.stl.export_stl(print_mesh))
    reloaded = trimesh.load(output_path, force="mesh", process=True)
    reloaded_metrics = metrics(reloaded)
    try:
        validate_print_metrics(reloaded_metrics, target_longest)
        if abs(reloaded_metrics["bounds_mm"][0][2]) > 1e-5:
            raise RuntimeError("STL 重新加载后不再位于 Z=0。")
        relative_volume_error = abs(
            reloaded_metrics["volume_mm3"] - aligned_metrics["volume_mm3"]
        ) / aligned_metrics["volume_mm3"]
        if relative_volume_error > 1e-6:
            raise RuntimeError("STL 导出前后体积变化超出容差。")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operation": "reviewed_glb_to_binary_stl_align_min_z_to_bed",
        "units": "millimeter",
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "input_sha256": expected_sha,
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256(output_path),
        "output_bytes": output_path.stat().st_size,
        "target_longest_mm": target_longest,
        "source_coordinate_units": "unitless_or_source_defined",
        "source_coordinate_metrics": source_coordinate_metrics,
        "scale_factor_to_mm": scale_factor_to_mm,
        "scaled_metrics": scaled_metrics,
        "aligned_metrics": aligned_metrics,
        "reloaded_stl_metrics": reloaded_metrics,
        "relative_volume_error": relative_volume_error,
        "orientation_check": orientation,
        "support_required": support_required,
        "decision": (
            "STL_GEOMETRY_PASS_SUPPORT_REQUIRED_WAITING_BAMBU_SLICER"
            if support_required
            else "STL_GEOMETRY_PASS_WAITING_BAMBU_SLICER"
        ),
        "limitations": [
            "STL 格式不存储单位；任务档案明确约定坐标单位为毫米。",
            "尚未在 Bambu Studio 中检查逐层路径、首层和实际支撑。",
            "若 support_required=true，禁止关闭支撑后直接打印。",
            "尚未发送打印。",
        ],
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
