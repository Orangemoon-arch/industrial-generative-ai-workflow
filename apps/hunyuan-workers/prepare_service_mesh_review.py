"""Create a versioned review GLB by dropping detached zero-area debris only."""

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
AREA_EPSILON = 1e-12


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
    parser = argparse.ArgumentParser(description="生成任务网格的无损 review GLB")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    return parser.parse_args()


def edge_counts(mesh: trimesh.Trimesh) -> tuple[int, int]:
    counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    return int((counts == 1).sum()), int((counts > 2).sum())


def mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    parts = mesh.split(only_watertight=False)
    boundary_edges, nonmanifold_edges = edge_counts(mesh)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(parts)),
        "area": float(mesh.area),
        "volume": float(mesh.volume),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "bounds": np.asarray(mesh.bounds).tolist(),
        "extents": np.asarray(mesh.extents).tolist(),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "degenerate_faces": int((np.asarray(mesh.area_faces) < AREA_EPSILON).sum()),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file() or not inside(input_path, RUNS_ROOT):
        raise ValueError("输入 GLB 必须位于 workspace/runs 内。")
    run_dir = next((path for path in input_path.parents if path.parent == RUNS_ROOT), None)
    if run_dir is None:
        raise ValueError("无法识别 GLB 所属任务目录。")

    output_path = args.output.resolve()
    report_path = args.report_output.resolve()
    for output in (output_path, report_path):
        if not inside(output, run_dir):
            raise ValueError("整理输出必须位于当前任务目录。")
        if output.exists():
            raise FileExistsError(f"拒绝覆盖已有输出：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    raw_mesh = trimesh.load(input_path, force="mesh", process=False)
    before = mesh_metrics(raw_mesh)
    parts = sorted(
        raw_mesh.split(only_watertight=False),
        key=lambda part: part.area,
        reverse=True,
    )
    if len(parts) < 2:
        raise ValueError("输入网格只有一个连通部件，无需生成整理副本。")
    detached_area = float(sum(part.area for part in parts[1:]))
    detached_volume = float(sum(abs(part.volume) for part in parts[1:]))
    if detached_area >= AREA_EPSILON or detached_volume >= AREA_EPSILON:
        raise RuntimeError("存在有实际面积或体积的次要部件，拒绝自动筛除。")

    review_mesh = parts[0].copy()
    review_mesh.remove_unreferenced_vertices()
    review_mesh.remove_infinite_values()
    after = mesh_metrics(review_mesh)

    if not after["watertight"]:
        raise RuntimeError("整理后主体不是水密网格，拒绝导出。")
    if not after["winding_consistent"]:
        raise RuntimeError("整理后法向不一致，拒绝导出。")
    if after["components"] != 1:
        raise RuntimeError("整理后仍有多个连通部件，拒绝导出。")
    if after["volume"] <= 0:
        raise RuntimeError("整理后有效体积不是正数，拒绝导出。")

    review_mesh.export(output_path, file_type="glb")
    exported = trimesh.load(output_path, force="mesh", process=False)
    verified = mesh_metrics(exported)
    if not verified["watertight"] or not verified["winding_consistent"]:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("重新加载导出 GLB 后验证失败，已撤销不完整输出。")

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operation": "keep_largest_watertight_component_drop_detached_zero_area_debris",
        "area_epsilon": AREA_EPSILON,
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "input_sha256": sha256(input_path),
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256(output_path),
        "removed_faces": before["faces"] - after["faces"],
        "removed_vertices": before["vertices"] - after["vertices"],
        "removed_components": before["components"] - after["components"],
        "detached_area_removed": detached_area,
        "detached_absolute_volume_removed": detached_volume,
        "note": "主体内近零面积面保留；强制删除会破坏水密性。",
        "before": before,
        "after_in_memory": after,
        "after_export_reload": verified,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
