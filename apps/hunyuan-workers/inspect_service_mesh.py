"""Create a CPU-only diagnostic report and orthographic preview for a task GLB."""

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
from PIL import Image, ImageDraw


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
    parser = argparse.ArgumentParser(description="任务 GLB 的 CPU 网格检查")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--preview-output", required=True, type=Path)
    parser.add_argument("--include-back", action="store_true")
    return parser.parse_args()


def component_record(mesh: trimesh.Trimesh, rank: int, total_area: float) -> dict[str, Any]:
    counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    extents = np.asarray(mesh.extents)
    longest = float(extents.max()) if len(extents) else 0.0
    return {
        "rank": rank,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "area": float(mesh.area),
        "area_share": float(mesh.area / total_area) if total_area else 0.0,
        "volume": float(mesh.volume),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "bounds": np.asarray(mesh.bounds).tolist(),
        "extents": extents.tolist(),
        "thinness_ratio": float(extents.min() / longest) if longest else None,
    }


def rotation_matrix(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    pitch = np.deg2rad(pitch_degrees)
    ry = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return rx @ ry


def render_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    rotation: np.ndarray,
    label: str,
    size: int = 760,
) -> Image.Image:
    transformed = vertices @ rotation.T
    transformed_normals = face_normals @ rotation.T
    xy = transformed[:, :2]
    minimum = xy.min(axis=0)
    maximum = xy.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    scale = (size - 80) / span.max()
    projected = (xy - (minimum + maximum) / 2.0) * scale
    projected[:, 0] += size / 2.0
    projected[:, 1] = size / 2.0 - projected[:, 1]

    depth = transformed[faces, 2].mean(axis=1)
    order = np.argsort(depth)
    light = np.array([0.25, -0.35, 0.902])
    light /= np.linalg.norm(light)
    intensity = np.clip(0.35 + 0.65 * np.abs(transformed_normals @ light), 0, 1)

    canvas = Image.new("RGB", (size, size), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    for face_index in order:
        polygon = [tuple(point) for point in projected[faces[face_index]]]
        shade = float(intensity[face_index])
        color = (
            int(58 + 72 * shade),
            int(105 + 92 * shade),
            int(126 + 96 * shade),
        )
        draw.polygon(polygon, fill=color)
    draw.rectangle((0, 0, size, 34), fill=(238, 241, 244))
    draw.text((12, 10), label, fill=(25, 35, 45))
    return canvas


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file() or not inside(input_path, RUNS_ROOT):
        raise ValueError("输入 GLB 必须位于 workspace/runs 内。")
    run_dir = next((p for p in input_path.parents if p.parent == RUNS_ROOT), None)
    if run_dir is None:
        raise ValueError("无法识别 GLB 所属任务目录。")
    for output in (args.report_output, args.preview_output):
        resolved = output.resolve()
        if not inside(resolved, run_dir):
            raise ValueError("检查输出必须位于当前任务目录。")
        if resolved.exists():
            raise FileExistsError(f"拒绝覆盖已有输出：{resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(input_path, force="mesh", process=False)
    areas = np.asarray(mesh.area_faces)
    degenerate = areas < 1e-12
    counts = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
    )
    parts = sorted(mesh.split(only_watertight=False), key=lambda part: part.area, reverse=True)
    total_area = float(mesh.area)
    components = [
        component_record(part, rank, total_area)
        for rank, part in enumerate(parts, start=1)
    ]

    valid_faces = np.asarray(mesh.faces)[~degenerate]
    valid_normals = np.asarray(mesh.face_normals)[~degenerate]
    views = [
        render_view(
            np.asarray(mesh.vertices),
            valid_faces,
            valid_normals,
            np.eye(3),
            "FRONT · XY",
        ),
        render_view(
            np.asarray(mesh.vertices),
            valid_faces,
            valid_normals,
            rotation_matrix(90, 0),
            "SIDE · ZY",
        ),
        render_view(
            np.asarray(mesh.vertices),
            valid_faces,
            valid_normals,
            rotation_matrix(35, -25),
            "OBLIQUE",
        ),
    ]
    if args.include_back:
        views.append(
            render_view(
                np.asarray(mesh.vertices),
                valid_faces,
                valid_normals,
                rotation_matrix(145, 25),
                "BACK · OBLIQUE",
            )
        )
    preview = Image.new("RGB", (sum(view.width for view in views), views[0].height))
    offset = 0
    for view in views:
        preview.paste(view, (offset, 0))
        offset += view.width
    preview.save(args.preview_output)

    extents = np.asarray(mesh.extents)
    face_quantiles = {
        str(value): float(np.quantile(areas, value))
        for value in (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
    }
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "input_sha256": sha256(input_path),
        "preview_path": str(args.preview_output.resolve().relative_to(PROJECT_ROOT)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(parts)),
        "area": total_area,
        "volume": float(mesh.volume),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "bounds": np.asarray(mesh.bounds).tolist(),
        "extents": extents.tolist(),
        "thinness_ratio": float(extents.min() / extents.max()),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "degenerate_faces": int(degenerate.sum()),
        "face_area_quantiles": face_quantiles,
        "largest_triangle_area_share": float(areas.max() / total_area),
        "component_records": components,
        "main_component_area_share": components[0]["area_share"],
        "main_component_watertight": components[0]["watertight"],
        "main_component_winding_consistent": components[0]["winding_consistent"],
    }
    report["preview_sha256"] = sha256(args.preview_output)
    write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
