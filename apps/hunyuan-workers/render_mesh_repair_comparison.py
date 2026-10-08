"""Render an honest before/after mesh comparison with shared camera framing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from inspect_service_mesh import PROJECT_ROOT, RUNS_ROOT, inside, rotation_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修复前后同视角、同缩放对照图")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def render(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    frame_minimum: np.ndarray,
    frame_maximum: np.ndarray,
    label: str,
    *,
    size: int = 720,
) -> Image.Image:
    vertices = np.asarray(mesh.vertices) @ rotation.T
    normals = np.asarray(mesh.face_normals) @ rotation.T
    faces = np.asarray(mesh.faces)
    valid = np.asarray(mesh.area_faces) >= 1e-12
    faces = faces[valid]
    normals = normals[valid]

    span = np.maximum(frame_maximum - frame_minimum, 1e-9)
    scale = (size - 90) / float(span.max())
    projected = (vertices[:, :2] - (frame_minimum + frame_maximum) / 2.0) * scale
    projected[:, 0] += size / 2.0
    projected[:, 1] = size / 2.0 - projected[:, 1]

    depth = vertices[faces, 2].mean(axis=1)
    order = np.argsort(depth)
    light = np.array([0.25, -0.35, 0.902])
    light /= np.linalg.norm(light)
    intensity = np.clip(0.35 + 0.65 * np.abs(normals @ light), 0, 1)
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
    draw.rectangle((0, 0, size, 42), fill=(238, 241, 244))
    draw.text((14, 14), label, fill=(25, 35, 45))
    return canvas


def frame_for(meshes: list[trimesh.Trimesh], rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack([np.asarray(mesh.vertices) @ rotation.T for mesh in meshes])[:, :2]
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    padding = max(float((maximum - minimum).max()) * 0.05, 1e-9)
    return minimum - padding, maximum + padding


def mark_removed_fragments(
    before_panel: Image.Image,
    after_panel: Image.Image,
    source: trimesh.Trimesh,
    rotation: np.ndarray,
    frame_minimum: np.ndarray,
    frame_maximum: np.ndarray,
) -> None:
    """Circle detached source components in the shared overall frame."""
    parts = sorted(
        source.split(only_watertight=False), key=lambda item: item.area, reverse=True
    )
    if len(parts) < 2:
        return
    size = before_panel.width
    span = np.maximum(frame_maximum - frame_minimum, 1e-9)
    scale = (size - 90) / float(span.max())
    center = (frame_minimum + frame_maximum) / 2.0
    before_draw = ImageDraw.Draw(before_panel)
    after_draw = ImageDraw.Draw(after_panel)
    for fragment in parts[1:]:
        point = np.asarray(fragment.centroid) @ rotation.T
        projected = (point[:2] - center) * scale
        x = float(projected[0] + size / 2.0)
        y = float(size / 2.0 - projected[1])
        radius = 18
        box = (x - radius, y - radius, x + radius, y + radius)
        before_draw.ellipse(box, outline=(220, 48, 48), width=4)
        before_draw.text((x + 24, y - 8), "detached artifact", fill=(180, 32, 32))
        after_draw.ellipse(box, outline=(40, 150, 90), width=3)
        after_draw.text((x + 24, y - 8), "removed", fill=(30, 120, 70))


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    candidate_path = args.candidate.resolve()
    output_path = args.output.resolve()
    for path in (source_path, candidate_path):
        if not path.is_file() or not inside(path, RUNS_ROOT):
            raise ValueError("源模型和候选模型必须位于 workspace/runs 内。")
    source_run = next((p for p in source_path.parents if p.parent == RUNS_ROOT), None)
    candidate_run = next((p for p in candidate_path.parents if p.parent == RUNS_ROOT), None)
    if source_run is None or source_run != candidate_run:
        raise ValueError("源模型和候选模型必须属于同一个 RUN。")
    if not inside(output_path, source_run):
        raise ValueError("对照图必须输出到当前 RUN。")
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = trimesh.load(source_path, force="mesh", process=False)
    candidate = trimesh.load(candidate_path, force="mesh", process=False)
    if not isinstance(source, trimesh.Trimesh) or not isinstance(candidate, trimesh.Trimesh):
        raise ValueError("源模型或候选模型无法加载为单一 Trimesh。")

    rotation = rotation_matrix(35, -25)
    overall_frame = frame_for([source, candidate], rotation)
    detail_frame = frame_for([candidate], rotation)
    panels = [
        render(source, rotation, *overall_frame, "BEFORE - overall shared frame"),
        render(candidate, rotation, *overall_frame, "AFTER - overall shared frame"),
        render(source, rotation, *detail_frame, "BEFORE - main body shared zoom"),
        render(candidate, rotation, *detail_frame, "AFTER - main body shared zoom"),
    ]
    mark_removed_fragments(
        panels[0], panels[1], source, rotation, overall_frame[0], overall_frame[1]
    )
    comparison = Image.new("RGB", (panels[0].width * 2, panels[0].height * 2))
    comparison.paste(panels[0], (0, 0))
    comparison.paste(panels[1], (panels[0].width, 0))
    comparison.paste(panels[2], (0, panels[0].height))
    comparison.paste(panels[3], (panels[0].width, panels[0].height))
    comparison.save(output_path)
    print(str(output_path.relative_to(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
