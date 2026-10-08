"""Prepare audited transparent multiview inputs for a Hunyuan3D-2mv smoke test."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )


def render_rgba(mesh: trimesh.Trimesh, yaw: float, size: int = 512) -> Image.Image:
    rotation = rotation_y(yaw)
    vertices = np.asarray(mesh.vertices) @ rotation.T
    normals = np.asarray(mesh.face_normals) @ rotation.T
    faces = np.asarray(mesh.faces)
    visible = normals[:, 2] > 1e-7
    faces = faces[visible]
    normals = normals[visible]
    if len(faces) == 0:
        raise ValueError("指定视角没有可见三角面。")

    xy = vertices[:, :2]
    minimum = xy.min(axis=0)
    maximum = xy.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    margin = 48
    scale = min((size - 2 * margin) / span[0], (size - 2 * margin) / span[1])
    projected = (xy - (minimum + maximum) / 2.0) * scale
    projected[:, 0] += size / 2.0
    projected[:, 1] = size / 2.0 - projected[:, 1]

    depth = vertices[faces, 2].mean(axis=1)
    order = np.argsort(depth)
    light = np.array([-0.30, -0.35, 0.88])
    light /= np.linalg.norm(light)
    intensity = np.clip(0.48 + 0.52 * np.abs(normals @ light), 0.0, 1.0)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    base = np.array([62, 119, 191], dtype=float)
    for face_index in order:
        polygon = [tuple(point) for point in projected[faces[face_index]]]
        rgb = np.clip(base * intensity[face_index], 0, 255).astype(np.uint8)
        draw.polygon(polygon, fill=(*[int(value) for value in rgb], 255))
    return canvas


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Project-local STL used only to render test views")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file() or PROJECT_ROOT not in source.parents:
        raise ValueError("输入必须是项目目录内已有的文件。")
    loaded = trimesh.load(source, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("输入没有可渲染三角网格。")

    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_ROOT / run_id
    for name in ("requests", "masks", "meshes", "reports", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    input_dir = run_dir / "masks/multiview_input_v1"
    input_dir.mkdir()

    views = {"front": 0.0, "left": 90.0, "back": 180.0}
    view_records = {}
    for view, yaw in views.items():
        output = input_dir / f"{view}.png"
        render_rgba(loaded, yaw).save(output)
        alpha = np.asarray(Image.open(output).getchannel("A"))
        view_records[view] = {
            "path": str(output.relative_to(PROJECT_ROOT)),
            "sha256": sha256(output),
            "yaw_degrees": yaw,
            "transparent_pixels": int(np.count_nonzero(alpha == 0)),
            "opaque_pixels": int(np.count_nonzero(alpha == 255)),
        }

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "purpose": "Hunyuan3D-2mv low-parameter backend smoke test",
        "print_status": "NOT_A_PRINT_APPROVAL",
        "source": {
            "path": str(source.relative_to(PROJECT_ROOT)),
            "sha256": sha256(source),
        },
        "render": {
            "method": "CPU orthographic projection",
            "size": [512, 512],
            "background": "transparent",
            "views": view_records,
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
