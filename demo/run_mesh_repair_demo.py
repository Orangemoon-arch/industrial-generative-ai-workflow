"""Reproducible CPU-only demonstration of guarded detached-artifact cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVICE_SOURCE = REPOSITORY_ROOT / "apps" / "industrial-ai-service"
sys.path.insert(0, str(SERVICE_SOURCE))

from mesh_repair_tool import detached_artifact_cleanup_candidate, metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic CPU-only guarded mesh-repair demo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "demo" / "output",
        help="Output directory; it must not contain existing files.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_damaged_mesh() -> trimesh.Trimesh:
    """Create one useful solid plus one tiny, distant watertight artifact."""
    main_body = trimesh.creation.box(extents=(2.0, 1.4, 0.7))
    artifact = trimesh.creation.icosphere(subdivisions=2, radius=0.018)
    artifact.apply_translation((0.0, 0.0, 1.8))
    return trimesh.util.concatenate((main_body, artifact))


def rotation_matrix(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    azimuth = np.radians(azimuth_degrees)
    elevation = np.radians(elevation_degrees)
    rotate_z = np.array(
        [
            [np.cos(azimuth), -np.sin(azimuth), 0.0],
            [np.sin(azimuth), np.cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotate_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elevation), -np.sin(elevation)],
            [0.0, np.sin(elevation), np.cos(elevation)],
        ]
    )
    return rotate_x @ rotate_z


def shared_frame(
    meshes: list[trimesh.Trimesh], rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack([np.asarray(mesh.vertices) @ rotation.T for mesh in meshes])[:, :2]
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    padding = max(float((maximum - minimum).max()) * 0.06, 1e-9)
    return minimum - padding, maximum + padding


def render_panel(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    title: str,
    *,
    size: int = 640,
) -> Image.Image:
    vertices = np.asarray(mesh.vertices) @ rotation.T
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals) @ rotation.T
    span = np.maximum(maximum - minimum, 1e-9)
    scale = (size - 80) / float(span.max())
    projected = (vertices[:, :2] - (minimum + maximum) / 2.0) * scale
    projected[:, 0] += size / 2.0
    projected[:, 1] = size / 2.0 - projected[:, 1]

    depth = vertices[faces, 2].mean(axis=1)
    order = np.argsort(depth)
    light = np.array([0.25, -0.35, 0.902])
    light /= np.linalg.norm(light)
    intensity = np.clip(0.35 + 0.65 * np.abs(normals @ light), 0.0, 1.0)

    canvas = Image.new("RGB", (size, size), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    for face_index in order:
        polygon = [tuple(point) for point in projected[faces[face_index]]]
        shade = float(intensity[face_index])
        color = (
            int(55 + 75 * shade),
            int(105 + 90 * shade),
            int(132 + 90 * shade),
        )
        draw.polygon(polygon, fill=color)
    draw.rectangle((0, 0, size, 46), fill=(232, 237, 242))
    draw.text((16, 16), title, fill=(22, 34, 48))
    return canvas


def add_artifact_markers(
    before: Image.Image,
    after: Image.Image,
    source: trimesh.Trimesh,
    rotation: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> None:
    parts = sorted(source.split(only_watertight=False), key=lambda item: item.area, reverse=True)
    span = np.maximum(maximum - minimum, 1e-9)
    scale = (before.width - 80) / float(span.max())
    center = (minimum + maximum) / 2.0
    before_draw = ImageDraw.Draw(before)
    after_draw = ImageDraw.Draw(after)
    for fragment in parts[1:]:
        point = np.asarray(fragment.centroid) @ rotation.T
        projected = (point[:2] - center) * scale
        x = float(projected[0] + before.width / 2.0)
        y = float(before.height / 2.0 - projected[1])
        marker = (x - 20, y - 20, x + 20, y + 20)
        before_draw.ellipse(marker, outline=(215, 45, 45), width=5)
        before_draw.text((x + 26, y - 9), "detached artifact", fill=(175, 30, 30))
        after_draw.ellipse(marker, outline=(35, 150, 85), width=4)
        after_draw.text((x + 26, y - 9), "removed", fill=(25, 115, 65))


def save_comparison(
    source: trimesh.Trimesh, candidate: trimesh.Trimesh, output_path: Path
) -> None:
    rotation = rotation_matrix(35.0, -25.0)
    minimum, maximum = shared_frame([source, candidate], rotation)
    before = render_panel(source, rotation, minimum, maximum, "BEFORE")
    after = render_panel(candidate, rotation, minimum, maximum, "AFTER")
    add_artifact_markers(before, after, source, rotation, minimum, maximum)
    comparison = Image.new("RGB", (before.width * 2, before.height), (255, 255, 255))
    comparison.paste(before, (0, 0))
    comparison.paste(after, (before.width, 0))
    comparison.save(output_path)


def assert_demo_result(
    before: dict[str, Any], after: dict[str, Any], operation: dict[str, Any]
) -> None:
    failures: list[str] = []
    if before["components"] != 2:
        failures.append(f"expected 2 input components, found {before['components']}")
    if not operation["accepted"]:
        failures.extend(operation["rejection_reasons"])
    if operation["discarded_component_count"] != 1:
        failures.append("expected exactly one discarded artifact")
    if after["components"] != 1:
        failures.append(f"expected 1 output component, found {after['components']}")
    if not after["watertight"] or not after["winding_consistent"]:
        failures.append("output must remain watertight and winding-consistent")
    for key in ("boundary_edges", "nonmanifold_edges", "degenerate_faces"):
        if after[key] != 0:
            failures.append(f"output {key} must be zero, found {after[key]}")
    if failures:
        raise RuntimeError("Demo gate failed: " + "; ".join(failures))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = output_dir / "damaged_with_detached_artifact.glb"
    candidate_path = output_dir / "repaired_candidate.glb"
    report_path = output_dir / "repair_report.json"
    comparison_path = output_dir / "before_after.png"

    source = build_damaged_mesh()
    source.export(source_path, file_type="glb")
    source_hash_before = file_sha256(source_path)
    loaded_source = trimesh.load(source_path, force="mesh", process=False)
    before = metrics(loaded_source)

    candidate, operation = detached_artifact_cleanup_candidate(loaded_source)
    candidate.export(candidate_path, file_type="glb")
    verified_candidate = trimesh.load(candidate_path, force="mesh", process=False)
    after = metrics(verified_candidate)
    assert_demo_result(before, after, operation)

    if file_sha256(source_path) != source_hash_before:
        raise RuntimeError("The original GLB changed during the demo.")

    save_comparison(loaded_source, verified_candidate, comparison_path)
    report = {
        "schema_version": "1.0",
        "operation": "detached_artifact_cleanup_demo",
        "status": "DEMO_PASSED_REQUIRES_HUMAN_REVIEW",
        "input": {
            "path": source_path.name,
            "sha256": source_hash_before,
            "metrics": before,
        },
        "candidate": {
            "path": candidate_path.name,
            "sha256": file_sha256(candidate_path),
            "metrics": after,
        },
        "repair_operation": operation,
        "comparison": comparison_path.name,
        "limitations": [
            "This demo removes only a tiny detached component from an expected single-part mesh.",
            "Passing topology gates does not prove engineering dimensions or semantic correctness.",
            "The candidate still requires visual review before downstream use.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("CPU mesh-repair demo passed")
    print(f"  components: {before['components']} -> {after['components']}")
    print(f"  watertight: {before['watertight']} -> {after['watertight']}")
    print(f"  discarded components: {operation['discarded_component_count']}")
    print(f"  output: {output_dir}")


if __name__ == "__main__":
    main()
