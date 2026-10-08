"""Extract an editable spur-gear parameter draft from a mesh.

The result is geometric evidence, not a CAD approval. When a reference mesh is
provided, the generated mesh is scaled by outside diameter and compared against
the reference's bore, root diameter and thickness.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()


def project_path(path: Path, *, must_exist: bool) -> Path:
    """Resolve a path while keeping all reads and writes inside the project."""
    resolved = path.resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"路径必须位于项目目录内：{resolved}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"文件不存在：{resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    resolved = project_path(path, must_exist=True)
    mesh = trimesh.load(resolved, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise TypeError(f"无法加载为有效三角网格：{resolved}")
    return mesh


def cyclic_envelope(
    points: np.ndarray, center: np.ndarray, *, bins: int = 8192
) -> np.ndarray:
    xy = points - center
    radii = np.linalg.norm(xy, axis=1)
    # Ignore bore/cap vertices before angular binning. Otherwise a bin which
    # happens to contain only an inner-ring vertex creates a false radial dip.
    # Standard external gears have a root radius comfortably above this cutoff.
    outer_cutoff = float(np.max(radii)) * 0.55
    outer = radii >= outer_cutoff
    xy = xy[outer]
    radii = radii[outer]
    angles = (np.arctan2(xy[:, 1], xy[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    indices = np.floor(angles / (2.0 * np.pi) * bins).astype(int) % bins
    envelope = np.full(bins, np.nan, dtype=float)
    for index, radius in zip(indices, radii):
        if np.isnan(envelope[index]) or radius > envelope[index]:
            envelope[index] = radius
    valid = np.flatnonzero(np.isfinite(envelope))
    if len(valid) < bins // 50:
        raise ValueError("外轮廓角度采样不足，无法可靠估算齿数。")
    envelope = np.interp(
        np.arange(bins),
        np.concatenate((valid - bins, valid, valid + bins)),
        np.concatenate((envelope[valid], envelope[valid], envelope[valid])),
    )
    return gaussian_filter1d(envelope, sigma=3.0, mode="wrap")


def infer_tooth_count(envelope: np.ndarray) -> dict[str, Any]:
    bins = len(envelope)
    trend = gaussian_filter1d(envelope, sigma=bins / 50.0, mode="wrap")
    signal = envelope - trend
    amplitudes = np.abs(np.fft.rfft(signal))
    ranked = sorted(
        ((float(amplitudes[index]), index) for index in range(6, min(81, len(amplitudes)))),
        reverse=True,
    )
    best_amplitude, best_count = ranked[0]
    second_amplitude = ranked[1][0] if len(ranked) > 1 else 0.0
    return {
        "estimated_tooth_count": int(best_count),
        "fft_amplitude": best_amplitude,
        "confidence_ratio_to_second": float(
            best_amplitude / max(second_amplitude, np.finfo(float).eps)
        ),
        "top_frequency_candidates": [
            {"tooth_count": int(index), "amplitude": amplitude}
            for amplitude, index in ranked[:6]
        ],
    }


def extract(mesh: trimesh.Trimesh) -> tuple[dict[str, Any], np.ndarray]:
    extents = np.asarray(mesh.extents, dtype=float)
    axis = int(np.argmin(extents))
    plane = [index for index in range(3) if index != axis]
    center3 = np.asarray(mesh.center_mass if mesh.is_watertight else mesh.centroid)
    center = center3[plane]
    points = np.asarray(mesh.vertices)[:, plane]
    radial = np.linalg.norm(points - center, axis=1)
    envelope = cyclic_envelope(points, center)
    tooth = infer_tooth_count(envelope)

    outside_radius = float(np.quantile(envelope, 0.95))
    root_radius = float(np.quantile(envelope, 0.05))
    bore_radius = float(np.quantile(radial, 0.005))
    tooth_count = tooth["estimated_tooth_count"]
    module = (2.0 * outside_radius) / (tooth_count + 2.0)
    pitch_diameter = module * tooth_count
    expected_root_diameter = module * (tooth_count - 2.5)
    result = {
        "axis_index": axis,
        "axis_name": "XYZ"[axis],
        "center_in_profile_plane": center.tolist(),
        **tooth,
        "tooth_count_signal_confident": bool(
            tooth["confidence_ratio_to_second"] >= 1.25
        ),
        "outside_diameter_mesh_units": 2.0 * outside_radius,
        "root_diameter_mesh_units": 2.0 * root_radius,
        "bore_diameter_mesh_units": 2.0 * bore_radius,
        "thickness_mesh_units": float(extents[axis]),
        "module_from_outside_diameter_mesh_units": module,
        "pitch_diameter_mesh_units": pitch_diameter,
        "standard_full_depth_root_diameter_mesh_units": expected_root_diameter,
        "root_diameter_deviation_from_standard_ratio": float(
            (2.0 * root_radius - expected_root_diameter)
            / max(expected_root_diameter, np.finfo(float).eps)
        ),
        "bore_to_outside_ratio": float(bore_radius / outside_radius),
        "thickness_to_outside_ratio": float(extents[axis] / (2.0 * outside_radius)),
        "method_notes": [
            "旋转轴取包围盒最薄方向。",
            "齿数取径向外轮廓去趋势后的 FFT 主频。",
            "径向包络会先排除低于最大半径 55% 的孔壁/端面内圈顶点。",
            "外径/齿根径分别取平滑径向包络的 95%/5% 分位。",
            "孔径取顶点径向距离的 0.5% 分位。",
        ],
    }
    return result, envelope


def compare_to_reference(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    scale = (
        reference["outside_diameter_mesh_units"]
        / candidate["outside_diameter_mesh_units"]
    )
    fields = (
        "outside_diameter_mesh_units",
        "root_diameter_mesh_units",
        "bore_diameter_mesh_units",
        "thickness_mesh_units",
        "module_from_outside_diameter_mesh_units",
        "pitch_diameter_mesh_units",
    )
    scaled: dict[str, float] = {}
    errors: dict[str, float] = {}
    for field in fields:
        scaled[field] = float(candidate[field] * scale)
        reference_value = float(reference[field])
        errors[field] = float(
            (scaled[field] - reference_value)
            / max(abs(reference_value), np.finfo(float).eps)
        )
    return {
        "scale_candidate_to_reference_by_outside_diameter": float(scale),
        "tooth_count_matches": bool(
            candidate["estimated_tooth_count"] == reference["estimated_tooth_count"]
        ),
        "candidate_scaled_to_reference_units": scaled,
        "relative_error_after_outside_diameter_alignment": errors,
        "interpretation": (
            "外径对齐只能用于比较比例；不能证明生成模型具有参考模型的实际单位。"
        ),
    }


def draw_profiles(
    output: Path,
    candidate: np.ndarray,
    reference: np.ndarray | None,
) -> None:
    width, height = 1600, 720
    margin = 90
    canvas = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), "RADIAL ENVELOPE / 0-360 DEG", fill=(24, 50, 75))
    draw.line((margin, height - margin, width - margin, height - margin), fill=(80, 90, 100), width=2)
    draw.line((margin, margin, margin, height - margin), fill=(80, 90, 100), width=2)

    profiles = [("candidate", candidate / max(candidate.max(), 1e-12), (0, 132, 128))]
    if reference is not None:
        profiles.append(("reference", reference / max(reference.max(), 1e-12), (220, 154, 48)))
    all_values = np.concatenate([profile for _, profile, _ in profiles])
    lower = float(np.quantile(all_values, 0.01)) - 0.02
    upper = float(np.quantile(all_values, 0.99)) + 0.02
    for name, profile, color in profiles:
        points = []
        for index in range(width - 2 * margin):
            source_index = int(index / (width - 2 * margin - 1) * (len(profile) - 1))
            value = float(profile[source_index])
            x = margin + index
            y = height - margin - (value - lower) / max(upper - lower, 1e-12) * (height - 2 * margin)
            points.append((x, y))
        draw.line(points, fill=color, width=2)
        legend_x = margin + (0 if name == "candidate" else 250)
        draw.line((legend_x, 55, legend_x + 45, 55), fill=color, width=5)
        draw.text((legend_x + 55, 46), name, fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从直齿轮网格提取参数草案")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--profile-preview", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = project_path(args.report, must_exist=False)
    preview_path = project_path(args.profile_preview, must_exist=False)
    for output in (report_path, preview_path):
        if output.exists():
            raise FileExistsError(f"拒绝覆盖已有输出：{output}")
    input_path = project_path(args.input, must_exist=True)
    candidate_mesh = load_mesh(input_path)
    candidate, candidate_profile = extract(candidate_mesh)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "spur_gear_mesh_parameter_draft",
        "status": "PARAMETER_DRAFT_REQUIRES_ENGINEERING_CONFIRMATION",
        "input": {"path": str(input_path), "sha256": sha256(input_path), **candidate},
        "limitations": [
            "只适用于旋转轴近似为包围盒最薄方向的外啮合直齿轮。",
            "自动齿数和尺寸必须结合视觉、参考模型或实测值确认。",
            "无法从生成网格可靠恢复压力角、齿侧间隙、公差和材料收缩补偿。",
        ],
    }
    reference_profile = None
    if args.reference:
        reference_path = project_path(args.reference, must_exist=True)
        reference_mesh = load_mesh(reference_path)
        reference, reference_profile = extract(reference_mesh)
        report["reference"] = {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            **reference,
        }
        report["comparison"] = compare_to_reference(candidate, reference)
    draw_profiles(preview_path, candidate_profile, reference_profile)
    report["profile_preview"] = str(preview_path)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
