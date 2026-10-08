"""Build a deterministic involute external spur-gear GLB candidate."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from build_ngw_planetary_demo import extrude_radial_annulus, sampled_external_radius


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()


def project_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"输出必须位于项目目录内：{resolved}")
    if resolved.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topology(mesh: trimesh.Trimesh) -> dict[str, object]:
    edges = np.sort(np.asarray(mesh.edges), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(mesh.split(only_watertight=False))),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "positive_volume": bool(mesh.volume > 0.0),
        "extents": np.asarray(mesh.extents, dtype=float).tolist(),
    }


def build_mesh(
    *,
    teeth: int,
    module_mm: float,
    pressure_angle_degrees: float,
    tooth_thinning_mm: float,
    bore_diameter_mm: float,
    face_width_mm: float,
    samples_per_tooth: int,
) -> tuple[trimesh.Trimesh, dict[str, float]]:
    if teeth < 6 or module_mm <= 0 or bore_diameter_mm <= 0 or face_width_mm <= 0:
        raise ValueError("齿数至少为 6，模数、孔径和齿宽必须为正数。")
    theta, radius, dimensions = sampled_external_radius(
        teeth,
        module_mm,
        pressure_angle_degrees,
        tooth_thinning_mm,
        samples_per_tooth,
    )
    if bore_diameter_mm >= dimensions["root_diameter_mm"]:
        raise ValueError("孔径必须小于齿根径。")
    mesh = extrude_radial_annulus(
        theta,
        radius,
        bore_diameter_mm / 2.0,
        face_width_mm,
        "parametric_involute_external_spur_gear",
    )
    mesh.visual.vertex_colors = np.tile(
        np.array([45, 112, 190, 255], dtype=np.uint8), (len(mesh.vertices), 1)
    )
    mesh.units = "mm"
    return mesh, dimensions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成参数化渐开线外直齿轮 GLB 候选")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--teeth", required=True, type=int)
    parser.add_argument("--module-mm", required=True, type=float)
    parser.add_argument("--pressure-angle-degrees", default=20.0, type=float)
    parser.add_argument("--tooth-thinning-mm", default=0.0, type=float)
    parser.add_argument("--bore-diameter-mm", required=True, type=float)
    parser.add_argument("--face-width-mm", required=True, type=float)
    parser.add_argument("--samples-per-tooth", default=64, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = project_output(args.output)
    report_path = project_output(args.report)
    mesh, dimensions = build_mesh(
        teeth=args.teeth,
        module_mm=args.module_mm,
        pressure_angle_degrees=args.pressure_angle_degrees,
        tooth_thinning_mm=args.tooth_thinning_mm,
        bore_diameter_mm=args.bore_diameter_mm,
        face_width_mm=args.face_width_mm,
        samples_per_tooth=args.samples_per_tooth,
    )
    before = topology(mesh)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output, file_type="glb")
    verified = trimesh.load(output, force="mesh", process=False)
    after = topology(verified)
    passed = all(
        (
            after["components"] == 1,
            after["watertight"],
            after["winding_consistent"],
            after["boundary_edges"] == 0,
            after["nonmanifold_edges"] == 0,
            after["positive_volume"],
        )
    )
    report = {
        "schema_version": "1.0",
        "operation": "deterministic_involute_spur_gear_reconstruction",
        "status": (
            "PARAMETRIC_RECONSTRUCTION_CANDIDATE_REQUIRES_PRINT_REVIEW"
            if passed
            else "RECONSTRUCTION_VALIDATION_FAILED_DO_NOT_USE"
        ),
        "parameters_mm": {
            "teeth": args.teeth,
            "module": args.module_mm,
            "pressure_angle_degrees": args.pressure_angle_degrees,
            "tooth_thinning": args.tooth_thinning_mm,
            "bore_diameter": args.bore_diameter_mm,
            "face_width": args.face_width_mm,
            "samples_per_tooth": args.samples_per_tooth,
            **dimensions,
        },
        "output": {"path": str(output), "sha256": sha256(output)},
        "before_export": before,
        "after_export_reload": after,
        "limitations": [
            "这是确定性渐开线网格重建，不是对原 Hunyuan 顶点的平滑修复。",
            "GLB 几何数值沿用项目毫米参数；进入其他软件时必须核对单位。",
            "仍需验证 18/27/72 啮合、打印收缩、侧隙和 Bambu Studio 逐层结果。",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("导出重载后的拓扑门禁失败。")


if __name__ == "__main__":
    main()
