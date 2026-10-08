"""Generate CPU-only fit coupons for the V1 D-shaft gear assembly."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import build_parametric_gear_assembly as base


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finish(vertices: list[list[float]], faces: list[list[int]], name: str) -> trimesh.Trimesh:
    mesh = base.finish_mesh(vertices, faces, name)
    base.validate_mesh(name, mesh)
    return mesh


def make_d_hole_ring(
    parameters: dict[str, float | int],
    diametral_clearance_mm: float,
    outer_diameter_mm: float,
) -> trimesh.Trimesh:
    count = int(parameters["teeth"]) * int(parameters["samples_per_tooth"])
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    shaft_radius = float(parameters["shaft_diameter_mm"]) / 2.0
    clearance = diametral_clearance_mm / 2.0
    flat_x = shaft_radius - float(parameters["d_flat_depth_mm"]) + clearance
    nominal = base.d_profile_xy(theta, shaft_radius + clearance, flat_x)
    lead = float(parameters["gear_bore_lead_in_mm"])
    lead_xy = base.d_profile_xy(theta, shaft_radius + clearance + lead, flat_x + lead)
    outer = base.polar_xy(theta, outer_diameter_mm / 2.0)
    thickness = 5.0
    lead_height = float(parameters["gear_bore_lead_height_mm"])

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    outer_bottom = base.append_ring(vertices, outer, 0.0)
    outer_top = base.append_ring(vertices, outer, thickness)
    inner_bottom = base.append_ring(vertices, lead_xy, 0.0)
    inner_lead = base.append_ring(vertices, nominal, lead_height)
    inner_top = base.append_ring(vertices, nominal, thickness)
    base.connect_rings(faces, outer_bottom, outer_top)
    base.connect_rings(faces, inner_bottom, inner_lead)
    base.connect_rings(faces, inner_lead, inner_top)
    base.annulus_faces(faces, outer_bottom, inner_bottom, reverse=True)
    base.annulus_faces(faces, outer_top, inner_top)
    return finish(vertices, faces, f"d_hole_clearance_{diametral_clearance_mm:.2f}")


def make_d_plug(parameters: dict[str, float | int]) -> trimesh.Trimesh:
    count = int(parameters["teeth"]) * int(parameters["samples_per_tooth"])
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    base_xy = base.polar_xy(theta, 10.0)
    shaft_radius = float(parameters["shaft_diameter_mm"]) / 2.0
    shaft_xy = base.d_profile_xy(
        theta,
        shaft_radius,
        shaft_radius - float(parameters["d_flat_depth_mm"]),
    )
    tip_xy = base.d_profile_xy(
        theta,
        shaft_radius - 0.4,
        shaft_radius - float(parameters["d_flat_depth_mm"]) - 0.4,
    )
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    bottom = base.append_ring(vertices, base_xy, 0.0)
    shoulder_outer = base.append_ring(vertices, base_xy, 3.0)
    shaft_bottom = base.append_ring(vertices, shaft_xy, 3.0)
    shaft_chamfer = base.append_ring(vertices, shaft_xy, 10.2)
    tip = base.append_ring(vertices, tip_xy, 11.0)
    base.connect_rings(faces, bottom, shoulder_outer)
    base.annulus_faces(faces, shoulder_outer, shaft_bottom)
    base.connect_rings(faces, shaft_bottom, shaft_chamfer)
    base.connect_rings(faces, shaft_chamfer, tip)
    base.cap_faces(vertices, faces, bottom, 0.0, reverse=True)
    base.cap_faces(vertices, faces, tip, 11.0, reverse=False)
    return finish(vertices, faces, "d_shaft_test_plug")


def make_round_plug(parameters: dict[str, float | int]) -> trimesh.Trimesh:
    count = int(parameters["teeth"]) * int(parameters["samples_per_tooth"])
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    base_xy = base.polar_xy(theta, 7.0)
    peg_radius = float(parameters["retaining_peg_diameter_mm"]) / 2.0
    peg_xy = base.polar_xy(theta, peg_radius)
    tip_xy = base.polar_xy(theta, peg_radius - 0.3)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    bottom = base.append_ring(vertices, base_xy, 0.0)
    shoulder_outer = base.append_ring(vertices, base_xy, 3.0)
    peg_bottom = base.append_ring(vertices, peg_xy, 3.0)
    peg_chamfer = base.append_ring(vertices, peg_xy, 8.4)
    tip = base.append_ring(vertices, tip_xy, 9.0)
    base.connect_rings(faces, bottom, shoulder_outer)
    base.annulus_faces(faces, shoulder_outer, peg_bottom)
    base.connect_rings(faces, peg_bottom, peg_chamfer)
    base.connect_rings(faces, peg_chamfer, tip)
    base.cap_faces(vertices, faces, bottom, 0.0, reverse=True)
    base.cap_faces(vertices, faces, tip, 9.0, reverse=False)
    return finish(vertices, faces, "round_peg_test_plug")


def print_oriented_round_socket(
    parameters: dict[str, float | int], diametral_clearance_mm: float, outer_diameter_mm: float
) -> trimesh.Trimesh:
    modified = dict(parameters)
    modified["cap_outer_diameter_mm"] = outer_diameter_mm
    modified["cap_socket_diametral_clearance_mm"] = diametral_clearance_mm
    socket = base.make_cap(modified)
    socket.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    socket.apply_translation([0.0, 0.0, float(modified["cap_height_mm"])])
    socket.apply_translation([0.0, 0.0, -float(socket.bounds[0, 2])])
    base.validate_mesh(f"round_socket_{diametral_clearance_mm:.2f}", socket)
    return socket


def mesh_record(mesh: trimesh.Trimesh) -> dict[str, Any]:
    metrics = base.mesh_metrics(mesh)
    metrics["minimum_z_mm"] = float(mesh.bounds[0, 2])
    return metrics


def export_stl(mesh: trimesh.Trimesh, path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖：{path}")
    path.write_bytes(mesh.export(file_type="stl"))
    reloaded = trimesh.load(path, force="mesh", process=True)
    base.validate_mesh(path.stem, reloaded)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "metrics": mesh_record(reloaded),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成齿轮轴装配配合试样")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    run_dir = RUNS_ROOT / args.run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("目标装配任务不存在。")
    if args.version < 1:
        raise ValueError("版本号必须大于0。")
    output_dir = run_dir / "fit_coupons" / f"v{args.version}"
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有试样版本：{output_dir}")
    output_dir.mkdir(parents=True)

    parameters = base.default_parameters()
    d_clearances = [0.30, 0.50, 0.70]
    round_clearances = [0.10, 0.30, 0.50]
    d_outer = [20.0, 22.0, 24.0]
    round_outer = [14.0, 16.0, 18.0]
    parts: list[dict[str, Any]] = []
    plate_meshes: list[trimesh.Trimesh] = []

    for index, (clearance, diameter) in enumerate(zip(d_clearances, d_outer), start=1):
        mesh = make_d_hole_ring(parameters, clearance, diameter)
        path = output_dir / f"d_hole_clearance_{clearance:.2f}mm_od_{diameter:.0f}mm_v{args.version}.stl"
        record = export_stl(mesh, path)
        record.update(
            {
                "role": "d_hole_coupon",
                "diametral_clearance_mm": clearance,
                "identification": f"外径 {diameter:.0f} mm",
            }
        )
        parts.append(record)
        placed = mesh.copy()
        placed.apply_translation([-39.0 + (index - 1) * 26.0, 16.0, 0.0])
        plate_meshes.append(placed)

    d_plug = make_d_plug(parameters)
    d_plug_path = output_dir / f"d_shaft_12.00mm_test_plug_v{args.version}.stl"
    d_plug_record = export_stl(d_plug, d_plug_path)
    d_plug_record.update({"role": "d_shaft_test_plug", "nominal_diameter_mm": 12.0})
    parts.append(d_plug_record)
    d_plug_placed = d_plug.copy()
    d_plug_placed.apply_translation([39.0, 16.0, 0.0])
    plate_meshes.append(d_plug_placed)

    for index, (clearance, diameter) in enumerate(zip(round_clearances, round_outer), start=1):
        mesh = print_oriented_round_socket(parameters, clearance, diameter)
        path = output_dir / f"round_socket_clearance_{clearance:.2f}mm_od_{diameter:.0f}mm_v{args.version}.stl"
        record = export_stl(mesh, path)
        record.update(
            {
                "role": "round_socket_coupon",
                "diametral_clearance_mm": clearance,
                "identification": f"外径 {diameter:.0f} mm",
                "print_orientation": "盲孔开口向上",
            }
        )
        parts.append(record)
        placed = mesh.copy()
        placed.apply_translation([-39.0 + (index - 1) * 26.0, -16.0, 0.0])
        plate_meshes.append(placed)

    round_plug = make_round_plug(parameters)
    round_plug_path = output_dir / f"round_peg_8.00mm_test_plug_v{args.version}.stl"
    round_plug_record = export_stl(round_plug, round_plug_path)
    round_plug_record.update({"role": "round_peg_test_plug", "nominal_diameter_mm": 8.0})
    parts.append(round_plug_record)
    round_plug_placed = round_plug.copy()
    round_plug_placed.apply_translation([39.0, -16.0, 0.0])
    plate_meshes.append(round_plug_placed)

    combined = trimesh.util.concatenate(plate_meshes)
    combined.remove_unreferenced_vertices()
    combined_path = output_dir / f"all_fit_coupons_on_plate_v{args.version}.stl"
    if combined_path.exists():
        raise FileExistsError(f"拒绝覆盖：{combined_path}")
    combined_path.write_bytes(combined.export(file_type="stl"))
    combined_reload = trimesh.load(combined_path, force="mesh", process=True)
    components = combined_reload.split(only_watertight=False)
    if len(components) != 8 or not all(component.is_watertight for component in components):
        raise RuntimeError("组合打印STL必须包含8个独立水密试样。")
    counts = np.bincount(
        combined_reload.edges_unique_inverse, minlength=len(combined_reload.edges_unique)
    )
    if int((counts == 1).sum()) or int((counts > 2).sum()):
        raise RuntimeError("组合打印STL存在边界边或非流形边。")

    scene = trimesh.Scene()
    colors = [
        (70, 130, 180, 255),
        (80, 155, 205, 255),
        (100, 180, 220, 255),
        (230, 134, 54, 255),
        (90, 170, 100, 255),
        (120, 190, 110, 255),
        (150, 205, 120, 255),
        (190, 110, 170, 255),
    ]
    for index, (mesh, color) in enumerate(zip(plate_meshes, colors), start=1):
        scene.add_geometry(base.colored_meter_mesh(mesh, color), node_name=f"coupon_{index}")
    glb_path = output_dir / f"all_fit_coupons_layout_v{args.version}_meters.glb"
    glb_path.write_bytes(trimesh.exchange.gltf.export_glb(scene))

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "version": args.version,
        "status": "REVIEW_READY_NOT_PRINT_APPROVED",
        "purpose": "用最小试样确定Bambu A1/PLA实际装配间隙，避免直接打印整套装配体。",
        "mapping": {
            "d_hole": {
                "male": "12.00 mm D形轴，D平面切深1.50 mm",
                "variants": [
                    {"outer_diameter_mm": 20, "diametral_clearance_mm": 0.30, "nominal_round_diameter_mm": 12.30},
                    {"outer_diameter_mm": 22, "diametral_clearance_mm": 0.50, "nominal_round_diameter_mm": 12.50},
                    {"outer_diameter_mm": 24, "diametral_clearance_mm": 0.70, "nominal_round_diameter_mm": 12.70},
                ],
            },
            "round_socket": {
                "male": "8.00 mm圆柱",
                "variants": [
                    {"outer_diameter_mm": 14, "diametral_clearance_mm": 0.10, "hole_diameter_mm": 8.10},
                    {"outer_diameter_mm": 16, "diametral_clearance_mm": 0.30, "hole_diameter_mm": 8.30},
                    {"outer_diameter_mm": 18, "diametral_clearance_mm": 0.50, "hole_diameter_mm": 8.50},
                ],
            },
        },
        "parts": parts,
        "combined_print": {
            "path": str(combined_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(combined_path),
            "bytes": combined_path.stat().st_size,
            "components": len(components),
            "all_components_watertight": all(component.is_watertight for component in components),
            "bounds_mm": np.asarray(combined_reload.bounds).tolist(),
            "extents_mm": np.asarray(combined_reload.extents).tolist(),
            "minimum_z_mm": float(combined_reload.bounds[0, 2]),
            "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()),
        },
        "review_glb": {
            "path": str(glb_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(glb_path),
            "bytes": glb_path.stat().st_size,
            "units": "metres in GLB; corresponds to millimetre STL dimensions",
        },
        "print_guidance": {
            "support_required": False,
            "orientation": "使用组合STL当前朝向；所有试样最低点为Z=0，D形孔竖直贯通，圆孔盲孔开口向上。",
            "provisional_process": "Bambu A1、0.4 mm喷嘴、PLA、0.20 mm层高；最终参数仍以Bambu Studio切片检查为准。",
            "gate": "必须在Bambu Studio核对组合STL为8个零件、尺寸约98×53×11 mm、首层完整且无自动支撑后，才可发送打印。",
        },
        "test_method": [
            "打印后不要混淆试样：D孔环按外径20/22/24 mm对应0.30/0.50/0.70 mm直径间隙。",
            "圆孔帽按外径14/16/18 mm对应0.10/0.30/0.50 mm直径间隙。",
            "待零件冷却后再试插，不要用钳子强行压入。",
            "分别记录插不入、过紧、顺滑且无明显晃动、过松四种结果。",
            "优先选择能手动顺畅插入、倒置轻晃不脱落或符合装配需求的最小间隙。",
        ],
    }
    report_path = run_dir / "reports" / f"gear_assembly_fit_coupon_report_v{args.version}.json"
    write_json(report_path, report)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_paths = [Path(item["path"]) for item in parts] + [combined_path.relative_to(PROJECT_ROOT), glb_path.relative_to(PROJECT_ROOT), report_path.relative_to(PROJECT_ROOT)]
    artifact_roles = [item["role"] for item in parts] + ["fit_coupon_combined_print_stl", "fit_coupon_review_glb", "fit_coupon_report"]
    for relative_path, role in zip(relative_paths, artifact_roles):
        absolute = PROJECT_ROOT / relative_path
        manifest.setdefault("artifacts", []).append(
            {
                "role": role,
                "version": args.version,
                "path": str(relative_path),
                "bytes": absolute.stat().st_size,
                "sha256": sha256(absolute),
            }
        )
    manifest.setdefault("stages", {})["fit_coupon"] = "REVIEW_READY_NOT_PRINT_APPROVED"
    manifest["selected_fit_coupon"] = {
        "version": args.version,
        "status": "REVIEW_READY_NOT_PRINT_APPROVED",
        "combined_stl_path": str(combined_path.relative_to(PROJECT_ROOT)),
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
        "review_glb_path": str(glb_path.relative_to(PROJECT_ROOT)),
    }
    manifest["status"] = "FIT_COUPON_REVIEW_READY"
    manifest["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest.setdefault("events", []).append(
        {
            "time": manifest["updated_at"],
            "event": "FIT_COUPON_V1_GENERATED",
            "detail": "生成D形孔和圆柱孔六种间隙试样、两个测试轴及8件组合打印STL；等待Bambu Studio检查。",
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "version": args.version, "combined_stl": str(combined_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
