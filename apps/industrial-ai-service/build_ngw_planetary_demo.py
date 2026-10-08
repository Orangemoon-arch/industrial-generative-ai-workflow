"""Generate a CPU-only NGW planetary-gear FDM demonstration set.

The tooth flanks are sampled from the standard involute equation.  The output is
for low-speed, hand-turned fit validation only; it is not a load-rated gearbox.
All dimensions are millimetres.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()
RUN_ID_PATTERN = re.compile(r"RUN-\d{8}-\d{6}-[a-z0-9]{6}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def involute_angle(radius: np.ndarray | float, base_radius: float) -> np.ndarray:
    radius_array = np.asarray(radius, dtype=float)
    value = np.sqrt(np.maximum((radius_array / base_radius) ** 2 - 1.0, 0.0))
    return value - np.arctan(value)


def sampled_external_radius(
    teeth: int,
    module: float,
    pressure_angle_deg: float,
    tooth_thinning_mm: float,
    samples_per_tooth: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pitch_radius = module * teeth / 2.0
    base_radius = pitch_radius * math.cos(math.radians(pressure_angle_deg))
    outer_radius = pitch_radius + module
    root_radius = pitch_radius - 1.25 * module
    half_pitch = math.pi / teeth
    half_tooth_pitch = math.pi / (2.0 * teeth) - tooth_thinning_mm / (2.0 * pitch_radius)
    inv_pitch = float(involute_angle(pitch_radius, base_radius))
    flank_start_radius = max(base_radius, root_radius)
    flank_radii = np.linspace(flank_start_radius, outer_radius, 256)
    flank_angles = half_tooth_pitch + inv_pitch - involute_angle(flank_radii, base_radius)
    order = np.argsort(flank_angles)
    flank_angles = flank_angles[order]
    flank_radii = flank_radii[order]
    tip_angle = float(flank_angles[0])
    base_angle = float(flank_angles[-1])

    count = teeth * samples_per_tooth
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    relative = ((theta + half_pitch) % (2.0 * half_pitch)) - half_pitch
    absolute = np.abs(relative)
    radius = np.full(count, root_radius, dtype=float)
    radius[absolute <= tip_angle] = outer_radius
    active = (absolute > tip_angle) & (absolute < base_angle)
    radius[active] = np.interp(absolute[active], flank_angles, flank_radii)
    dimensions = {
        "pitch_diameter_mm": 2.0 * pitch_radius,
        "base_diameter_mm": 2.0 * base_radius,
        "outside_diameter_mm": 2.0 * outer_radius,
        "root_diameter_mm": 2.0 * root_radius,
    }
    return theta, radius, dimensions


def sampled_internal_radius(
    teeth: int,
    module: float,
    pressure_angle_deg: float,
    space_widening_mm: float,
    samples_per_tooth: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pitch_radius = module * teeth / 2.0
    base_radius = pitch_radius * math.cos(math.radians(pressure_angle_deg))
    tip_radius = pitch_radius - module
    root_radius = pitch_radius + 1.25 * module
    if tip_radius <= base_radius:
        raise ValueError("Internal-gear tip radius must exceed its base radius for this generator")
    half_pitch = math.pi / teeth
    half_space_pitch = math.pi / (2.0 * teeth) + space_widening_mm / (2.0 * pitch_radius)
    inv_pitch = float(involute_angle(pitch_radius, base_radius))
    flank_radii = np.linspace(tip_radius, root_radius, 256)
    flank_angles = half_space_pitch + inv_pitch - involute_angle(flank_radii, base_radius)
    order = np.argsort(flank_angles)
    flank_angles = flank_angles[order]
    flank_radii = flank_radii[order]
    root_angle = float(flank_angles[0])
    tip_angle = float(flank_angles[-1])

    count = teeth * samples_per_tooth
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    relative = ((theta + half_pitch) % (2.0 * half_pitch)) - half_pitch
    absolute = np.abs(relative)
    radius = np.full(count, tip_radius, dtype=float)
    radius[absolute <= root_angle] = root_radius
    active = (absolute > root_angle) & (absolute < tip_angle)
    radius[active] = np.interp(absolute[active], flank_angles, flank_radii)
    dimensions = {
        "pitch_diameter_mm": 2.0 * pitch_radius,
        "internal_tip_diameter_mm": 2.0 * tip_radius,
        "internal_root_diameter_mm": 2.0 * root_radius,
        "base_diameter_mm": 2.0 * base_radius,
    }
    return theta, radius, dimensions


def xy_from_polar(theta: np.ndarray, radius: np.ndarray | float) -> np.ndarray:
    return np.column_stack((np.asarray(radius) * np.cos(theta), np.asarray(radius) * np.sin(theta)))


def append_ring(vertices: list[list[float]], xy: np.ndarray, z: float) -> np.ndarray:
    start = len(vertices)
    vertices.extend([[float(x), float(y), float(z)] for x, y in xy])
    return np.arange(start, start + len(xy), dtype=np.int64)


def connect_closed(faces: list[list[int]], ring_a: np.ndarray, ring_b: np.ndarray, reverse: bool = False) -> None:
    for index in range(len(ring_a)):
        nxt = (index + 1) % len(ring_a)
        pair = [
            [int(ring_a[index]), int(ring_a[nxt]), int(ring_b[nxt])],
            [int(ring_a[index]), int(ring_b[nxt]), int(ring_b[index])],
        ]
        faces.extend([triangle[::-1] for triangle in pair] if reverse else pair)


def annulus_cap(
    faces: list[list[int]], outer: np.ndarray, inner: np.ndarray, reverse: bool = False
) -> None:
    for index in range(len(outer)):
        nxt = (index + 1) % len(outer)
        pair = [
            [int(outer[index]), int(outer[nxt]), int(inner[nxt])],
            [int(outer[index]), int(inner[nxt]), int(inner[index])],
        ]
        faces.extend([triangle[::-1] for triangle in pair] if reverse else pair)


def finish_mesh(vertices: list[list[float]], faces: list[list[int]], name: str) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(faces), process=True)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh.metadata["name"] = name
    return mesh


def extrude_radial_annulus(
    theta: np.ndarray,
    outer_radius: np.ndarray | float,
    inner_radius: np.ndarray | float,
    height: float,
    name: str,
) -> trimesh.Trimesh:
    outer_xy = xy_from_polar(theta, outer_radius)
    inner_xy = xy_from_polar(theta, inner_radius)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    outer_bottom = append_ring(vertices, outer_xy, 0.0)
    outer_top = append_ring(vertices, outer_xy, height)
    inner_bottom = append_ring(vertices, inner_xy, 0.0)
    inner_top = append_ring(vertices, inner_xy, height)
    connect_closed(faces, outer_bottom, outer_top)
    connect_closed(faces, inner_bottom, inner_top, reverse=True)
    annulus_cap(faces, outer_top, inner_top)
    annulus_cap(faces, outer_bottom, inner_bottom, reverse=True)
    return finish_mesh(vertices, faces, name)


def extrude_sector_strip(
    theta: np.ndarray,
    outer_radius: float,
    inner_radius: np.ndarray,
    height: float,
    name: str,
) -> trimesh.Trimesh:
    outer_xy = xy_from_polar(theta, outer_radius)
    inner_xy = xy_from_polar(theta, inner_radius)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    ob = append_ring(vertices, outer_xy, 0.0)
    ot = append_ring(vertices, outer_xy, height)
    ib = append_ring(vertices, inner_xy, 0.0)
    it = append_ring(vertices, inner_xy, height)
    for index in range(len(theta) - 1):
        nxt = index + 1
        faces.extend([[int(ob[index]), int(ob[nxt]), int(ot[nxt])], [int(ob[index]), int(ot[nxt]), int(ot[index])]])
        faces.extend([[int(ib[index]), int(it[nxt]), int(ib[nxt])], [int(ib[index]), int(it[index]), int(it[nxt])]])
        faces.extend([[int(ot[index]), int(ot[nxt]), int(it[nxt])], [int(ot[index]), int(it[nxt]), int(it[index])]])
        faces.extend([[int(ob[index]), int(ib[nxt]), int(ob[nxt])], [int(ob[index]), int(ib[index]), int(ib[nxt])]])
    for index in (0, len(theta) - 1):
        if index == 0:
            faces.extend([[int(ob[index]), int(ot[index]), int(it[index])], [int(ob[index]), int(it[index]), int(ib[index])]])
        else:
            faces.extend([[int(ob[index]), int(it[index]), int(ot[index])], [int(ob[index]), int(ib[index]), int(it[index])]])
    return finish_mesh(vertices, faces, name)


def validate_mesh(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "body_count": int(mesh.body_count),
        "volume_mm3": float(mesh.volume),
        "bounds_mm": np.asarray(mesh.bounds).round(4).tolist(),
        "extents_mm": np.asarray(mesh.extents).round(4).tolist(),
    }


def make_preview(
    path: Path,
    ring_theta: np.ndarray,
    ring_inner_radius: np.ndarray,
    sun_theta: np.ndarray,
    sun_radius: np.ndarray,
    planet_theta: np.ndarray,
    planet_radius: np.ndarray,
    params: dict[str, Any],
    center_radius: float,
) -> None:
    size = 1400
    margin = 70
    scale = (size - 2 * margin) / params["ring_outer_diameter_mm"]
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image, "RGBA")

    def project(xy: np.ndarray) -> list[tuple[float, float]]:
        return [(size / 2 + float(x) * scale, size / 2 - float(y) * scale) for x, y in xy]

    outer = xy_from_polar(ring_theta, params["ring_outer_diameter_mm"] / 2.0)
    inner = xy_from_polar(ring_theta, ring_inner_radius)
    draw.polygon(project(outer), fill=(90, 120, 155, 220), outline=(30, 50, 80, 255))
    draw.polygon(project(inner), fill=(255, 255, 255, 255), outline=(30, 50, 80, 255))

    sun_xy = xy_from_polar(sun_theta, sun_radius)
    draw.polygon(project(sun_xy), fill=(247, 190, 70, 240), outline=(120, 70, 10, 255))
    sun_bore = xy_from_polar(
        np.linspace(0.0, 2.0 * math.pi, 160, endpoint=False), params["sun_bore_diameter_mm"] / 2.0
    )
    draw.polygon(project(sun_bore), fill=(255, 255, 255, 255), outline=(120, 70, 10, 255))

    base_planet = xy_from_polar(planet_theta, planet_radius)
    bore_theta = np.linspace(0.0, 2.0 * math.pi, 160, endpoint=False)
    for index in range(params["planet_count"]):
        gamma = 2.0 * math.pi * index / params["planet_count"]
        rotation = np.array([[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]])
        center = np.array([center_radius * math.cos(gamma), center_radius * math.sin(gamma)])
        planet_xy = base_planet @ rotation.T + center
        bore_xy = xy_from_polar(bore_theta, params["planet_bore_diameter_mm"] / 2.0) + center
        draw.polygon(project(planet_xy), fill=(224, 100, 75, 220), outline=(110, 35, 25, 255))
        draw.polygon(project(bore_xy), fill=(255, 255, 255, 255), outline=(110, 35, 25, 255))
    draw.text((30, 30), "NGW 18/27/72 - assembly preview (not print approval)", fill=(20, 20, 20, 255))
    image.save(path)


def build(output_root: Path) -> Path:
    params: dict[str, Any] = {
        "module_mm": 1.5,
        "pressure_angle_deg": 20.0,
        "sun_teeth": 18,
        "planet_teeth": 27,
        "ring_teeth": 72,
        "planet_count": 3,
        "face_width_mm": 8.0,
        "nominal_pair_backlash_mm": 0.30,
        "sun_bore_diameter_mm": 8.2,
        "planet_bore_diameter_mm": 5.3,
        "ring_outer_diameter_mm": 122.0,
        "samples_per_tooth": 32,
        "coupon_span_deg": 25.0,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{stamp}-{signature}"
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RuntimeError("Invalid run id")
    run_dir = (output_root / run_id).resolve()
    if output_root.resolve() not in run_dir.parents:
        raise RuntimeError("Output escaped runs root")
    run_dir.mkdir(parents=True, exist_ok=False)

    per_gear_allowance = params["nominal_pair_backlash_mm"] / 2.0
    sun_theta, sun_radius, sun_dims = sampled_external_radius(
        params["sun_teeth"], params["module_mm"], params["pressure_angle_deg"],
        per_gear_allowance, params["samples_per_tooth"]
    )
    planet_theta, planet_radius, planet_dims = sampled_external_radius(
        params["planet_teeth"], params["module_mm"], params["pressure_angle_deg"],
        per_gear_allowance, params["samples_per_tooth"]
    )
    ring_theta, ring_inner_radius, ring_dims = sampled_internal_radius(
        params["ring_teeth"], params["module_mm"], params["pressure_angle_deg"],
        per_gear_allowance, params["samples_per_tooth"]
    )
    sun = extrude_radial_annulus(
        sun_theta, sun_radius, params["sun_bore_diameter_mm"] / 2.0,
        params["face_width_mm"], "sun_18t"
    )
    planet = extrude_radial_annulus(
        planet_theta, planet_radius, params["planet_bore_diameter_mm"] / 2.0,
        params["face_width_mm"], "planet_27t"
    )
    ring = extrude_radial_annulus(
        ring_theta, params["ring_outer_diameter_mm"] / 2.0, ring_inner_radius,
        params["face_width_mm"], "ring_72t"
    )

    span = math.radians(params["coupon_span_deg"])
    coupon_theta = np.linspace(-span / 2.0, span / 2.0, 321)
    relative = np.mod(coupon_theta, 2.0 * math.pi)
    coupon_inner = np.interp(relative, ring_theta, ring_inner_radius, period=2.0 * math.pi)
    ring_coupon = extrude_sector_strip(
        coupon_theta, params["ring_outer_diameter_mm"] / 2.0, coupon_inner,
        params["face_width_mm"], "ring_sector_coupon"
    )

    outputs = {
        "sun": run_dir / "ngw_sun_18t_v1.stl",
        "planet": run_dir / "ngw_planet_27t_v1.stl",
        "ring": run_dir / "ngw_ring_72t_v1.stl",
        "ring_sector_coupon": run_dir / "ngw_ring_sector_coupon_v1.stl",
    }
    meshes = {"sun": sun, "planet": planet, "ring": ring, "ring_sector_coupon": ring_coupon}
    for key, path in outputs.items():
        meshes[key].export(path)

    # One convenient import for Bambu Studio, while preserving two physically
    # separate shells so the parts remain removable after printing.
    coupon_print_plate = trimesh.util.concatenate([planet.copy(), ring_coupon.copy()])
    coupon_print_plate_path = run_dir / "ngw_planet_and_ring_coupon_same_plate_v1.stl"
    coupon_print_plate.export(coupon_print_plate_path)

    center_radius = params["module_mm"] * (params["sun_teeth"] + params["planet_teeth"]) / 2.0
    scene = trimesh.Scene()
    scene.add_geometry(ring.copy(), node_name="ring_72t")
    scene.add_geometry(sun.copy(), node_name="sun_18t")
    for index in range(params["planet_count"]):
        gamma = 2.0 * math.pi * index / params["planet_count"]
        item = planet.copy()
        transform = trimesh.transformations.rotation_matrix(gamma, [0, 0, 1])
        transform[:3, 3] = [center_radius * math.cos(gamma), center_radius * math.sin(gamma), 0.0]
        item.apply_transform(transform)
        scene.add_geometry(item, node_name=f"planet_{index + 1}")
    assembly_glb = run_dir / "ngw_18_27_72_assembly_preview_v1.glb"
    scene.export(assembly_glb)
    preview_png = run_dir / "ngw_18_27_72_top_view_v1.png"
    make_preview(
        preview_png, ring_theta, ring_inner_radius, sun_theta, sun_radius,
        planet_theta, planet_radius, params, center_radius
    )

    validation = {key: validate_mesh(mesh) for key, mesh in meshes.items()}
    equations = {
        "concentric_tooth_relation": params["ring_teeth"] == params["sun_teeth"] + 2 * params["planet_teeth"],
        "equal_spacing_phase_integer": (params["sun_teeth"] + params["ring_teeth"]) % params["planet_count"] == 0,
        "fixed_ring_ratio": 1.0 + params["ring_teeth"] / params["sun_teeth"],
        "planet_center_radius_mm": center_radius,
        "planet_center_pitch_circle_diameter_mm": 2.0 * center_radius,
        "planet_clearance_condition_left": params["planet_teeth"] + 2,
        "planet_clearance_condition_right": (params["sun_teeth"] + params["planet_teeth"]) * math.sin(math.pi / params["planet_count"]),
    }
    all_meshes_valid = all(
        item["watertight"] and item["winding_consistent"] and item["body_count"] == 1 and item["volume_mm3"] > 0
        for item in validation.values()
    )
    report = {
        "run_id": run_id,
        "status": "WAITING_COUPON_SLICER_AND_PHYSICAL_FIT_REVIEW",
        "scope": "LOW_SPEED_HAND_TURNED_DEMONSTRATOR_ONLY",
        "parameters": params,
        "derived_dimensions": {"sun": sun_dims, "planet": planet_dims, "ring": ring_dims},
        "equation_checks": equations,
        "mesh_validation": validation,
        "same_plate_validation": validate_mesh(coupon_print_plate),
        "all_meshes_valid": all_meshes_valid,
        "files": {},
        "printing_gate": [
            "First slice and print only the 27T planet plus ring-sector coupon.",
            "Manually verify smooth mesh at several tooth positions without forcing.",
            "Do not print the full ring until the 0.30 mm nominal pair backlash is physically accepted.",
            "This run contains no carrier or load-rated shaft design and is not a complete gearbox.",
        ],
    }
    for key, path in {
        **outputs,
        "same_plate_planet_and_coupon": coupon_print_plate_path,
        "assembly_preview": assembly_glb,
        "top_view": preview_png,
    }.items():
        report["files"][key] = {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)}
    report_path = run_dir / "ngw_design_and_validation_v1.json"
    write_json(report_path, report)
    readme = run_dir / "README_先打印试片.md"
    readme.write_text(
        "# NGW 18/27/72 首版试片\n\n"
        "当前状态：等待 Bambu Studio 切片和实体啮合验证，不是打印批准。\n\n"
        "首轮优先只导入 `ngw_planet_and_ring_coupon_same_plate_v1.stl`。"
        "该文件含两个彼此分开的实体，打印后仍可分别取下。保持毫米单位；切片后逐层检查齿根和内齿是否连续。"
        "打印完成后让行星轮与内齿试片在多个齿位轻轻啮合，不得强压。\n\n"
        "通过标准：能插入、能沿齿面移动、没有明显顶齿或卡死。"
        "若太紧或太松，保留本 RUN，另建新版本调整齿厚余量。\n\n"
        "完整内齿圈、太阳轮和装配预览仅供尺寸与结构检查。"
        "本 RUN 尚未包含行星架，严禁宣称为承载或生产齿轮箱。\n",
        encoding="utf-8",
    )
    final_manifest = {
        "run_id": run_id,
        "report": {"path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": sha256(report_path)},
        "instructions": {"path": str(readme.relative_to(PROJECT_ROOT)), "sha256": sha256(readme)},
        "all_meshes_valid": all_meshes_valid,
        "status": report["status"],
    }
    write_json(run_dir / "manifest.json", final_manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=RUNS_ROOT)
    args = parser.parse_args()
    run_dir = build(args.output_root)
    print(run_dir)


if __name__ == "__main__":
    main()
