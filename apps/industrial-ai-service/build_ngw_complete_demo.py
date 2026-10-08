"""Build the complete, hand-turned NGW 18/27/72 demonstration assembly.

This consumes the tooth geometry whose 0.30 mm nominal pair backlash passed the
physical planet-to-ring coupon.  It is not a load-rated gearbox.
"""

from __future__ import annotations

import os

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import Delaunay

import build_ngw_planetary_demo as gearlib


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def triangulated_plate_with_holes(
    outer_radius: float,
    holes: list[tuple[float, float, float]],
    height: float,
    name: str,
    grid_spacing: float = 1.5,
) -> trimesh.Trimesh:
    """Extrude a convex circular plate with circular through-holes."""
    boundary: list[list[float]] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False):
        boundary.append([outer_radius * math.cos(angle), outer_radius * math.sin(angle)])
    for cx, cy, radius in holes:
        samples = max(72, int(math.ceil(2.0 * math.pi * radius / 0.25)))
        for angle in np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False):
            boundary.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])

    grid: list[list[float]] = []
    values = np.arange(-outer_radius + grid_spacing, outer_radius, grid_spacing)
    for x in values:
        for y in values:
            if x * x + y * y >= (outer_radius - 0.35) ** 2:
                continue
            if any((x - cx) ** 2 + (y - cy) ** 2 <= (radius + 0.35) ** 2 for cx, cy, radius in holes):
                continue
            grid.append([float(x), float(y)])
    points = np.asarray(boundary + grid, dtype=float)
    triangulation = Delaunay(points)

    kept: list[list[int]] = []
    for raw in triangulation.simplices:
        triangle = points[raw]
        center = triangle.mean(axis=0)
        edge_midpoints = np.asarray([
            (triangle[0] + triangle[1]) / 2.0,
            (triangle[1] + triangle[2]) / 2.0,
            (triangle[2] + triangle[0]) / 2.0,
        ])
        probes = np.vstack([center, edge_midpoints])
        if np.any(np.sum(probes**2, axis=1) > outer_radius**2 + 1e-8):
            continue
        if any(np.any((probes[:, 0] - cx) ** 2 + (probes[:, 1] - cy) ** 2 < radius**2 - 1e-8) for cx, cy, radius in holes):
            continue
        indices = [int(value) for value in raw]
        a, b, c = points[indices]
        if np.cross(b - a, c - a) < 0:
            indices[1], indices[2] = indices[2], indices[1]
        kept.append(indices)

    edge_counts: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for a, b, c in kept:
        for start, end in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((start, end)))
            count, _ = edge_counts.get(key, (0, (start, end)))
            edge_counts[key] = (count + 1, (start, end))
    boundary_edges = [oriented for count, oriented in edge_counts.values() if count == 1]

    count = len(points)
    vertices = np.vstack([
        np.column_stack([points, np.zeros(count)]),
        np.column_stack([points, np.full(count, height)]),
    ])
    faces: list[list[int]] = []
    for a, b, c in kept:
        faces.append([a + count, b + count, c + count])
        faces.append([c, b, a])
    for start, end in boundary_edges:
        faces.extend([
            [start, end, end + count],
            [start, end + count, start + count],
        ])
    return gearlib.finish_mesh(vertices.tolist(), faces, name)


def revolved_solid(profile: list[tuple[float, float]], name: str, samples: int = 128) -> trimesh.Trimesh:
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    rings = [gearlib.append_ring(vertices, gearlib.xy_from_polar(theta, radius), z) for radius, z in profile]
    for first, second in zip(rings, rings[1:]):
        gearlib.connect_closed(faces, first, second)
    # Solid bottom and top caps.
    for ring, z, reverse in ((rings[0], profile[0][1], True), (rings[-1], profile[-1][1], False)):
        center = len(vertices)
        vertices.append([0.0, 0.0, float(z)])
        for index in range(samples):
            nxt = (index + 1) % samples
            triangle = [center, int(ring[index]), int(ring[nxt])]
            faces.append(triangle[::-1] if reverse else triangle)
    return gearlib.finish_mesh(vertices, faces, name)


def blind_cap(outer_diameter: float, socket_diameter: float, height: float, depth: float, name: str) -> trimesh.Trimesh:
    samples = 160
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    outer = gearlib.xy_from_polar(theta, outer_diameter / 2.0)
    inner = gearlib.xy_from_polar(theta, socket_diameter / 2.0)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    ob = gearlib.append_ring(vertices, outer, 0.0)
    ot = gearlib.append_ring(vertices, outer, height)
    ib = gearlib.append_ring(vertices, inner, 0.0)
    it = gearlib.append_ring(vertices, inner, depth)
    gearlib.connect_closed(faces, ob, ot)
    gearlib.connect_closed(faces, ib, it, reverse=True)
    gearlib.annulus_cap(faces, ob, ib, reverse=True)
    # Solid top face and the downward-facing ceiling of the blind socket.
    top_center = len(vertices)
    vertices.append([0.0, 0.0, height])
    for index in range(samples):
        nxt = (index + 1) % samples
        faces.append([top_center, int(ot[index]), int(ot[nxt])])
    socket_center = len(vertices)
    vertices.append([0.0, 0.0, depth])
    for index in range(samples):
        nxt = (index + 1) % samples
        faces.append([socket_center, int(it[nxt]), int(it[index])])
    return gearlib.finish_mesh(vertices, faces, name)


def translated(mesh: trimesh.Trimesh, xyz: tuple[float, float, float]) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_translation(xyz)
    return result


def socket_up(mesh: trimesh.Trimesh, height: float) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    result.apply_translation((0.0, 0.0, height))
    return result


def build() -> Path:
    params: dict[str, Any] = {
        "module_mm": 1.5,
        "pressure_angle_deg": 20.0,
        "sun_teeth": 18,
        "planet_teeth": 27,
        "ring_teeth": 72,
        "planet_count": 3,
        "gear_face_width_mm": 8.0,
        "nominal_pair_backlash_mm": 0.30,
        "sun_bore_mm": 8.2,
        "planet_bore_mm": 5.3,
        "ring_outer_mm": 122.0,
        "carrier_outer_mm": 90.0,
        "carrier_thickness_mm": 3.0,
        "carrier_center_bore_mm": 8.4,
        "carrier_pin_hole_mm": 5.4,
        "planet_pin_shaft_mm": 5.0,
        "planet_pin_shaft_length_mm": 15.0,
        "planet_pin_head_mm": 9.0,
        "planet_pin_head_thickness_mm": 2.0,
        "planet_spacer_thickness_mm": 0.6,
        "samples_per_tooth": 32,
        "physical_coupon_source": "RUN-20260904-151500-67ae54 / COUPON_PHYSICAL_MESH_PASS_BY_USER",
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{stamp}-{signature}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    allowance = params["nominal_pair_backlash_mm"] / 2.0
    st, sr, _ = gearlib.sampled_external_radius(params["sun_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    pt, pr, _ = gearlib.sampled_external_radius(params["planet_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    rt, rr, _ = gearlib.sampled_internal_radius(params["ring_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    sun = gearlib.extrude_radial_annulus(st, sr, params["sun_bore_mm"] / 2.0, params["gear_face_width_mm"], "sun")
    planet = gearlib.extrude_radial_annulus(pt, pr, params["planet_bore_mm"] / 2.0, params["gear_face_width_mm"], "planet")
    ring = gearlib.extrude_radial_annulus(rt, params["ring_outer_mm"] / 2.0, rr, params["gear_face_width_mm"], "ring")

    center_radius = params["module_mm"] * (params["sun_teeth"] + params["planet_teeth"]) / 2.0
    holes = [(0.0, 0.0, params["carrier_center_bore_mm"] / 2.0)]
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        holes.append((center_radius * math.cos(angle), center_radius * math.sin(angle), params["carrier_pin_hole_mm"] / 2.0))
    carrier = triangulated_plate_with_holes(params["carrier_outer_mm"] / 2.0, holes, params["carrier_thickness_mm"], "carrier")
    pin = revolved_solid([
        (params["planet_pin_head_mm"] / 2.0, 0.0),
        (params["planet_pin_head_mm"] / 2.0, params["planet_pin_head_thickness_mm"]),
        (params["planet_pin_shaft_mm"] / 2.0, params["planet_pin_head_thickness_mm"]),
        (params["planet_pin_shaft_mm"] / 2.0, params["planet_pin_head_thickness_mm"] + params["planet_pin_shaft_length_mm"]),
    ], "planet_pin")
    pin_cap = blind_cap(9.0, 5.2, 5.0, 3.8, "planet_pin_cap")
    washer_theta = np.linspace(0.0, 2.0 * math.pi, 192, endpoint=False)
    washer = gearlib.extrude_radial_annulus(washer_theta, 4.5, params["planet_bore_mm"] / 2.0, params["planet_spacer_thickness_mm"], "planet_spacer")
    sun_handle = revolved_solid([(11.0, 0.0), (11.0, 4.0), (4.0, 4.0), (4.0, 19.0)], "sun_handle")
    sun_cap = blind_cap(13.0, 8.2, 6.0, 4.5, "sun_cap")

    parts = {
        "ring": ring,
        "sun": sun,
        "planet": planet,
        "carrier": carrier,
        "planet_pin": pin,
        "planet_pin_cap": pin_cap,
        "planet_spacer": washer,
        "sun_handle": sun_handle,
        "sun_cap": sun_cap,
    }
    files: dict[str, dict[str, str]] = {}
    for key, mesh in parts.items():
        path = run_dir / f"ngw_{key}_v1.stl"
        mesh.export(path)
        files[key] = {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": gearlib.sha256(path)}

    # Plate A: large gears, all kept as separate shells with generous spacing.
    plate_a_items = [translated(ring, (-60.0, 0.0, 0.0)), translated(sun, (85.0, 80.0, 0.0))]
    for y in (-55.0, 0.0, 55.0):
        plate_a_items.append(translated(planet, (40.0, y, 0.0)))
    plate_a = trimesh.util.concatenate(plate_a_items)
    plate_a_path = run_dir / "ngw_same_plate_A_gears_v1.stl"
    plate_a.export(plate_a_path)
    files["same_plate_A_gears"] = {"path": str(plate_a_path.relative_to(PROJECT_ROOT)), "sha256": gearlib.sha256(plate_a_path)}

    # Plate B: carrier and repeated small hardware.
    plate_b_items = [
        translated(carrier, (-45.0, 0.0, 0.0)),
        translated(sun_handle, (20.0, -25.0, 0.0)),
        translated(socket_up(sun_cap, 6.0), (20.0, 10.0, 0.0)),
    ]
    for index, y in enumerate((-30.0, 0.0, 30.0)):
        plate_b_items.extend([
            translated(pin, (48.0, y, 0.0)),
            translated(socket_up(pin_cap, 5.0), (65.0, y, 0.0)),
            translated(washer, (80.0, y, 0.0)),
        ])
    plate_b = trimesh.util.concatenate(plate_b_items)
    plate_b_path = run_dir / "ngw_same_plate_B_carrier_hardware_v1.stl"
    plate_b.export(plate_b_path)
    files["same_plate_B_hardware"] = {"path": str(plate_b_path.relative_to(PROJECT_ROOT)), "sha256": gearlib.sha256(plate_b_path)}

    scene = trimesh.Scene()
    scene.add_geometry(ring.copy(), node_name="ring")
    scene.add_geometry(translated(carrier, (0.0, 0.0, -3.6)), node_name="carrier")
    scene.add_geometry(sun.copy(), node_name="sun")
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        item = planet.copy()
        transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        transform[:3, 3] = [center_radius * math.cos(angle), center_radius * math.sin(angle), 0.0]
        item.apply_transform(transform)
        scene.add_geometry(item, node_name=f"planet_{index + 1}")
    preview = run_dir / "ngw_complete_assembly_preview_v1.glb"
    scene.export(preview)
    files["assembly_preview"] = {"path": str(preview.relative_to(PROJECT_ROOT)), "sha256": gearlib.sha256(preview)}

    validations = {key: gearlib.validate_mesh(mesh) for key, mesh in parts.items()}
    report = {
        "run_id": run_id,
        "status": "WAITING_BAMBU_SLICER_REVIEW_AND_FULL_ASSEMBLY_PRINT",
        "scope": "LOW_SPEED_HAND_TURNED_DEMONSTRATOR_ONLY",
        "parameters": params,
        "equation_checks": {
            "tooth_relation": params["ring_teeth"] == params["sun_teeth"] + 2 * params["planet_teeth"],
            "three_planet_phase": (params["sun_teeth"] + params["ring_teeth"]) % 3 == 0,
            "fixed_ring_ratio": 1.0 + params["ring_teeth"] / params["sun_teeth"],
            "planet_center_radius_mm": center_radius,
        },
        "part_validation": validations,
        "plate_validation": {
            "A": gearlib.validate_mesh(plate_a),
            "B": gearlib.validate_mesh(plate_b),
        },
        "files": files,
        "assembly_notes": [
            "Use three copies each of planet, planet pin, pin cap, and spacer.",
            "Place the carrier behind the gears; put one 0.6 mm spacer between each planet and carrier.",
            "Insert each pin from the gear side and retain it behind the carrier with a cap.",
            "The sun handle passes through the sun and carrier center; retain it behind with the sun cap.",
            "Hold the ring fixed and turn the sun handle slowly by hand only.",
        ],
        "gates": [
            "Inspect both combined STL files in Bambu Studio and split to objects if desired.",
            "Verify no two shells overlap and every part is flat on the bed.",
            "Do not send to printer until layer preview is checked.",
            "Stop assembly if pins or caps require force; revise fit instead.",
        ],
    }
    report_path = run_dir / "ngw_complete_design_and_validation_v1.json"
    write_json(report_path, report)
    instructions = run_dir / "README_完整样机打印与装配.md"
    instructions.write_text(
        "# NGW 18/27/72 完整低速演示样机\n\n"
        "试片已由用户确认通过；当前仍需 Bambu Studio 切片检查和完整实体装配验证。\n\n"
        "建议先打印 `ngw_same_plate_A_gears_v1.stl`，确认完整内齿圈、太阳轮和三个行星轮。"
        "再打印 `ngw_same_plate_B_carrier_hardware_v1.stl`。两个同板文件都包含多个彼此分开的实体。\n\n"
        "装配时每个行星轮与行星架之间放一个 0.6 mm 垫片；销轴从齿轮正面穿入，背面用小帽固定。"
        "太阳轮手柄穿过太阳轮和行星架中心，背面用大帽固定。固定内齿圈后仅用手缓慢旋转太阳轮。\n\n"
        "这是教学演示件，不承载、不接电机、不用于机械臂关节。若任何孔轴或固定帽需要强压，立即停止并调整参数。\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "status": report["status"],
        "report": {"path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": gearlib.sha256(report_path)},
        "instructions": {"path": str(instructions.relative_to(PROJECT_ROOT)), "sha256": gearlib.sha256(instructions)},
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


if __name__ == "__main__":
    print(build())
