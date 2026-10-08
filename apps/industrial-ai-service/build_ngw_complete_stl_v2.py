"""Build the complete image-referenced NGW 18/27/72 print candidate.

The carrier hole is locked at 5.6 mm from a physical PLA coupon.  The script
exports six unique STL types and two A1-oriented combined plates representing
ten physical parts.  Nothing is sent to a printer.
"""

from __future__ import annotations

import os

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Delaunay
from skimage import measure


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_ngw_complete_demo as completelib  # noqa: E402
import build_ngw_image_reference_review as reviewlib  # noqa: E402
import build_ngw_planetary_demo as gearlib  # noqa: E402
import build_ngw_watertight_and_pin_coupon as watertightlib  # noqa: E402


RED = reviewlib.RED
BLUE = reviewlib.BLUE
YELLOW = reviewlib.YELLOW
GREEN = reviewlib.GREEN


def inside_outer(points: np.ndarray, params: dict[str, Any], center_radius: float) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    result = x * x + y * y <= params["carrier_center_hub_radius_mm"] ** 2 + 1e-9
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        ex = center_radius * math.cos(angle)
        ey = center_radius * math.sin(angle)
        result |= watertightlib.distance_to_segment_squared(x, y, ex, ey) <= (params["carrier_arm_width_mm"] / 2.0) ** 2 + 1e-9
        result |= (x - ex) ** 2 + (y - ey) ** 2 <= params["carrier_planet_hub_radius_mm"] ** 2 + 1e-9
    return result


def inside_holes(points: np.ndarray, params: dict[str, Any], center_radius: float) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    result = x * x + y * y < params["carrier_center_interface_radius_mm"] ** 2 - 1e-8
    pin_radius = params["carrier_pin_hole_diameter_mm"] / 2.0
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        ex = center_radius * math.cos(angle)
        ey = center_radius * math.sin(angle)
        result |= (x - ex) ** 2 + (y - ey) ** 2 < pin_radius**2 - 1e-8
    return result


def analytic_carrier(params: dict[str, Any], center_radius: float) -> trimesh.Trimesh:
    points: list[list[float]] = []
    for radius, samples in [
        (params["carrier_center_hub_radius_mm"], 240),
        (params["carrier_center_interface_radius_mm"], 192),
    ]:
        for angle in np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False):
            points.append([radius * math.cos(angle), radius * math.sin(angle)])
    pin_radius = params["carrier_pin_hole_diameter_mm"] / 2.0
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        ex = center_radius * math.cos(angle)
        ey = center_radius * math.sin(angle)
        for radius, samples in [(params["carrier_planet_hub_radius_mm"], 192), (pin_radius, 144)]:
            for local in np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False):
                points.append([ex + radius * math.cos(local), ey + radius * math.sin(local)])
        nx, ny = -math.sin(angle), math.cos(angle)
        for t in np.linspace(0.0, 1.0, 100):
            for sign in (-1.0, 1.0):
                points.append([
                    t * ex + sign * nx * params["carrier_arm_width_mm"] / 2.0,
                    t * ey + sign * ny * params["carrier_arm_width_mm"] / 2.0,
                ])
    limit = center_radius + params["carrier_planet_hub_radius_mm"]
    for x in np.arange(-limit, limit + 0.01, 0.9):
        for y in np.arange(-limit, limit + 0.01, 0.9):
            points.append([float(x), float(y)])
    array = np.unique(np.round(np.asarray(points, dtype=float), 8), axis=0)
    triangulation = Delaunay(array)
    kept: list[list[int]] = []
    for raw in triangulation.simplices:
        triangle = array[raw]
        probes = np.vstack(
            [
                triangle,
                triangle.mean(axis=0),
                (triangle[0] + triangle[1]) / 2.0,
                (triangle[1] + triangle[2]) / 2.0,
                (triangle[2] + triangle[0]) / 2.0,
            ]
        )
        if not np.all(inside_outer(probes, params, center_radius)):
            continue
        if np.any(inside_holes(probes, params, center_radius)):
            continue
        indices = [int(value) for value in raw]
        a, b, c = array[indices]
        if np.cross(b - a, c - a) < 0:
            indices[1], indices[2] = indices[2], indices[1]
        kept.append(indices)
    if not kept:
        raise RuntimeError("Carrier triangulation produced no faces")

    edge_counts: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for a, b, c in kept:
        for start, end in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((start, end)))
            count, _ = edge_counts.get(key, (0, (start, end)))
            edge_counts[key] = (count + 1, (start, end))
    boundary_edges = [edge for count, edge in edge_counts.values() if count == 1]

    z0 = params["carrier_z0_mm"]
    z1 = z0 + params["carrier_thickness_mm"]
    count = len(array)
    vertices = np.vstack(
        [np.column_stack([array, np.full(count, z0)]), np.column_stack([array, np.full(count, z1)])]
    ).tolist()
    faces: list[list[int]] = []
    for a, b, c in kept:
        faces.append([c, b, a])
        faces.append([a + count, b + count, c + count])
    interface_radius = params["carrier_center_interface_radius_mm"]
    for start, end in boundary_edges:
        p0, p1 = array[start], array[end]
        radii = np.linalg.norm(np.vstack([p0, p1]), axis=1)
        is_center_interface = np.all(np.abs(radii - interface_radius) < 1e-5)
        if is_center_interface:
            continue
        faces.extend([[start, end, end + count], [start, end + count, start + count]])

    theta = np.linspace(0.0, 2.0 * math.pi, 192, endpoint=False)
    interface_xy = gearlib.xy_from_polar(theta, interface_radius)
    shaft_radius = params["carrier_output_shaft_diameter_mm"] / 2.0
    shaft_xy = gearlib.xy_from_polar(theta, shaft_radius)
    interface_bottom = gearlib.append_ring(vertices, interface_xy, z0)
    interface_top = gearlib.append_ring(vertices, interface_xy, z1)
    shaft_bottom = gearlib.append_ring(vertices, shaft_xy, z1)
    shaft_top = gearlib.append_ring(vertices, shaft_xy, z1 + params["carrier_output_shaft_length_mm"])
    bottom_center = len(vertices)
    vertices.append([0.0, 0.0, z0])
    top_center = len(vertices)
    vertices.append([0.0, 0.0, z1 + params["carrier_output_shaft_length_mm"]])
    for index in range(len(theta)):
        nxt = (index + 1) % len(theta)
        faces.append([bottom_center, int(interface_bottom[nxt]), int(interface_bottom[index])])
        faces.extend(
            [
                [int(interface_top[index]), int(interface_top[nxt]), int(shaft_bottom[nxt])],
                [int(interface_top[index]), int(shaft_bottom[nxt]), int(shaft_bottom[index])],
            ]
        )
        faces.extend(
            [
                [int(shaft_bottom[index]), int(shaft_bottom[nxt]), int(shaft_top[nxt])],
                [int(shaft_bottom[index]), int(shaft_top[nxt]), int(shaft_top[index])],
            ]
        )
        faces.append([top_center, int(shaft_top[index]), int(shaft_top[nxt])])
    mesh = gearlib.finish_mesh(vertices, faces, "analytic_carrier_output_5p6_holes")
    mesh.units = "mm"
    return mesh


def sdf_carrier(params: dict[str, Any], center_radius: float) -> trimesh.Trimesh:
    """Create one watertight carrier from a continuous signed-distance field."""
    pitch = params["carrier_sdf_pitch_mm"]
    limit = center_radius + params["carrier_planet_hub_radius_mm"] + 2.0 * pitch
    z0 = params["carrier_z0_mm"]
    z1 = z0 + params["carrier_thickness_mm"]
    zend = z1 + params["carrier_output_shaft_length_mm"]
    xs = np.arange(-limit, limit + pitch * 0.5, pitch)
    ys = np.arange(-limit, limit + pitch * 0.5, pitch)
    zs = np.arange(z0 - 2.0 * pitch, zend + 2.0 * pitch, pitch)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    outer = params["carrier_center_hub_radius_mm"] - np.sqrt(x * x + y * y)
    hole_clearance = np.full_like(outer, np.inf)
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        ex = center_radius * math.cos(angle)
        ey = center_radius * math.sin(angle)
        arm = params["carrier_arm_width_mm"] / 2.0 - np.sqrt(
            watertightlib.distance_to_segment_squared(x, y, ex, ey)
        )
        hub = params["carrier_planet_hub_radius_mm"] - np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
        outer = np.maximum(outer, np.maximum(arm, hub))
        hole_clearance = np.minimum(
            hole_clearance,
            np.sqrt((x - ex) ** 2 + (y - ey) ** 2) - params["carrier_pin_hole_diameter_mm"] / 2.0,
        )
    radial = np.sqrt(x * x + y * y)
    field = np.empty((len(xs), len(ys), len(zs)), dtype=np.float32)
    for index, z in enumerate(zs):
        plate = np.minimum(np.minimum(outer, z - z0), z1 - z)
        plate = np.minimum(plate, hole_clearance)
        shaft = np.minimum(
            np.minimum(params["carrier_output_shaft_diameter_mm"] / 2.0 - radial, z - (z1 - 0.5)),
            zend - z,
        )
        field[:, :, index] = np.maximum(plate, shaft).astype(np.float32)
    # Avoid extracting exactly on the z-grid planes.  A zero level coincides
    # with the flat plate/shaft caps and creates thousands of zero-area STL
    # triangles after export.  The tiny positive iso offset is only 0.01 mm.
    vertices, faces, _, _ = measure.marching_cubes(field, level=0.01, spacing=(pitch, pitch, pitch))
    vertices += np.array([xs[0], ys[0], zs[0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh.metadata["name"] = "sdf_carrier_output_5p6_holes"
    mesh.units = "mm"
    return mesh


def measured_carrier_holes(mesh: trimesh.Trimesh, center_radius: float) -> list[dict[str, float]]:
    section = mesh.section(
        plane_origin=(0.0, 0.0, float(mesh.bounds[0, 2] + 1.5)),
        plane_normal=(0.0, 0.0, 1.0),
    )
    if section is None:
        raise RuntimeError("Carrier mid-plane section is empty")
    loops = [np.asarray(loop)[:, :2] for loop in section.discrete if len(loop) >= 8]
    records: list[dict[str, float]] = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        center = np.array([center_radius * math.cos(angle), center_radius * math.sin(angle)])
        loop = min(loops, key=lambda item: float(np.linalg.norm(item.mean(axis=0) - center)))
        extents = loop.max(axis=0) - loop.min(axis=0)
        records.append(
            {
                "index": index + 1,
                "diameter_x_mm": float(extents[0]),
                "diameter_y_mm": float(extents[1]),
                "mean_diameter_mm": float(extents.mean()),
            }
        )
    return records


def print_oriented(mesh: trimesh.Trimesh, flip: bool = False) -> trimesh.Trimesh:
    result = mesh.copy()
    if flip:
        result.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (1.0, 0.0, 0.0)))
    result.apply_translation((0.0, 0.0, -float(result.bounds[0, 2])))
    result.units = "mm"
    return result


def placed(mesh: trimesh.Trimesh, x: float, y: float) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_translation((x, y, 0.0))
    return result


def ensure_aabb_separation(items: list[tuple[str, trimesh.Trimesh]], minimum_gap: float = 1.0) -> None:
    for first in range(len(items)):
        name_a, mesh_a = items[first]
        for second in range(first + 1, len(items)):
            name_b, mesh_b = items[second]
            overlap_x = min(mesh_a.bounds[1, 0], mesh_b.bounds[1, 0]) - max(mesh_a.bounds[0, 0], mesh_b.bounds[0, 0])
            overlap_y = min(mesh_a.bounds[1, 1], mesh_b.bounds[1, 1]) - max(mesh_a.bounds[0, 1], mesh_b.bounds[0, 1])
            if overlap_x > -minimum_gap and overlap_y > -minimum_gap:
                raise RuntimeError(f"Print plate AABB gap too small: {name_a} vs {name_b}")


def export_checked(mesh: trimesh.Trimesh, path: Path, expected_components: int) -> dict[str, Any]:
    mesh.export(path)
    reloaded = trimesh.load(path, force="mesh", process=True)
    reloaded.merge_vertices()
    parts = reloaded.split(only_watertight=False)
    metrics = reviewlib.mesh_metrics(reloaded)
    component_metrics = [reviewlib.mesh_metrics(part) for part in parts]
    if len(parts) != expected_components:
        raise RuntimeError(f"{path.name}: expected {expected_components} components, got {len(parts)}")
    if not all(item["watertight"] and item["winding_consistent"] and item["positive_volume"] for item in component_metrics):
        raise RuntimeError(f"{path.name}: invalid component")
    return {"mesh": metrics, "component_checks": component_metrics, "sha256": reviewlib.sha256(path)}


def plate_preview(
    plate_a: trimesh.Trimesh,
    plate_b: trimesh.Trimesh,
    output: Path,
) -> None:
    a = reviewlib.render_items([("plate_A", plate_a, RED)], 0, 0, "打印板A：红杯体、太阳轮、D轴、3个行星轮", size=(900, 620))
    b = reviewlib.render_items([("plate_B", plate_b, GREEN)], 0, 0, "打印板B：行星架输出件、3根阶梯销", size=(900, 620))
    board = Image.new("RGB", (1800, 690), "white")
    board.paste(a, (0, 0))
    board.paste(b, (900, 0))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype(str(reviewlib.FONT_BOLD), 23)
    draw.text((22, 638), "两板均为毫米单位、平放/头部朝下的免支撑候选；必须在Bambu Studio逐层确认后才能打印。", fill=(30, 30, 30), font=font)
    board.save(output)


def build() -> Path:
    params: dict[str, Any] = {
        "module_mm": 1.5,
        "pressure_angle_deg": 20.0,
        "sun_teeth": 18,
        "planet_teeth": 27,
        "ring_teeth": 72,
        "planet_count": 3,
        "nominal_pair_backlash_mm": 0.30,
        "ring_outer_diameter_mm": 122.0,
        "ring_tooth_width_mm": 9.2,
        "rear_plate_thickness_mm": 3.0,
        "rear_plate_center_bore_mm": 9.0,
        "gear_face_width_mm": 8.0,
        "gear_z0_mm": 0.6,
        "planet_bore_diameter_mm": 5.3,
        "sun_input_shaft_diameter_mm": 8.0,
        "sun_input_d_flat_depth_mm": 1.0,
        "sun_bore_diametral_clearance_mm": 0.30,
        "sun_input_rear_z_mm": -13.0,
        "sun_input_flange_diameter_mm": 16.0,
        "sun_input_flange_thickness_mm": 3.0,
        "carrier_z0_mm": 9.2,
        "carrier_thickness_mm": 3.0,
        "carrier_arm_width_mm": 10.0,
        "carrier_planet_hub_radius_mm": 7.0,
        "carrier_center_hub_radius_mm": 9.0,
        "carrier_center_interface_radius_mm": 8.0,
        "carrier_pin_hole_diameter_mm": 5.6,
        "carrier_sdf_pitch_mm": 0.20,
        "carrier_output_shaft_diameter_mm": 9.0,
        "carrier_output_shaft_length_mm": 34.0,
        "planet_pin_journal_diameter_mm": 5.0,
        "planet_pin_press_diameter_mm": 5.3,
        "planet_pin_head_diameter_mm": 9.0,
        "planet_pin_rear_z_mm": 0.3,
        "planet_pin_front_z_mm": 14.2,
        "samples_per_tooth": 32,
        "pin_hole_physical_selection": "RUN-20260905-112030-d6110d / 5.6 mm hole accepted",
        "gear_fit_evidence": "RUN-20260904-151500-67ae54 / COUPON_PHYSICAL_MESH_PASS_BY_USER",
    }
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{signature}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    unique_dir = run_dir / "individual_stl"
    plate_dir = run_dir / "combined_plates"
    unique_dir.mkdir()
    plate_dir.mkdir()

    allowance = params["nominal_pair_backlash_mm"] / 2.0
    st, sr, _ = gearlib.sampled_external_radius(params["sun_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    pt, pr, _ = gearlib.sampled_external_radius(params["planet_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    rt, rr, _ = gearlib.sampled_internal_radius(params["ring_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    shaft_radius = params["sun_input_shaft_diameter_mm"] / 2.0
    sun = gearlib.extrude_radial_annulus(
        st,
        sr,
        reviewlib.d_profile(st, shaft_radius + params["sun_bore_diametral_clearance_mm"] / 2.0, params["sun_input_d_flat_depth_mm"]),
        params["gear_face_width_mm"],
        "sun_18t_d_bore",
    )
    planet = gearlib.extrude_radial_annulus(pt, pr, params["planet_bore_diameter_mm"] / 2.0, params["gear_face_width_mm"], "planet_27t")
    sun.apply_translation((0.0, 0.0, params["gear_z0_mm"]))
    planet.apply_translation((0.0, 0.0, params["gear_z0_mm"]))
    red_cup = watertightlib.ring_cup(
        rt, rr, params["ring_outer_diameter_mm"] / 2.0, params["rear_plate_center_bore_mm"] / 2.0,
        -params["rear_plate_thickness_mm"], params["ring_tooth_width_mm"]
    )
    sun_shaft = watertightlib.d_flanged_shaft(params)
    center_radius = params["module_mm"] * (params["sun_teeth"] + params["planet_teeth"]) / 2.0
    carrier = sdf_carrier(params, center_radius)
    carrier_hole_measurements = measured_carrier_holes(carrier, center_radius)
    if any(abs(item["mean_diameter_mm"] - params["carrier_pin_hole_diameter_mm"]) > 0.12 for item in carrier_hole_measurements):
        raise RuntimeError(f"Carrier hole SDF deviation too large: {carrier_hole_measurements}")
    pin = completelib.revolved_solid(
        [
            (params["planet_pin_journal_diameter_mm"] / 2.0, params["planet_pin_rear_z_mm"]),
            (params["planet_pin_journal_diameter_mm"] / 2.0, params["carrier_z0_mm"]),
            (params["planet_pin_press_diameter_mm"] / 2.0, params["carrier_z0_mm"]),
            (params["planet_pin_press_diameter_mm"] / 2.0, params["carrier_z0_mm"] + params["carrier_thickness_mm"]),
            (params["planet_pin_head_diameter_mm"] / 2.0, params["carrier_z0_mm"] + params["carrier_thickness_mm"]),
            (params["planet_pin_head_diameter_mm"] / 2.0, params["planet_pin_front_z_mm"]),
        ],
        "planet_step_pin",
    )

    assembly_planets: list[trimesh.Trimesh] = []
    assembly_pins: list[trimesh.Trimesh] = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        x, y = center_radius * math.cos(angle), center_radius * math.sin(angle)
        gear = planet.copy()
        transform = trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0))
        transform[:3, 3] = (x, y, 0.0)
        gear.apply_transform(transform)
        assembly_planets.append(gear)
        assembly_pins.append(reviewlib.translated(pin, (x, y, 0.0)))

    assembled = [("red_ring_cup", red_cup, RED), ("yellow_sun", sun, YELLOW), ("yellow_input", sun_shaft, YELLOW)]
    assembled.extend((f"blue_planet_{i+1}", mesh, BLUE) for i, mesh in enumerate(assembly_planets))
    assembled.append(("green_carrier", carrier, GREEN))
    assembled.extend((f"green_pin_{i+1}", mesh, GREEN) for i, mesh in enumerate(assembly_pins))
    assembly_scene = trimesh.Scene()
    for name, mesh, rgba in assembled:
        reviewlib.add_scene_geometry(assembly_scene, name, mesh, rgba)
    assembly_glb = run_dir / "ngw_complete_v2_assembly_review_meters.glb"
    assembly_scene.export(assembly_glb)

    print_parts = {
        "01_red_ring_cup_v2.stl": print_oriented(red_cup),
        "02_yellow_sun_18t_v2.stl": print_oriented(sun),
        "03_yellow_d_input_shaft_v2.stl": print_oriented(sun_shaft),
        "04_blue_planet_27t_v2_PRINT_3.stl": print_oriented(planet),
        "05_green_carrier_output_5p6_v2.stl": print_oriented(carrier),
        "06_green_step_pin_5p3_v2_PRINT_3.stl": print_oriented(pin, flip=True),
    }
    individual_checks: dict[str, Any] = {}
    for filename, mesh in print_parts.items():
        individual_checks[filename] = export_checked(mesh, unique_dir / filename, 1)

    plate_a_items = [
        ("red", placed(print_parts["01_red_ring_cup_v2.stl"], -60.0, 0.0)),
        ("planet_1", placed(print_parts["04_blue_planet_27t_v2_PRINT_3.stl"], 45.0, -55.0)),
        ("planet_2", placed(print_parts["04_blue_planet_27t_v2_PRINT_3.stl"], 45.0, 0.0)),
        ("planet_3", placed(print_parts["04_blue_planet_27t_v2_PRINT_3.stl"], 45.0, 55.0)),
        ("sun", placed(print_parts["02_yellow_sun_18t_v2.stl"], 95.0, -27.0)),
        ("sun_shaft", placed(print_parts["03_yellow_d_input_shaft_v2.stl"], 95.0, 27.0)),
    ]
    ensure_aabb_separation(plate_a_items, 2.0)
    plate_a = trimesh.util.concatenate([mesh for _, mesh in plate_a_items])
    plate_a_path = plate_dir / "ngw_complete_v2_plate_A_red_yellow_blue.stl"
    plate_a_check = export_checked(plate_a, plate_a_path, 6)

    plate_b_items = [
        ("carrier", placed(print_parts["05_green_carrier_output_5p6_v2.stl"], -28.0, 0.0)),
        ("pin_1", placed(print_parts["06_green_step_pin_5p3_v2_PRINT_3.stl"], 32.0, -25.0)),
        ("pin_2", placed(print_parts["06_green_step_pin_5p3_v2_PRINT_3.stl"], 32.0, 0.0)),
        ("pin_3", placed(print_parts["06_green_step_pin_5p3_v2_PRINT_3.stl"], 32.0, 25.0)),
    ]
    ensure_aabb_separation(plate_b_items, 2.0)
    plate_b = trimesh.util.concatenate([mesh for _, mesh in plate_b_items])
    plate_b_path = plate_dir / "ngw_complete_v2_plate_B_green_carrier_pins.stl"
    plate_b_check = export_checked(plate_b, plate_b_path, 4)

    for name, check in (("A", plate_a_check), ("B", plate_b_check)):
        extents = np.asarray(check["mesh"]["extents_mm"])
        if extents[0] > 256.0 or extents[1] > 256.0:
            raise RuntimeError(f"Plate {name} exceeds A1 XY area: {extents}")
        if abs(check["mesh"]["bounds_mm"][0][2]) > 1e-5:
            raise RuntimeError(f"Plate {name} is not aligned to Z=0")

    plate_preview_path = run_dir / "ngw_complete_v2_plate_layout_preview.png"
    plate_preview(plate_a, plate_b, plate_preview_path)
    assembly_preview_path = run_dir / "ngw_complete_v2_assembly_fourview.png"
    exploded = [(name, reviewlib.translated(mesh, (0.0, 0.0, index * 8.0)), rgba) for index, (name, mesh, rgba) in enumerate(assembled)]
    from build_ngw_printable_structure_review import write_review_board

    write_review_board(
        assembled,
        exploded,
        assembly_preview_path,
        exploded_title="制造分件展开：10件，STL候选已生成",
    )

    report = {
        "run_id": run_id,
        "status": "FULL_10_PART_STL_CANDIDATE_WAITING_BAMBU_SLICER_REVIEW",
        "parameters": params,
        "physical_fit_selection": {
            "selected_carrier_hole_mm": 5.6,
            "pin_press_section_mm": 5.3,
            "result": "inserts fully, removable with resistance, no obvious lateral wobble",
        },
        "carrier_hole_section_measurements_mm": carrier_hole_measurements,
        "bill_of_materials": [
            {"file": "01_red_ring_cup_v2.stl", "quantity": 1},
            {"file": "02_yellow_sun_18t_v2.stl", "quantity": 1},
            {"file": "03_yellow_d_input_shaft_v2.stl", "quantity": 1},
            {"file": "04_blue_planet_27t_v2_PRINT_3.stl", "quantity": 3},
            {"file": "05_green_carrier_output_5p6_v2.stl", "quantity": 1},
            {"file": "06_green_step_pin_5p3_v2_PRINT_3.stl", "quantity": 3},
        ],
        "individual_stl_checks": individual_checks,
        "combined_plate_checks": {"A": plate_a_check, "B": plate_b_check},
        "files": {
            "assembly_glb": {"path": str(assembly_glb.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(assembly_glb)},
            "assembly_preview": {"path": str(assembly_preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(assembly_preview_path)},
            "plate_layout_preview": {"path": str(plate_preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(plate_preview_path)},
            "plate_A": {"path": str(plate_a_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(plate_a_path)},
            "plate_B": {"path": str(plate_b_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(plate_b_path)},
        },
        "remaining_gates": [
            "Open both combined plates in Bambu Studio as millimetres.",
            "Confirm plate A has 6 objects and plate B has 4 objects.",
            "Use A1, 0.4 mm nozzle and PLA; do not auto-arrange or rescale.",
            "Inspect first layer, internal teeth, D bore, 5.6 mm carrier holes and every layer before printing.",
            "Do not print if the slicer repairs geometry, merges adjacent pieces, adds unexpected support, or reports floating regions.",
        ],
    }
    report_path = run_dir / "ngw_complete_v2_design_and_validation.json"
    reviewlib.write_json(report_path, report)
    guide_path = run_dir / "README_Bambu检查后再打印.md"
    guide_path.write_text(
        "# NGW 18/27/72 完整 10 件套 V2\n\n"
        "当前只是完整 STL 候选，尚未通过 Bambu Studio 逐层检查。不要直接发送打印。\n\n"
        "先导入 `combined_plates/ngw_complete_v2_plate_A_red_yellow_blue.stl`，应识别 6 个实体：红杯体、太阳轮、D输入轴和3个行星轮。"
        "再导入 `combined_plates/ngw_complete_v2_plate_B_green_carrier_pins.stl`，应识别 4 个实体：行星架输出件和3根阶梯销。\n\n"
        "设置 A1、0.4 mm 喷嘴、PLA、毫米单位；保持文件现有朝向，不自动缩放或重新排布，默认不需要支撑。"
        "逐层检查首层、内齿、太阳轮D孔、行星架3个5.6 mm孔、销轴以及是否存在悬空或切片修复提示。\n\n"
        "两板均确认无误后再打印。装配时不得强压：黄色D轴从红后板背面插入并装太阳轮；放入3个行星轮；"
        "绿色行星架从前侧对准，3根绿销从前侧插入。固定红杯体，缓慢转动黄色输入轴，观察绿色行星架输出。\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "status": report["status"],
        "report": {"path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(report_path)},
        "guide": {"path": str(guide_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(guide_path)},
        **report["files"],
    }
    reviewlib.write_json(run_dir / "manifest.json", manifest)
    return run_dir


if __name__ == "__main__":
    print(build())
