"""Build the corrected image-topology NGW 18/27/72 print candidate.

Axial order, front to rear: integrated yellow sun input and gear plane,
green carrier behind the planets, red rear plate, then the green carrier output
shaft through the rear plate.  The old V2 run is never modified.
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
from skimage import measure


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_ngw_complete_demo as completelib  # noqa: E402
import build_ngw_complete_stl_v2 as v2lib  # noqa: E402
import build_ngw_image_reference_review as reviewlib  # noqa: E402
import build_ngw_planetary_demo as gearlib  # noqa: E402
import build_ngw_watertight_and_pin_coupon as watertightlib  # noqa: E402


RED = reviewlib.RED
BLUE = reviewlib.BLUE
YELLOW = reviewlib.YELLOW
GREEN = reviewlib.GREEN


def stepped_radial_solid(
    theta: np.ndarray,
    profiles: list[tuple[float, np.ndarray | float]],
    name: str,
) -> trimesh.Trimesh:
    """Create one watertight solid from ordered radial profiles."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    rings: list[np.ndarray] = []
    for z, radius in profiles:
        rings.append(gearlib.append_ring(vertices, gearlib.xy_from_polar(theta, radius), z))
    for first, second in zip(rings, rings[1:]):
        gearlib.connect_closed(faces, first, second)
    for ring, z, reverse in ((rings[0], profiles[0][0], True), (rings[-1], profiles[-1][0], False)):
        center = len(vertices)
        vertices.append([0.0, 0.0, float(z)])
        for index in range(len(ring)):
            nxt = (index + 1) % len(ring)
            face = [center, int(ring[index]), int(ring[nxt])]
            faces.append(face[::-1] if reverse else face)
    mesh = gearlib.finish_mesh(vertices, faces, name)
    mesh.units = "mm"
    return mesh


def integrated_sun_input(
    theta: np.ndarray,
    tooth_radius: np.ndarray,
    params: dict[str, Any],
) -> trimesh.Trimesh:
    """One yellow part: front cylinder, 18T sun and rear round pilot."""
    handle_radius = params["sun_front_cylinder_diameter_mm"] / 2.0
    pilot_radius = params["sun_rear_pilot_diameter_mm"] / 2.0
    return stepped_radial_solid(
        theta,
        [
            (-params["sun_front_cylinder_length_mm"], handle_radius),
            (0.0, handle_radius),
            (0.0, tooth_radius),
            (params["gear_face_width_mm"], tooth_radius),
            (params["gear_face_width_mm"], pilot_radius),
            (params["sun_rear_pilot_end_z_mm"], pilot_radius),
        ],
        "yellow_sun_18t_integrated_front_cylinder_and_rear_pilot",
    )


def corrected_red_cup(
    ring_theta: np.ndarray,
    ring_inner_radius: np.ndarray,
    params: dict[str, Any],
) -> trimesh.Trimesh:
    """Mirror the proven analytic cup so its rear plate sits behind carrier."""
    rear_plate_front = params["rear_plate_front_z_mm"]
    base = watertightlib.ring_cup(
        ring_theta,
        ring_inner_radius,
        params["ring_outer_diameter_mm"] / 2.0,
        params["rear_plate_center_bore_mm"] / 2.0,
        -params["rear_plate_thickness_mm"],
        rear_plate_front,
    )
    transform = np.eye(4)
    transform[2, 2] = -1.0
    transform[2, 3] = rear_plate_front
    base.apply_transform(transform)
    trimesh.repair.fix_normals(base, multibody=True)
    base.metadata["name"] = "red_ring_cup_rear_plate_behind_carrier"
    base.units = "mm"
    return base


def corrected_carrier(params: dict[str, Any], center_radius: float) -> trimesh.Trimesh:
    """SDF-union carrier with three pin holes and a front blind pilot socket."""
    pitch = params["carrier_sdf_pitch_mm"]
    limit = center_radius + params["carrier_planet_hub_radius_mm"] + 2.0 * pitch
    z0 = params["carrier_z0_mm"]
    z1 = z0 + params["carrier_thickness_mm"]
    shaft_end = z1 + params["carrier_output_shaft_length_mm"]
    xs = np.arange(-limit, limit + pitch * 0.5, pitch)
    ys = np.arange(-limit, limit + pitch * 0.5, pitch)
    zs = np.arange(z0 - 2.0 * pitch, shaft_end + 2.0 * pitch, pitch)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    radial = np.sqrt(x * x + y * y)

    outer = params["carrier_center_hub_radius_mm"] - radial
    pin_clearance = np.full_like(outer, np.inf)
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        ex = center_radius * math.cos(angle)
        ey = center_radius * math.sin(angle)
        arm = params["carrier_arm_width_mm"] / 2.0 - np.sqrt(
            watertightlib.distance_to_segment_squared(x, y, ex, ey)
        )
        hub = params["carrier_planet_hub_radius_mm"] - np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
        outer = np.maximum(outer, np.maximum(arm, hub))
        pin_clearance = np.minimum(
            pin_clearance,
            np.sqrt((x - ex) ** 2 + (y - ey) ** 2) - params["carrier_pin_hole_diameter_mm"] / 2.0,
        )

    field = np.empty((len(xs), len(ys), len(zs)), dtype=np.float32)
    socket_radius = params["carrier_center_socket_diameter_mm"] / 2.0
    for index, z in enumerate(zs):
        plate = np.minimum(np.minimum(outer, z - z0), z1 - z)
        plate = np.minimum(plate, pin_clearance)
        shaft = np.minimum(
            np.minimum(params["carrier_output_shaft_diameter_mm"] / 2.0 - radial, z - (z1 - 0.5)),
            shaft_end - z,
        )
        value = np.maximum(plate, shaft)
        if z <= params["carrier_center_socket_end_z_mm"]:
            value = np.minimum(value, radial - socket_radius)
        field[:, :, index] = value.astype(np.float32)

    vertices, faces, _, _ = measure.marching_cubes(field, level=0.01, spacing=(pitch, pitch, pitch))
    vertices += np.array([xs[0], ys[0], zs[0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh.metadata["name"] = "green_rear_carrier_output_with_pilot_socket"
    mesh.units = "mm"
    return mesh


def corrected_pin(params: dict[str, Any]) -> trimesh.Trimesh:
    """Head at front, journal through planet, insertion section in rear carrier."""
    return completelib.revolved_solid(
        [
            (params["planet_pin_head_diameter_mm"] / 2.0, params["planet_pin_head_front_z_mm"]),
            (params["planet_pin_head_diameter_mm"] / 2.0, params["planet_pin_head_rear_z_mm"]),
            (params["planet_pin_journal_diameter_mm"] / 2.0, params["planet_pin_head_rear_z_mm"]),
            (params["planet_pin_journal_diameter_mm"] / 2.0, params["carrier_z0_mm"]),
            (params["planet_pin_press_diameter_mm"] / 2.0, params["carrier_z0_mm"]),
            (params["planet_pin_press_diameter_mm"] / 2.0, params["planet_pin_press_end_z_mm"]),
            (params["planet_pin_tip_diameter_mm"] / 2.0, params["planet_pin_tip_end_z_mm"]),
        ],
        "green_front_insert_step_pin",
    )


def write_review_board(
    assembled: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]],
    exploded: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]],
    output: Path,
) -> None:
    panels = [
        reviewlib.render_items(assembled, 152, -16, "修正版正面斜视：按原图颜色"),
        reviewlib.render_items(assembled, 180, 0, "正视：绿架在蓝轮后面"),
        reviewlib.render_items(assembled, 90, 0, "侧视：黄输入在前、绿轴穿红背板"),
        reviewlib.render_items(exploded, 104, -12, "9件轴向展开：本轮已生成STL"),
    ]
    board = Image.new("RGB", (1520, 1320), "white")
    for index, panel in enumerate(panels):
        board.paste(panel, ((index % 2) * 760, (index // 2) * 620))
    draw = ImageDraw.Draw(board)
    title = ImageFont.truetype(str(reviewlib.FONT_BOLD), 26)
    body = ImageFont.truetype(str(reviewlib.FONT), 20)
    draw.text((22, 1243), "前→后：黄色一体输入/齿轮层 → 绿色三臂架 → 红色背板 → 绿色长输出轴", fill=(24, 24, 24), font=title)
    draw.text((22, 1285), "共9件；旧蓝色行星轮×3可复用。太阳轮圆导向柱与行星架圆孔为间隙定位，不传扭。", fill=(75, 50, 20), font=body)
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
        "gear_face_width_mm": 8.0,
        "planet_bore_diameter_mm": 5.3,
        "sun_front_cylinder_diameter_mm": 10.0,
        "sun_front_cylinder_length_mm": 10.0,
        "sun_rear_pilot_diameter_mm": 6.0,
        "sun_rear_pilot_end_z_mm": 10.5,
        "carrier_z0_mm": 8.6,
        "carrier_thickness_mm": 3.0,
        "carrier_arm_width_mm": 10.0,
        "carrier_planet_hub_radius_mm": 7.0,
        "carrier_center_hub_radius_mm": 9.0,
        "carrier_pin_hole_diameter_mm": 5.6,
        "carrier_center_socket_diameter_mm": 6.6,
        "carrier_center_socket_end_z_mm": 10.8,
        "carrier_sdf_pitch_mm": 0.20,
        "carrier_output_shaft_diameter_mm": 9.0,
        "carrier_output_shaft_length_mm": 34.0,
        "rear_plate_front_z_mm": 12.2,
        "rear_plate_thickness_mm": 3.0,
        "rear_plate_center_bore_mm": 9.8,
        "planet_pin_journal_diameter_mm": 5.0,
        "planet_pin_press_diameter_mm": 5.3,
        "planet_pin_head_diameter_mm": 9.0,
        "planet_pin_head_front_z_mm": -2.3,
        "planet_pin_head_rear_z_mm": -0.3,
        "planet_pin_press_end_z_mm": 11.4,
        "planet_pin_tip_diameter_mm": 4.8,
        "planet_pin_tip_end_z_mm": 11.7,
        "samples_per_tooth": 32,
        "source_reference": "workspace/docs/image.png",
        "approved_topology_review": "workspace/docs/NGW原图正确轴向拓扑审查_v1.png",
        "gear_fit_evidence": "RUN-20260904-151500-67ae54 / COUPON_PHYSICAL_MESH_PASS_BY_USER",
        "pin_fit_evidence": "RUN-20260905-112030-d6110d / 5.6 hole with 5.3 pin accepted",
    }
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{signature}"
    run_dir = RUNS_ROOT / run_id
    unique_dir = run_dir / "individual_stl"
    plate_dir = run_dir / "combined_plates"
    unique_dir.mkdir(parents=True, exist_ok=False)
    plate_dir.mkdir()

    allowance = params["nominal_pair_backlash_mm"] / 2.0
    st, sr, _ = gearlib.sampled_external_radius(params["sun_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    pt, pr, _ = gearlib.sampled_external_radius(params["planet_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    rt, rr, _ = gearlib.sampled_internal_radius(params["ring_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    sun = integrated_sun_input(st, sr, params)
    planet = gearlib.extrude_radial_annulus(pt, pr, params["planet_bore_diameter_mm"] / 2.0, params["gear_face_width_mm"], "blue_planet_27t_reusable")
    red_cup = corrected_red_cup(rt, rr, params)
    center_radius = params["module_mm"] * (params["sun_teeth"] + params["planet_teeth"]) / 2.0
    carrier = corrected_carrier(params, center_radius)
    pin = corrected_pin(params)

    carrier_holes = v2lib.measured_carrier_holes(carrier, center_radius)
    if any(abs(item["mean_diameter_mm"] - params["carrier_pin_hole_diameter_mm"]) > 0.12 for item in carrier_holes):
        raise RuntimeError(f"Carrier hole deviation too large: {carrier_holes}")

    planets: list[trimesh.Trimesh] = []
    pins: list[trimesh.Trimesh] = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        x, y = center_radius * math.cos(angle), center_radius * math.sin(angle)
        gear = planet.copy()
        transform = trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0))
        transform[:3, 3] = (x, y, 0.0)
        gear.apply_transform(transform)
        planets.append(gear)
        pins.append(reviewlib.translated(pin, (x, y, 0.0)))

    assembled: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]] = [
        ("red_rear_backed_ring_cup", red_cup, RED),
        ("yellow_integrated_sun_input", sun, YELLOW),
    ]
    assembled.extend((f"blue_planet_{i + 1}", item, BLUE) for i, item in enumerate(planets))
    assembled.append(("green_rear_carrier_output", carrier, GREEN))
    assembled.extend((f"green_front_pin_{i + 1}", item, GREEN) for i, item in enumerate(pins))

    scene = trimesh.Scene()
    for name, mesh, rgba in assembled:
        reviewlib.add_scene_geometry(scene, name, mesh, rgba)
    assembly_glb = run_dir / "ngw_corrected_v3_assembly_review_meters.glb"
    scene.export(assembly_glb)

    exploded_offsets = {
        "red_rear_backed_ring_cup": 30.0,
        "green_rear_carrier_output": 14.0,
        "yellow_integrated_sun_input": -18.0,
    }
    exploded = []
    for name, mesh, rgba in assembled:
        offset = exploded_offsets.get(name, -7.0 if "pin" in name else 0.0)
        exploded.append((name, reviewlib.translated(mesh, (0.0, 0.0, offset)), rgba))
    preview_path = run_dir / "ngw_corrected_v3_assembly_fourview.png"
    write_review_board(assembled, exploded, preview_path)

    print_parts = {
        "01_red_rear_backed_ring_cup_v3.stl": v2lib.print_oriented(red_cup, flip=True),
        "02_yellow_integrated_sun_input_v3_SUPPORT_REQUIRED.stl": v2lib.print_oriented(sun),
        "03_blue_planet_27t_v3_REUSE_OR_PRINT_3.stl": v2lib.print_oriented(planet),
        "04_green_rear_carrier_output_v3.stl": v2lib.print_oriented(carrier),
        "05_green_front_step_pin_v3_PRINT_3.stl": v2lib.print_oriented(pin),
    }
    individual_checks: dict[str, Any] = {}
    for filename, mesh in print_parts.items():
        individual_checks[filename] = v2lib.export_checked(mesh, unique_dir / filename, 1)

    plate_a_items = [
        ("red", v2lib.placed(print_parts["01_red_rear_backed_ring_cup_v3.stl"], -60.0, 0.0)),
        ("planet_1", v2lib.placed(print_parts["03_blue_planet_27t_v3_REUSE_OR_PRINT_3.stl"], 45.0, -55.0)),
        ("planet_2", v2lib.placed(print_parts["03_blue_planet_27t_v3_REUSE_OR_PRINT_3.stl"], 45.0, 0.0)),
        ("planet_3", v2lib.placed(print_parts["03_blue_planet_27t_v3_REUSE_OR_PRINT_3.stl"], 45.0, 55.0)),
    ]
    v2lib.ensure_aabb_separation(plate_a_items, 2.0)
    plate_a = trimesh.util.concatenate([mesh for _, mesh in plate_a_items])
    plate_a_path = plate_dir / "ngw_corrected_v3_plate_A_red_and_optional_blue.stl"
    plate_a_check = v2lib.export_checked(plate_a, plate_a_path, 4)

    plate_b_items = [
        ("carrier", v2lib.placed(print_parts["04_green_rear_carrier_output_v3.stl"], -28.0, 0.0)),
        ("pin_1", v2lib.placed(print_parts["05_green_front_step_pin_v3_PRINT_3.stl"], 32.0, -25.0)),
        ("pin_2", v2lib.placed(print_parts["05_green_front_step_pin_v3_PRINT_3.stl"], 32.0, 0.0)),
        ("pin_3", v2lib.placed(print_parts["05_green_front_step_pin_v3_PRINT_3.stl"], 32.0, 25.0)),
    ]
    v2lib.ensure_aabb_separation(plate_b_items, 2.0)
    plate_b = trimesh.util.concatenate([mesh for _, mesh in plate_b_items])
    plate_b_path = plate_dir / "ngw_corrected_v3_plate_B_green_carrier_and_pins.stl"
    plate_b_check = v2lib.export_checked(plate_b, plate_b_path, 4)

    yellow_path = unique_dir / "02_yellow_integrated_sun_input_v3_SUPPORT_REQUIRED.stl"
    axial_checks = {
        "gear_to_carrier_gap_mm": params["carrier_z0_mm"] - params["gear_face_width_mm"],
        "carrier_to_rear_plate_gap_mm": params["rear_plate_front_z_mm"] - (params["carrier_z0_mm"] + params["carrier_thickness_mm"]),
        "pin_tip_to_rear_plate_gap_mm": params["rear_plate_front_z_mm"] - params["planet_pin_tip_end_z_mm"],
        "output_shaft_to_rear_bore_diametral_clearance_mm": params["rear_plate_center_bore_mm"] - params["carrier_output_shaft_diameter_mm"],
        "sun_pilot_to_carrier_socket_diametral_clearance_mm": params["carrier_center_socket_diameter_mm"] - params["sun_rear_pilot_diameter_mm"],
        "sun_pilot_tip_to_socket_bottom_gap_mm": params["carrier_center_socket_end_z_mm"] - params["sun_rear_pilot_end_z_mm"],
        "carrier_outer_radius_mm": center_radius + params["carrier_planet_hub_radius_mm"],
        "minimum_ring_inner_tooth_radius_mm": float(np.min(rr)),
        "carrier_radial_clearance_to_ring_teeth_mm": float(np.min(rr)) - (center_radius + params["carrier_planet_hub_radius_mm"]),
    }
    if min(axial_checks[key] for key in ("gear_to_carrier_gap_mm", "carrier_to_rear_plate_gap_mm", "pin_tip_to_rear_plate_gap_mm", "sun_pilot_tip_to_socket_bottom_gap_mm")) < 0.29:
        raise RuntimeError(f"Axial clearance too small: {axial_checks}")
    if axial_checks["carrier_radial_clearance_to_ring_teeth_mm"] <= 1.0:
        raise RuntimeError(f"Carrier collides radially with ring: {axial_checks}")

    for name, check in (("A", plate_a_check), ("B", plate_b_check)):
        extents = np.asarray(check["mesh"]["extents_mm"])
        if extents[0] > 256.0 or extents[1] > 256.0:
            raise RuntimeError(f"Plate {name} exceeds A1 area: {extents}")

    report = {
        "run_id": run_id,
        "status": "CORRECTED_9_PART_STL_CANDIDATE_WAITING_BAMBU_REVIEW",
        "parameters": params,
        "axial_and_radial_checks": axial_checks,
        "carrier_hole_section_measurements_mm": carrier_holes,
        "bill_of_materials": [
            {"file": "01_red_rear_backed_ring_cup_v3.stl", "quantity": 1, "reuse": False},
            {"file": "02_yellow_integrated_sun_input_v3_SUPPORT_REQUIRED.stl", "quantity": 1, "reuse": False},
            {"file": "03_blue_planet_27t_v3_REUSE_OR_PRINT_3.stl", "quantity": 3, "reuse": True},
            {"file": "04_green_rear_carrier_output_v3.stl", "quantity": 1, "reuse": False},
            {"file": "05_green_front_step_pin_v3_PRINT_3.stl", "quantity": 3, "reuse": False},
        ],
        "individual_stl_checks": individual_checks,
        "combined_plate_checks": {"A": plate_a_check, "B": plate_b_check},
        "files": {
            "assembly_glb": {"path": str(assembly_glb.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(assembly_glb)},
            "assembly_preview": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(preview_path)},
            "plate_A": {"path": str(plate_a_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(plate_a_path)},
            "plate_B": {"path": str(plate_b_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(plate_b_path)},
            "yellow_support_part": {"path": str(yellow_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(yellow_path)},
        },
        "printing_gate": [
            "Old blue planets may be reused; if reused, do not print the three optional blue parts on plate A.",
            "The integrated yellow sun input has a radial overhang and requires a separate Bambu support review.",
            "Confirm plate A has four objects and plate B has four objects when using the supplied combined files.",
            "Check the red internal teeth, carrier blind center socket, three 5.6 mm holes, green shaft and pin steps layer by layer.",
            "Do not send any file to the printer until the user confirms the Bambu preview.",
        ],
    }
    report_path = run_dir / "ngw_corrected_v3_design_and_validation.json"
    reviewlib.write_json(report_path, report)
    guide_path = run_dir / "README_修正版先看预览再切片.md"
    guide_path.write_text(
        "# NGW 原图拓扑修正版 V3（原尺寸）\n\n"
        "这是一套新的 9 件候选，旧 V2 禁止混装。三颗旧蓝色行星轮可直接复用，所以实际只需新打印红杯体、黄色太阳轮一体输入、绿色行星架输出和三根新销，共 6 件。\n\n"
        "先查看 `ngw_corrected_v3_assembly_fourview.png`。若使用旧蓝轮，板 A 组合文件只作为完整套件备份，不要直接整板重复打印；红杯体请使用独立 STL。"
        "板 B 含绿色行星架和三根新销。黄色一体太阳轮必须单独导入，它有悬挑，需要在 Bambu Studio 中单独检查支撑。\n\n"
        "装配顺序：红杯体开口朝前；黄色太阳轮一体输入和三颗蓝轮放入齿轮层；绿色行星架从后方对准；三根绿销从正面穿过蓝轮并插入绿架；最后让绿色长轴穿过红背板中心孔。"
        "只做低速手动验证，不接电机、不承载。\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "status": report["status"],
        "stl_generated": True,
        "report": {"path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(report_path)},
        "guide": {"path": str(guide_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(guide_path)},
        **report["files"],
    }
    reviewlib.write_json(run_dir / "manifest.json", manifest)
    return run_dir


if __name__ == "__main__":
    print(build())
