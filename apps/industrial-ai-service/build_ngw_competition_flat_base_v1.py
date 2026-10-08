#!/usr/bin/env python3
"""Build the approved NGW competition flat-base red cup and green carrier."""

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
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_ngw_complete_stl_v2 as v2lib  # noqa: E402
import build_ngw_corrected_topology_stl_v3 as v3lib  # noqa: E402
import build_ngw_image_reference_review as reviewlib  # noqa: E402
import build_ngw_planetary_demo as gearlib  # noqa: E402
import build_ngw_watertight_and_pin_coupon as watertightlib  # noqa: E402


RED = reviewlib.RED
BLUE = reviewlib.BLUE
YELLOW = reviewlib.YELLOW
GREEN = reviewlib.GREEN


def competition_carrier(params: dict[str, Any], center_radius: float) -> trimesh.Trimesh:
    """One green mesh: rear carrier, three tapered front pins and flush coarse shaft."""
    pitch = params["carrier_sdf_pitch_mm"]
    limit = center_radius + params["carrier_planet_hub_radius_mm"] + 2.0 * pitch
    z0 = params["carrier_z0_mm"]
    z1 = z0 + params["carrier_thickness_mm"]
    pin_front = params["planet_pin_tip_front_z_mm"]
    pin_taper_end = params["planet_pin_taper_end_z_mm"]
    shaft_end = params["rear_plate_front_z_mm"] + params["rear_plate_thickness_mm"]

    xs = np.arange(-limit, limit + pitch * 0.5, pitch)
    ys = np.arange(-limit, limit + pitch * 0.5, pitch)
    zs = np.arange(pin_front - 2.0 * pitch, shaft_end + 2.0 * pitch, pitch)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    radial = np.sqrt(x * x + y * y)

    outer = params["carrier_center_hub_radius_mm"] - radial
    pin_centers: list[tuple[float, float]] = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        ex = center_radius * math.cos(angle)
        ey = center_radius * math.sin(angle)
        pin_centers.append((ex, ey))
        arm = params["carrier_arm_width_mm"] / 2.0 - np.sqrt(
            watertightlib.distance_to_segment_squared(x, y, ex, ey)
        )
        hub = params["carrier_planet_hub_radius_mm"] - np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
        outer = np.maximum(outer, np.maximum(arm, hub))

    field = np.empty((len(xs), len(ys), len(zs)), dtype=np.float32)
    socket_radius = params["carrier_center_socket_diameter_mm"] / 2.0
    journal_radius = params["planet_pin_journal_diameter_mm"] / 2.0
    tip_radius = params["planet_pin_tip_diameter_mm"] / 2.0
    for index, z in enumerate(zs):
        plate = np.minimum(np.minimum(outer, z - z0), z1 - z)
        shaft = np.minimum(
            np.minimum(params["carrier_output_shaft_diameter_mm"] / 2.0 - radial, z - (z1 - 0.5)),
            shaft_end - z,
        )
        value = np.maximum(plate, shaft)

        if pin_front <= z <= z1:
            if z < pin_taper_end:
                ratio = (z - pin_front) / (pin_taper_end - pin_front)
                pin_radius = tip_radius + ratio * (journal_radius - tip_radius)
            else:
                pin_radius = journal_radius
            for ex, ey in pin_centers:
                pin_radial = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
                pin_field = np.minimum(
                    np.minimum(pin_radius - pin_radial, z - pin_front),
                    z1 - z,
                )
                value = np.maximum(value, pin_field)

        # Preserve the proven blind round socket for the existing yellow sun pilot.
        if z <= params["carrier_center_socket_end_z_mm"]:
            value = np.minimum(value, radial - socket_radius)
        field[:, :, index] = value.astype(np.float32)

    vertices, faces, _, _ = measure.marching_cubes(
        field, level=params["carrier_iso_level_mm"], spacing=(pitch, pitch, pitch)
    )
    vertices += np.array([xs[0], ys[0], zs[0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh.metadata["name"] = "green_competition_carrier_integrated_pins_flat_output"
    mesh.units = "mm"
    return mesh


def measured_round_section(mesh: trimesh.Trimesh, center: tuple[float, float], z: float,
                           radial_limit: float, z_tolerance: float = 0.11) -> dict[str, float]:
    vertices = np.asarray(mesh.vertices)
    local_x = vertices[:, 0] - center[0]
    local_y = vertices[:, 1] - center[1]
    mask = (
        (np.abs(vertices[:, 2] - z) <= z_tolerance)
        & (local_x * local_x + local_y * local_y <= radial_limit * radial_limit)
    )
    selected = vertices[mask]
    if len(selected) < 8:
        raise RuntimeError(f"Not enough section vertices at z={z}: {len(selected)}")
    diameter_x = float(selected[:, 0].max() - selected[:, 0].min())
    diameter_y = float(selected[:, 1].max() - selected[:, 1].min())
    return {"diameter_x_mm": diameter_x, "diameter_y_mm": diameter_y,
            "mean_diameter_mm": (diameter_x + diameter_y) / 2.0}


def write_review_board(assembled, exploded, red_print, green_print, output: Path) -> None:
    panels = [
        reviewlib.render_items(assembled, 152, -16, "比赛平底版正面：原颜色和齿轮关系"),
        reviewlib.render_items(assembled, 90, 0, "侧视：Ø14绿端与红底面齐平"),
        reviewlib.render_items(exploded, 104, -12, "装配顺序：红杯→绿架→蓝轮→黄太阳轮"),
        reviewlib.render_items([
            ("red_print", red_print, RED), ("green_print", reviewlib.translated(green_print, (145.0, 0.0, 0.0)), GREEN)
        ], 28, -18, "新打印件：红杯无支撑；绿架需检查支撑"),
    ]
    board = Image.new("RGB", (1520, 1320), "white")
    for index, panel in enumerate(panels):
        board.paste(panel, ((index % 2) * 760, (index // 2) * 620))
    draw = ImageDraw.Draw(board)
    title = ImageFont.truetype(str(reviewlib.FONT_BOLD), 25)
    body = ImageFont.truetype(str(reviewlib.FONT_BOLD), 19)
    draw.text((22, 1243), "锁定：一体行星轴Ø4.6｜绿色粗轴Ø14｜红孔Ø15｜输出端完全平齐、无D孔/六角孔", fill=(24,24,24), font=title)
    draw.text((22, 1284), "只需新打印红色杯体和绿色三臂架；黄色太阳轮和三颗蓝轮复用。先在Bambu Studio逐层确认。", fill=(125,55,30), font=body)
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
        "carrier_z0_mm": 8.8,
        "carrier_thickness_mm": 3.0,
        "carrier_arm_width_mm": 10.0,
        "carrier_planet_hub_radius_mm": 7.0,
        "carrier_center_hub_radius_mm": 9.0,
        "carrier_center_socket_diameter_mm": 6.6,
        "carrier_center_socket_end_z_mm": 10.8,
        "carrier_sdf_pitch_mm": 0.20,
        "carrier_iso_level_mm": 0.01,
        "carrier_output_shaft_diameter_mm": 14.0,
        "rear_plate_front_z_mm": 12.6,
        "rear_plate_thickness_mm": 3.0,
        "rear_plate_center_bore_mm": 15.0,
        "planet_pin_journal_diameter_mm": 4.6,
        "planet_pin_tip_diameter_mm": 3.2,
        "planet_pin_tip_front_z_mm": -2.0,
        "planet_pin_taper_end_z_mm": 0.0,
        "flat_output_face": True,
        "output_socket": None,
        "source_v3_run": "RUN-20260905-164113-f797f7",
        "fit_coupon_run": "RUN-20260907-104114-ecb94a",
        "physical_fit_selection": "middle values accepted: pin 4.6, shaft 14 / hole 15.0",
    }
    signature = hashlib.sha256(json.dumps(params, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{signature}"
    run_dir = RUNS_ROOT / run_id
    stl_dir = run_dir / "individual_stl"
    run_dir.mkdir(parents=True, exist_ok=False)
    stl_dir.mkdir()

    allowance = params["nominal_pair_backlash_mm"] / 2.0
    st, sr, _ = gearlib.sampled_external_radius(params["sun_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, 32)
    pt, pr, _ = gearlib.sampled_external_radius(params["planet_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, 32)
    rt, rr, _ = gearlib.sampled_internal_radius(params["ring_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, 32)

    red = v3lib.corrected_red_cup(rt, rr, params)
    yellow = v3lib.integrated_sun_input(st, sr, params)
    blue_template = gearlib.extrude_radial_annulus(pt, pr, params["planet_bore_diameter_mm"] / 2.0,
                                                   params["gear_face_width_mm"], "blue_planet_27t_reused")
    center_radius = params["module_mm"] * (params["sun_teeth"] + params["planet_teeth"]) / 2.0
    green = competition_carrier(params, center_radius)

    planets = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        planets.append(reviewlib.translated(blue_template, (center_radius * math.cos(angle), center_radius * math.sin(angle), 0.0)))
    assembled = [("red_flat_cup", red, RED), ("yellow_reused", yellow, YELLOW)]
    assembled.extend((f"blue_reused_{i+1}", item, BLUE) for i, item in enumerate(planets))
    assembled.append(("green_flat_carrier", green, GREEN))

    scene = trimesh.Scene()
    for name, mesh, rgba in assembled:
        reviewlib.add_scene_geometry(scene, name, mesh, rgba)
    assembly_glb = run_dir / "ngw_competition_flat_base_assembly_review_meters.glb"
    scene.export(assembly_glb)

    offsets = {"red_flat_cup": 28.0, "green_flat_carrier": 13.0, "yellow_reused": -15.0}
    exploded = [(name, reviewlib.translated(mesh, (0.0, 0.0, offsets.get(name, 0.0))), rgba)
                for name, mesh, rgba in assembled]

    red_print = v2lib.print_oriented(red, flip=True)
    green_print = v2lib.print_oriented(green, flip=True)
    red_path = stl_dir / "01_red_flat_base_ring_cup_competition_v1.stl"
    green_path = stl_dir / "02_green_flat_carrier_integrated_pins_competition_v1_SUPPORT_REQUIRED.stl"
    red_check = v2lib.export_checked(red_print, red_path, 1)
    green_check = v2lib.export_checked(green_print, green_path, 1)

    preview_path = run_dir / "ngw_competition_flat_base_fourview_v1.png"
    write_review_board(assembled, exploded, red_print, green_print, preview_path)

    pin_sections = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        center = (center_radius * math.cos(angle), center_radius * math.sin(angle))
        pin_sections.append(measured_round_section(green, center, 4.0, 3.2))
    output_section = measured_round_section(green, (0.0, 0.0), 14.0, 8.0)
    designed_back = params["rear_plate_front_z_mm"] + params["rear_plate_thickness_mm"]
    green_back = float(green.bounds[1, 2])
    flat_offset = green_back - designed_back
    checks = {
        "gear_to_carrier_gap_mm": params["carrier_z0_mm"] - params["gear_face_width_mm"],
        "carrier_to_red_plate_gap_mm": params["rear_plate_front_z_mm"] - (params["carrier_z0_mm"] + params["carrier_thickness_mm"]),
        "output_shaft_to_red_bore_diametral_clearance_mm": params["rear_plate_center_bore_mm"] - params["carrier_output_shaft_diameter_mm"],
        "yellow_pilot_to_carrier_socket_diametral_clearance_mm": params["carrier_center_socket_diameter_mm"] - params["sun_rear_pilot_diameter_mm"],
        "yellow_pilot_tip_to_socket_bottom_gap_mm": params["carrier_center_socket_end_z_mm"] - params["sun_rear_pilot_end_z_mm"],
        "green_output_face_minus_red_back_plane_mm": flat_offset,
        "measured_pin_sections_mm": pin_sections,
        "measured_output_section_mm": output_section,
    }
    if abs(flat_offset) > 0.05:
        raise RuntimeError(f"Green output is not flush enough: {flat_offset}")
    if any(abs(item["mean_diameter_mm"] - 4.6) > 0.12 for item in pin_sections):
        raise RuntimeError(f"Integrated pin deviation: {pin_sections}")
    if abs(output_section["mean_diameter_mm"] - 14.0) > 0.15:
        raise RuntimeError(f"Output shaft deviation: {output_section}")

    report = {
        "run_id": run_id,
        "status": "COMPETITION_FLAT_BASE_TWO_PART_STL_CANDIDATE_WAITING_BAMBU_REVIEW",
        "parameters": params,
        "reuse": ["existing yellow integrated sun x1", "existing blue planet 27T x3"],
        "new_prints": [red_path.name, green_path.name],
        "geometry_checks": checks,
        "individual_stl_checks": {red_path.name: red_check, green_path.name: green_check},
        "files": {
            "red_stl": {"path": str(red_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(red_path)},
            "green_stl": {"path": str(green_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(green_path)},
            "assembly_glb": {"path": str(assembly_glb.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(assembly_glb)},
            "preview_png": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(preview_path)},
        },
        "printing_gate": [
            "Import in millimetres without scaling.",
            "Print red and green as separate colour jobs.",
            "Red cup is a no-support candidate; inspect the first layer, centre bore and internal teeth.",
            "Green carrier is oriented with the flat output face on the bed; use build-plate-only support under the carrier arms and inspect pin tapers.",
            "Do not send either STL until the Bambu layer preview is confirmed.",
        ],
    }
    report_path = run_dir / "ngw_competition_flat_base_validation_v1.json"
    reviewlib.write_json(report_path, report)
    guide_path = run_dir / "README_比赛平底版先看预览再切片.md"
    guide_path.write_text(
        "# NGW 比赛平底版：两个新打印件\n\n"
        "先查看 `ngw_competition_flat_base_fourview_v1.png`。本版不留 D 孔或六角孔，绿色输出端与红色背板底面齐平。\n\n"
        "只需新打印：\n\n"
        "1. `individual_stl/01_red_flat_base_ring_cup_competition_v1.stl`；\n"
        "2. `individual_stl/02_green_flat_carrier_integrated_pins_competition_v1_SUPPORT_REQUIRED.stl`。\n\n"
        "复用旧黄色太阳轮和三颗旧蓝轮。红、绿两件颜色不同，应分别切片打印。\n\n"
        "Bambu Studio：A1、0.4 mm 喷嘴、PLA、毫米单位，不缩放。红杯保持当前朝向，先按无支撑检查；绿架保持粗轴平面朝下，启用‘仅从打印板生成支撑’，重点检查绿架三条臂下方，禁止在三根细轴之间生成难以拆除的包裹支撑。逐层确认红色内齿、Ø15 中心孔、绿色 Ø14 粗轴平面、三根 Ø4.6 轴及 2 mm 锥形尖端。\n\n"
        "装配：红杯平底放置 → 放入绿架粗轴 → 三颗旧蓝轮套入一体轴 → 放入旧黄色太阳轮。该比赛版用于水平平放和机器人自上而下装配，不保证翻转后齿轮不脱落。\n",
        encoding="utf-8",
    )
    report["files"]["guide"] = {"path": str(guide_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(guide_path)}
    reviewlib.write_json(report_path, report)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    build()
