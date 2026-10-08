"""Build a versioned printable-split review for the image-referenced NGW demo.

Only coloured GLB/PNG/JSON review artifacts are produced.  No STL is exported.
All dimensions are millimetres before GLB conversion to metres.
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


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_ngw_image_reference_review as reviewlib  # noqa: E402
import build_ngw_planetary_demo as gearlib  # noqa: E402
import build_ngw_complete_demo as completelib  # noqa: E402


RED = reviewlib.RED
BLUE = reviewlib.BLUE
YELLOW = reviewlib.YELLOW
GREEN = reviewlib.GREEN
FONT = reviewlib.FONT
FONT_BOLD = reviewlib.FONT_BOLD


def copy_at(mesh: trimesh.Trimesh, xyz: tuple[float, float, float]) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_translation(xyz)
    return result


def make_carrier_parts(params: dict[str, Any], center_radius: float) -> list[tuple[str, trimesh.Trimesh]]:
    z0 = params["carrier_z0_mm"]
    z1 = z0 + params["carrier_thickness_mm"]
    parts: list[tuple[str, trimesh.Trimesh]] = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        arm = trimesh.creation.box(
            extents=(center_radius, params["carrier_arm_width_mm"], params["carrier_thickness_mm"])
        )
        arm.apply_transform(trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0)))
        arm.apply_translation(
            (
                center_radius * math.cos(angle) / 2.0,
                center_radius * math.sin(angle) / 2.0,
                (z0 + z1) / 2.0,
            )
        )
        parts.append((f"carrier_arm_{index + 1}", arm))
        hub = reviewlib.annulus(
            params["carrier_planet_hub_radius_mm"],
            params["carrier_pin_hole_diameter_mm"] / 2.0,
            z0,
            z1,
            f"carrier_pin_hub_{index + 1}",
        )
        hub.apply_translation((center_radius * math.cos(angle), center_radius * math.sin(angle), 0.0))
        parts.append((f"carrier_pin_hub_{index + 1}", hub))
    parts.append(
        (
            "carrier_center_hub",
            reviewlib.cylinder(params["carrier_center_hub_radius_mm"], z0, z1, "carrier_center_hub"),
        )
    )
    parts.append(
        (
            "carrier_output_shaft",
            reviewlib.cylinder(
                params["carrier_output_shaft_diameter_mm"] / 2.0,
                z1 - 0.4,
                z1 + params["carrier_output_shaft_length_mm"],
                "carrier_output_shaft",
            ),
        )
    )
    return parts


def make_planet_pin(params: dict[str, Any]) -> trimesh.Trimesh:
    return completelib.revolved_solid(
        [
            (params["planet_pin_journal_diameter_mm"] / 2.0, params["planet_pin_rear_z_mm"]),
            (params["planet_pin_journal_diameter_mm"] / 2.0, params["carrier_z0_mm"]),
            (params["planet_pin_press_diameter_mm"] / 2.0, params["carrier_z0_mm"]),
            (
                params["planet_pin_press_diameter_mm"] / 2.0,
                params["carrier_z0_mm"] + params["carrier_thickness_mm"],
            ),
            (params["planet_pin_head_diameter_mm"] / 2.0, params["carrier_z0_mm"] + params["carrier_thickness_mm"]),
            (params["planet_pin_head_diameter_mm"] / 2.0, params["planet_pin_front_z_mm"]),
        ],
        "planet_step_pin",
    )


def make_sun_input_parts(params: dict[str, Any]) -> list[tuple[str, trimesh.Trimesh]]:
    radius = params["sun_input_shaft_diameter_mm"] / 2.0
    theta = np.linspace(0.0, 2.0 * math.pi, 256, endpoint=False)
    shaft = reviewlib.solid_prism(
        theta,
        reviewlib.d_profile(theta, radius, params["sun_input_d_flat_depth_mm"]),
        params["sun_input_rear_z_mm"],
        params["gear_z0_mm"] + params["gear_face_width_mm"],
        "sun_input_d_shaft",
    )
    flange = reviewlib.cylinder(
        params["sun_input_flange_diameter_mm"] / 2.0,
        params["sun_input_rear_z_mm"] - params["sun_input_flange_thickness_mm"],
        params["sun_input_rear_z_mm"] + 0.4,
        "sun_input_rear_flange",
    )
    return [("sun_input_d_shaft", shaft), ("sun_input_rear_flange", flange)]


def write_review_board(
    assembled: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]],
    exploded: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]],
    output: Path,
    exploded_title: str = "制造分件展开：10件，仍未生成STL",
) -> None:
    panels = [
        reviewlib.render_items(assembled, 28, -16, "装配斜视：可见颜色与原图对应"),
        reviewlib.render_items(assembled, 0, 0, "正视：绿色销头位于三个行星轮中心"),
        reviewlib.render_items(assembled, 90, 0, "侧视：黄轴从背面输入，绿轴向前输出"),
        reviewlib.render_items(exploded, 76, -12, exploded_title),
    ]
    board = Image.new("RGB", (1520, 1320), "white")
    for index, panel in enumerate(panels):
        board.paste(panel, ((index % 2) * 760, (index // 2) * 620))
    draw = ImageDraw.Draw(board)
    title = ImageFont.truetype(str(FONT_BOLD), 27)
    body = ImageFont.truetype(str(FONT), 20)
    draw.text((22, 1243), "分件：红色杯体×1｜黄色太阳轮×1＋D输入轴×1｜蓝色行星轮×3｜绿色行星架输出件×1＋阶梯销×3", fill=(24, 24, 24), font=title)
    draw.text((22, 1285), "保持方式：红后板与绿行星架夹住齿轮；销轴压配在行星架、与行星轮间隙转动。当前仅供结构确认。", fill=(90, 42, 20), font=body)
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
        "carrier_pin_hole_diameter_mm": 5.2,
        "carrier_center_hub_radius_mm": 9.0,
        "carrier_output_shaft_diameter_mm": 9.0,
        "carrier_output_shaft_length_mm": 34.0,
        "planet_pin_journal_diameter_mm": 5.0,
        "planet_pin_press_diameter_mm": 5.3,
        "planet_pin_head_diameter_mm": 9.0,
        "planet_pin_rear_z_mm": 0.3,
        "planet_pin_front_z_mm": 14.2,
        "samples_per_tooth": 32,
        "source_review_run": "RUN-20260905-101932-3b8603",
        "physical_tooth_fit_evidence": "RUN-20260904-151500-67ae54 / COUPON_PHYSICAL_MESH_PASS_BY_USER",
    }
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{signature}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    allowance = params["nominal_pair_backlash_mm"] / 2.0
    st, sr, _ = gearlib.sampled_external_radius(params["sun_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    pt, pr, _ = gearlib.sampled_external_radius(params["planet_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    rt, rr, _ = gearlib.sampled_internal_radius(params["ring_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    shaft_radius = params["sun_input_shaft_diameter_mm"] / 2.0
    bore_radius = shaft_radius + params["sun_bore_diametral_clearance_mm"] / 2.0
    sun = gearlib.extrude_radial_annulus(
        st,
        sr,
        reviewlib.d_profile(st, bore_radius, params["sun_input_d_flat_depth_mm"]),
        params["gear_face_width_mm"],
        "sun_18t_d_bore",
    )
    planet = gearlib.extrude_radial_annulus(
        pt, pr, params["planet_bore_diameter_mm"] / 2.0, params["gear_face_width_mm"], "planet_27t"
    )
    ring = gearlib.extrude_radial_annulus(
        rt, params["ring_outer_diameter_mm"] / 2.0, rr, params["ring_tooth_width_mm"], "ring_72t"
    )
    sun.apply_translation((0.0, 0.0, params["gear_z0_mm"]))
    planet.apply_translation((0.0, 0.0, params["gear_z0_mm"]))
    rear_plate = reviewlib.annulus(
        params["ring_outer_diameter_mm"] / 2.0,
        params["rear_plate_center_bore_mm"] / 2.0,
        -params["rear_plate_thickness_mm"],
        0.0,
        "ring_rear_plate",
    )
    sun_input_parts = make_sun_input_parts(params)
    center_radius = params["module_mm"] * (params["sun_teeth"] + params["planet_teeth"]) / 2.0
    carrier_parts = make_carrier_parts(params, center_radius)
    pin_template = make_planet_pin(params)

    planets: list[tuple[str, trimesh.Trimesh]] = []
    pins: list[tuple[str, trimesh.Trimesh]] = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        xy = (center_radius * math.cos(angle), center_radius * math.sin(angle))
        gear = planet.copy()
        transform = trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0))
        transform[:3, 3] = [xy[0], xy[1], 0.0]
        gear.apply_transform(transform)
        planets.append((f"planet_{index + 1}", gear))
        pins.append((f"planet_step_pin_{index + 1}", copy_at(pin_template, (xy[0], xy[1], 0.0))))

    assembled: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]] = [
        ("ring_internal_teeth", ring, RED),
        ("ring_rear_plate", rear_plate, RED),
        ("sun_18t", sun, YELLOW),
    ]
    assembled.extend((name, mesh, YELLOW) for name, mesh in sun_input_parts)
    assembled.extend((name, mesh, BLUE) for name, mesh in planets)
    assembled.extend((name, mesh, GREEN) for name, mesh in carrier_parts)
    assembled.extend((name, mesh, GREEN) for name, mesh in pins)

    exploded: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]] = [
        ("ring_internal_teeth", copy_at(ring, (0.0, 0.0, -8.0)), RED),
        ("ring_rear_plate", copy_at(rear_plate, (0.0, 0.0, -8.0)), RED),
        ("sun_18t", copy_at(sun, (0.0, 0.0, 7.0)), YELLOW),
    ]
    exploded.extend((name, copy_at(mesh, (0.0, 0.0, 7.0)), YELLOW) for name, mesh in sun_input_parts)
    exploded.extend((name, copy_at(mesh, (0.0, 0.0, 14.0)), BLUE) for name, mesh in planets)
    exploded.extend((name, copy_at(mesh, (0.0, 0.0, 27.0)), GREEN) for name, mesh in carrier_parts)
    exploded.extend((name, copy_at(mesh, (0.0, 0.0, 43.0)), GREEN) for name, mesh in pins)

    assembly_scene = trimesh.Scene()
    for name, mesh, rgba in assembled:
        reviewlib.add_scene_geometry(assembly_scene, name, mesh, rgba)
    assembly_path = run_dir / "ngw_printable_split_assembly_review_v1_meters.glb"
    assembly_scene.export(assembly_path)
    exploded_scene = trimesh.Scene()
    for name, mesh, rgba in exploded:
        reviewlib.add_scene_geometry(exploded_scene, name, mesh, rgba)
    exploded_path = run_dir / "ngw_printable_split_exploded_review_v1_meters.glb"
    exploded_scene.export(exploded_path)
    preview_path = run_dir / "ngw_printable_split_fourview_v1.png"
    write_review_board(assembled, exploded, preview_path)

    checks = {
        "tooth_relation": params["ring_teeth"] == params["sun_teeth"] + 2 * params["planet_teeth"],
        "three_planet_phase": (params["sun_teeth"] + params["ring_teeth"]) % 3 == 0,
        "fixed_ring_ratio": 1.0 + params["ring_teeth"] / params["sun_teeth"],
        "planet_count": len(planets),
        "manufacturing_part_count": 10,
        "rear_gear_clearance_mm": params["gear_z0_mm"],
        "gear_carrier_clearance_mm": params["carrier_z0_mm"] - (params["gear_z0_mm"] + params["gear_face_width_mm"]),
        "planet_pin_journal_diametral_clearance_mm": params["planet_bore_diameter_mm"] - params["planet_pin_journal_diameter_mm"],
        "carrier_pin_diametral_interference_mm": params["planet_pin_press_diameter_mm"] - params["carrier_pin_hole_diameter_mm"],
        "sun_d_bore_diametral_clearance_mm": params["sun_bore_diametral_clearance_mm"],
        "rear_plate_shaft_diametral_clearance_mm": params["rear_plate_center_bore_mm"] - params["sun_input_shaft_diameter_mm"],
        "pin_rear_clearance_to_plate_mm": params["planet_pin_rear_z_mm"],
    }
    required_positive = [
        "rear_gear_clearance_mm",
        "gear_carrier_clearance_mm",
        "planet_pin_journal_diametral_clearance_mm",
        "carrier_pin_diametral_interference_mm",
        "sun_d_bore_diametral_clearance_mm",
        "rear_plate_shaft_diametral_clearance_mm",
        "pin_rear_clearance_to_plate_mm",
    ]
    if not checks["tooth_relation"] or not checks["three_planet_phase"] or any(checks[key] <= 0 for key in required_positive):
        raise RuntimeError("Printable split design constraint failed")

    primitive_checks = {
        name: reviewlib.mesh_metrics(mesh)
        for name, mesh, _ in assembled
    }
    report = {
        "run_id": run_id,
        "status": "PRINTABLE_SPLIT_STRUCTURE_REVIEW_WAITING_USER_CONFIRMATION_NO_STL",
        "source_review_run": params["source_review_run"],
        "parameters": params,
        "colour_mapping": {
            "red": "fixed ring cup: internal ring plus rear plate",
            "blue": "three freely rotating planet gears",
            "yellow": "short sun gear and separate rear D input shaft with flange",
            "green": "carrier/output-shaft body and three removable stepped planet pins",
        },
        "bill_of_materials": [
            {"part": "red fixed ring cup", "quantity": 1},
            {"part": "yellow 18T short sun gear", "quantity": 1},
            {"part": "yellow rear D input shaft with flange", "quantity": 1},
            {"part": "blue 27T planet gear", "quantity": 3},
            {"part": "green three-arm carrier with output shaft", "quantity": 1},
            {"part": "green stepped planet pin", "quantity": 3},
        ],
        "checks": checks,
        "primitive_mesh_checks_mm": primitive_checks,
        "assembly_sequence_review": [
            "Place the red ring cup with its rear plate down/back.",
            "Insert the yellow D input shaft from the rear and fit the short yellow sun on its D section.",
            "Place the three blue planets in mesh with the sun and internal ring.",
            "Place the green carrier at the front and align its three pin holes with the planet bores.",
            "Insert the three green stepped pins from the front; press sections hold in the carrier while 5.0 mm journals rotate in 5.3 mm planet bores.",
        ],
        "printing_assumptions_to_test_before_full_set": [
            "5.3 mm printed pin into 5.2 mm printed carrier hole is a tentative 0.1 mm diametral press fit and needs a small coupon.",
            "5.0 mm journal in 5.3 mm planet bore is the existing 0.3 mm diametral running-clearance assumption.",
            "The red ring and rear plate, and the green carrier arms/hubs/output shaft, are logical single parts but remain overlapping review primitives until final watertight union.",
            "No bearing, circlip, screw or front cover is included; the red rear plate and green carrier provide axial retention for the gears.",
        ],
        "files": {
            "assembly_glb": {"path": str(assembly_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(assembly_path)},
            "exploded_glb": {"path": str(exploded_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(exploded_path)},
            "fourview_png": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(preview_path)},
        },
        "gates": [
            "No STL was generated.",
            "User must confirm the 10-part split and front-inserted stepped pins.",
            "After confirmation, create watertight logical unions and print only fit coupons before the full set.",
        ],
    }
    report_path = run_dir / "ngw_printable_split_design_review_v1.json"
    reviewlib.write_json(report_path, report)
    manifest = {
        "run_id": run_id,
        "status": report["status"],
        "report": {"path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(report_path)},
        "assembly_glb": report["files"]["assembly_glb"],
        "exploded_glb": report["files"]["exploded_glb"],
        "fourview_png": report["files"]["fourview_png"],
        "stl_generated": False,
    }
    reviewlib.write_json(run_dir / "manifest.json", manifest)
    return run_dir


if __name__ == "__main__":
    print(build())
