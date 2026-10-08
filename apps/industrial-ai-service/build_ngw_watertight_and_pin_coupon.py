"""Build watertight NGW review parts and a pin-hole fit coupon.

The complete ten-part STL set remains gated.  This script exports coloured GLB
review scenes plus only one printable STL containing a pin and three hole rings.
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
import build_ngw_image_reference_review as reviewlib  # noqa: E402
import build_ngw_planetary_demo as gearlib  # noqa: E402


RED = reviewlib.RED
BLUE = reviewlib.BLUE
YELLOW = reviewlib.YELLOW
GREEN = reviewlib.GREEN


def ring_cup(
    theta: np.ndarray,
    inner_tooth_radius: np.ndarray,
    outer_radius: float,
    rear_bore_radius: float,
    rear_z: float,
    front_z: float,
) -> trimesh.Trimesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    outer_xy = gearlib.xy_from_polar(theta, outer_radius)
    tooth_xy = gearlib.xy_from_polar(theta, inner_tooth_radius)
    bore_xy = gearlib.xy_from_polar(theta, rear_bore_radius)
    outer_back = gearlib.append_ring(vertices, outer_xy, rear_z)
    outer_front = gearlib.append_ring(vertices, outer_xy, front_z)
    bore_back = gearlib.append_ring(vertices, bore_xy, rear_z)
    bore_top = gearlib.append_ring(vertices, bore_xy, 0.0)
    tooth_back = gearlib.append_ring(vertices, tooth_xy, 0.0)
    tooth_front = gearlib.append_ring(vertices, tooth_xy, front_z)
    gearlib.connect_closed(faces, outer_back, outer_front)
    gearlib.connect_closed(faces, bore_back, bore_top, reverse=True)
    gearlib.annulus_cap(faces, outer_back, bore_back, reverse=True)
    gearlib.annulus_cap(faces, tooth_back, bore_top)
    gearlib.connect_closed(faces, tooth_back, tooth_front, reverse=True)
    gearlib.annulus_cap(faces, outer_front, tooth_front)
    mesh = gearlib.finish_mesh(vertices, faces, "watertight_ring_cup")
    mesh.units = "mm"
    return mesh


def d_flanged_shaft(params: dict[str, Any]) -> trimesh.Trimesh:
    theta = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
    circle = gearlib.xy_from_polar(theta, params["sun_input_flange_diameter_mm"] / 2.0)
    shaft_radius = params["sun_input_shaft_diameter_mm"] / 2.0
    d_xy = gearlib.xy_from_polar(
        theta,
        reviewlib.d_profile(theta, shaft_radius, params["sun_input_d_flat_depth_mm"]),
    )
    z0 = params["sun_input_rear_z_mm"] - params["sun_input_flange_thickness_mm"]
    z1 = params["sun_input_rear_z_mm"]
    z2 = params["gear_z0_mm"] + params["gear_face_width_mm"]
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    flange_back = gearlib.append_ring(vertices, circle, z0)
    flange_front = gearlib.append_ring(vertices, circle, z1)
    shaft_back = gearlib.append_ring(vertices, d_xy, z1)
    shaft_front = gearlib.append_ring(vertices, d_xy, z2)
    gearlib.connect_closed(faces, flange_back, flange_front)
    gearlib.connect_closed(faces, flange_front, shaft_back)
    gearlib.connect_closed(faces, shaft_back, shaft_front)
    for ring, z, reverse in ((flange_back, z0, True), (shaft_front, z2, False)):
        center = len(vertices)
        vertices.append([0.0, 0.0, z])
        for index in range(len(ring)):
            nxt = (index + 1) % len(ring)
            face = [center, int(ring[index]), int(ring[nxt])]
            faces.append(face[::-1] if reverse else face)
    mesh = gearlib.finish_mesh(vertices, faces, "watertight_sun_d_input_shaft")
    mesh.units = "mm"
    return mesh


def distance_to_segment_squared(
    x: np.ndarray, y: np.ndarray, end_x: float, end_y: float
) -> np.ndarray:
    length_squared = end_x * end_x + end_y * end_y
    t = np.clip((x * end_x + y * end_y) / length_squared, 0.0, 1.0)
    dx = x - t * end_x
    dy = y - t * end_y
    return dx * dx + dy * dy


def watertight_carrier(params: dict[str, Any], center_radius: float) -> trimesh.Trimesh:
    pitch = params["carrier_voxel_pitch_mm"]
    margin = pitch * 2.0
    xy_limit = center_radius + params["carrier_planet_hub_radius_mm"] + margin
    z_min = params["carrier_z0_mm"] - margin
    z_max = (
        params["carrier_z0_mm"]
        + params["carrier_thickness_mm"]
        + params["carrier_output_shaft_length_mm"]
        + margin
    )
    xs = np.arange(-xy_limit, xy_limit + pitch * 0.5, pitch)
    ys = np.arange(-xy_limit, xy_limit + pitch * 0.5, pitch)
    zs = np.arange(z_min, z_max + pitch * 0.5, pitch)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    shape = x * x + y * y <= params["carrier_center_hub_radius_mm"] ** 2
    holes = np.zeros_like(shape)
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        end_x = center_radius * math.cos(angle)
        end_y = center_radius * math.sin(angle)
        shape |= distance_to_segment_squared(x, y, end_x, end_y) <= (params["carrier_arm_width_mm"] / 2.0) ** 2
        shape |= (x - end_x) ** 2 + (y - end_y) ** 2 <= params["carrier_planet_hub_radius_mm"] ** 2
        holes |= (x - end_x) ** 2 + (y - end_y) ** 2 < (params["carrier_pin_hole_diameter_mm"] / 2.0) ** 2
    volume = np.zeros((len(xs), len(ys), len(zs)), dtype=np.uint8)
    plate_z = (zs >= params["carrier_z0_mm"]) & (
        zs <= params["carrier_z0_mm"] + params["carrier_thickness_mm"]
    )
    volume[:, :, plate_z] = (shape & ~holes)[:, :, None]
    shaft_xy = x * x + y * y <= (params["carrier_output_shaft_diameter_mm"] / 2.0) ** 2
    shaft_z = (zs >= params["carrier_z0_mm"] + params["carrier_thickness_mm"] - 0.5) & (
        zs <= params["carrier_z0_mm"] + params["carrier_thickness_mm"] + params["carrier_output_shaft_length_mm"]
    )
    volume[:, :, shaft_z] |= shaft_xy[:, :, None]
    vertices, faces, _, _ = measure.marching_cubes(volume.astype(np.float32), level=0.5, spacing=(pitch, pitch, pitch))
    vertices += np.array([xs[0], ys[0], zs[0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh.metadata["name"] = "watertight_carrier_output"
    mesh.units = "mm"
    return mesh


def fit_coupon(params: dict[str, Any]) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    pin = completelib.revolved_solid(
        [
            (params["coupon_pin_head_diameter_mm"] / 2.0, 0.0),
            (params["coupon_pin_head_diameter_mm"] / 2.0, params["coupon_pin_head_thickness_mm"]),
            (params["planet_pin_press_diameter_mm"] / 2.0, params["coupon_pin_head_thickness_mm"]),
            (
                params["planet_pin_press_diameter_mm"] / 2.0,
                params["coupon_pin_head_thickness_mm"] + params["coupon_pin_length_mm"],
            ),
        ],
        "coupon_press_pin_5p3",
    )
    layouts = [(5.3, 14.0, -9.0), (5.2, 16.0, 11.0), (5.1, 18.0, 33.0)]
    items = [reviewlib.translated(pin, (-30.0, 0.0, 0.0))]
    identities: list[dict[str, float]] = []
    for hole, outer, x in layouts:
        ring = reviewlib.annulus(outer / 2.0, hole / 2.0, 0.0, params["coupon_ring_height_mm"], f"coupon_hole_{hole:.1f}")
        items.append(reviewlib.translated(ring, (x, 0.0, 0.0)))
        identities.append({"hole_diameter_mm": hole, "outer_diameter_mm": outer, "center_x_mm": x})
    combined = trimesh.util.concatenate(items)
    combined.metadata["name"] = "ngw_carrier_pin_fit_coupon"
    combined.units = "mm"
    return combined, {"pin_center_x_mm": -30.0, "rings": identities}


def make_coupon_preview(coupon: trimesh.Trimesh, output: Path) -> None:
    items = [("pin_hole_coupon", coupon, GREEN)]
    top = reviewlib.render_items(items, 0, 0, "俯视：左侧测试销，右侧三个孔环", size=(900, 500))
    oblique = reviewlib.render_items(items, 28, -18, "斜视：保持此竖直免支撑方向", size=(900, 500))
    board = Image.new("RGB", (1800, 570), "white")
    board.paste(top, (0, 0))
    board.paste(oblique, (900, 0))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype(str(reviewlib.FONT_BOLD), 24)
    draw.text(
        (24, 520),
        "孔环识别：外径14=孔5.3（先试）｜外径16=孔5.2｜外径18=孔5.1（最后试）",
        fill=(30, 30, 30),
        font=font,
    )
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
        "carrier_voxel_pitch_mm": 0.25,
        "planet_pin_journal_diameter_mm": 5.0,
        "planet_pin_press_diameter_mm": 5.3,
        "planet_pin_head_diameter_mm": 9.0,
        "planet_pin_rear_z_mm": 0.3,
        "planet_pin_front_z_mm": 14.2,
        "coupon_pin_head_diameter_mm": 9.0,
        "coupon_pin_head_thickness_mm": 2.0,
        "coupon_pin_length_mm": 8.0,
        "coupon_ring_height_mm": 5.0,
        "samples_per_tooth": 32,
        "source_split_review_run": "RUN-20260905-103559-3ed827",
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
    sun = gearlib.extrude_radial_annulus(
        st,
        sr,
        reviewlib.d_profile(st, shaft_radius + params["sun_bore_diametral_clearance_mm"] / 2.0, params["sun_input_d_flat_depth_mm"]),
        params["gear_face_width_mm"],
        "sun_18t_d_bore",
    )
    planet = gearlib.extrude_radial_annulus(
        pt, pr, params["planet_bore_diameter_mm"] / 2.0, params["gear_face_width_mm"], "planet_27t"
    )
    sun.apply_translation((0.0, 0.0, params["gear_z0_mm"]))
    planet.apply_translation((0.0, 0.0, params["gear_z0_mm"]))
    red_cup = ring_cup(
        rt,
        rr,
        params["ring_outer_diameter_mm"] / 2.0,
        params["rear_plate_center_bore_mm"] / 2.0,
        -params["rear_plate_thickness_mm"],
        params["ring_tooth_width_mm"],
    )
    sun_shaft = d_flanged_shaft(params)
    center_radius = params["module_mm"] * (params["sun_teeth"] + params["planet_teeth"]) / 2.0
    carrier = watertight_carrier(params, center_radius)
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

    planets: list[tuple[str, trimesh.Trimesh]] = []
    pins: list[tuple[str, trimesh.Trimesh]] = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        x = center_radius * math.cos(angle)
        y = center_radius * math.sin(angle)
        gear = planet.copy()
        transform = trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0))
        transform[:3, 3] = (x, y, 0.0)
        gear.apply_transform(transform)
        planets.append((f"planet_{index + 1}", gear))
        pins.append((f"planet_pin_{index + 1}", reviewlib.translated(pin, (x, y, 0.0))))

    assembled: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]] = [
        ("red_ring_cup", red_cup, RED),
        ("yellow_sun", sun, YELLOW),
        ("yellow_d_input_shaft", sun_shaft, YELLOW),
    ]
    assembled.extend((name, mesh, BLUE) for name, mesh in planets)
    assembled.append(("green_carrier_output", carrier, GREEN))
    assembled.extend((name, mesh, GREEN) for name, mesh in pins)

    exploded: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]] = [
        ("red_ring_cup", reviewlib.translated(red_cup, (0.0, 0.0, -8.0)), RED),
        ("yellow_sun", reviewlib.translated(sun, (0.0, 0.0, 7.0)), YELLOW),
        ("yellow_d_input_shaft", reviewlib.translated(sun_shaft, (0.0, 0.0, 7.0)), YELLOW),
    ]
    exploded.extend((name, reviewlib.translated(mesh, (0.0, 0.0, 14.0)), BLUE) for name, mesh in planets)
    exploded.append(("green_carrier_output", reviewlib.translated(carrier, (0.0, 0.0, 27.0)), GREEN))
    exploded.extend((name, reviewlib.translated(mesh, (0.0, 0.0, 43.0)), GREEN) for name, mesh in pins)

    assembly_scene = trimesh.Scene()
    for name, mesh, rgba in assembled:
        reviewlib.add_scene_geometry(assembly_scene, name, mesh, rgba)
    assembly_path = run_dir / "ngw_watertight_parts_assembly_review_v1_meters.glb"
    assembly_scene.export(assembly_path)
    exploded_scene = trimesh.Scene()
    for name, mesh, rgba in exploded:
        reviewlib.add_scene_geometry(exploded_scene, name, mesh, rgba)
    exploded_path = run_dir / "ngw_watertight_parts_exploded_review_v1_meters.glb"
    exploded_scene.export(exploded_path)

    preview_path = run_dir / "ngw_watertight_parts_fourview_v1.png"
    from build_ngw_printable_structure_review import write_review_board

    write_review_board(assembled, exploded, preview_path)

    coupon, coupon_identity = fit_coupon(params)
    coupon_path = run_dir / "ngw_planet_pin_press_fit_coupon_v1.stl"
    coupon.export(coupon_path)
    coupon_preview_path = run_dir / "ngw_planet_pin_press_fit_coupon_preview_v1.png"
    make_coupon_preview(coupon, coupon_preview_path)
    reloaded_coupon = trimesh.load(coupon_path, force="mesh", process=True)
    reloaded_coupon.merge_vertices()
    coupon_checks = reviewlib.mesh_metrics(reloaded_coupon)
    coupon_parts = reloaded_coupon.split(only_watertight=False)
    coupon_component_checks = [reviewlib.mesh_metrics(part) for part in coupon_parts]
    if len(coupon_parts) != 4 or not all(item["watertight"] and item["winding_consistent"] and item["positive_volume"] for item in coupon_component_checks):
        raise RuntimeError("Fit coupon must reload as four valid watertight components")

    part_checks = {name: reviewlib.mesh_metrics(mesh) for name, mesh, _ in assembled}
    invalid = {
        name: metrics
        for name, metrics in part_checks.items()
        if not (metrics["watertight"] and metrics["winding_consistent"] and metrics["positive_volume"])
    }
    if invalid:
        raise RuntimeError(f"Invalid review parts: {sorted(invalid)}")
    if part_checks["red_ring_cup"]["components"] != 1 or part_checks["green_carrier_output"]["components"] != 1:
        raise RuntimeError("Red cup and green carrier must each be one connected component")

    report = {
        "run_id": run_id,
        "status": "PIN_HOLE_COUPON_READY_WAITING_SLICER_AND_PHYSICAL_TEST",
        "full_set_stl_status": "NOT_GENERATED_WAITING_COUPON_RESULT",
        "source_split_review_run": params["source_split_review_run"],
        "parameters": params,
        "part_mesh_checks_mm": part_checks,
        "coupon": {
            "purpose": "Choose the carrier press-fit hole for one 5.3 mm stepped pin.",
            "identity": coupon_identity,
            "test_order": [
                "First try outer diameter 14 mm / hole 5.3 mm.",
                "Then try outer diameter 16 mm / hole 5.2 mm.",
                "Only if still loose, try outer diameter 18 mm / hole 5.1 mm.",
            ],
            "result_labels": ["cannot insert", "too tight", "firm removable", "firm permanent", "loose"],
            "combined_mesh_check": coupon_checks,
            "component_checks": coupon_component_checks,
        },
        "files": {
            "assembly_glb": {"path": str(assembly_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(assembly_path)},
            "exploded_glb": {"path": str(exploded_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(exploded_path)},
            "fourview_png": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(preview_path)},
            "pin_hole_coupon_stl": {"path": str(coupon_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(coupon_path)},
            "pin_hole_coupon_preview": {"path": str(coupon_preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(coupon_preview_path)},
        },
        "gates": [
            "Only the four-component fit coupon STL is approved for slicer review.",
            "The ten-part complete STL set does not exist yet.",
            "Use PLA, 0.4 mm nozzle, no support, and keep the provided upright orientation.",
            "After cooling, test from 5.3 mm hole to 5.2 mm and then 5.1 mm; do not force a stuck pin.",
        ],
    }
    report_path = run_dir / "ngw_watertight_and_coupon_report_v1.json"
    reviewlib.write_json(report_path, report)
    guide_path = run_dir / "README_先打印销孔试片.md"
    guide_path.write_text(
        "# NGW 行星架销孔配合试片\n\n"
        "本轮只打印 `ngw_planet_pin_press_fit_coupon_v1.stl`，不要寻找完整组件 STL；完整组件尚未导出。\n\n"
        "试片包含 4 个分开的实体：一根 5.3 mm 测试销，以及三个孔环。孔环按外径识别："
        "外径 14 mm = 5.3 mm 孔，外径 16 mm = 5.2 mm 孔，外径 18 mm = 5.1 mm 孔。\n\n"
        "Bambu Studio 中保持毫米单位和当前竖直朝向，PLA、0.4 mm 喷嘴，不需要支撑。先检查 4 个实体、首层连续和孔内没有支撑，再发送打印。\n\n"
        "完全冷却后，从最松的 5.3 mm 孔开始，再试 5.2 mm；只有仍明显松动才试 5.1 mm。记录每档为：插不入、过紧、牢固可拆、牢固永久或松动。不要用蛮力把销压入，以免无法取出。\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "status": report["status"],
        "full_set_stl_status": report["full_set_stl_status"],
        "report": {"path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(report_path)},
        "guide": {"path": str(guide_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(guide_path)},
        **report["files"],
    }
    reviewlib.write_json(run_dir / "manifest.json", manifest)
    return run_dir


if __name__ == "__main__":
    print(build())
