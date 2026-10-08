"""Build a CPU-only NGW image-reference review assembly.

This stage intentionally exports coloured GLB review scenes and PNG projections
only.  It does not export STL.  Visible colour/topology follows the supplied
reference image; hidden axial details are explicit, reversible design assumptions.
Dimensions are millimetres until GLB export, where geometry is converted to metres.
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

import build_ngw_planetary_demo as gearlib  # noqa: E402


RED = (153, 70, 52, 255)
BLUE = (45, 101, 181, 255)
YELLOW = (241, 181, 38, 255)
GREEN = (88, 157, 48, 255)
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def colour(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    result = mesh.copy()
    result.visual = trimesh.visual.ColorVisuals(
        mesh=result, face_colors=np.tile(np.asarray(rgba, dtype=np.uint8), (len(result.faces), 1))
    )
    return result


def cylinder(radius: float, z0: float, z1: float, name: str, sections: int = 128) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=sections)
    mesh.apply_translation((0.0, 0.0, (z0 + z1) / 2.0))
    mesh.metadata["name"] = name
    mesh.units = "mm"
    return mesh


def d_profile(theta: np.ndarray, radius: float, flat_depth: float) -> np.ndarray:
    flat_x = radius - flat_depth
    radial = np.full_like(theta, radius, dtype=float)
    cosine = np.cos(theta)
    active = (cosine > 1e-9) & (radius * cosine > flat_x)
    radial[active] = flat_x / cosine[active]
    return radial


def solid_prism(theta: np.ndarray, radius: np.ndarray | float, z0: float, z1: float, name: str) -> trimesh.Trimesh:
    xy = gearlib.xy_from_polar(theta, radius)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    bottom = gearlib.append_ring(vertices, xy, z0)
    top = gearlib.append_ring(vertices, xy, z1)
    gearlib.connect_closed(faces, bottom, top)
    for ring, z, reverse in ((bottom, z0, True), (top, z1, False)):
        center = len(vertices)
        vertices.append([0.0, 0.0, float(z)])
        for index in range(len(ring)):
            nxt = (index + 1) % len(ring)
            triangle = [center, int(ring[index]), int(ring[nxt])]
            faces.append(triangle[::-1] if reverse else triangle)
    return gearlib.finish_mesh(vertices, faces, name)


def annulus(outer_radius: float, inner_radius: float, z0: float, z1: float, name: str) -> trimesh.Trimesh:
    theta = np.linspace(0.0, 2.0 * math.pi, 256, endpoint=False)
    mesh = gearlib.extrude_radial_annulus(theta, outer_radius, inner_radius, z1 - z0, name)
    mesh.apply_translation((0.0, 0.0, z0))
    return mesh


def translated(mesh: trimesh.Trimesh, xyz: tuple[float, float, float]) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_translation(xyz)
    return result


def carrier_visual_parts(params: dict[str, Any], center_radius: float) -> list[tuple[str, trimesh.Trimesh]]:
    z0 = params["carrier_z0_mm"]
    z1 = z0 + params["carrier_thickness_mm"]
    parts: list[tuple[str, trimesh.Trimesh]] = []
    arm_length = center_radius
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        arm = trimesh.creation.box(
            extents=(arm_length, params["carrier_arm_width_mm"], params["carrier_thickness_mm"])
        )
        arm.apply_transform(trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0)))
        arm.apply_translation(
            (
                0.5 * center_radius * math.cos(angle),
                0.5 * center_radius * math.sin(angle),
                (z0 + z1) / 2.0,
            )
        )
        parts.append((f"carrier_arm_{index + 1}", arm))
        px = center_radius * math.cos(angle)
        py = center_radius * math.sin(angle)
        hub = cylinder(params["carrier_planet_hub_radius_mm"], z0, z1, f"carrier_hub_{index + 1}")
        hub.apply_translation((px, py, 0.0))
        parts.append((f"carrier_planet_hub_{index + 1}", hub))
        pin = cylinder(params["planet_pin_diameter_mm"] / 2.0, params["gear_z0_mm"], z1, f"planet_pin_{index + 1}")
        pin.apply_translation((px, py, 0.0))
        parts.append((f"planet_pin_{index + 1}", pin))
    parts.append(("carrier_center_hub", cylinder(params["carrier_center_hub_radius_mm"], z0, z1 + 2.0, "carrier_center_hub")))
    parts.append(("carrier_output_shaft", cylinder(params["carrier_output_shaft_diameter_mm"] / 2.0, z1, z1 + params["carrier_output_shaft_length_mm"], "carrier_output_shaft")))
    return parts


def mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    components = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "positive_volume": bool(mesh.volume > 0),
        "bounds_mm": np.asarray(mesh.bounds).round(6).tolist(),
        "extents_mm": np.asarray(mesh.extents).round(6).tolist(),
    }


def rotation_matrix(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    pitch = np.deg2rad(pitch_degrees)
    ry = np.array([[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]])
    return rx @ ry


def render_items(
    items: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]],
    yaw: float,
    pitch: float,
    label: str,
    size: tuple[int, int] = (760, 620),
) -> Image.Image:
    rotation = rotation_matrix(yaw, pitch)
    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    normals_all: list[np.ndarray] = []
    colours_all: list[np.ndarray] = []
    offset = 0
    for _, mesh, rgba in items:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        vertices_all.append(vertices)
        faces_all.append(faces + offset)
        normals_all.append(np.asarray(mesh.face_normals))
        colours_all.append(np.tile(np.asarray(rgba[:3]), (len(faces), 1)))
        offset += len(vertices)
    vertices = np.vstack(vertices_all) @ rotation.T
    faces = np.vstack(faces_all)
    normals = np.vstack(normals_all) @ rotation.T
    base_colours = np.vstack(colours_all)
    # Orthographic back-face culling avoids drawing hidden/internal triangles on
    # top of visible faces in this lightweight CPU preview renderer.
    visible = normals[:, 2] > 1e-7
    faces = faces[visible]
    normals = normals[visible]
    base_colours = base_colours[visible]
    xy = vertices[:, :2]
    minimum, maximum = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    scale = min((size[0] - 70) / span[0], (size[1] - 105) / span[1])
    projected = (xy - (minimum + maximum) / 2.0) * scale
    projected[:, 0] += size[0] / 2.0
    projected[:, 1] = size[1] / 2.0 - projected[:, 1] + 18
    depth = vertices[faces, 2].mean(axis=1)
    order = np.argsort(depth)
    light = np.array([0.25, -0.35, 0.90])
    light /= np.linalg.norm(light)
    intensity = np.clip(0.42 + 0.58 * np.abs(normals @ light), 0.0, 1.0)
    canvas = Image.new("RGB", size, (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    for face_index in order:
        polygon = [tuple(point) for point in projected[faces[face_index]]]
        rgb = np.clip(base_colours[face_index] * intensity[face_index], 0, 255).astype(np.uint8)
        draw.polygon(polygon, fill=tuple(int(x) for x in rgb))
    font = ImageFont.truetype(str(FONT_BOLD), 24)
    draw.rectangle((0, 0, size[0], 46), fill=(231, 235, 239))
    draw.text((16, 9), label, fill=(20, 28, 36), font=font)
    return canvas


def make_review_board(
    assembled: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]],
    exploded: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]],
    output: Path,
) -> None:
    views = [
        render_items(assembled, 27, -17, "装配斜视：对应原图右侧方向"),
        render_items(assembled, 0, 0, "正视：核对红/蓝/黄/绿"),
        render_items(assembled, 90, 0, "侧视：核对轴向层次"),
        render_items(exploded, 72, -12, "轴向展开：仅用于分清零件归属"),
    ]
    board = Image.new("RGB", (1520, 1320), "white")
    for index, view in enumerate(views):
        board.paste(view, ((index % 2) * 760, (index // 2) * 620))
    draw = ImageDraw.Draw(board)
    title = ImageFont.truetype(str(FONT_BOLD), 30)
    body = ImageFont.truetype(str(FONT), 21)
    draw.rectangle((0, 1240, 1520, 1320), fill=(255, 255, 255))
    draw.text((24, 1246), "颜色：红=固定内齿圈+后侧板  蓝=3个行星轮  黄=短太阳轮+背面短输入轴  绿=行星架+右侧长输出轴", fill=(25, 25, 25), font=title)
    draw.text((24, 1287), "审查件，不是STL；黄色短输入轴、后侧板一体化和轴向间隙均为待确认的功能设计假设。", fill=(95, 40, 20), font=body)
    board.save(output)


def add_scene_geometry(scene: trimesh.Scene, name: str, mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> None:
    item = colour(mesh, rgba)
    item.apply_scale(0.001)
    item.units = "m"
    scene.add_geometry(item, node_name=name, geom_name=name)


def build() -> Path:
    params: dict[str, Any] = {
        "module_mm": 1.5,
        "pressure_angle_deg": 20.0,
        "sun_teeth": 18,
        "planet_teeth": 27,
        "ring_teeth": 72,
        "planet_count": 3,
        "nominal_pair_backlash_mm": 0.30,
        "sun_input_shaft_diameter_mm": 8.0,
        "sun_input_d_flat_depth_mm": 1.0,
        "sun_bore_diametral_clearance_mm": 0.30,
        "planet_bore_diameter_mm": 5.3,
        "ring_outer_diameter_mm": 122.0,
        "ring_tooth_width_mm": 9.2,
        "gear_face_width_mm": 8.0,
        "gear_z0_mm": 0.6,
        "rear_plate_thickness_mm": 3.0,
        "rear_plate_center_bore_mm": 9.0,
        "carrier_z0_mm": 9.2,
        "carrier_thickness_mm": 3.0,
        "carrier_arm_width_mm": 13.0,
        "carrier_planet_hub_radius_mm": 7.5,
        "carrier_center_hub_radius_mm": 9.0,
        "planet_pin_diameter_mm": 5.0,
        "carrier_output_shaft_diameter_mm": 9.0,
        "carrier_output_shaft_length_mm": 34.0,
        "sun_input_rear_extension_mm": 10.0,
        "axial_clearance_to_rear_plate_mm": 0.6,
        "axial_clearance_to_carrier_mm": 0.6,
        "samples_per_tooth": 32,
        "reference_image": "workspace/docs/image.png",
        "physical_tooth_fit_evidence": "RUN-20260904-151500-67ae54 / COUPON_PHYSICAL_MESH_PASS_BY_USER",
    }
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{signature}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    allowance = params["nominal_pair_backlash_mm"] / 2.0
    st, sr, sun_dims = gearlib.sampled_external_radius(params["sun_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    pt, pr, planet_dims = gearlib.sampled_external_radius(params["planet_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    rt, rr, ring_dims = gearlib.sampled_internal_radius(params["ring_teeth"], params["module_mm"], params["pressure_angle_deg"], allowance, params["samples_per_tooth"])
    sun_shaft_radius = params["sun_input_shaft_diameter_mm"] / 2.0
    sun_bore_radius = sun_shaft_radius + params["sun_bore_diametral_clearance_mm"] / 2.0
    sun_bore = d_profile(
        st,
        sun_bore_radius,
        params["sun_input_d_flat_depth_mm"],
    )
    sun = gearlib.extrude_radial_annulus(st, sr, sun_bore, params["gear_face_width_mm"], "sun_18t_d_bore")
    planet = gearlib.extrude_radial_annulus(pt, pr, params["planet_bore_diameter_mm"] / 2.0, params["gear_face_width_mm"], "planet_27t")
    ring = gearlib.extrude_radial_annulus(rt, params["ring_outer_diameter_mm"] / 2.0, rr, params["ring_tooth_width_mm"], "ring_72t")
    sun.apply_translation((0.0, 0.0, params["gear_z0_mm"]))
    planet.apply_translation((0.0, 0.0, params["gear_z0_mm"]))
    rear_plate = annulus(
        params["ring_outer_diameter_mm"] / 2.0,
        params["rear_plate_center_bore_mm"] / 2.0,
        -params["rear_plate_thickness_mm"],
        0.0,
        "ring_rear_plate",
    )
    shaft_theta = np.linspace(0.0, 2.0 * math.pi, 256, endpoint=False)
    sun_input = solid_prism(
        shaft_theta,
        d_profile(shaft_theta, sun_shaft_radius, params["sun_input_d_flat_depth_mm"]),
        -params["rear_plate_thickness_mm"] - params["sun_input_rear_extension_mm"],
        params["gear_z0_mm"] + params["gear_face_width_mm"],
        "sun_input_shaft_assumption",
    )
    center_radius = params["module_mm"] * (params["sun_teeth"] + params["planet_teeth"]) / 2.0
    carrier_parts = carrier_visual_parts(params, center_radius)

    planets: list[tuple[str, trimesh.Trimesh]] = []
    for index in range(3):
        angle = index * 2.0 * math.pi / 3.0
        item = planet.copy()
        transform = trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0))
        transform[:3, 3] = [center_radius * math.cos(angle), center_radius * math.sin(angle), 0.0]
        item.apply_transform(transform)
        planets.append((f"planet_{index + 1}", item))

    assembled: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]] = [
        ("ring_internal_teeth", ring, RED),
        ("ring_rear_plate", rear_plate, RED),
        ("sun_input_shaft_assumption", sun_input, YELLOW),
        ("sun_18t", sun, YELLOW),
    ]
    assembled.extend((name, mesh, BLUE) for name, mesh in planets)
    assembled.extend((name, mesh, GREEN) for name, mesh in carrier_parts)

    exploded: list[tuple[str, trimesh.Trimesh, tuple[int, int, int, int]]] = [
        ("ring_internal_teeth", translated(ring, (0.0, 0.0, -6.0)), RED),
        ("ring_rear_plate", translated(rear_plate, (0.0, 0.0, -6.0)), RED),
        ("sun_input_shaft_assumption", translated(sun_input, (0.0, 0.0, 7.0)), YELLOW),
        ("sun_18t", translated(sun, (0.0, 0.0, 7.0)), YELLOW),
    ]
    exploded.extend((name, translated(mesh, (0.0, 0.0, 7.0)), BLUE) for name, mesh in planets)
    exploded.extend((name, translated(mesh, (0.0, 0.0, 24.0)), GREEN) for name, mesh in carrier_parts)

    assembly_scene = trimesh.Scene()
    for name, mesh, rgba in assembled:
        add_scene_geometry(assembly_scene, name, mesh, rgba)
    assembly_path = run_dir / "ngw_image_reference_assembly_review_v1_meters.glb"
    assembly_scene.export(assembly_path)

    exploded_scene = trimesh.Scene()
    for name, mesh, rgba in exploded:
        add_scene_geometry(exploded_scene, name, mesh, rgba)
    exploded_path = run_dir / "ngw_image_reference_exploded_review_v1_meters.glb"
    exploded_scene.export(exploded_path)
    preview_path = run_dir / "ngw_image_reference_fourview_v1.png"
    make_review_board(assembled, exploded, preview_path)

    reference_path = PROJECT_ROOT / params["reference_image"]
    logical_meshes = {
        "ring_internal_teeth": ring,
        "ring_rear_plate": rear_plate,
        "sun_18t": sun,
        "sun_input_shaft_assumption": sun_input,
        "planet_template_27t": planet,
    }
    checks = {
        "tooth_relation": params["ring_teeth"] == params["sun_teeth"] + 2 * params["planet_teeth"],
        "three_planet_phase": (params["sun_teeth"] + params["ring_teeth"]) % 3 == 0,
        "fixed_ring_ratio": 1.0 + params["ring_teeth"] / params["sun_teeth"],
        "planet_center_radius_mm": center_radius,
        "rear_axial_clearance_mm": params["gear_z0_mm"],
        "front_axial_clearance_mm": params["carrier_z0_mm"] - (params["gear_z0_mm"] + params["gear_face_width_mm"]),
        "sun_d_bore_diametral_clearance_mm": params["sun_bore_diametral_clearance_mm"],
    }
    if not checks["tooth_relation"] or not checks["three_planet_phase"]:
        raise RuntimeError("NGW tooth or three-planet phase constraint failed")
    if checks["rear_axial_clearance_mm"] <= 0 or checks["front_axial_clearance_mm"] <= 0:
        raise RuntimeError("Axial clearance must be positive")
    report = {
        "run_id": run_id,
        "status": "IMAGE_REFERENCE_3D_REVIEW_WAITING_USER_CONFIRMATION_NO_STL",
        "scope": "VISIBLE_REFERENCE_TOPOLOGY_PLUS_EXPLICIT_FUNCTIONAL_ASSUMPTIONS",
        "reference": {"path": params["reference_image"], "sha256": sha256(reference_path)},
        "colour_mapping": {
            "red": "fixed internal ring gear plus rear side plate",
            "blue": "three planet gears",
            "yellow": "short sun gear plus tentative short rear input shaft",
            "green": "three-arm carrier, planet pins, and long right/front output shaft",
        },
        "parameters": params,
        "equation_and_clearance_checks": checks,
        "derived_tooth_dimensions": {"sun": sun_dims, "planet": planet_dims, "ring": ring_dims},
        "mesh_checks_mm": {name: mesh_metrics(mesh) for name, mesh in logical_meshes.items()},
        "visible_reference_facts": [
            "One red internal ring structure with a continuous red face behind the gears.",
            "Exactly three blue planet gears.",
            "One short yellow central sun gear; no long yellow member is visible.",
            "Green planet centres and a green carrier with a long shaft extending to the right/front.",
        ],
        "functional_design_assumptions_requiring_user_review": [
            "The red rear side plate is modelled as integrated with the fixed ring body.",
            "The yellow sun receives input through a short D shaft exiting the rear side, hidden in the original front view.",
            "The green carrier sits 0.6 mm in front of the gear faces and uses integrated 5.0 mm planet pins.",
            "The green carrier output shaft is 9.0 mm diameter and extends 34.0 mm to the right/front.",
            "Bearings, retaining clips, split lines and final printable unions are intentionally not modelled at this review stage.",
        ],
        "files": {
            "assembly_glb": {"path": str(assembly_path.relative_to(PROJECT_ROOT)), "sha256": sha256(assembly_path)},
            "exploded_glb": {"path": str(exploded_path.relative_to(PROJECT_ROOT)), "sha256": sha256(exploded_path)},
            "fourview_png": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": sha256(preview_path)},
        },
        "gates": [
            "No STL was generated.",
            "User must review colour mapping, rear plate, short yellow sun/input, green carrier position, and output shaft.",
            "After topology approval, printable unions, retention and tolerances must be designed and checked separately.",
        ],
    }
    report_path = run_dir / "ngw_image_reference_design_review_v1.json"
    write_json(report_path, report)
    manifest = {
        "run_id": run_id,
        "status": report["status"],
        "report": {"path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": sha256(report_path)},
        "assembly_glb": report["files"]["assembly_glb"],
        "exploded_glb": report["files"]["exploded_glb"],
        "fourview_png": report["files"]["fourview_png"],
        "stl_generated": False,
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


if __name__ == "__main__":
    print(build())
