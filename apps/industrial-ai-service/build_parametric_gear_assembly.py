"""Build a CPU-only, versioned gear-on-shaft teaching assembly.

The generated tooth profile is intentionally a deterministic teaching profile,
not a standards-compliant involute transmission gear. Dimensions are millimetres.
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


def connect_rings(faces: list[list[int]], ring_a: np.ndarray, ring_b: np.ndarray) -> None:
    count = len(ring_a)
    for index in range(count):
        nxt = (index + 1) % count
        faces.append([int(ring_a[index]), int(ring_a[nxt]), int(ring_b[nxt])])
        faces.append([int(ring_a[index]), int(ring_b[nxt]), int(ring_b[index])])


def annulus_faces(
    faces: list[list[int]], outer: np.ndarray, inner: np.ndarray, reverse: bool = False
) -> None:
    count = len(outer)
    for index in range(count):
        nxt = (index + 1) % count
        triangles = [
            [int(outer[index]), int(outer[nxt]), int(inner[nxt])],
            [int(outer[index]), int(inner[nxt]), int(inner[index])],
        ]
        if reverse:
            triangles = [triangle[::-1] for triangle in triangles]
        faces.extend(triangles)


def cap_faces(
    vertices: list[list[float]], faces: list[list[int]], ring: np.ndarray, z: float, reverse: bool
) -> None:
    xy = np.asarray([vertices[int(index)][:2] for index in ring])
    center_index = len(vertices)
    vertices.append([float(xy[:, 0].mean()), float(xy[:, 1].mean()), float(z)])
    for index in range(len(ring)):
        nxt = (index + 1) % len(ring)
        triangle = [center_index, int(ring[index]), int(ring[nxt])]
        faces.append(triangle[::-1] if reverse else triangle)


def append_ring(vertices: list[list[float]], xy: np.ndarray, z: float) -> np.ndarray:
    start = len(vertices)
    vertices.extend([[float(x), float(y), float(z)] for x, y in xy])
    return np.arange(start, start + len(xy), dtype=np.int64)


def polar_xy(theta: np.ndarray, radius: np.ndarray | float) -> np.ndarray:
    return np.column_stack([np.asarray(radius) * np.cos(theta), np.asarray(radius) * np.sin(theta)])


def d_profile_xy(theta: np.ndarray, radius: float, flat_x: float) -> np.ndarray:
    radial = np.full_like(theta, radius, dtype=float)
    cosine = np.cos(theta)
    active = (cosine > 1e-9) & ((radius * cosine) > flat_x)
    radial[active] = flat_x / cosine[active]
    return polar_xy(theta, radial)


def teaching_gear_outer_xy(
    theta: np.ndarray, teeth: int, tip_radius: float, root_radius: float
) -> np.ndarray:
    phase = ((theta * teeth / (2.0 * math.pi) + 0.5) % 1.0) - 0.5
    absolute = np.abs(phase)
    radius = np.full_like(theta, root_radius, dtype=float)
    radius[absolute <= 0.20] = tip_radius
    transition = (absolute > 0.20) & (absolute < 0.34)
    blend = (absolute[transition] - 0.20) / 0.14
    radius[transition] = tip_radius + (root_radius - tip_radius) * blend
    return polar_xy(theta, radius)


def make_gear(params: dict[str, float | int]) -> trimesh.Trimesh:
    teeth = int(params["teeth"])
    count = teeth * int(params["samples_per_tooth"])
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    outer = teaching_gear_outer_xy(
        theta, teeth, float(params["gear_outer_diameter_mm"]) / 2.0, float(params["gear_root_diameter_mm"]) / 2.0
    )
    shaft_radius = float(params["shaft_diameter_mm"]) / 2.0
    clearance = float(params["gear_bore_diametral_clearance_mm"]) / 2.0
    nominal_inner = d_profile_xy(
        theta,
        shaft_radius + clearance,
        shaft_radius - float(params["d_flat_depth_mm"]) + clearance,
    )
    lead = float(params["gear_bore_lead_in_mm"])
    lead_inner = d_profile_xy(
        theta,
        shaft_radius + clearance + lead,
        shaft_radius - float(params["d_flat_depth_mm"]) + clearance + lead,
    )
    thickness = float(params["gear_thickness_mm"])
    lead_height = float(params["gear_bore_lead_height_mm"])

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    outer_bottom = append_ring(vertices, outer, 0.0)
    outer_top = append_ring(vertices, outer, thickness)
    inner_bottom = append_ring(vertices, lead_inner, 0.0)
    inner_lead = append_ring(vertices, nominal_inner, lead_height)
    inner_top = append_ring(vertices, nominal_inner, thickness)
    connect_rings(faces, outer_bottom, outer_top)
    connect_rings(faces, inner_bottom, inner_lead)
    connect_rings(faces, inner_lead, inner_top)
    annulus_faces(faces, outer_bottom, inner_bottom, reverse=True)
    annulus_faces(faces, outer_top, inner_top, reverse=False)
    return finish_mesh(vertices, faces, "gear")


def make_shaft(params: dict[str, float | int]) -> trimesh.Trimesh:
    count = int(params["teeth"]) * int(params["samples_per_tooth"])
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    base_radius = float(params["shaft_base_diameter_mm"]) / 2.0
    shaft_radius = float(params["shaft_diameter_mm"]) / 2.0
    flat_x = shaft_radius - float(params["d_flat_depth_mm"])
    peg_radius = float(params["retaining_peg_diameter_mm"]) / 2.0
    base_height = float(params["shaft_base_height_mm"])
    gear_height = float(params["gear_thickness_mm"])
    peg_height = float(params["retaining_peg_height_mm"])
    chamfer_height = float(params["shaft_tip_chamfer_height_mm"])

    circle_base = polar_xy(theta, base_radius)
    d_shaft = d_profile_xy(theta, shaft_radius, flat_x)
    circle_peg = polar_xy(theta, peg_radius)
    circle_tip = polar_xy(theta, peg_radius - float(params["shaft_tip_chamfer_radial_mm"]))

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    base_bottom = append_ring(vertices, circle_base, 0.0)
    base_top = append_ring(vertices, circle_base, base_height)
    shaft_bottom = append_ring(vertices, d_shaft, base_height)
    shaft_top = append_ring(vertices, d_shaft, base_height + gear_height)
    peg_bottom = append_ring(vertices, circle_peg, base_height + gear_height)
    peg_chamfer = append_ring(
        vertices, circle_peg, base_height + gear_height + peg_height - chamfer_height
    )
    peg_top = append_ring(vertices, circle_tip, base_height + gear_height + peg_height)

    connect_rings(faces, base_bottom, base_top)
    annulus_faces(faces, base_top, shaft_bottom)
    connect_rings(faces, shaft_bottom, shaft_top)
    annulus_faces(faces, shaft_top, peg_bottom)
    connect_rings(faces, peg_bottom, peg_chamfer)
    connect_rings(faces, peg_chamfer, peg_top)
    cap_faces(vertices, faces, base_bottom, 0.0, reverse=True)
    cap_faces(
        vertices,
        faces,
        peg_top,
        base_height + gear_height + peg_height,
        reverse=False,
    )
    return finish_mesh(vertices, faces, "shaft")


def make_cap(params: dict[str, float | int]) -> trimesh.Trimesh:
    count = int(params["teeth"]) * int(params["samples_per_tooth"])
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    outer_radius = float(params["cap_outer_diameter_mm"]) / 2.0
    peg_radius = float(params["retaining_peg_diameter_mm"]) / 2.0
    clearance = float(params["cap_socket_diametral_clearance_mm"]) / 2.0
    lead = float(params["cap_socket_lead_in_mm"])
    height = float(params["cap_height_mm"])
    depth = float(params["cap_socket_depth_mm"])
    lead_height = float(params["cap_socket_lead_height_mm"])

    outer = polar_xy(theta, outer_radius)
    socket = polar_xy(theta, peg_radius + clearance)
    socket_lead = polar_xy(theta, peg_radius + clearance + lead)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    outer_bottom = append_ring(vertices, outer, 0.0)
    outer_top = append_ring(vertices, outer, height)
    socket_bottom = append_ring(vertices, socket_lead, 0.0)
    socket_lead_ring = append_ring(vertices, socket, lead_height)
    socket_ceiling = append_ring(vertices, socket, depth)
    connect_rings(faces, outer_bottom, outer_top)
    annulus_faces(faces, outer_bottom, socket_bottom, reverse=True)
    connect_rings(faces, socket_bottom, socket_lead_ring)
    connect_rings(faces, socket_lead_ring, socket_ceiling)
    cap_faces(vertices, faces, socket_ceiling, depth, reverse=True)
    cap_faces(vertices, faces, outer_top, height, reverse=False)
    return finish_mesh(vertices, faces, "cap")


def finish_mesh(vertices: list[list[float]], faces: list[list[int]], name: str) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh.metadata["name"] = name
    mesh.units = "mm"
    return mesh


def mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    areas = np.asarray(mesh.area_faces)
    parts = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "positive_volume": bool(mesh.volume > 0),
        "volume_mm3": float(mesh.volume),
        "bounds_mm": np.asarray(mesh.bounds).tolist(),
        "extents_mm": np.asarray(mesh.extents).tolist(),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "degenerate_faces_below_1e-12_mm2": int((areas < 1e-12).sum()),
    }


def validate_mesh(name: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    metrics = mesh_metrics(mesh)
    required = {
        "components": 1,
        "watertight": True,
        "winding_consistent": True,
        "positive_volume": True,
        "boundary_edges": 0,
        "nonmanifold_edges": 0,
        "degenerate_faces_below_1e-12_mm2": 0,
    }
    failures = [key for key, expected in required.items() if metrics[key] != expected]
    if failures:
        raise RuntimeError(f"{name} 网格门禁失败：{', '.join(failures)}")
    return metrics


def render_outline(meshes: list[tuple[str, trimesh.Trimesh, tuple[int, int, int]]], output: Path) -> None:
    size = 760
    canvas = Image.new("RGB", (size * 2, size), (248, 249, 250))
    views = [("ASSEMBLED · SIDE", (0, 2)), ("ASSEMBLED · TOP", (0, 1))]
    for view_index, (label, axes) in enumerate(views):
        draw = ImageDraw.Draw(canvas)
        panel_x = view_index * size
        all_points = np.concatenate([mesh.vertices[:, axes] for _, mesh, _ in meshes], axis=0)
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        span = np.maximum(maximum - minimum, 1e-9)
        scale = (size - 100) / span.max()
        for _, mesh, color in meshes:
            points = mesh.vertices[:, axes]
            points = (points - (minimum + maximum) / 2.0) * scale
            points[:, 0] += panel_x + size / 2.0
            points[:, 1] = size / 2.0 - points[:, 1]
            for face in mesh.faces:
                polygon = [tuple(points[int(index)]) for index in face]
                draw.polygon(polygon, fill=color)
        draw.rectangle((panel_x, 0, panel_x + size, 36), fill=(238, 241, 244))
        draw.text((panel_x + 12, 11), label, fill=(25, 35, 45))
    canvas.save(output)


def colored_meter_mesh(mesh_mm: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    result = mesh_mm.copy()
    result.apply_scale(0.001)
    result.units = "m"
    result.visual.face_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(result.faces), 1))
    return result


def export_glb(mesh_mm: trimesh.Trimesh, output: Path, color: tuple[int, int, int, int]) -> None:
    meter_mesh = colored_meter_mesh(mesh_mm, color)
    output.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene(meter_mesh)))


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def make_gear_cap_print_layout(
    params: dict[str, float | int],
    gear: trimesh.Trimesh,
    cap_socket_up: trimesh.Trimesh,
    gap_mm: float = 3.0,
) -> trimesh.Trimesh:
    """Place the gear and socket-up cap on one plate as two separate solids."""
    if not math.isfinite(gap_mm) or gap_mm < 3.0:
        raise ValueError("组合打印件间距必须至少为 3 mm。")
    placed_gear = gear.copy()
    placed_cap = cap_socket_up.copy()
    cap_offset_x = (
        float(params["gear_outer_diameter_mm"]) / 2.0
        + gap_mm
        + float(params["cap_outer_diameter_mm"]) / 2.0
    )
    placed_cap.apply_translation([cap_offset_x, 0.0, 0.0])
    combined = trimesh.util.concatenate([placed_gear, placed_cap])
    combined.remove_unreferenced_vertices()
    components = combined.split(only_watertight=False)
    if len(components) != 2:
        raise RuntimeError("齿轮与限位帽组合打印 STL 必须包含两个独立实体。")
    for index, component in enumerate(components, start=1):
        validate_mesh(f"gear-cap print component {index}", component)
    if abs(float(combined.bounds[0, 2])) > 1e-8:
        raise RuntimeError("齿轮与限位帽组合打印 STL 未对齐到 Z=0。")
    return combined


def default_parameters() -> dict[str, float | int]:
    return {
        "teeth": 16,
        "samples_per_tooth": 24,
        "gear_outer_diameter_mm": 60.0,
        "gear_root_diameter_mm": 49.0,
        "gear_thickness_mm": 10.0,
        "shaft_diameter_mm": 12.0,
        "d_flat_depth_mm": 1.5,
        "gear_bore_diametral_clearance_mm": 0.50,
        "gear_bore_lead_in_mm": 0.40,
        "gear_bore_lead_height_mm": 0.80,
        "shaft_base_diameter_mm": 24.0,
        "shaft_base_height_mm": 4.0,
        "retaining_peg_diameter_mm": 8.0,
        "retaining_peg_height_mm": 4.0,
        "shaft_tip_chamfer_height_mm": 0.80,
        "shaft_tip_chamfer_radial_mm": 0.40,
        "cap_outer_diameter_mm": 18.0,
        "cap_height_mm": 6.0,
        "cap_socket_depth_mm": 4.0,
        "cap_socket_diametral_clearance_mm": 0.30,
        "cap_socket_lead_in_mm": 0.20,
        "cap_socket_lead_height_mm": 0.60,
    }


def scaled_parameters(
    scale: float,
    gear_bore_clearance_mm: float | None = None,
    cap_socket_clearance_mm: float | None = None,
) -> dict[str, float | int]:
    """Scale nominal geometry and optionally override absolute fit clearances."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale 必须是大于 0 的有限数值。")
    params = default_parameters()
    fixed_fit_parameters = {
        "gear_bore_diametral_clearance_mm",
        "cap_socket_diametral_clearance_mm",
    }
    fixed_count_parameters = {"teeth", "samples_per_tooth"}
    for key, value in list(params.items()):
        if key not in fixed_fit_parameters and key not in fixed_count_parameters:
            params[key] = float(value) * scale
    fit_overrides = {
        "gear_bore_diametral_clearance_mm": gear_bore_clearance_mm,
        "cap_socket_diametral_clearance_mm": cap_socket_clearance_mm,
    }
    for key, value in fit_overrides.items():
        if value is None:
            continue
        if not math.isfinite(value) or not 0.05 <= value <= 1.00:
            raise ValueError(f"{key} 必须是 0.05 到 1.00 mm 的有限数值。")
        params[key] = float(value)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="生成三件式 D 形孔齿轮轴教学装配")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="名义几何缩放倍率；配合间隙保持实测值")
    parser.add_argument("--version", type=int, default=1, help="输出文件版本号")
    parser.add_argument(
        "--gear-bore-clearance-mm",
        type=float,
        default=None,
        help="D 形孔相对 D 形轴的直径间隙；默认 0.50 mm",
    )
    parser.add_argument(
        "--cap-socket-clearance-mm",
        type=float,
        default=None,
        help="限位帽盲孔相对顶部圆柱的直径间隙；默认 0.30 mm",
    )
    parser.add_argument(
        "--fit-validation-source-run",
        default="",
        help="已完成实物配合测试的来源 RUN；只记录追溯关系，不读取或覆盖其模型",
    )
    parser.add_argument(
        "--shaft-reprint-required",
        action="store_true",
        help="新建整套组件时标记短轴需要单独打印；组合 STL 仍只排齿轮和限位帽",
    )
    args = parser.parse_args()
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run-id 格式无效。")
    if args.version < 1:
        raise ValueError("version 必须大于等于 1。")
    if args.fit_validation_source_run and not RUN_ID_PATTERN.fullmatch(args.fit_validation_source_run):
        raise ValueError("fit-validation-source-run 格式无效。")
    if args.fit_validation_source_run and not (RUNS_ROOT / args.fit_validation_source_run).is_dir():
        raise FileNotFoundError("配合测试来源 RUN 不存在。")
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有任务目录：{run_dir}")

    params = scaled_parameters(
        args.scale,
        gear_bore_clearance_mm=args.gear_bore_clearance_mm,
        cap_socket_clearance_mm=args.cap_socket_clearance_mm,
    )
    version = args.version
    version_tag = f"v{version}"

    for directory in ("cad", "stl", "reports", "assembly", "requests"):
        (run_dir / directory).mkdir(parents=True, exist_ok=False)

    gear = make_gear(params)
    shaft = make_shaft(params)
    cap = make_cap(params)
    meshes = {"gear": gear, "shaft": shaft, "cap": cap}
    metrics = {name: validate_mesh(name, mesh) for name, mesh in meshes.items()}

    colors = {
        "gear": (77, 139, 201, 255),
        "shaft": (230, 134, 54, 255),
        "cap": (92, 173, 107, 255),
    }
    artifacts: list[dict[str, Any]] = []
    for name, mesh in meshes.items():
        stl_path = run_dir / "stl" / f"{name}_{version_tag}_print_candidate.stl"
        stl_path.write_bytes(mesh.export(file_type="stl"))
        # Binary STL stores each triangle independently. Merge coincident vertices
        # during reload before evaluating topology, otherwise every triangle looks
        # like a disconnected open component even when the exported solid is valid.
        reloaded = trimesh.load(stl_path, force="mesh", process=True)
        reload_metrics = validate_mesh(f"{name} STL reload", reloaded)
        metrics[name]["stl_reload"] = reload_metrics
        artifacts.append(artifact_record(stl_path, f"{name}_print_stl"))

        glb_path = run_dir / "cad" / f"{name}_{version_tag}_review_meters.glb"
        export_glb(mesh, glb_path, colors[name])
        artifacts.append(artifact_record(glb_path, f"{name}_review_glb"))

    # The cap is exported a second time in a support-friendlier print orientation:
    # socket opening upward, with its closed top face on the print bed.
    cap_print = cap.copy()
    cap_print.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
    cap_print.apply_translation([0.0, 0.0, float(params["cap_height_mm"])])
    cap_print_path = run_dir / "stl" / f"cap_{version_tag}_socket_up_print_candidate.stl"
    cap_print_path.write_bytes(cap_print.export(file_type="stl"))
    validate_mesh("cap socket-up STL reload", trimesh.load(cap_print_path, force="mesh", process=True))
    artifacts.append(artifact_record(cap_print_path, "cap_recommended_print_orientation_stl"))

    gear_cap_print = make_gear_cap_print_layout(params, gear, cap_print)
    gear_cap_print_path = run_dir / "stl" / f"gear_cap_{version_tag}_combined_print_candidate.stl"
    gear_cap_print_path.write_bytes(gear_cap_print.export(file_type="stl"))
    gear_cap_reload = trimesh.load(gear_cap_print_path, force="mesh", process=True)
    gear_cap_components = gear_cap_reload.split(only_watertight=False)
    if len(gear_cap_components) != 2:
        raise RuntimeError("组合打印 STL 重载后必须保持两个独立实体。")
    for index, component in enumerate(gear_cap_components, start=1):
        validate_mesh(f"gear-cap STL reload component {index}", component)
    artifacts.append(artifact_record(gear_cap_print_path, "gear_cap_combined_print_stl"))

    gear_assembled = gear.copy()
    gear_assembled.apply_translation([0.0, 0.0, float(params["shaft_base_height_mm"])])
    cap_assembled = cap.copy()
    cap_assembled.apply_translation(
        [0.0, 0.0, float(params["shaft_base_height_mm"]) + float(params["gear_thickness_mm"])]
    )
    assembled_scene = trimesh.Scene()
    assembled_scene.add_geometry(colored_meter_mesh(shaft, colors["shaft"]), node_name="shaft")
    assembled_scene.add_geometry(colored_meter_mesh(gear_assembled, colors["gear"]), node_name="gear")
    assembled_scene.add_geometry(colored_meter_mesh(cap_assembled, colors["cap"]), node_name="cap")
    assembly_glb = run_dir / "assembly" / f"gear_shaft_cap_assembled_{version_tag}_meters.glb"
    assembly_glb.write_bytes(trimesh.exchange.gltf.export_glb(assembled_scene))
    artifacts.append(artifact_record(assembly_glb, "assembled_review_glb"))

    gear_exploded = gear.copy()
    gear_exploded.apply_translation([0.0, 0.0, 25.0 * args.scale])
    cap_exploded = cap.copy()
    cap_exploded.apply_translation([0.0, 0.0, 44.0 * args.scale])
    exploded_scene = trimesh.Scene()
    exploded_scene.add_geometry(colored_meter_mesh(shaft, colors["shaft"]), node_name="shaft")
    exploded_scene.add_geometry(colored_meter_mesh(gear_exploded, colors["gear"]), node_name="gear")
    exploded_scene.add_geometry(colored_meter_mesh(cap_exploded, colors["cap"]), node_name="cap")
    exploded_glb = run_dir / "assembly" / f"gear_shaft_cap_exploded_{version_tag}_meters.glb"
    exploded_glb.write_bytes(trimesh.exchange.gltf.export_glb(exploded_scene))
    artifacts.append(artifact_record(exploded_glb, "exploded_review_glb"))

    preview = run_dir / "reports" / f"gear_shaft_cap_assembly_preview_{version_tag}.png"
    render_outline(
        [
            ("shaft", shaft, colors["shaft"][:3]),
            ("gear", gear_assembled, colors["gear"][:3]),
            ("cap", cap_assembled, colors["cap"][:3]),
        ],
        preview,
    )
    artifacts.append(artifact_record(preview, "assembly_preview"))

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    assembled_height = (
        float(params["shaft_base_height_mm"])
        + float(params["gear_thickness_mm"])
        + float(params["cap_height_mm"])
    )
    fit_validated = bool(args.fit_validation_source_run)
    fit_description = (
        f"D 孔 {float(params['gear_bore_diametral_clearance_mm']):g} mm、"
        f"帽孔 {float(params['cap_socket_diametral_clearance_mm']):g} mm"
    )
    design = {
        "schema_version": "1.1",
        "created_at": now,
        "run_id": args.run_id,
        "design_version": version,
        "nominal_geometry_scale": args.scale,
        "design_status": "FIT_PARAMETERS_SELECTED_WAITING_SLICER_REVIEW" if fit_validated else "REVIEW_READY_NOT_PRINT_APPROVED",
        "purpose": "机械臂与灵巧手装配教学验证；不用于动力传动或承载",
        "geometry_method": "CPU deterministic parametric mesh",
        "tooth_profile": "教学用确定性梯形齿廓，非标准渐开线，不用于齿轮啮合传动",
        "bom": [
            {"index": 1, "part": "16 齿 D 形孔齿轮", "quantity": 1, "printed": True},
            {"index": 2, "part": "带底座 D 形短轴", "quantity": 1, "printed": True},
            {"index": 3, "part": "圆孔可拆限位帽", "quantity": 1, "printed": True},
        ],
        "parameters_mm": params,
        "fits": {
            "gear_to_shaft": {
                "type": "D 形孔方向定位 + 轴向间隙插装",
                "diametral_clearance_mm": params["gear_bore_diametral_clearance_mm"],
                "radial_and_flat_clearance_mm": float(params["gear_bore_diametral_clearance_mm"]) / 2.0,
                "lead_in_mm": params["gear_bore_lead_in_mm"],
            },
            "cap_to_peg": {
                "type": "圆柱盲孔可拆插装；保持力需试样确认",
                "diametral_clearance_mm": params["cap_socket_diametral_clearance_mm"],
                "radial_clearance_mm": float(params["cap_socket_diametral_clearance_mm"]) / 2.0,
                "lead_in_mm": params["cap_socket_lead_in_mm"],
            },
        },
        "assembled_dimensions_mm": [
            float(params["gear_outer_diameter_mm"]),
            float(params["gear_outer_diameter_mm"]),
            assembled_height,
        ],
        "combined_print": {
            "path": str(gear_cap_print_path.relative_to(PROJECT_ROOT)),
            "components": 2,
            "extents_mm": np.asarray(gear_cap_reload.extents).tolist(),
            "minimum_z_mm": float(gear_cap_reload.bounds[0, 2]),
            "orientation": "齿轮平放；限位帽盲孔朝上；两个实体间距至少 3 mm",
            "shaft_reprint_required": bool(args.shaft_reprint_required),
        },
        "grasp_features": {
            "gear": "优先夹持上下端面靠近轮毂的区域，避开轮齿",
            "shaft": "底座固定在装配工装中，第一版不由机械臂悬空抓持",
            "cap": f"夹持 {float(params['cap_outer_diameter_mm']):g} mm 外圆，沿轴线压装",
        },
        "assembly_sequence": [
            "将短轴底座放入并固定在装配区定位工装",
            "识别齿轮 D 形孔与短轴 D 形截面的角度",
            "从齿轮端面/轮毂区域抓取，避免碰触轮齿",
            "沿轴线接近并完成 D 面角度对准",
            f"低速轴向插入，直到齿轮下端面接触 {float(params['shaft_base_diameter_mm']):g} mm 底座肩面",
            f"抓取限位帽并将圆孔对准 {float(params['retaining_peg_diameter_mm']):g} mm 顶部圆柱",
            "沿轴线压至限位帽下端面接触齿轮上端面",
            "视觉检查同轴、端面贴合、零件完整，并分流到成品区或异常区",
        ],
        "success_criteria": [
            "三件零件齐全且顺序正确",
            "齿轮下端面与轴肩贴合，可见轴向间隙不大于 0.5 mm",
            "限位帽下端面与齿轮上端面贴合，可见轴向间隙不大于 0.5 mm",
            "齿轮 D 面与短轴 D 面完成方向配合，齿轮不能自由空转",
            "轮齿无断裂、崩边和抓取碰撞痕迹",
        ],
        "print_gate": {
            "approved": False,
            "reason": (
                (
                    f"{fit_description} 直径间隙引用来源 RUN 的同尺寸 PLA/A1 实物通过记录；"
                    "每次新打印仍须在 Bambu Studio 复核并完成本件试装。"
                )
                if fit_validated and math.isclose(args.scale, 1.0)
                else (
                    f"{fit_description} 直径间隙引用来源 RUN 的 PLA/A1 试装结论；"
                    "本版名义几何尺寸发生缩放，属于参数迁移而非同尺寸实物验证，仍须在 Bambu Studio 复核并通过正式试装。"
                )
                if fit_validated
                else f"{fit_description} 直径间隙尚未完成实物验证，必须先做配合小样并检查后再打印整套。"
            ),
            "fit_validation_source_run": args.fit_validation_source_run or None,
            "required_next": [
                "在 Blender 或网页查看 assembled/exploded GLB",
                "确认三件外形和 D 面方向符合装配教学意图",
                "在 Bambu Studio 中逐件检查尺寸、朝向、首层和支撑",
                "切片人工确认后再发送打印",
            ],
        },
        "robot_control": "PLANNING_ONLY_NO_ROBOT_CONTROL",
        "mesh_metrics": metrics,
    }
    design_path = run_dir / "reports" / f"gear_shaft_cap_design_report_{version_tag}.json"
    write_json(design_path, design)
    artifacts.append(artifact_record(design_path, "design_report"))

    request_path = run_dir / "requests" / f"parametric_gear_assembly_{version_tag}.json"
    write_json(
        request_path,
        {
            "schema_version": "1.1",
            "run_id": args.run_id,
            "design_version": version,
            "nominal_geometry_scale": args.scale,
            "fit_validation_source_run": args.fit_validation_source_run or None,
            "parameters_mm": params,
            "shaft_reprint_required": bool(args.shaft_reprint_required),
        },
    )
    artifacts.append(artifact_record(request_path, "parametric_request"))

    manifest = {
        "schema_version": "1.1",
        "run_id": args.run_id,
        "created_at": now,
        "updated_at": now,
        "status": "PARAMETRIC_ASSEMBLY_REVIEW_READY",
        "mode": f"cpu-parametric-gear-assembly-{version_tag}",
        "request": {
            "assembly_name": "16 齿 D 形孔齿轮轴教学组件",
            "part_type": "直齿圆柱齿轮",
            "purpose": "装配场景演示",
            "target_size_mm": float(params["gear_outer_diameter_mm"]),
            "target_outer_size_mm": float(params["gear_outer_diameter_mm"]),
            "nominal_geometry_scale": args.scale,
            "source_run": "RUN-20260828-143514-13bf9d",
            "fit_validation_source_run": args.fit_validation_source_run or None,
            "shaft_reprint_required": bool(args.shaft_reprint_required),
        },
        "stages": {
            "engineering_definition": f"{version_tag.upper()}_DEFINED",
            "parametric_cad": "GENERATED",
            "mesh_review": "WAITING_HUMAN_CONFIRMATION",
            "fit_coupon": "PARAMETERS_SELECTED_FROM_SOURCE_RUN" if fit_validated else "NOT_STARTED",
            "print": "WAITING_BAMBU_SLICER_REVIEW",
            "assembly": "PLANNING_ONLY_NO_ROBOT_CONTROL",
        },
        "artifacts": artifacts,
        "events": [
            {"time": now, "event": "ASSEMBLY_RUN_CREATED", "detail": "创建独立三件式参数化装配任务。"},
            {"time": now, "event": "PARAMETRIC_CAD_GENERATED", "detail": f"CPU 生成 {args.scale:g} 倍齿轮、短轴、限位帽及装配/爆炸预览；使用 {fit_description}。"},
            {"time": now, "event": "MESH_GATE_PASSED", "detail": "三个独立 STL 均通过单体、水密、法向、边界和非流形门禁。"},
        ],
    }
    if fit_validated:
        manifest["events"].append(
            {
                "time": now,
                "event": "FIT_PARAMETERS_ACCEPTED",
                "detail": f"引用 {args.fit_validation_source_run} 的 A1/PLA 试装记录；本版使用 {fit_description}。",
            }
        )
    write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"run_id": args.run_id, "run_dir": str(run_dir), "status": manifest["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
