#!/usr/bin/env python3
"""Build the NGW V4 assembly-path coupon only; no full V4 parts are emitted."""

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
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_ngw_complete_demo as completelib  # noqa: E402
import build_ngw_complete_stl_v2 as v2lib  # noqa: E402
import build_ngw_image_reference_review as reviewlib  # noqa: E402
import build_ngw_planetary_demo as gearlib  # noqa: E402


def blind_cap(outer_diameter: float, socket_diameter: float, height: float = 2.2, depth: float = 1.7) -> trimesh.Trimesh:
    """Closed cap with a blind socket, printed socket-up for inspection."""
    samples = 160
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    outer_xy = gearlib.xy_from_polar(theta, outer_diameter / 2.0)
    inner_xy = gearlib.xy_from_polar(theta, socket_diameter / 2.0)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    outer_bottom = gearlib.append_ring(vertices, outer_xy, 0.0)
    outer_top = gearlib.append_ring(vertices, outer_xy, height)
    inner_open = gearlib.append_ring(vertices, inner_xy, 0.0)
    inner_bottom = gearlib.append_ring(vertices, inner_xy, depth)
    gearlib.connect_closed(faces, outer_bottom, outer_top)
    gearlib.annulus_cap(faces, outer_bottom, inner_open, reverse=True)
    gearlib.connect_closed(faces, inner_open, inner_bottom, reverse=True)
    top_center = len(vertices)
    vertices.append([0.0, 0.0, height])
    blind_center = len(vertices)
    vertices.append([0.0, 0.0, depth])
    for index in range(samples):
        nxt = (index + 1) % samples
        faces.append([top_center, int(outer_top[index]), int(outer_top[nxt])])
        faces.append([blind_center, int(inner_bottom[nxt]), int(inner_bottom[index])])
    cap = gearlib.finish_mesh(vertices, faces, f"cap_socket_{socket_diameter:.1f}")
    cap.units = "mm"
    return cap


def c_clip(outer_diameter: float, inner_diameter: float, thickness: float, gap_angle_deg: float = 100.0) -> trimesh.Trimesh:
    """Watertight flat annular-sector C clip with an open mouth along +X."""
    samples = 128
    start = math.radians(gap_angle_deg / 2.0)
    stop = 2.0 * math.pi - start
    theta = np.linspace(start, stop, samples)
    outer_r = outer_diameter / 2.0
    inner_r = inner_diameter / 2.0
    vertices: list[list[float]] = []
    for z in (0.0, thickness):
        for radius in (outer_r, inner_r):
            vertices.extend([[radius * math.cos(t), radius * math.sin(t), z] for t in theta])
    # Index blocks: bottom outer, bottom inner, top outer, top inner.
    bo = np.arange(0, samples)
    bi = np.arange(samples, 2 * samples)
    to = np.arange(2 * samples, 3 * samples)
    ti = np.arange(3 * samples, 4 * samples)
    faces: list[list[int]] = []
    for i in range(samples - 1):
        j = i + 1
        # Bottom and top annular surfaces.
        faces.extend([[int(bo[i]), int(bi[j]), int(bi[i])], [int(bo[i]), int(bo[j]), int(bi[j])]])
        faces.extend([[int(to[i]), int(ti[i]), int(ti[j])], [int(to[i]), int(ti[j]), int(to[j])]])
        # Outer and inner curved walls.
        faces.extend([[int(bo[i]), int(to[i]), int(to[j])], [int(bo[i]), int(to[j]), int(bo[j])]])
        faces.extend([[int(bi[i]), int(bi[j]), int(ti[j])], [int(bi[i]), int(ti[j]), int(ti[i])]])
    # Two end walls at the open mouth.
    for i in (0, samples - 1):
        if i == 0:
            faces.extend([[int(bo[i]), int(bi[i]), int(ti[i])], [int(bo[i]), int(ti[i]), int(to[i])]])
        else:
            faces.extend([[int(bo[i]), int(ti[i]), int(bi[i])], [int(bo[i]), int(to[i]), int(ti[i])]])
    clip = gearlib.finish_mesh(vertices, faces, f"c_clip_id_{inner_diameter:.1f}")
    clip.units = "mm"
    return clip


def pin_fixture() -> trimesh.Trimesh:
    # 5.0 journal represents the future carrier-integrated pin. The 0.6 mm
    # shoulder reproduces the carrier-to-gear axial gap. A 3.0 mm peg accepts
    # the removable cap after the user's existing 8 mm blue planet is fitted.
    mesh = completelib.revolved_solid(
        [(9.0, 0.0), (9.0, 3.0), (3.5, 3.0), (3.5, 3.6),
         (2.5, 3.6), (2.5, 11.9), (1.5, 11.9), (1.5, 13.7)],
        "v4_integrated_pin_fixture",
        samples=192,
    )
    mesh.units = "mm"
    return mesh


def grooved_post(base_diameter: float, shaft_diameter: float, groove_diameter: float,
                  groove_z0: float, groove_width: float, top_z: float, name: str) -> trimesh.Trimesh:
    profile = [
        (base_diameter / 2.0, 0.0), (base_diameter / 2.0, 3.0),
        (shaft_diameter / 2.0, 3.0), (shaft_diameter / 2.0, groove_z0),
        (groove_diameter / 2.0, groove_z0), (groove_diameter / 2.0, groove_z0 + groove_width),
        (shaft_diameter / 2.0, groove_z0 + groove_width), (shaft_diameter / 2.0, top_z),
    ]
    mesh = completelib.revolved_solid(profile, name, samples=192)
    mesh.units = "mm"
    return mesh


def build_coupon(params: dict[str, Any]) -> tuple[trimesh.Trimesh, list[tuple[str, trimesh.Trimesh]]]:
    parts: list[tuple[str, trimesh.Trimesh]] = []
    parts.append(("A_pin_fixture", reviewlib.translated(pin_fixture(), (-47.0, 22.0, 0.0))))
    parts.append(("B_yellow_groove_post", reviewlib.translated(
        grooved_post(18.0, 6.0, 4.8, 10.8, 1.4, 14.2, "yellow_groove_post"), (0.0, 22.0, 0.0))))
    parts.append(("C_green_groove_post", reviewlib.translated(
        grooved_post(22.0, 9.0, 7.4, 11.5, 1.8, 15.8, "green_groove_post"), (47.0, 22.0, 0.0))))

    cap_specs = [(3.2, 8.0, -27.0), (3.3, 9.0, 0.0), (3.4, 10.0, 28.0)]
    for socket, outer, x in cap_specs:
        parts.append((f"D_cap_socket_{socket:.1f}", reviewlib.translated(blind_cap(outer, socket), (x, -8.0, 0.0))))

    yellow_specs = [(4.9, 8.6, -30.0), (5.0, 9.0, 0.0), (5.1, 9.4, 30.0)]
    for inner, outer, x in yellow_specs:
        parts.append((f"E_yellow_clip_id_{inner:.1f}", reviewlib.translated(c_clip(outer, inner, 1.2), (x, -28.0, 0.0))))

    green_specs = [(7.5, 13.4, -32.0), (7.7, 14.0, 0.0), (7.9, 14.6, 33.0)]
    for inner, outer, x in green_specs:
        parts.append((f"F_green_clip_id_{inner:.1f}", reviewlib.translated(c_clip(outer, inner, 1.6), (x, -49.0, 0.0))))

    v2lib.ensure_aabb_separation(parts, 1.5)
    combined = trimesh.util.concatenate([mesh for _, mesh in parts])
    combined.metadata["name"] = "ngw_v4_retention_fit_coupon"
    combined.units = "mm"
    return combined, parts


def make_preview(parts: list[tuple[str, trimesh.Trimesh]], output: Path) -> None:
    green_items = [(name, mesh, reviewlib.GREEN) for name, mesh in parts]
    top = reviewlib.render_items(green_items, 0, 0, "V4限位试片俯视：保持整体布局导入", size=(900, 620))
    oblique = reviewlib.render_items(green_items, 28, -18, "V4限位试片斜视：三个立柱、九个小件", size=(900, 620))
    board = Image.new("RGB", (1800, 740), "white")
    board.paste(top, (0, 0))
    board.paste(oblique, (900, 0))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype(str(reviewlib.FONT_BOLD), 22)
    small = ImageFont.truetype(str(reviewlib.FONT_BOLD), 19)
    draw.text((22, 634), "上排：细轴夹具｜黄6mm槽轴｜绿9mm槽轴　中排：端帽孔3.2/3.3/3.4", fill=(25,25,25), font=font)
    draw.text((22, 672), "下两排：黄卡环ID4.9/5.0/5.1｜绿卡环ID7.5/7.7/7.9；从最大ID开始试，不可强掰。", fill=(25,25,25), font=small)
    draw.text((22, 706), "只打印这张试片；用现有蓝轮实测5.0细轴与端帽。完整V4主件尚未生成。", fill=(155,48,35), font=font)
    board.save(output)


def build() -> Path:
    params: dict[str, Any] = {
        "source_v3_run": "RUN-20260905-164113-f797f7",
        "approved_review": "workspace/docs/NGW_V4隐藏限位与复用方案审查_v1.png",
        "planet_existing_nominal_bore_mm": 5.3,
        "planet_existing_thickness_mm": 8.0,
        "integrated_pin_journal_mm": 5.0,
        "cap_peg_mm": 3.0,
        "cap_socket_candidates_mm": [3.2, 3.3, 3.4],
        "yellow_shaft_mm": 6.0,
        "yellow_groove_mm": [4.8, 1.4],
        "yellow_clip_inner_candidates_mm": [4.9, 5.0, 5.1],
        "green_shaft_mm": 9.0,
        "green_groove_mm": [7.4, 1.8],
        "green_clip_inner_candidates_mm": [7.5, 7.7, 7.9],
        "material_assumption": "PLA, Bambu Lab A1, 0.4 mm nozzle",
    }
    signature = hashlib.sha256(json.dumps(params, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{signature}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    coupon, parts = build_coupon(params)
    stl_path = run_dir / "ngw_v4_retention_fit_coupon_v1.stl"
    check = v2lib.export_checked(coupon, stl_path, expected_components=len(parts))
    preview_path = run_dir / "ngw_v4_retention_fit_coupon_preview_v1.png"
    make_preview(parts, preview_path)

    report = {
        "run_id": run_id,
        "status": "V4_RETENTION_COUPON_READY_WAITING_SLICER_AND_PHYSICAL_TEST",
        "full_v4_stl_status": "NOT_GENERATED_WAITING_RETENTION_COUPON_RESULT",
        "parameters": params,
        "component_count": len(parts),
        "component_names": [name for name, _ in parts],
        "mesh_check": check,
        "test_order": {
            "cap": "Place one existing blue planet on fixture A, then try cap socket 3.4, 3.3, 3.2 from loose to tight.",
            "yellow_clip": "Try ID 5.1, 5.0, 4.9 on fixture B from loose to tight.",
            "green_clip": "Try ID 7.9, 7.7, 7.5 on fixture C from loose to tight.",
        },
        "pass_criteria": [
            "Existing blue planet slides fully over the 5.0 mm journal and rotates without binding or obvious radial wobble.",
            "Selected cap presses onto the 3.0 mm peg, stays under a gentle pull, and can be removed without damage.",
            "Selected C clip enters the groove without cracking and does not leave the groove under a gentle axial push.",
        ],
        "files": {
            "coupon_stl": {"path": str(stl_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(stl_path)},
            "preview_png": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(preview_path)},
        },
    }
    report_path = run_dir / "ngw_v4_retention_coupon_report_v1.json"
    reviewlib.write_json(report_path, report)

    guide = run_dir / "README_先打印V4限位试片.md"
    guide.write_text(
        "# V4 隐藏限位小试片\n\n"
        "本 RUN 只允许打印 `ngw_v4_retention_fit_coupon_v1.stl`，完整 V4 主件尚未生成。\n\n"
        "## 切片\n\n"
        "- Bambu Lab A1、0.4 mm 喷嘴、PLA、毫米单位。\n"
        "- 保持当前朝向，不缩放；不加支撑。\n"
        "- 导入后应看到 12 个分离实体：3 个带底座立柱、3 个端帽、3 个黄轴卡环候选、3 个绿轴卡环候选。\n"
        "- 切片预览确认三个盲孔没有被填死、两个槽轴沟槽清楚、六个 C 环开口没有粘死。\n\n"
        "## 完全冷却后测试\n\n"
        "1. 把一颗现有蓝轮套到左上 5.0 mm 细轴夹具上。应能到底、自由转动且没有明显左右晃动。\n"
        "2. 端帽按孔 3.4 → 3.3 → 3.2 mm，从最松到最紧试。不要强压；外径 10/9/8 mm 分别对应孔 3.4/3.3/3.2。\n"
        "3. 黄轴卡环按内径 5.1 → 5.0 → 4.9 mm 测试；外径 9.4/9.0/8.6 mm 用于辨认。\n"
        "4. 绿轴卡环按内径 7.9 → 7.7 → 7.5 mm 测试；外径 14.6/14.0/13.4 mm 用于辨认。\n"
        "5. 卡环应能从侧面进入槽，不开裂；轻推轴向时不从槽中跳出。C 环只在开口方向轻微张开，不能扭折。\n\n"
        "## 反馈格式\n\n"
        "- 蓝轮套细轴：插不入 / 过紧 / 转动合适 / 明显松晃。\n"
        "- 端帽：3.4、3.3、3.2 分别反馈插不入 / 过紧 / 牢固可拆 / 松动。\n"
        "- 黄卡环：5.1、5.0、4.9 分别反馈装不入或开裂 / 合适 / 会脱出。\n"
        "- 绿卡环：7.9、7.7、7.5 分别反馈装不入或开裂 / 合适 / 会脱出。\n",
        encoding="utf-8",
    )
    # Rewrite report once so the guide is also in the manifest.
    report["files"]["guide"] = {"path": str(guide.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(guide)}
    reviewlib.write_json(report_path, report)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    build()
