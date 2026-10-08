#!/usr/bin/env python3
"""Build only the NGW competition flat-base fit coupon; no full parts."""

from __future__ import annotations

import os

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import trimesh
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_ngw_complete_demo as completelib  # noqa: E402
import build_ngw_complete_stl_v2 as v2lib  # noqa: E402
import build_ngw_image_reference_review as reviewlib  # noqa: E402


def tapered_pin_fixture(diameter: float) -> trimesh.Trimesh:
    """Base plus 8 mm full journal and 2 mm lead-in taper for an old blue gear."""
    mesh = completelib.revolved_solid(
        [
            (9.0, 0.0), (9.0, 3.0),
            (diameter / 2.0, 3.0), (diameter / 2.0, 11.0),
            (1.6, 13.0),
        ],
        f"competition_planet_pin_{diameter:.1f}",
        samples=192,
    )
    mesh.units = "mm"
    return mesh


def coarse_shaft_fixture() -> trimesh.Trimesh:
    """Short Ø14 shaft with a 1 mm perimeter chamfer, matching the flat-base concept."""
    mesh = completelib.revolved_solid(
        [
            (13.0, 0.0), (13.0, 3.0),
            (7.0, 3.0), (7.0, 6.5),
            (6.0, 7.5),
        ],
        "competition_green_output_shaft_14",
        samples=192,
    )
    mesh.units = "mm"
    return mesh


def build_coupon() -> tuple[trimesh.Trimesh, list[tuple[str, trimesh.Trimesh]], list[dict[str, float]]]:
    items: list[tuple[str, trimesh.Trimesh]] = []
    pin_layout = [
        {"diameter_mm": 4.4, "base_diameter_mm": 18.0, "x_mm": -58.0},
        {"diameter_mm": 4.6, "base_diameter_mm": 18.0, "x_mm": 0.0},
        {"diameter_mm": 4.8, "base_diameter_mm": 18.0, "x_mm": 58.0},
    ]
    for item in pin_layout:
        mesh = reviewlib.translated(tapered_pin_fixture(item["diameter_mm"]), (item["x_mm"], 24.0, 0.0))
        items.append((f"pin_{item['diameter_mm']:.1f}", mesh))

    shaft = reviewlib.translated(coarse_shaft_fixture(), (-55.0, -24.0, 0.0))
    items.append(("shaft_14.0", shaft))

    hole_layout = [
        {"hole_mm": 14.8, "outer_mm": 26.0, "x_mm": -18.0},
        {"hole_mm": 15.0, "outer_mm": 28.0, "x_mm": 20.0},
        {"hole_mm": 15.2, "outer_mm": 30.0, "x_mm": 61.0},
    ]
    for item in hole_layout:
        ring = reviewlib.annulus(
            item["outer_mm"] / 2.0,
            item["hole_mm"] / 2.0,
            0.0,
            3.0,
            f"red_backplate_hole_{item['hole_mm']:.1f}",
        )
        items.append((f"hole_{item['hole_mm']:.1f}", reviewlib.translated(ring, (item["x_mm"], -24.0, 0.0))))

    v2lib.ensure_aabb_separation(items, 2.0)
    coupon = trimesh.util.concatenate([mesh for _, mesh in items])
    coupon.metadata["name"] = "ngw_competition_flat_base_fit_coupon"
    coupon.units = "mm"
    return coupon, items, pin_layout + hole_layout


def make_preview(items: list[tuple[str, trimesh.Trimesh]], output: Path) -> None:
    colored = []
    for name, mesh in items:
        color = reviewlib.RED if name.startswith("hole") else reviewlib.GREEN
        colored.append((name, mesh, color))
    top = reviewlib.render_items(colored, 0, 0, "比赛版配合试片俯视：保持布局导入", size=(900, 560))
    oblique = reviewlib.render_items(colored, 28, -18, "斜视：三档细轴、一个粗轴、三档孔", size=(900, 560))
    board = Image.new("RGB", (1800, 700), "white")
    board.paste(top, (0, 0))
    board.paste(oblique, (900, 0))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype(str(reviewlib.FONT_BOLD), 22)
    draw.text((22, 578), "上排：左/中/右 = Ø4.4 / 4.6 / 4.8 细轴，用同一颗现有蓝轮依次测试", fill=(25,25,25), font=font)
    draw.text((22, 618), "下排：Ø14 绿轴｜外径26=孔14.8｜外径28=孔15.0｜外径30=孔15.2", fill=(25,25,25), font=font)
    draw.text((22, 658), "从最松开始：细轴4.4→4.6→4.8；粗轴先试孔15.2→15.0→14.8。完整比赛版尚未生成。", fill=(155,48,35), font=font)
    board.save(output)


def build() -> Path:
    params: dict[str, Any] = {
        "source_v3_run": "RUN-20260905-164113-f797f7",
        "approved_competition_review": "workspace/docs/NGW比赛平底版结构审查_v1.png",
        "flat_output_face": True,
        "output_socket": None,
        "existing_blue_planet_nominal_hole_mm": 5.3,
        "existing_blue_planet_thickness_mm": 8.0,
        "integrated_pin_candidates_mm": [4.4, 4.6, 4.8],
        "pin_full_journal_length_mm": 8.0,
        "pin_lead_in_length_mm": 2.0,
        "green_output_shaft_mm": 14.0,
        "red_backplate_hole_candidates_mm": [14.8, 15.0, 15.2],
        "red_backplate_test_thickness_mm": 3.0,
        "material_assumption": "PLA, Bambu Lab A1, 0.4 mm nozzle",
    }
    signature = hashlib.sha256(json.dumps(params, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{signature}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    coupon, items, layout = build_coupon()
    stl_path = run_dir / "ngw_competition_flat_base_fit_coupon_v1.stl"
    check = v2lib.export_checked(coupon, stl_path, expected_components=len(items))
    preview_path = run_dir / "ngw_competition_flat_base_fit_coupon_preview_v1.png"
    make_preview(items, preview_path)

    guide_path = run_dir / "README_先打印比赛平底版配合试片.md"
    guide_path.write_text(
        "# NGW 比赛平底版配合试片\n\n"
        "本 RUN 只允许打印 `ngw_competition_flat_base_fit_coupon_v1.stl`。红杯体和绿色三臂架正式 STL 尚未生成。\n\n"
        "## Bambu Studio 检查\n\n"
        "- A1、0.4 mm 喷嘴、PLA、毫米单位，不缩放、不改变朝向、无支撑。\n"
        "- 导入后应为 7 个独立实体：上排 3 根带底座细轴；下排 1 根 Ø14 粗轴和 3 个孔环。\n"
        "- 确认三根细轴的锥形尖端存在，三个大孔没有被填死。\n\n"
        "## 完全冷却后测试\n\n"
        "1. 拿同一颗现有蓝色行星轮，从上排左侧 Ø4.4 开始套入，再测试 Ø4.6、Ø4.8。\n"
        "2. 目标是机器人无需压入即可到底、可以自由转动，同时径向晃动不要大到影响齿轮啮合。\n"
        "3. 用下排左侧 Ø14 粗轴先试最右侧外径30/孔15.2，再试外径28/孔15.0，最后试外径26/孔14.8。\n"
        "4. 目标是粗轴依靠倒角容易插入，不需要按压，转动顺畅且没有明显歪斜。禁止强压或强拆。\n\n"
        "## 反馈格式\n\n"
        "- 蓝轮与细轴：4.4、4.6、4.8 分别反馈‘插不入 / 过紧 / 合适 / 过松’。\n"
        "- Ø14 粗轴与红孔：15.2、15.0、14.8 分别反馈‘插不入 / 过紧 / 合适 / 过松’。\n"
        "- 如果两档都能用，优先选择机器人更容易无接触找正、无需用力的一档。\n",
        encoding="utf-8",
    )

    report = {
        "run_id": run_id,
        "status": "COMPETITION_FLAT_BASE_FIT_COUPON_READY_WAITING_PHYSICAL_TEST",
        "full_competition_stl_status": "NOT_GENERATED_WAITING_FIT_RESULT",
        "parameters": params,
        "component_count": len(items),
        "component_names": [name for name, _ in items],
        "layout": layout,
        "mesh_check": check,
        "test_order": {
            "planet_pin": [4.4, 4.6, 4.8],
            "coarse_shaft_hole": [15.2, 15.0, 14.8],
        },
        "files": {
            "coupon_stl": {"path": str(stl_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(stl_path)},
            "preview_png": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(preview_path)},
            "guide": {"path": str(guide_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(guide_path)},
        },
    }
    report_path = run_dir / "ngw_competition_flat_base_fit_coupon_report_v1.json"
    reviewlib.write_json(report_path, report)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    build()
