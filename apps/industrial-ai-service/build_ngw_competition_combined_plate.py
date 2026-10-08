#!/usr/bin/env python3
"""Combine the two validated competition parts into one two-shell STL plate."""

from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
SOURCE_RUN = PROJECT_ROOT / "workspace/runs/RUN-20260907-114349-24e920"
OUT_DIR = SOURCE_RUN / "combined_plate"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_ngw_complete_stl_v2 as v2lib  # noqa: E402
import build_ngw_image_reference_review as reviewlib  # noqa: E402


RED_PATH = SOURCE_RUN / "individual_stl/01_red_flat_base_ring_cup_competition_v1.stl"
GREEN_PATH = SOURCE_RUN / "individual_stl/02_green_flat_carrier_integrated_pins_competition_v1_SUPPORT_REQUIRED.stl"


def load_one(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=True)
    mesh.merge_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) != 1 or not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise RuntimeError(f"Invalid source mesh: {path}")
    return mesh


def place_side_by_side(red: trimesh.Trimesh, green: trimesh.Trimesh, gap: float = 12.0):
    red = red.copy()
    green = green.copy()
    red_width = float(red.extents[0])
    green_width = float(green.extents[0])
    total_width = red_width + gap + green_width
    left = -total_width / 2.0
    red_shift_x = left - float(red.bounds[0, 0])
    green_target_min = left + red_width + gap
    green_shift_x = green_target_min - float(green.bounds[0, 0])
    red_shift_y = -float(red.bounds[:, 1].mean())
    green_shift_y = -float(green.bounds[:, 1].mean())
    red.apply_translation((red_shift_x, red_shift_y, -float(red.bounds[0, 2])))
    green.apply_translation((green_shift_x, green_shift_y, -float(green.bounds[0, 2])))
    actual_gap = float(green.bounds[0, 0] - red.bounds[1, 0])
    if actual_gap < gap - 1e-6:
        raise RuntimeError(f"Layout gap too small: {actual_gap}")
    return red, green, actual_gap


def preview(red: trimesh.Trimesh, green: trimesh.Trimesh, output: Path) -> None:
    items = [("red_cup", red, reviewlib.RED), ("green_carrier", green, reviewlib.GREEN)]
    top = reviewlib.render_items(items, 0, 0, "组合打印板俯视：两个独立实体", size=(900, 600))
    oblique = reviewlib.render_items(items, 28, -18, "斜视：红杯＋粗轴朝下的绿架", size=(900, 600))
    board = Image.new("RGB", (1800, 690), "white")
    board.paste(top, (0, 0))
    board.paste(oblique, (900, 0))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype(str(reviewlib.FONT_BOLD), 22)
    draw.text((22, 618), "一个STL、两个分离实体、间距12mm；默认同色打印。绿架启用仅从打印板生成支撑，红杯无需支撑。", fill=(30,30,30), font=font)
    draw.text((22, 656), "Bambu Studio逐层确认支撑没有包住三根细轴，再发送打印。", fill=(155,48,35), font=font)
    board.save(output)


def build() -> Path:
    OUT_DIR.mkdir(exist_ok=False)
    red, green, gap = place_side_by_side(load_one(RED_PATH), load_one(GREEN_PATH))
    combined = trimesh.util.concatenate([red, green])
    combined.units = "mm"
    stl_path = OUT_DIR / "ngw_competition_flat_base_red_green_combined_plate_v1.stl"
    check = v2lib.export_checked(combined, stl_path, expected_components=2)
    if check["mesh"]["extents_mm"][0] > 256 or check["mesh"]["extents_mm"][1] > 256:
        raise RuntimeError(f"Combined plate exceeds A1 area: {check['mesh']['extents_mm']}")

    preview_path = OUT_DIR / "ngw_competition_flat_base_combined_plate_preview_v1.png"
    preview(red, green, preview_path)
    guide_path = OUT_DIR / "README_组合板切片说明.md"
    guide_path.write_text(
        "# 比赛平底版红绿组合打印板\n\n"
        "打印文件：`ngw_competition_flat_base_red_green_combined_plate_v1.stl`。一个 STL 内包含红杯体和绿架两个互不相连的实体，间距 12 mm；原独立 STL 均保留。\n\n"
        "## Bambu Studio\n\n"
        "- A1、0.4 mm 喷嘴、PLA、毫米单位，不缩放、不自动重新摆放。\n"
        "- 导入后应看到两个实体，总平面尺寸约 198.6 × 122 mm。\n"
        "- 默认使用同一种耗材颜色，这是一次打印节省时间的方案。\n"
        "- 开启支撑并选择‘仅从打印板生成支撑’。红杯本身不需要支撑，支撑应主要出现在绿架三条臂下方。\n"
        "- 逐层确认：红色内齿和 Ø15 孔完整；绿色粗轴平面贴打印板；三根 Ø4.6 细轴竖直且没有被支撑包住。\n"
        "- 如果切片软件把两个实体自动重排、翻转或缩放，撤销后重新导入。\n\n"
        "打印完成后完全冷却再取件，先拆绿架下方支撑，再检查三根细轴是否完整。\n",
        encoding="utf-8",
    )
    report = {
        "status": "COMPETITION_COMBINED_PLATE_READY_WAITING_BAMBU_REVIEW",
        "source_run": str(SOURCE_RUN.relative_to(PROJECT_ROOT)),
        "source_files": [str(RED_PATH.relative_to(PROJECT_ROOT)), str(GREEN_PATH.relative_to(PROJECT_ROOT))],
        "layout": {"gap_mm": gap, "extents_mm": check["mesh"]["extents_mm"], "expected_components": 2},
        "mesh_check": check,
        "files": {
            "combined_stl": {"path": str(stl_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(stl_path)},
            "preview": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(preview_path)},
            "guide": {"path": str(guide_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(guide_path)},
        },
    }
    reviewlib.write_json(OUT_DIR / "combined_plate_validation_v1.json", report)
    print(stl_path)
    return stl_path


if __name__ == "__main__":
    build()
