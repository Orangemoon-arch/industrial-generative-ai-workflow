"""Build the second NGW carrier pin-hole coupon after the tight v1 result.

The coupon contains three identical 5.3 mm pins and three hole rings sized
5.4/5.5/5.6 mm.  No complete assembly STL is generated.
"""

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
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_ngw_complete_demo as completelib  # noqa: E402
import build_ngw_image_reference_review as reviewlib  # noqa: E402


def make_pin() -> trimesh.Trimesh:
    pin = completelib.revolved_solid(
        [(4.5, 0.0), (4.5, 2.0), (2.65, 2.0), (2.65, 10.0)],
        "coupon_v2_pin_5p3",
    )
    pin.units = "mm"
    return pin


def build_coupon() -> tuple[trimesh.Trimesh, list[dict[str, float]]]:
    pin = make_pin()
    groups = [
        {"hole_mm": 5.4, "outer_mm": 14.0, "x_mm": -34.0},
        {"hole_mm": 5.5, "outer_mm": 16.0, "x_mm": 0.0},
        {"hole_mm": 5.6, "outer_mm": 18.0, "x_mm": 36.0},
    ]
    items: list[trimesh.Trimesh] = []
    for group in groups:
        items.append(reviewlib.translated(pin, (group["x_mm"], -11.0, 0.0)))
        ring = reviewlib.annulus(
            group["outer_mm"] / 2.0,
            group["hole_mm"] / 2.0,
            0.0,
            5.0,
            f"coupon_v2_hole_{group['hole_mm']:.1f}",
        )
        items.append(reviewlib.translated(ring, (group["x_mm"], 11.0, 0.0)))
    coupon = trimesh.util.concatenate(items)
    coupon.metadata["name"] = "ngw_pin_hole_coupon_v2"
    coupon.units = "mm"
    return coupon, groups


def make_preview(coupon: trimesh.Trimesh, output: Path) -> None:
    items = [("coupon_v2", coupon, reviewlib.GREEN)]
    top = reviewlib.render_items(items, 0, 0, "俯视：每个孔环配一根独立5.3销", size=(900, 520))
    oblique = reviewlib.render_items(items, 28, -18, "斜视：保持竖直方向，无支撑", size=(900, 520))
    board = Image.new("RGB", (1800, 590), "white")
    board.paste(top, (0, 0))
    board.paste(oblique, (900, 0))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype(str(reviewlib.FONT_BOLD), 24)
    draw.text(
        (22, 539),
        "孔环：外径14=孔5.4｜外径16=孔5.5｜外径18=孔5.6；建议先试最松的外径18",
        fill=(25, 25, 25),
        font=font,
    )
    board.save(output)


def mesh_record(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return reviewlib.mesh_metrics(mesh)


def build() -> Path:
    params = {
        "pin_diameter_mm": 5.3,
        "pin_head_diameter_mm": 9.0,
        "pin_head_thickness_mm": 2.0,
        "pin_shaft_length_mm": 8.0,
        "hole_diameters_mm": [5.4, 5.5, 5.6],
        "ring_outer_diameters_mm": [14.0, 16.0, 18.0],
        "ring_height_mm": 5.0,
        "source_coupon_run": "RUN-20260905-104737-cef824",
        "source_result": "5.3 hole too tight and difficult to remove; 5.2 and 5.1 cannot insert",
    }
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{signature}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    coupon, groups = build_coupon()
    stl_path = run_dir / "ngw_planet_pin_press_fit_coupon_v2.stl"
    coupon.export(stl_path)
    preview_path = run_dir / "ngw_planet_pin_press_fit_coupon_preview_v2.png"
    make_preview(coupon, preview_path)

    reloaded = trimesh.load(stl_path, force="mesh", process=True)
    reloaded.merge_vertices()
    parts = reloaded.split(only_watertight=False)
    overall = mesh_record(reloaded)
    component_checks = [mesh_record(part) for part in parts]
    if len(parts) != 6:
        raise RuntimeError(f"Expected 6 coupon components, found {len(parts)}")
    if not all(item["watertight"] and item["winding_consistent"] and item["positive_volume"] for item in component_checks):
        raise RuntimeError("Every coupon component must be watertight and valid")

    report = {
        "run_id": run_id,
        "status": "PIN_HOLE_COUPON_V2_READY_WAITING_SLICER_AND_PHYSICAL_TEST",
        "full_set_stl_status": "NOT_GENERATED_WAITING_COUPON_V2_RESULT",
        "parameters": params,
        "layout_identity": groups,
        "test_order": [
            "Outer diameter 18 mm ring / 5.6 mm hole first.",
            "Outer diameter 16 mm ring / 5.5 mm hole second.",
            "Outer diameter 14 mm ring / 5.4 mm hole last.",
        ],
        "result_labels": ["cannot insert", "too tight", "firm removable", "firm permanent", "loose"],
        "mesh_check": overall,
        "component_checks": component_checks,
        "files": {
            "coupon_stl": {"path": str(stl_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(stl_path)},
            "preview_png": {"path": str(preview_path.relative_to(PROJECT_ROOT)), "sha256": reviewlib.sha256(preview_path)},
        },
        "gates": [
            "Only this six-component coupon STL is allowed for slicer review.",
            "Do not force any pin and keep each pin paired with its own ring.",
            "No complete NGW assembly STL was generated.",
        ],
    }
    report_path = run_dir / "ngw_pin_coupon_v2_report.json"
    reviewlib.write_json(report_path, report)
    guide_path = run_dir / "README_第二轮销孔试片.md"
    guide_path.write_text(
        "# NGW 行星架销孔第二轮配合试片\n\n"
        "第一轮结果：5.3 mm 孔过紧且难拆，5.2/5.1 mm 孔插不入。完整组件继续暂停。\n\n"
        "本 STL 有 6 个分开实体，每个孔环旁边各有一根独立 5.3 mm 销。外径 14 mm 环是 5.4 mm 孔；"
        "外径 16 mm 环是 5.5 mm 孔；外径 18 mm 环是 5.6 mm 孔。\n\n"
        "Bambu Studio：A1、0.4 mm 喷嘴、PLA、毫米单位、保持竖直方向、无支撑；确认 6 个实体和孔内无支撑。\n\n"
        "冷却后先试外径 18 mm / 5.6 mm 孔，再试外径 16 mm / 5.5 mm，最后才试外径 14 mm / 5.4 mm。"
        "三根销互相独立，即使一根卡住也不要强拆。分别记录：插不入、过紧、牢固可拆、牢固永久或松动。\n",
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
