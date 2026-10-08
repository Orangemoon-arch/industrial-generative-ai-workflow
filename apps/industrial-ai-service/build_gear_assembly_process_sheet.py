"""Build a deterministic Chinese assembly-process sheet for the three-part sample."""

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

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"
RUN_ID_PATTERN = re.compile(r"RUN-\d{8}-\d{6}-[a-z0-9]{6}")
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

BLUE = "#397EC1"
ORANGE = "#E98A2E"
GREEN = "#62AD67"
DARK = "#17324D"
MUTED = "#60758A"
PANEL = "#F6F9FC"
LINE = "#CAD6E2"
RED = "#C44545"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def gear_points(cx: float, cy: float, outer: float, root: float, teeth: int = 16) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(teeth * 4):
        phase = index % 4
        radius = outer if phase in (1, 2) else root
        angle = -math.pi / 2.0 + index * 2.0 * math.pi / (teeth * 4)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def draw_gear(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: str = BLUE) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    outer = min(x1 - x0, y1 - y0) * 0.47
    draw.polygon(gear_points(cx, cy, outer, outer * 0.82), fill=color, outline=DARK, width=3)
    bore_r = outer * 0.24
    flat_x = cx + bore_r * 0.60
    bore: list[tuple[float, float]] = []
    for i in range(120):
        angle = i * 2.0 * math.pi / 120
        px = min(cx + bore_r * math.cos(angle), flat_x)
        py = cy + bore_r * math.sin(angle)
        bore.append((px, py))
    draw.polygon(bore, fill="white", outline=DARK)


def draw_shaft_side(draw: ImageDraw.ImageDraw, cx: int, base_y: int, scale: float = 1.0) -> None:
    base_w, base_h = int(120 * scale), int(32 * scale)
    shaft_w, shaft_h = int(52 * scale), int(105 * scale)
    peg_w, peg_h = int(34 * scale), int(38 * scale)
    draw.rounded_rectangle(
        [cx - base_w // 2, base_y - base_h, cx + base_w // 2, base_y],
        radius=max(4, int(10 * scale)), fill=ORANGE, outline=DARK, width=3,
    )
    shaft_top = base_y - base_h - shaft_h
    draw.rectangle(
        [cx - shaft_w // 2, shaft_top, cx + shaft_w // 2, base_y - base_h],
        fill=ORANGE, outline=DARK, width=3,
    )
    draw.rounded_rectangle(
        [cx - peg_w // 2, shaft_top - peg_h, cx + peg_w // 2, shaft_top + 4],
        radius=max(3, int(7 * scale)), fill=ORANGE, outline=DARK, width=3,
    )


def draw_down_arrow(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int) -> None:
    draw.line([x, y0, x, y1 - 14], fill=MUTED, width=7)
    draw.polygon([(x - 14, y1 - 20), (x + 14, y1 - 20), (x, y1)], fill=MUTED)


def centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, text_font: ImageFont.FreeTypeFont, fill: str = DARK) -> None:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1]), text, font=text_font, fill=fill)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], number: int, title: str) -> tuple[int, int, int, int]:
    draw.rounded_rectangle(box, radius=20, fill=PANEL, outline=LINE, width=3)
    x0, y0, x1, y1 = box
    draw.ellipse([x0 + 20, y0 + 18, x0 + 72, y0 + 70], fill=DARK)
    centered_text(draw, (x0 + 46, y0 + 22), str(number), font(30), fill="white")
    draw.text((x0 + 88, y0 + 22), title, font=font(30), fill=DARK)
    return x0 + 20, y0 + 82, x1 - 20, y1 - 20


def draw_sheet(run_id: str, params: dict[str, Any], version: int, output: Path) -> None:
    image = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1920, 118], fill=DARK)
    draw.text((60, 22), f"D形轴齿轮三件套｜装配工艺设计图 V{version}", font=font(46), fill="white")
    draw.text((60, 79), f"任务 {run_id}　教学装配与流程验证｜禁止用于动力传动或承载", font=font(22), fill="#D9E6F2")

    margin, gap = 42, 24
    panel_w = (1920 - margin * 2 - gap * 2) // 3
    panel_h = 390
    boxes = []
    for row in range(2):
        for col in range(3):
            x0 = margin + col * (panel_w + gap)
            y0 = 145 + row * (panel_h + gap)
            boxes.append((x0, y0, x0 + panel_w, y0 + panel_h))

    content = panel(draw, boxes[0], 1, "确认BOM与零件")
    x0, y0, x1, y1 = content
    draw_gear(draw, (x0 + 5, y0 + 30, x0 + 190, y0 + 215))
    draw_shaft_side(draw, x0 + 315, y0 + 225, 0.75)
    draw.ellipse([x0 + 410, y0 + 85, x0 + 535, y0 + 155], fill=GREEN, outline=DARK, width=3)
    draw.text((x0 + 20, y1 - 55), "齿轮 ×1　D形短轴 ×1　限位帽 ×1", font=font(22), fill=MUTED)

    content = panel(draw, boxes[1], 2, "固定短轴与工装")
    x0, y0, x1, y1 = content
    draw.rectangle([x0 + 80, y1 - 55, x1 - 80, y1 - 15], fill="#AAB7C4", outline=DARK, width=3)
    draw_shaft_side(draw, (x0 + x1) // 2, y1 - 55, 1.0)
    draw.text((x0 + 35, y0 + 5), "底座完全落入定位工装，轴线保持竖直", font=font(22), fill=MUTED)

    content = panel(draw, boxes[2], 3, "识别D面并对准")
    x0, y0, x1, y1 = content
    draw_gear(draw, (x0 + 125, y0 + 20, x1 - 125, y1 - 45))
    draw.arc([x0 + 75, y0 + 5, x1 - 75, y1 - 25], 205, 315, fill=RED, width=8)
    draw.polygon([(x1 - 102, y0 + 82), (x1 - 62, y0 + 75), (x1 - 82, y0 + 112)], fill=RED)
    draw.text((x0 + 25, y1 - 40), "旋转齿轮，使孔内D面与轴D面同向", font=font(22), fill=MUTED)

    content = panel(draw, boxes[3], 4, "沿轴线插入齿轮")
    x0, y0, x1, y1 = content
    cx = (x0 + x1) // 2
    draw_shaft_side(draw, cx, y1 - 72, 0.9)
    draw.rounded_rectangle([cx - 150, y0 + 72, cx + 150, y0 + 122], radius=10, fill=BLUE, outline=DARK, width=3)
    draw_down_arrow(draw, cx, y0 + 2, y0 + 65)
    draw.text((x0 + 28, y1 - 40), "低速插入，直到齿轮端面贴合轴肩", font=font(22), fill=MUTED)

    content = panel(draw, boxes[4], 5, "安装限位帽")
    x0, y0, x1, y1 = content
    cx = (x0 + x1) // 2
    draw_shaft_side(draw, cx, y1 - 72, 0.82)
    draw.rounded_rectangle([cx - 145, y0 + 125, cx + 145, y0 + 172], radius=10, fill=BLUE, outline=DARK, width=3)
    draw.ellipse([cx - 68, y0 + 28, cx + 68, y0 + 95], fill=GREEN, outline=DARK, width=3)
    draw_down_arrow(draw, cx, y0 + 98, y0 + 120)
    draw.text((x0 + 28, y1 - 40), "盲孔对准顶部圆柱，压至端面贴合", font=font(22), fill=MUTED)

    content = panel(draw, boxes[5], 6, "复核并记录结果")
    x0, y0, x1, y1 = content
    cx = x0 + 145
    draw_shaft_side(draw, cx, y1 - 25, 0.78)
    draw.rounded_rectangle([cx - 135, y0 + 115, cx + 135, y0 + 162], radius=9, fill=BLUE, outline=DARK, width=3)
    draw.ellipse([cx - 61, y0 + 75, cx + 61, y0 + 126], fill=GREEN, outline=DARK, width=3)
    criteria = ["齿轮空转 ≤ 2°", "倒置轻晃帽不脱落", "可手动装入并拆卸", "无断齿、翘边和孔变形"]
    for idx, item in enumerate(criteria):
        yy = y0 + 45 + idx * 54
        draw.ellipse([x0 + 310, yy, x0 + 342, yy + 32], fill=GREEN)
        centered_text(draw, (x0 + 326, yy - 3), "✓", font(25), fill="white")
        draw.text((x0 + 357, yy - 1), item, font=font(21), fill=DARK)

    footer_y = 1000
    draw.line([42, footer_y - 16, 1878, footer_y - 16], fill=LINE, width=2)
    footer = (
        f"关键参数：齿轮 {float(params['gear_outer_diameter_mm']):g} mm｜"
        f"D轴 {float(params['shaft_diameter_mm']):g} mm｜"
        f"顶部圆柱 {float(params['retaining_peg_diameter_mm']):g} mm｜"
        f"D孔间隙 {float(params['gear_bore_diametral_clearance_mm']):.2f} mm｜"
        f"帽孔间隙 {float(params['cap_socket_diametral_clearance_mm']):.2f} mm"
    )
    draw.text((48, footer_y), footer, font=font(22), fill=DARK)
    draw.text((48, footer_y + 39), "状态：等待Bambu Studio检查与实体试装；机械臂/灵巧手控制未接入。", font=font(20), fill=RED)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def append_artifact(manifest: dict[str, Any], path: Path, role: str, version: int) -> None:
    relative = str(path.relative_to(PROJECT_ROOT))
    if any(item.get("path") == relative for item in manifest.get("artifacts", [])):
        raise FileExistsError(f"清单中已存在归档：{relative}")
    manifest.setdefault("artifacts", []).append(
        {
            "role": role,
            "version": version,
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成三件式齿轮轴装配工艺设计图")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run-id 格式无效。")
    if args.version < 1:
        raise ValueError("version 必须大于等于 1。")
    run_dir = RUNS_ROOT / args.run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("装配任务不存在。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_artifact = next(
        (item for item in manifest.get("artifacts", []) if item.get("role") == "design_report"),
        None,
    )
    if not report_artifact:
        raise ValueError("装配任务缺少设计报告。")
    report_path = (PROJECT_ROOT / report_artifact["path"]).resolve()
    if run_dir.resolve() not in report_path.parents or not report_path.is_file():
        raise ValueError("设计报告路径无效。")
    if report_artifact.get("sha256") != sha256(report_path):
        raise ValueError("设计报告 SHA256 不匹配。")
    design = json.loads(report_path.read_text(encoding="utf-8"))
    params = design["parameters_mm"]

    png_path = run_dir / "reports" / f"gear_shaft_cap_assembly_process_v{args.version}.png"
    guide_path = run_dir / "reports" / f"gear_shaft_cap_assembly_process_v{args.version}.md"
    template_path = run_dir / "reviews" / f"assembly_fit_acceptance_template_v{args.version}.md"
    for path in (png_path, guide_path, template_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有文件：{path}")

    draw_sheet(args.run_id, params, args.version, png_path)
    guide_path.write_text(
        f"# D形轴齿轮三件套装配工艺说明 V{args.version}\n\n"
        "1. 核对齿轮、D形短轴和限位帽各一件。\n"
        "2. 将短轴底座固定于装配工装，保持轴线竖直。\n"
        "3. 识别齿轮孔和短轴的D形平面，旋转至同向。\n"
        "4. 沿轴线低速插入齿轮，直至齿轮端面贴合轴肩。\n"
        "5. 将限位帽盲孔对准顶部圆柱，沿轴线压至端面贴合。\n"
        "6. 记录齿轮剩余空转角度、插入力、倒置脱落、拆卸和可见缺陷。\n\n"
        "验收建议：齿轮空转不大于2°，限位帽倒置轻晃不脱落，两件均可手动装入并无损拆卸。\n\n"
        "安全边界：教学装配与流程验证，不用于动力传动或承载；当前不生成或执行机器人控制命令。\n",
        encoding="utf-8",
    )
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        "# 三件套打印与试装验收记录\n\n"
        f"- 任务：`{args.run_id}`\n"
        "- 打印机/材料：Bambu Lab A1 / PLA\n"
        f"- D孔直径间隙：`{float(params['gear_bore_diametral_clearance_mm']):.2f} mm`\n"
        f"- 帽孔直径间隙：`{float(params['cap_socket_diametral_clearance_mm']):.2f} mm`\n"
        "- Bambu Studio工程或切片截图：待填写\n"
        "- 齿轮插入：待填写（无法插入/过紧/略紧可接受/顺畅/过松）\n"
        "- 齿轮剩余空转角度：待填写\n"
        "- 限位帽插入：待填写（无法插入/过紧/略紧可接受/顺畅/过松）\n"
        "- 倒置轻晃是否脱落：待填写\n"
        "- 是否可手动无损拆卸：待填写\n"
        "- 翘边、象脚、断齿或孔变形：待填写\n"
        "- 最终结论：待填写（通过/继续调整）\n",
        encoding="utf-8",
    )

    append_artifact(manifest, png_path, "assembly_process_sheet", args.version)
    append_artifact(manifest, guide_path, "assembly_process_guide", args.version)
    append_artifact(manifest, template_path, "assembly_fit_acceptance_template", args.version)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["updated_at"] = now
    manifest.setdefault("stages", {})["assembly_process_sheet"] = "GENERATED_WAITING_REVIEW"
    manifest.setdefault("events", []).append(
        {
            "time": now,
            "event": "ASSEMBLY_PROCESS_SHEET_GENERATED",
            "detail": f"生成装配工艺设计图、说明和试装验收模板 v{args.version}。",
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "process_sheet": str(png_path),
                "acceptance_template": str(template_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
