"""Create a strict RGBA cutout for a reviewed white-background service image.

The script is CPU-only. It separates the subject by its RGB distance from the
median border color, retains the largest foreground component, and reports the
enclosed transparent regions so fan/rotor openings can be checked before 3D
generation. Existing outputs are never overwritten.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()


def inside(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    return resolved == parent or parent in resolved.parents


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为纯色浅背景图片生成严格二值 Alpha 遮罩"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--rgba-output", required=True, type=Path)
    parser.add_argument("--preview-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=40.0)
    parser.add_argument("--border-width", type=int, default=20)
    parser.add_argument("--minimum-hole-pixels", type=int, default=100)
    parser.add_argument(
        "--trim-below-row",
        type=int,
        default=None,
        help="可选：将该行及以下像素清为背景，用于去除与主体相连的底部阴影。",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    if not input_path.is_file() or not inside(input_path, RUNS_ROOT):
        raise ValueError("输入图片必须存在且位于 workspace/runs 内。")
    input_run = next(
        (parent for parent in input_path.parents if parent.parent == RUNS_ROOT), None
    )
    if input_run is None:
        raise ValueError("无法识别输入图片所属任务目录。")
    for output in (args.rgba_output, args.preview_output, args.report_output):
        resolved = output.resolve()
        if not inside(resolved, input_run):
            raise ValueError("所有遮罩输出必须位于输入图片所属任务目录内。")
        if resolved.exists():
            raise FileExistsError(f"拒绝覆盖已有输出：{resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.border_width <= 100:
        raise ValueError("border-width 必须在 1 到 100 之间。")
    if not 1 <= args.minimum_hole_pixels:
        raise ValueError("minimum-hole-pixels 必须为正整数。")


def main() -> None:
    args = parse_args()
    validate_paths(args)

    source = np.asarray(Image.open(args.input).convert("RGB"))
    height, width, _ = source.shape
    border_width = min(args.border_width, height // 4, width // 4)
    border = np.concatenate(
        [
            source[:border_width].reshape(-1, 3),
            source[-border_width:].reshape(-1, 3),
            source[:, :border_width].reshape(-1, 3),
            source[:, -border_width:].reshape(-1, 3),
        ]
    )
    background_rgb = np.median(border, axis=0)
    color_distance = np.sqrt(
        ((source.astype(np.float32) - background_rgb) ** 2).sum(axis=2)
    )
    initial_foreground = color_distance > args.threshold

    structure = np.ones((3, 3), dtype=int)
    labels, component_count = ndimage.label(
        initial_foreground, structure=structure
    )
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    if not component_sizes.any():
        raise ValueError("当前阈值没有检测到前景主体。")
    main_label = int(component_sizes.argmax())
    foreground = labels == main_label
    main_component_pixels_raw = int(foreground.sum())
    trimmed_bottom_pixels = 0
    if args.trim_below_row is not None:
        if not 1 <= args.trim_below_row < height:
            raise ValueError("trim-below-row 必须位于图片有效行范围内。")
        trimmed_bottom_pixels = int(foreground[args.trim_below_row :].sum())
        foreground[args.trim_below_row :] = False
    main_foreground_pixels_before_hole_fill = int(foreground.sum())

    initial_background_labels, _ = ndimage.label(~foreground, structure=structure)
    initial_background_sizes = np.bincount(initial_background_labels.ravel())
    initial_border_labels = set(
        np.unique(
            np.concatenate(
                [
                    initial_background_labels[0],
                    initial_background_labels[-1],
                    initial_background_labels[:, 0],
                    initial_background_labels[:, -1],
                ]
            )
        ).tolist()
    )
    initial_enclosed = [
        (label, int(size))
        for label, size in enumerate(initial_background_sizes)
        if label != 0 and label not in initial_border_labels
    ]
    small_hole_labels = [
        label
        for label, size in initial_enclosed
        if size < args.minimum_hole_pixels
    ]
    filled_small_hole_pixels = 0
    for label in small_hole_labels:
        pixels = initial_background_labels == label
        filled_small_hole_pixels += int(pixels.sum())
        foreground[pixels] = True

    final_labels, final_component_count = ndimage.label(
        foreground, structure=structure
    )
    final_component_sizes = np.bincount(final_labels.ravel())
    final_component_sizes[0] = 0
    final_main_label = int(final_component_sizes.argmax())
    final_main = final_labels == final_main_label
    detached_after_trim_pixels = int(foreground.sum() - final_main.sum())
    foreground = final_main

    background_labels, _ = ndimage.label(~foreground, structure=structure)
    background_sizes = np.bincount(background_labels.ravel())
    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    background_labels[0],
                    background_labels[-1],
                    background_labels[:, 0],
                    background_labels[:, -1],
                ]
            )
        ).tolist()
    )
    enclosed_sizes = sorted(
        [
            int(size)
            for label, size in enumerate(background_sizes)
            if label != 0 and label not in border_labels
        ],
        reverse=True,
    )
    large_hole_sizes = [
        size for size in enclosed_sizes if size >= args.minimum_hole_pixels
    ]

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = source
    rgba[:, :, 3] = foreground.astype(np.uint8) * 255
    rgba[~foreground, :3] = 0
    Image.fromarray(rgba, "RGBA").save(args.rgba_output)

    tile = 24
    yy, xx = np.indices((height, width))
    checker = np.where(((xx // tile + yy // tile) % 2)[..., None] == 0, 224, 176)
    checker = np.repeat(checker.astype(np.uint8), 3, axis=2)
    composite = np.where(foreground[:, :, None], source, checker)
    alpha_view = np.repeat(foreground[:, :, None], 3, axis=2).astype(np.uint8) * 255
    preview = np.concatenate([source, composite, alpha_view], axis=1)
    Image.fromarray(preview, "RGB").save(args.preview_output)

    rows, columns = np.nonzero(foreground)
    foreground_bbox = [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]
    total_initial = int(initial_foreground.sum())
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "border_median_rgb_distance_topology_cleanup_v3",
        "input_path": relative(args.input),
        "input_sha256": sha256(args.input),
        "rgba_path": relative(args.rgba_output),
        "preview_path": relative(args.preview_output),
        "image_size": [width, height],
        "parameters": {
            "threshold": args.threshold,
            "border_width": border_width,
            "minimum_hole_pixels": args.minimum_hole_pixels,
            "trim_below_row": args.trim_below_row,
        },
        "background_rgb_median": background_rgb.round(3).tolist(),
        "initial_foreground_components": int(component_count),
        "initial_foreground_pixels": total_initial,
        "main_component_pixels_raw": main_component_pixels_raw,
        "trimmed_bottom_pixels": trimmed_bottom_pixels,
        "main_foreground_pixels_before_hole_fill": (
            main_foreground_pixels_before_hole_fill
        ),
        "main_foreground_pixels": int(foreground.sum()),
        "main_share_of_initial_foreground": round(
            float(main_component_pixels_raw / total_initial), 8
        ),
        "foreground_fraction": round(float(foreground.mean()), 8),
        "foreground_bbox_xyxy": foreground_bbox,
        "corner_alpha": [
            int(rgba[0, 0, 3]),
            int(rgba[0, -1, 3]),
            int(rgba[-1, 0, 3]),
            int(rgba[-1, -1, 3]),
        ],
        "initial_enclosed_transparent_region_count": len(initial_enclosed),
        "filled_small_hole_count": len(small_hole_labels),
        "filled_small_hole_pixels": filled_small_hole_pixels,
        "foreground_components_before_final_filter": int(final_component_count),
        "detached_after_trim_pixels_removed": detached_after_trim_pixels,
        "final_foreground_components": 1,
        "enclosed_transparent_region_count": len(enclosed_sizes),
        "enclosed_transparent_region_sizes": enclosed_sizes,
        "large_transparent_hole_count": len(large_hole_sizes),
        "large_transparent_hole_sizes": large_hole_sizes,
    }
    report["rgba_sha256"] = sha256(args.rgba_output)
    report["preview_sha256"] = sha256(args.preview_output)
    write_json(args.report_output, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
