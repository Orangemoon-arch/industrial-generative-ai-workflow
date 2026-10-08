"""Split an audited 2x2 RGBA multiview sheet into a versioned 2mv RUN."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"
VIEW_LAYOUT = {
    "front": (0, 0),
    "left": (1, 0),
    "back": (0, 1),
    "right": (1, 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_quadrant(image: Image.Image, column: int, row: int) -> tuple[Image.Image, dict]:
    width, height = image.size
    x_edges = (0, width // 2, width)
    y_edges = (0, height // 2, height)
    quadrant = image.crop((x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1]))
    alpha = np.asarray(quadrant.getchannel("A"))
    foreground = alpha > 8
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        raise ValueError(f"第 {row + 1} 行第 {column + 1} 列没有前景。")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = quadrant.crop((x0, y0, x1, y1))
    target_extent = 420
    scale = min(target_extent / cropped.width, target_extent / cropped.height)
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    canvas.alpha_composite(resized, ((512 - resized.width) // 2, (512 - resized.height) // 2))
    normalized_alpha = np.asarray(canvas.getchannel("A"))
    return canvas, {
        "source_quadrant": [column, row],
        "source_foreground_bbox": [x0, y0, x1, y1],
        "source_foreground_pixels": int(foreground.sum()),
        "normalized_foreground_pixels": int((normalized_alpha > 8).sum()),
        "alpha_min": int(normalized_alpha.min()),
        "alpha_max": int(normalized_alpha.max()),
    }


def mask_iou(first: Image.Image, second: Image.Image) -> float:
    a = np.asarray(first.getchannel("A")) > 32
    b = np.asarray(second.getchannel("A")) > 32
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def count_radial_peaks(image: Image.Image) -> int:
    mask = (np.asarray(image.getchannel("A")) > 32).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    moments = cv2.moments(mask)
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    vectors = contour - center
    angles = (np.arctan2(vectors[:, 1], vectors[:, 0]) + 2 * np.pi) % (2 * np.pi)
    radii = np.linalg.norm(vectors, axis=1)
    bins = 1080
    profile = np.full(bins, np.nan)
    indexes = np.floor(angles / (2 * np.pi) * bins).astype(int) % bins
    for index, radius in zip(indexes, radii):
        if np.isnan(profile[index]) or radius > profile[index]:
            profile[index] = radius
    valid = np.flatnonzero(~np.isnan(profile))
    profile = np.interp(np.arange(bins), valid, profile[valid], period=bins)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (0, 1), sigmaX=3).ravel()
    extended = np.concatenate([profile, profile, profile])
    peaks = []
    minimum_distance = bins // 40
    threshold = float(profile.mean() + 0.30 * profile.std())
    for index in range(1, len(extended) - 1):
        if extended[index] > threshold and extended[index] >= extended[index - 1] and extended[index] > extended[index + 1]:
            if not peaks or index - peaks[-1] >= minimum_distance:
                peaks.append(index)
            elif extended[index] > extended[peaks[-1]]:
                peaks[-1] = index
    return sum(bins <= index < 2 * bins for index in peaks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    sheet_path = args.sheet.resolve()
    if not sheet_path.is_file() or PROJECT_ROOT not in sheet_path.parents:
        raise ValueError("四视图原图必须位于项目目录内。")
    sheet = Image.open(sheet_path).convert("RGBA")
    source_alpha = np.asarray(sheet.getchannel("A"))
    if source_alpha.min() != 0 or source_alpha.max() != 255:
        raise ValueError("四视图原图必须同时包含透明背景和完全不透明前景。")

    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_ROOT / run_id
    for name in ("requests", "masks", "meshes", "reports", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    input_dir = run_dir / "masks/multiview_input_v1"
    input_dir.mkdir()

    images = {}
    records = {}
    for view, (column, row) in VIEW_LAYOUT.items():
        normalized, metrics = normalize_quadrant(sheet, column, row)
        output = input_dir / f"{view}.png"
        normalized.save(output)
        images[view] = normalized
        records[view] = {
            "path": str(output.relative_to(PROJECT_ROOT)),
            "sha256": sha256(output),
            **metrics,
        }

    consistency = {
        "front_back_mask_iou": round(mask_iou(images["front"], images["back"]), 6),
        "left_right_mask_iou": round(mask_iou(images["left"], images["right"]), 6),
        "front_radial_peak_count_estimate": count_radial_peaks(images["front"]),
        "back_radial_peak_count_estimate": count_radial_peaks(images["back"]),
    }
    prompt = None
    if args.prompt_file:
        prompt_path = args.prompt_file.resolve()
        if not prompt_path.is_file() or PROJECT_ROOT not in prompt_path.parents:
            raise ValueError("提示词文件必须位于项目目录内。")
        prompt = {
            "path": str(prompt_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(prompt_path),
        }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "purpose": "AI-generated multiview consistency benchmark for Hunyuan3D-2mv",
        "source_generator": "Codex built-in image generation; not ERNIE",
        "competition_pipeline_status": "BENCHMARK_ONLY_NOT_ERNIE",
        "print_status": "NOT_A_PRINT_APPROVAL",
        "source": {
            "path": str(sheet_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(sheet_path),
            "size": list(sheet.size),
            "prompt": prompt,
        },
        "views": records,
        "consistency": consistency,
    }
    write_json(run_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
