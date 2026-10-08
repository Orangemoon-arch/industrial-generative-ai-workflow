"""Versioned manual multimodal capture records for task-3 preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


CAPTURE_CHECKPOINTS = [
    "before_pick",
    "after_grasp",
    "before_assembly",
    "after_assembly",
    "final_sort",
]

CAPTURE_ZONES = ["取料区", "装配区", "成品区", "异常区"]

CAPTURE_LABELS = [
    "UNLABELED",
    "PART_PRESENT",
    "GRASP_SUCCESS",
    "GRASP_FAILURE",
    "ALIGNED",
    "MISALIGNED",
    "ASSEMBLY_SUCCESS",
    "ASSEMBLY_FAILURE",
    "QUALIFIED",
    "REJECTED",
]

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(value: Any, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label}不能为空。")
    if len(text) > maximum:
        raise ValueError(f"{label}过长，请控制在 {maximum} 个字符以内。")
    return text


def _uploaded_path(upload: Any) -> Path:
    value = upload
    if hasattr(value, "path"):
        value = value.path
    elif isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("请先上传一张图片或一段短视频。")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError("上传的媒体文件不存在。")
    return path


def _is_within(path: Path, root: Path) -> bool:
    resolved_root = root.resolve()
    return path == resolved_root or resolved_root in path.parents


def _media_format(path: Path) -> tuple[str, str]:
    with path.open("rb") as handle:
        header = handle.read(32)
    suffix = path.suffix.lower()
    if header.startswith(b"\x89PNG\r\n\x1a\n") and suffix == ".png":
        return "image", ".png"
    if header.startswith(b"\xff\xd8\xff") and suffix in {".jpg", ".jpeg"}:
        return "image", ".jpg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP" and suffix == ".webp":
        return "image", ".webp"
    if len(header) >= 12 and header[4:8] == b"ftyp" and suffix in {".mp4", ".mov"}:
        return "video", suffix
    if header.startswith(b"\x1aE\xdf\xa3") and suffix == ".webm":
        return "video", ".webm"
    raise ValueError("只接受内容与扩展名一致的 PNG、JPG、WEBP、MP4、MOV 或 WEBM。")


def archive_manual_capture(
    *,
    project_root: Path,
    run_dir: Path,
    run_id: str,
    upload: Any,
    checkpoint: str,
    part_id: str,
    zone: str,
    label: str,
    notes: str = "",
    device_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy one user-provided media file into a versioned task capture folder."""
    checkpoint_value = str(checkpoint or "").strip()
    zone_value = str(zone or "").strip()
    label_value = str(label or "").strip()
    if checkpoint_value not in CAPTURE_CHECKPOINTS:
        raise ValueError("采集节点无效。")
    if zone_value not in CAPTURE_ZONES:
        raise ValueError("区域标签无效。")
    if label_value not in CAPTURE_LABELS:
        raise ValueError("人工标签无效。")
    part_value = _required_text(part_id, "零件 ID", 120)
    notes_value = str(notes or "").strip()
    if len(notes_value) > 1000:
        raise ValueError("备注过长，请控制在 1000 个字符以内。")

    source = _uploaded_path(upload)
    gradio_upload_root = Path(tempfile.gettempdir()) / "gradio"
    if not (_is_within(source, project_root) or _is_within(source, gradio_upload_root)):
        raise ValueError("上传源文件不在项目目录或 Gradio 临时上传目录中。")
    size = source.stat().st_size
    if size <= 0 or size > MAX_UPLOAD_BYTES:
        raise ValueError("媒体文件必须大于 0 且不超过 20 MiB。")
    media_type, extension = _media_format(source)

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    capture_id = f"CAP-{stamp}-{uuid.uuid4().hex[:6]}"
    capture_dir = run_dir / "multimodal" / "captures" / capture_id
    capture_dir.mkdir(parents=True, exist_ok=False)
    media_path = capture_dir / f"{media_type}{extension}"
    temporary = media_path.with_suffix(media_path.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(media_path)
    media_sha = _sha256(media_path)

    snapshot_reference = None
    if isinstance(device_snapshot, dict) and device_snapshot.get("path"):
        snapshot_reference = {
            "path": device_snapshot.get("path"),
            "sha256": device_snapshot.get("sha256"),
            "source": device_snapshot.get("source"),
        }

    record = {
        "schema_version": "1.0",
        "capture_id": capture_id,
        "run_id": run_id,
        "created_at": _now_iso(),
        "source": "MANUAL_WEB_UPLOAD",
        "checkpoint": checkpoint_value,
        "part_id": part_value,
        "zone": zone_value,
        "human_label": label_value,
        "notes": notes_value,
        "media": {
            "type": media_type,
            "original_name": source.name,
            "path": str(media_path.resolve().relative_to(project_root.resolve())),
            "bytes": media_path.stat().st_size,
            "sha256": media_sha,
        },
        "device_snapshot": snapshot_reference,
        "calibration": {"status": "UNAVAILABLE", "calibration_id": None},
        "model_review": {"qwen_status": "NOT_RUN"},
        "data_quality": "MANUAL_LABEL_UNVERIFIED",
        "safety": {
            "captured_from_live_adapter": False,
            "device_command_generated": False,
            "robot_control_allowed": False,
        },
    }
    record_path = capture_dir / "record.json"
    temporary_record = record_path.with_suffix(".json.tmp")
    temporary_record.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_record.replace(record_path)
    return {
        "record": record,
        "record_path": str(record_path.resolve().relative_to(project_root.resolve())),
        "record_sha256": _sha256(record_path),
        "media_path": record["media"]["path"],
        "media_sha256": media_sha,
    }


def capture_summary(records: list[dict[str, Any]] | None) -> str:
    items = records if isinstance(records, list) else []
    image_count = sum(
        1 for item in items if item.get("media", {}).get("type") == "image"
    )
    video_count = sum(
        1 for item in items if item.get("media", {}).get("type") == "video"
    )
    return (
        "### 多模态采集摘要\n\n"
        f"- 已归档记录：**{len(items)}** 条\n"
        f"- 图片：{image_count} 条；短视频：{video_count} 条\n"
        "- 当前来源：人工网页上传，不代表相机已连接\n"
        "- 当前标定：未完成；Qwen：未运行；人工标签：待质检\n\n"
        "所有记录仅用于任务 3 数据框架验证，不包含机械臂或灵巧手控制命令。"
    )


def capture_table(records: list[dict[str, Any]] | None) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in records if isinstance(records, list) else []:
        media = item.get("media", {}) if isinstance(item, dict) else {}
        rows.append(
            [
                str(item.get("capture_id", "")),
                str(item.get("checkpoint", "")),
                str(item.get("part_id", "")),
                str(item.get("zone", "")),
                str(item.get("human_label", "")),
                str(media.get("type", "")),
                str(item.get("created_at", "")),
            ]
        )
    return rows
