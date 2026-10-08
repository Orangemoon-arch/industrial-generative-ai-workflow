"""Subprocess adapter for NF4 Qwen3-VL candidate-image review."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from gpu_executor import project_gpu_lock, require_available_gpu


class QwenExecutionError(RuntimeError):
    """Raised when a Qwen review attempt fails."""


def _inside(path: Path, parent: Path) -> bool:
    return path.resolve() == parent.resolve() or parent.resolve() in path.resolve().parents


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def next_review(run_dir: Path) -> int:
    request_dir = run_dir / "requests"
    existing = list(request_dir.glob("qwen_review_*.json")) if request_dir.is_dir() else []
    numbers: list[int] = []
    for path in existing:
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def run_qwen_review(
    *,
    project_root: Path,
    run_dir: Path,
    image_path: Path,
    part_type: str,
    expected_count: int | None,
    counted_feature: str,
    visual_checks: list[str],
    required_structure: str,
    forbidden_elements: str,
    minimum_free_mib: int = 7500,
    timeout_seconds: int = 1200,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    image_path = image_path.resolve()
    runs_root = (project_root / "workspace/runs").resolve()
    if not _inside(run_dir, runs_root):
        raise ValueError("Qwen 任务目录必须位于 workspace/runs 内。")
    if not _inside(image_path, run_dir) or not image_path.is_file():
        raise ValueError("Qwen 只能检查当前任务目录中的候选图片。")
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Qwen 候选图片格式不受支持。")
    if expected_count is not None and not 1 <= expected_count <= 100:
        raise ValueError("预期重复结构数量必须为 1 到 100。")
    counted_feature = counted_feature.strip() or "主要重复结构"
    clean_checks = [str(item).strip() for item in visual_checks if str(item).strip()]

    qwen_dir = project_root / "apps/qwen3-vl"
    python_path = qwen_dir / ".venv/bin/python"
    worker_path = qwen_dir / "run_qwen_service_review.py"
    model_path = project_root / "models/qwen3-vl-8b-instruct-ms"
    for required in (python_path, worker_path, model_path):
        if not required.exists():
            raise FileNotFoundError(f"Qwen 运行依赖不存在：{required}")

    review_number = next_review(run_dir)
    review_name = f"qwen_review_{review_number:03d}"
    request_path = run_dir / "requests" / f"{review_name}.json"
    result_path = run_dir / "reviews" / f"{review_name}.json"
    log_path = run_dir / "logs" / f"{review_name}.log"
    request = {
        "schema_version": "1.0",
        "model_path": str(model_path.relative_to(project_root)),
        "image_path": str(image_path.relative_to(project_root)),
        "result_path": str(result_path.relative_to(project_root)),
        "part_type": part_type.strip(),
        "expected_repeat_count": expected_count,
        "counted_feature": counted_feature,
        "visual_checks": clean_checks,
        "required_structure": required_structure.strip(),
        "forbidden_elements": forbidden_elements.strip(),
        "quantization": "NF4_4bit_double_quant",
        "review_protocol": "blind_observation_v2",
        "image_max_size": 768,
        "max_new_tokens": 512,
    }
    _write_json(request_path, request)

    lock_path = runs_root / ".gpu-model.lock"
    with project_gpu_lock(lock_path):
        snapshot = require_available_gpu(minimum_free_mib)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            try:
                completed = subprocess.run(
                    [str(python_path), str(worker_path), "--request", str(request_path)],
                    cwd=qwen_dir,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise QwenExecutionError(
                    f"Qwen 检查超过 {timeout_seconds} 秒，已由执行器停止。"
                ) from exc

    if completed.returncode != 0:
        raise QwenExecutionError(
            f"Qwen 工作进程退出码为 {completed.returncode}，详见项目内日志。"
        )
    if not result_path.is_file():
        raise QwenExecutionError("Qwen 工作进程结束但没有生成结构化检查 JSON。")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    review = result.get("review")
    if not isinstance(review, dict) or review.get("decision") not in {
        "PASS",
        "REVIEW",
        "REJECT",
    }:
        raise QwenExecutionError("Qwen 返回的结构化检查等级无效。")
    if result.get("image_path") != str(image_path.relative_to(project_root)):
        raise QwenExecutionError("Qwen 返回的目标图片与请求不一致。")

    return {
        "review_number": review_number,
        "request_path": str(request_path.relative_to(project_root)),
        "result_path": str(result_path.relative_to(project_root)),
        "log_path": str(log_path.relative_to(project_root)),
        "gpu_preflight": snapshot.to_dict(),
        "result": result,
    }
