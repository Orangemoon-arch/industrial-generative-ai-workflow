"""Subprocess adapter for the validated low-VRAM ERNIE worker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from gpu_executor import project_gpu_lock, require_available_gpu


class ErnieExecutionError(RuntimeError):
    """Raised when an ERNIE worker attempt fails."""


ERNIE_VARIANTS: dict[str, dict[str, Any]] = {
    "ernie-image": {
        "display_name": "ERNIE-Image",
        "relative_path": "models/ernie-image-ms",
        "default_steps": 50,
        "guidance_scale": 4.0,
    },
    "ernie-image-turbo": {
        "display_name": "ERNIE-Image-Turbo",
        "relative_path": "models/ernie-image-turbo-ms",
        "default_steps": 8,
        "guidance_scale": 1.0,
    },
}


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


def next_attempt(run_dir: Path) -> int:
    request_dir = run_dir / "requests"
    existing = list(request_dir.glob("ernie_attempt_*.json")) if request_dir.is_dir() else []
    numbers: list[int] = []
    for path in existing:
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def model_is_complete(model_path: Path) -> bool:
    required_paths = [
        model_path / "model_index.json",
        model_path / "scheduler",
        model_path / "text_encoder",
        model_path / "tokenizer",
        model_path / "transformer",
        model_path / "vae",
    ]
    required_weight_roots = [
        model_path / "text_encoder",
        model_path / "transformer",
        model_path / "vae",
    ]
    return (
        all(path.exists() for path in required_paths)
        and all(any(path.rglob("*.safetensors")) for path in required_weight_roots)
        and not any(
            incomplete
            for path in required_weight_roots
            for incomplete in path.rglob("*.incomplete")
        )
    )


def run_ernie(
    *,
    project_root: Path,
    run_dir: Path,
    prompt: str,
    base_seed: int,
    candidate_count: int,
    width: int = 1024,
    height: int = 1024,
    inference_steps: int | None = None,
    model_variant: str = "ernie-image",
    guidance_scale: float | None = None,
    use_pe: bool = False,
    minimum_free_mib: int = 3500,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    runs_root = (project_root / "workspace/runs").resolve()
    if not _inside(run_dir, runs_root):
        raise ValueError("ERNIE 任务目录必须位于 workspace/runs 内。")
    if not prompt.strip() or len(prompt) > 4000:
        raise ValueError("提示词长度必须为 1 到 4000 个字符。")
    if not 1 <= candidate_count <= 4:
        raise ValueError("候选图片数量必须为 1 到 4。")
    if width not in {512, 768, 1024} or height not in {512, 768, 1024}:
        raise ValueError("图片宽高只允许 512、768 或 1024。")
    variant = ERNIE_VARIANTS.get(model_variant)
    if variant is None:
        raise ValueError("ERNIE 模型只允许 ernie-image 或 ernie-image-turbo。")
    steps = int(inference_steps or variant["default_steps"])
    guidance = float(
        variant["guidance_scale"] if guidance_scale is None else guidance_scale
    )
    if not 1 <= steps <= 100:
        raise ValueError("推理步数必须为 1 到 100。")
    if not 0.0 <= guidance <= 20.0:
        raise ValueError("guidance scale 必须为 0 到 20。")
    if not 0 <= base_seed <= 2**32 - candidate_count:
        raise ValueError("随机种子超出允许范围。")

    ernie_dir = project_root / "apps/ernie-image"
    python_path = ernie_dir / ".venv/bin/python"
    worker_path = ernie_dir / "run_ernie_service_task.py"
    model_path = project_root / str(variant["relative_path"])
    for required in (python_path, worker_path):
        if not required.exists():
            raise FileNotFoundError(f"ERNIE 运行依赖不存在：{required}")
    if not model_is_complete(model_path):
        raise FileNotFoundError(f"ERNIE 模型目录尚未完整下载：{model_path}")

    attempt = next_attempt(run_dir)
    attempt_name = f"ernie_attempt_{attempt:03d}"
    request_path = run_dir / "requests" / f"{attempt_name}.json"
    output_dir = run_dir / "images" / attempt_name
    result_path = run_dir / "reports" / f"{attempt_name}_result.json"
    log_path = run_dir / "logs" / f"{attempt_name}.log"
    request = {
        "schema_version": "1.0",
        "model_variant": model_variant,
        "model_name": variant["display_name"],
        "model_path": str(model_path.relative_to(project_root)),
        "output_dir": str(output_dir.relative_to(project_root)),
        "result_path": str(result_path.relative_to(project_root)),
        "prompt": prompt.strip(),
        "seeds": [base_seed + index for index in range(candidate_count)],
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "use_pe": bool(use_pe),
        "low_vram_mode": "sequential_cpu_offload",
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
                    cwd=ernie_dir,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ErnieExecutionError(
                    f"ERNIE 任务超过 {timeout_seconds} 秒，已由执行器停止。"
                ) from exc

    if completed.returncode != 0:
        raise ErnieExecutionError(
            f"ERNIE 工作进程退出码为 {completed.returncode}，详见项目内日志。"
        )
    if not result_path.is_file():
        raise ErnieExecutionError("ERNIE 工作进程结束但没有生成结果 JSON。")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != candidate_count:
        raise ErnieExecutionError("ERNIE 返回的候选图片数量与请求不一致。")
    for item in outputs:
        image_path = (project_root / item["path"]).resolve()
        if not _inside(image_path, output_dir) or not image_path.is_file():
            raise ErnieExecutionError("ERNIE 返回了无效或越界的图片路径。")

    return {
        "attempt": attempt,
        "request_path": str(request_path.relative_to(project_root)),
        "result_path": str(result_path.relative_to(project_root)),
        "log_path": str(log_path.relative_to(project_root)),
        "gpu_preflight": snapshot.to_dict(),
        "result": result,
    }
