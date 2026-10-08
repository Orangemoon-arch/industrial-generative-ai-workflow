"""Independent HTTP API V1 for the industrial generative-AI workflow.

The API process never imports a GPU model. Long-running work is submitted to a
single in-process executor and delegated to the existing guarded subprocess
adapters. Every job is persisted below workspace/runs before it is queued.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from ernie_adapter import ERNIE_VARIANTS, ErnieExecutionError, model_is_complete, run_ernie
from gpu_executor import GpuUnavailableError
from hunyuan_adapter import (
    HunyuanExecutionError,
    backend_is_available,
    run_hunyuan,
    run_hunyuan_multiview,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "workspace/runs"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
API_SCHEMA_VERSION = "1.1"
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="industrial-ai-api")
JOB_LOCK = threading.Lock()


class TextToImageRequest(BaseModel):
    """Task 1 request; ERNIE-Image is the default competition backend."""

    prompt: str = Field(min_length=1, max_length=4000)
    model: Literal["ernie-image", "ernie-image-turbo"] = "ernie-image"
    base_seed: int = Field(default=20260907, ge=0, le=2**32 - 4)
    candidate_count: int = Field(default=1, ge=1, le=4)
    width: Literal[512, 768, 1024] = 768
    height: Literal[512, 768, 1024] = 768
    inference_steps: int = Field(default=50, ge=1, le=100)


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    root = parent.resolve()
    return resolved == root or root in resolved.parents


def _job_path(job_id: str) -> Path:
    if not job_id.startswith("RUN-") or "/" in job_id or "\\" in job_id:
        raise ValueError("无效任务编号。")
    path = (RUNS_ROOT / job_id / "reports/api_job_v1.json").resolve()
    if not _inside(path, RUNS_ROOT):
        raise ValueError("任务路径越界。")
    return path


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_job(job_id: str) -> dict[str, Any]:
    path = _job_path(job_id)
    if not path.is_file():
        raise FileNotFoundError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = now_iso()
    _write_json_atomic(_job_path(job["job_id"]), job)


def _update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with JOB_LOCK:
        job = _read_job(job_id)
        job.update(updates)
        _save_job(job)
        return job


def _new_job(kind: str, request: dict[str, Any]) -> tuple[str, Path]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        job_id = f"RUN-{stamp}-{uuid.uuid4().hex[:6]}"
        run_dir = RUNS_ROOT / job_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            continue
    for name in ("requests", "images", "masks", "meshes", "reports", "logs"):
        (run_dir / name).mkdir()
    created = now_iso()
    job = {
        "schema_version": API_SCHEMA_VERSION,
        "owner": "industrial-ai-independent-api-v1",
        "job_id": job_id,
        "kind": kind,
        "status": "QUEUED",
        "created_at": created,
        "updated_at": created,
        "request": request,
        "artifacts": [],
        "metrics": {},
        "error": None,
        "notice": (
            "任务一按请求选择 ERNIE-Image 或保留的 Turbo 回退后端。"
            if kind == "text_to_image"
            else "输出为 Hunyuan 原始 GLB，必须继续完成网格检查和人工确认。"
        ),
    }
    _save_job(job)
    return job_id, run_dir


def _artifact(role: str, relative_path: str, *, sha256: str | None = None) -> dict[str, Any]:
    path = (PROJECT_ROOT / relative_path).resolve()
    if not _inside(path, RUNS_ROOT) or not path.is_file():
        raise RuntimeError(f"产物路径无效：{relative_path}")
    return {
        "id": role,
        "role": role,
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256 or sha256_file(path),
        "download_url": "",
    }


def _execute_text_to_image(job_id: str) -> None:
    try:
        job = _update_job(job_id, status="RUNNING")
        request = job["request"]
        run_dir = _job_path(job_id).parents[1]
        execution = run_ernie(
            project_root=PROJECT_ROOT,
            run_dir=run_dir,
            prompt=request["prompt"],
            base_seed=request["base_seed"],
            candidate_count=request["candidate_count"],
            width=request["width"],
            height=request["height"],
            inference_steps=request["inference_steps"],
            model_variant=request["model"],
        )
        result = execution["result"]
        artifacts = []
        for index, output in enumerate(result["outputs"], start=1):
            artifacts.append(
                _artifact(
                    f"candidate_{index:02d}",
                    output["path"],
                    sha256=output.get("sha256"),
                )
            )
        artifacts.extend(
            [
                _artifact("request", execution["request_path"]),
                _artifact("result", execution["result_path"]),
                _artifact("log", execution["log_path"]),
            ]
        )
        _update_job(
            job_id,
            status="COMPLETED",
            artifacts=artifacts,
            metrics={
                "model": result.get("model"),
                "total_seconds": result.get("total_seconds"),
                "peak_allocated_gib": result.get("peak_allocated_gib"),
                "peak_reserved_gib": result.get("peak_reserved_gib"),
                "gpu_preflight": execution.get("gpu_preflight"),
            },
        )
    except GpuUnavailableError as exc:
        _update_job(job_id, status="BLOCKED_GPU", error=str(exc))
    except (ErnieExecutionError, FileNotFoundError, ValueError, RuntimeError) as exc:
        _update_job(job_id, status="FAILED", error=str(exc))
    except Exception as exc:  # Defensive persistence for unexpected worker failures.
        _update_job(job_id, status="FAILED", error=f"未预期错误：{type(exc).__name__}: {exc}")


def _execute_image_to_3d(job_id: str) -> None:
    try:
        job = _update_job(job_id, status="RUNNING")
        request = job["request"]
        run_dir = _job_path(job_id).parents[1]
        image_path = (PROJECT_ROOT / request["image_path"]).resolve()
        execution = run_hunyuan(
            project_root=PROJECT_ROOT,
            run_dir=run_dir,
            image_path=image_path,
            image_sha256=request["image_sha256"],
            seed=request["seed"],
            inference_steps=request["inference_steps"],
            octree_resolution=request["octree_resolution"],
            num_chunks=request["num_chunks"],
        )
        result = execution["result"]
        artifacts = [
            _artifact(
                "input_rgba",
                request["image_path"],
                sha256=request["image_sha256"],
            ),
            _artifact("raw_glb", execution["output_path"], sha256=result["output_sha256"]),
            _artifact("request", execution["request_path"]),
            _artifact("result", execution["result_path"]),
            _artifact("log", execution["log_path"]),
        ]
        _update_job(
            job_id,
            status="COMPLETED_RAW_WAITING_REVIEW",
            artifacts=artifacts,
            metrics={
                "model": result.get("model"),
                "total_seconds": result.get("total_seconds"),
                "peak_allocated_gib": result.get("peak_allocated_gib"),
                "peak_reserved_gib": result.get("peak_reserved_gib"),
                "mesh": result.get("mesh"),
                "gpu_preflight": execution.get("gpu_preflight"),
            },
        )
    except GpuUnavailableError as exc:
        _update_job(job_id, status="BLOCKED_GPU", error=str(exc))
    except (HunyuanExecutionError, FileNotFoundError, ValueError, RuntimeError) as exc:
        _update_job(job_id, status="FAILED", error=str(exc))
    except Exception as exc:  # Defensive persistence for unexpected worker failures.
        _update_job(job_id, status="FAILED", error=f"未预期错误：{type(exc).__name__}: {exc}")


def _execute_multiview_to_3d(job_id: str) -> None:
    try:
        job = _update_job(job_id, status="RUNNING")
        request = job["request"]
        run_dir = _job_path(job_id).parents[1]
        image_paths = {
            view: (PROJECT_ROOT / item["path"]).resolve()
            for view, item in request["images"].items()
        }
        image_hashes = {
            view: item["sha256"] for view, item in request["images"].items()
        }
        execution = run_hunyuan_multiview(
            project_root=PROJECT_ROOT,
            run_dir=run_dir,
            image_paths=image_paths,
            image_sha256=image_hashes,
            seed=request["seed"],
            inference_steps=request["inference_steps"],
            octree_resolution=request["octree_resolution"],
            num_chunks=request["num_chunks"],
        )
        result = execution["result"]
        artifacts = [
            _artifact(f"input_{view}", item["path"], sha256=item["sha256"])
            for view, item in request["images"].items()
        ]
        artifacts.extend(
            [
                _artifact("raw_glb", execution["output_path"], sha256=result["output_sha256"]),
                _artifact("request", execution["request_path"]),
                _artifact("result", execution["result_path"]),
                _artifact("log", execution["log_path"]),
            ]
        )
        _update_job(
            job_id,
            status="COMPLETED_RAW_WAITING_REVIEW",
            artifacts=artifacts,
            metrics={
                "interface_name": result.get("interface_name"),
                "model": result.get("model"),
                "backend_model": result.get("backend_model"),
                "input_mode": result.get("input_mode"),
                "views": list(result.get("inputs", {})),
                "total_seconds": result.get("total_seconds"),
                "peak_allocated_gib": result.get("peak_allocated_gib"),
                "peak_reserved_gib": result.get("peak_reserved_gib"),
                "mesh": result.get("mesh"),
                "gpu_preflight": execution.get("gpu_preflight"),
            },
        )
    except GpuUnavailableError as exc:
        _update_job(job_id, status="BLOCKED_GPU", error=str(exc))
    except (HunyuanExecutionError, FileNotFoundError, ValueError, RuntimeError) as exc:
        _update_job(job_id, status="FAILED", error=str(exc))
    except Exception as exc:
        _update_job(job_id, status="FAILED", error=f"未预期错误：{type(exc).__name__}: {exc}")


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    public = json.loads(json.dumps(job, ensure_ascii=False))
    for artifact in public.get("artifacts", []):
        artifact["download_url"] = (
            f"/api/v1/jobs/{job['job_id']}/artifacts/{artifact['id']}"
        )
    return public


app = FastAPI(
    title="工业生成式 AI 独立接口",
    version="1.1.0",
    description=(
        "任务一文生图与任务二图生三维的本地 HTTP API V1。"
        "所有 GPU 任务使用单队列顺序执行；任务一默认使用规程指定的 ERNIE-Image，"
        "并保留 ERNIE-Image-Turbo 作为显式回退后端。"
    ),
)


@app.get("/health", tags=["系统"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "api_version": "1.1.0",
        "queue_concurrency": 1,
        "models": {
            "task1": {
                "default_backend": "ERNIE-Image",
                "ernie_image_available": model_is_complete(
                    PROJECT_ROOT / str(ERNIE_VARIANTS["ernie-image"]["relative_path"])
                ),
                "turbo_fallback_available": model_is_complete(
                    PROJECT_ROOT
                    / str(ERNIE_VARIANTS["ernie-image-turbo"]["relative_path"])
                ),
            },
            "task2": {
                "single_view_backend": "Hunyuan3D-2.1 Shape",
                "single_view_available": backend_is_available(PROJECT_ROOT, "single_view_2_1"),
                "multiview_interface": "Hunyuan3D-2.1-compatible",
                "multiview_backend": "Hunyuan3D-2mv",
                "multiview_available": backend_is_available(PROJECT_ROOT, "multi_view_2mv"),
            },
        },
    }


@app.post(
    "/api/v1/text-to-image/jobs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["任务一：文生图"],
)
def submit_text_to_image(request: TextToImageRequest) -> JobAccepted:
    payload = request.model_dump()
    payload["prompt"] = payload["prompt"].strip()
    if not payload["prompt"]:
        raise HTTPException(status_code=422, detail="提示词不能为空。")
    if payload["base_seed"] > 2**32 - payload["candidate_count"]:
        raise HTTPException(status_code=422, detail="随机种子与候选数量组合超出范围。")
    variant = ERNIE_VARIANTS[payload["model"]]
    if not model_is_complete(PROJECT_ROOT / str(variant["relative_path"])):
        raise HTTPException(status_code=503, detail=f"{variant['display_name']} 尚未完整下载。")
    job_id, _ = _new_job("text_to_image", payload)
    JOB_EXECUTOR.submit(_execute_text_to_image, job_id)
    return JobAccepted(
        job_id=job_id,
        status="QUEUED",
        status_url=f"/api/v1/jobs/{job_id}",
    )


@app.post(
    "/api/v1/multiview-to-3d/jobs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["任务二：多视图生三维"],
)
async def submit_multiview_to_3d(
    front: UploadFile = File(description="正面透明背景 RGBA PNG"),
    left: UploadFile = File(description="左侧透明背景 RGBA PNG"),
    back: UploadFile = File(description="背面透明背景 RGBA PNG"),
    right: UploadFile | None = File(default=None, description="右侧透明背景 RGBA PNG（可选）"),
    seed: int = Form(default=1234, ge=0, le=2**32 - 1),
    inference_steps: int = Form(default=30, ge=1, le=50),
    octree_resolution: Literal[128, 256, 384] = Form(default=256),
    num_chunks: int = Form(default=8000, ge=1000, le=20000),
) -> JobAccepted:
    if not backend_is_available(PROJECT_ROOT, "multi_view_2mv"):
        raise HTTPException(
            status_code=503,
            detail="Hunyuan3D-2mv 权重尚未放入 models/hunyuan3d-2mv-ms。",
        )
    uploads = {"front": front, "left": left, "back": back}
    if right is not None:
        uploads["right"] = right
    contents: dict[str, tuple[str, bytes]] = {}
    for view, upload in uploads.items():
        filename = Path(upload.filename or f"{view}.png").name
        if Path(filename).suffix.lower() != ".png":
            raise HTTPException(status_code=415, detail=f"{view} 输入必须是 PNG 文件。")
        content = await upload.read(MAX_UPLOAD_BYTES + 1)
        await upload.close()
        if not content:
            raise HTTPException(status_code=422, detail=f"{view} 上传文件为空。")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{view} 文件不能超过 20 MiB。")
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(status_code=415, detail=f"{view} 文件内容不是有效 PNG。")
        contents[view] = (filename, content)

    request = {
        "interface_name": "Hunyuan3D-2.1-compatible",
        "backend_model": "Hunyuan3D-2mv",
        "input_mode": "multi_view",
        "seed": seed,
        "inference_steps": inference_steps,
        "octree_resolution": octree_resolution,
        "num_chunks": num_chunks,
        "images": {},
    }
    job_id, run_dir = _new_job("multiview_to_3d", request)
    try:
        for view, (filename, content) in contents.items():
            input_path = run_dir / "masks" / f"api_multiview_{view}_v1.png"
            input_path.write_bytes(content)
            with Image.open(input_path) as opened:
                opened.verify()
            with Image.open(input_path) as opened:
                if opened.mode != "RGBA":
                    raise ValueError(f"{view} 输入必须是 RGBA PNG。")
                alpha_min, alpha_max = opened.getchannel("A").getextrema()
                if alpha_min != 0 or alpha_max != 255:
                    raise ValueError(f"{view} 输入必须同时包含透明背景和不透明主体。")
            relative = str(input_path.relative_to(PROJECT_ROOT))
            request["images"][view] = {
                "filename": filename,
                "path": relative,
                "sha256": sha256_file(input_path),
            }
    except Exception as exc:
        _update_job(job_id, status="REJECTED_INPUT", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with JOB_LOCK:
        job = _read_job(job_id)
        job["request"] = request
        job["artifacts"] = [
            _artifact(f"input_{view}", item["path"], sha256=item["sha256"])
            for view, item in request["images"].items()
        ]
        _save_job(job)
    JOB_EXECUTOR.submit(_execute_multiview_to_3d, job_id)
    return JobAccepted(
        job_id=job_id,
        status="QUEUED",
        status_url=f"/api/v1/jobs/{job_id}",
    )


@app.post(
    "/api/v1/image-to-3d/jobs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["任务二：图生三维"],
)
async def submit_image_to_3d(
    image: UploadFile = File(description="已人工确认的透明背景 RGBA PNG"),
    seed: int = Form(default=1234, ge=0, le=2**32 - 1),
    inference_steps: int = Form(default=30, ge=1, le=50),
    octree_resolution: Literal[128, 256, 384] = Form(default=256),
    num_chunks: int = Form(default=8000, ge=1000, le=20000),
) -> JobAccepted:
    filename = Path(image.filename or "input.png").name
    if Path(filename).suffix.lower() != ".png":
        raise HTTPException(status_code=415, detail="任务二输入必须是 PNG 文件。")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not content:
        raise HTTPException(status_code=422, detail="上传文件为空。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="上传文件不能超过 20 MiB。")
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=415, detail="文件内容不是有效 PNG。")

    request = {
        "filename": filename,
        "seed": seed,
        "inference_steps": inference_steps,
        "octree_resolution": octree_resolution,
        "num_chunks": num_chunks,
    }
    job_id, run_dir = _new_job("image_to_3d", request)
    input_path = run_dir / "masks/api_input_rgba_v1.png"
    try:
        input_path.write_bytes(content)
        with Image.open(input_path) as opened:
            opened.verify()
        with Image.open(input_path) as opened:
            if opened.mode != "RGBA":
                raise ValueError("任务二要求 RGBA PNG，当前图片没有 Alpha 通道。")
    except Exception as exc:
        _update_job(job_id, status="REJECTED_INPUT", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    relative = str(input_path.relative_to(PROJECT_ROOT))
    digest = sha256_file(input_path)
    with JOB_LOCK:
        job = _read_job(job_id)
        job["request"].update({"image_path": relative, "image_sha256": digest})
        job["artifacts"] = [_artifact("input_rgba", relative, sha256=digest)]
        _save_job(job)
    JOB_EXECUTOR.submit(_execute_image_to_3d, job_id)
    return JobAccepted(
        job_id=job_id,
        status="QUEUED",
        status_url=f"/api/v1/jobs/{job_id}",
    )


@app.get("/api/v1/jobs/{job_id}", tags=["任务查询"])
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return _public_job(_read_job(job_id))
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="没有找到该 API 任务。")


@app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_id}", tags=["产物下载"])
def download_artifact(job_id: str, artifact_id: str) -> FileResponse:
    try:
        job = _read_job(job_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="没有找到该 API 任务。")
    artifact = next(
        (item for item in job.get("artifacts", []) if item.get("id") == artifact_id),
        None,
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="没有找到该任务产物。")
    path = (PROJECT_ROOT / artifact["path"]).resolve()
    run_dir = _job_path(job_id).parents[1]
    if not _inside(path, run_dir) or not path.is_file():
        raise HTTPException(status_code=404, detail="产物文件不存在或路径无效。")
    if sha256_file(path) != artifact.get("sha256"):
        raise HTTPException(status_code=409, detail="产物 SHA256 校验失败，拒绝下载。")
    return FileResponse(path, filename=path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="工业生成式 AI 独立 HTTP API V1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=27866, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
