"""Run one versioned ERNIE request produced by industrial-ai-service."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import ErnieImagePipeline


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()


def inside(path: Path, parent: Path) -> bool:
    return path.resolve() == parent.resolve() or parent.resolve() in path.resolve().parents


def resolve_project_path(value: str, *, must_exist: bool = False) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if not inside(path, PROJECT_ROOT):
        raise ValueError("请求包含项目目录以外的路径。")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_request(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not inside(path, RUNS_ROOT) or not path.is_file():
        raise ValueError("请求文件必须位于 workspace/runs 内。")
    request = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_path",
        "output_dir",
        "result_path",
        "prompt",
        "seeds",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "use_pe",
    }
    missing = sorted(required - request.keys())
    if missing:
        raise ValueError("请求缺少字段：" + ", ".join(missing))
    return request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = load_request(args.request)

    model_path = resolve_project_path(request["model_path"], must_exist=True)
    output_dir = resolve_project_path(request["output_dir"])
    result_path = resolve_project_path(request["result_path"])
    if not inside(output_dir, RUNS_ROOT) or not inside(result_path, RUNS_ROOT):
        raise ValueError("输出必须位于 workspace/runs 内。")
    if output_dir.exists():
        raise FileExistsError(f"拒绝复用已有输出目录：{output_dir}")

    prompt = str(request["prompt"]).strip()
    seeds = [int(value) for value in request["seeds"]]
    width = int(request["width"])
    height = int(request["height"])
    steps = int(request["num_inference_steps"])
    guidance_scale = float(request["guidance_scale"])
    use_pe = bool(request["use_pe"])
    model_name = str(request.get("model_name", "ERNIE-Image")).strip()
    if model_name not in {"ERNIE-Image", "ERNIE-Image-Turbo"}:
        raise ValueError("ERNIE 模型名称无效。")
    if not prompt or len(prompt) > 4000:
        raise ValueError("提示词长度无效。")
    if not 1 <= len(seeds) <= 4:
        raise ValueError("候选数量必须为 1 到 4。")
    if width not in {512, 768, 1024} or height not in {512, 768, 1024}:
        raise ValueError("图片尺寸无效。")
    if not 1 <= steps <= 100:
        raise ValueError("推理步数无效。")

    print(f"开始加载 {model_name}……", flush=True)
    total_started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    pipe = ErnieImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        pe=None,
        pe_tokenizer=None,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.enable_sequential_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    load_seconds = time.perf_counter() - load_started
    output_dir.mkdir(parents=True, exist_ok=False)

    outputs: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"生成第 {index}/{len(seeds)} 张，seed={seed}……", flush=True)
        image_started = time.perf_counter()
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                use_pe=use_pe,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
        output_path = output_dir / f"candidate_{index:02d}_seed_{seed}.png"
        if output_path.exists():
            raise FileExistsError(f"拒绝覆盖已有图片：{output_path}")
        image.save(output_path)
        outputs.append(
            {
                "index": index,
                "seed": seed,
                "path": str(output_path.relative_to(PROJECT_ROOT)),
                "bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
                "generation_seconds": round(time.perf_counter() - image_started, 3),
            }
        )

    torch.cuda.synchronize()
    result = {
        "schema_version": "1.0",
        "model": model_name,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "prompt": prompt,
        "parameters": {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "use_pe": use_pe,
            "low_vram_mode": "sequential_cpu_offload",
        },
        "load_seconds": round(load_seconds, 3),
        "total_seconds": round(time.perf_counter() - total_started, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "outputs": outputs,
    }
    write_json(result_path, result)
    print(f"任务完成：{result_path}", flush=True)


if __name__ == "__main__":
    main()
