"""Run one versioned Hunyuan3D Shape request in the dedicated environment."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()

BACKEND_CONTRACTS = {
    "single_view_2_1": {
        "model": "Hunyuan3D-2.1",
        "display_name": "Hunyuan3D-2.1 Shape",
        "model_path": "models/hunyuan3d-2.1-shape-ms",
        "subfolder": "hunyuan3d-dit-v2-1",
        "input_mode": "single_view",
    },
    "multi_view_2mv": {
        "model": "Hunyuan3D-2mv",
        "display_name": "Hunyuan3D-2.1 兼容多视图（2mv 后端）",
        "model_path": "models/hunyuan3d-2mv-ms",
        "subfolder": "hunyuan3d-dit-v2-mv",
        "input_mode": "multi_view",
    },
}
VIEW_ORDER = ("front", "left", "back", "right")


def inside(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    return resolved == parent.resolve() or parent.resolve() in resolved.parents


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


def write_multiview_compat_config(source: Path, destination: Path) -> str:
    """Translate only the old official package prefix; preserve the source config."""
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    replaced = 0
    for section in ("model", "vae", "conditioner", "scheduler", "image_processor", "pipeline"):
        item = config.get(section)
        if not isinstance(item, dict) or not isinstance(item.get("target"), str):
            raise ValueError(f"2mv 官方配置缺少 {section}.target。")
        target = item["target"]
        prefix = "hy3dgen.shapegen."
        if not target.startswith(prefix):
            raise ValueError(f"2mv 官方配置出现未知 target：{target}")
        item["target"] = "hy3dshape." + target[len(prefix):]
        replaced += 1
    if replaced != 6:
        raise ValueError("2mv 兼容配置转换不完整。")
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有兼容配置：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return sha256(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hunyuan Shape 服务任务工作进程")
    parser.add_argument("--request", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request_path = args.request.resolve()
    if not request_path.is_file() or not inside(
        request_path, PROJECT_ROOT / "workspace/runs"
    ):
        raise ValueError("请求文件必须位于 workspace/runs 内。")
    request = json.loads(request_path.read_text(encoding="utf-8"))

    # Schema 1.0 requests are retained for traceability and interpreted as 2.1 single-view.
    backend_key = request.get("backend_key", "single_view_2_1")
    if backend_key not in BACKEND_CONTRACTS:
        raise ValueError(f"未知 Hunyuan 后端：{backend_key}")
    backend = BACKEND_CONTRACTS[backend_key]
    if request.get("model_path") != backend["model_path"]:
        raise ValueError("模型路径与声明的 Hunyuan 后端不一致。")
    model_subfolder = request.get("model_subfolder", backend["subfolder"])
    if model_subfolder != backend["subfolder"]:
        raise ValueError("模型子目录与声明的 Hunyuan 后端不一致。")
    model_path = (PROJECT_ROOT / request["model_path"]).resolve()
    output_path = (PROJECT_ROOT / request["output_path"]).resolve()
    result_path = (PROJECT_ROOT / request["result_path"]).resolve()
    compat_config_path = result_path.with_name(
        result_path.stem + "_2mv_compat_config.yaml"
    )
    run_dir = next(
        (parent for parent in request_path.parents if parent.parent == PROJECT_ROOT / "workspace/runs"),
        None,
    )
    if run_dir is None:
        raise ValueError("无法识别任务目录。")
    image_specs = request.get("images")
    if image_specs is None:
        image_specs = {
            "front": {
                "path": request["image_path"],
                "sha256": request["image_sha256"],
            }
        }
    if not isinstance(image_specs, dict):
        raise ValueError("Hunyuan images 必须是按视角组织的对象。")
    if backend["input_mode"] == "single_view" and tuple(image_specs) != ("front",):
        raise ValueError("2.1 单图后端必须且只能接收 front 输入。")
    if backend["input_mode"] == "multi_view":
        if not {"front", "left", "back"}.issubset(image_specs):
            raise ValueError("2mv 多视图输入至少需要 front、left、back。")
        if len(image_specs) not in {3, 4} or set(image_specs) - set(VIEW_ORDER):
            raise ValueError("2mv 多视图输入只支持 front、left、back、right 中的 3 或 4 张。")

    image_paths: dict[str, Path] = {}
    input_hashes: dict[str, str] = {}
    images: dict[str, Image.Image] = {}
    for view in VIEW_ORDER:
        if view not in image_specs:
            continue
        spec = image_specs[view]
        image_path = (PROJECT_ROOT / spec["path"]).resolve()
        if not inside(image_path, run_dir / "masks") or not image_path.is_file():
            raise ValueError(f"{view} 输入不在当前任务 masks 目录。")
        input_sha = sha256(image_path)
        if input_sha != spec["sha256"]:
            raise ValueError(f"{view} 输入 SHA256 与批准记录不一致。")
        image = Image.open(image_path)
        if image.mode != "RGBA":
            raise ValueError(f"{view} 输入必须是 RGBA PNG。")
        alpha = np.asarray(image.getchannel("A"))
        if alpha.min() != 0 or alpha.max() != 255:
            raise ValueError(f"{view} 输入必须同时包含透明背景和不透明主体。")
        image_paths[view] = image_path
        input_hashes[view] = input_sha
        images[view] = image
    guarded_outputs = [output_path, result_path]
    if backend["input_mode"] == "multi_view":
        guarded_outputs.append(compat_config_path)
    for output in guarded_outputs:
        if not inside(output, run_dir):
            raise ValueError("输出路径越出当前任务目录。")
        if output.exists():
            raise FileExistsError(f"拒绝覆盖已有输出：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地 Hunyuan 模型不存在：{model_path}")

    total_start = time.time()
    load_start = time.time()
    use_safetensors = bool(request.get("use_safetensors", False))
    compat_config_sha = None
    if backend["input_mode"] == "multi_view":
        source_config = model_path / model_subfolder / "config.yaml"
        extension = "safetensors" if use_safetensors else "ckpt"
        checkpoint_path = model_path / model_subfolder / f"model.fp16.{extension}"
        compat_config_sha = write_multiview_compat_config(
            source_config,
            compat_config_path,
        )
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            str(checkpoint_path),
            str(compat_config_path),
            device="cpu",
            dtype=torch.float16,
            use_safetensors=use_safetensors,
            from_pretrained_kwargs={
                "model_path": str(model_path),
                "subfolder": model_subfolder,
                "use_safetensors": use_safetensors,
                "variant": "fp16",
                "dtype": torch.float16,
                "device": "cpu",
            },
        )
    else:
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(model_path),
            subfolder=model_subfolder,
            device="cpu",
            dtype=torch.float16,
            use_safetensors=use_safetensors,
        )
    pipeline.components = {
        "conditioner": pipeline.conditioner,
        "model": pipeline.model,
        "vae": pipeline.vae,
    }
    pipeline.enable_model_cpu_offload(device="cuda")
    pipeline.device = torch.device("cuda")
    load_seconds = time.time() - load_start

    torch.cuda.reset_peak_memory_stats()
    generation_start = time.time()
    generator = torch.Generator(device="cpu").manual_seed(int(request["seed"]))
    pipeline_input: Image.Image | dict[str, Image.Image]
    pipeline_input = images["front"] if backend["input_mode"] == "single_view" else images
    with torch.inference_mode():
        mesh = pipeline(
            image=pipeline_input,
            num_inference_steps=int(request["num_inference_steps"]),
            octree_resolution=int(request["octree_resolution"]),
            num_chunks=int(request["num_chunks"]),
            generator=generator,
            output_type="trimesh",
        )[0]
    generation_seconds = time.time() - generation_start
    mesh.export(output_path)

    parts = mesh.split(only_watertight=False)
    result = {
        "schema_version": "1.1",
        "interface_name": request.get("interface_name", "Hunyuan3D-2.1-compatible"),
        "model": backend["display_name"],
        "backend_model": backend["model"],
        "backend_key": backend_key,
        "input_mode": backend["input_mode"],
        "inputs": {
            view: {
                "path": str(image_paths[view].relative_to(PROJECT_ROOT)),
                "sha256": input_hashes[view],
            }
            for view in image_paths
        },
        "output_path": request["output_path"],
        "output_sha256": sha256(output_path),
        "output_bytes": output_path.stat().st_size,
        "parameters": {
            "seed": int(request["seed"]),
            "num_inference_steps": int(request["num_inference_steps"]),
            "octree_resolution": int(request["octree_resolution"]),
            "num_chunks": int(request["num_chunks"]),
            "dtype": request["dtype"],
            "low_vram_mode": request["low_vram_mode"],
            "remove_background": False,
            "model_subfolder": model_subfolder,
            "views": list(image_paths),
        },
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "total_seconds": round(time.time() - total_start, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "components": int(len(parts)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "volume": float(mesh.volume),
            "area": float(mesh.area),
            "bounds": np.asarray(mesh.bounds).tolist(),
            "extents": np.asarray(mesh.extents).tolist(),
        },
    }
    if compat_config_sha is not None:
        result["compatibility_config"] = {
            "path": str(compat_config_path.relative_to(PROJECT_ROOT)),
            "sha256": compat_config_sha,
            "source_path": str((model_path / model_subfolder / "config.yaml").relative_to(PROJECT_ROOT)),
            "translation": "hy3dgen.shapegen.* -> hy3dshape.*",
        }
    if backend["input_mode"] == "single_view":
        result["input_path"] = str(image_paths["front"].relative_to(PROJECT_ROOT))
        result["input_sha256"] = input_hashes["front"]
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
