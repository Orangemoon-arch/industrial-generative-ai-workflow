"""Review one generated candidate and emit deterministic structured JSON."""

from __future__ import annotations

import os

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("INDUSTRIAL_AI_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = (PROJECT_ROOT / "workspace/runs").resolve()
CUDA_LIB = APP_DIR / "cuda-lib"
os.environ["LIBRARY_PATH"] = str(CUDA_LIB) + ":" + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = str(CUDA_LIB) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

import qwen_review_rules as review_rules


def inside(path: Path, parent: Path) -> bool:
    return path.resolve() == parent.resolve() or parent.resolve() in path.resolve().parents


def resolve_project_path(value: str, *, must_exist: bool = False) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if not inside(path, PROJECT_ROOT):
        raise ValueError("请求包含项目目录以外的路径。")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


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


def load_request(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    if not inside(path, RUNS_ROOT) or not path.is_file():
        raise ValueError("请求文件必须位于 workspace/runs 内。")
    request = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_path",
        "image_path",
        "result_path",
        "part_type",
        "expected_repeat_count",
        "required_structure",
        "forbidden_elements",
        "image_max_size",
        "max_new_tokens",
    }
    missing = sorted(required - request.keys())
    if missing:
        raise ValueError("请求缺少字段：" + ", ".join(missing))
    return path, request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request_path, request = load_request(args.request)
    run_dir = request_path.parents[1]

    model_path = resolve_project_path(request["model_path"], must_exist=True)
    image_path = resolve_project_path(request["image_path"], must_exist=True)
    result_path = resolve_project_path(request["result_path"])
    if not inside(image_path, run_dir) or not inside(result_path, run_dir):
        raise ValueError("图片和结果必须位于同一个任务目录。")

    expected_value = request["expected_repeat_count"]
    expected_count = None if expected_value is None else int(expected_value)
    max_size = int(request["image_max_size"])
    max_new_tokens = int(request["max_new_tokens"])
    if expected_count is not None and not 1 <= expected_count <= 100:
        raise ValueError("预期数量无效。")
    if max_size not in {512, 768, 1024}:
        raise ValueError("图片缩放上限无效。")
    if not 128 <= max_new_tokens <= 768:
        raise ValueError("输出 token 上限无效。")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    print("开始以 NF4 4-bit 加载 Qwen3-VL……", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map="auto",
        max_memory={0: "6200MiB", "cpu": "22GiB"},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    image = Image.open(image_path).convert("RGB")
    original_size = list(image.size)
    image.thumbnail((max_size, max_size))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": review_rules.build_instruction(request)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        name: value.to("cuda") if hasattr(value, "to") else value
        for name, value in inputs.items()
    }

    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    raw_output = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    review = review_rules.normalize_review(
        review_rules.extract_json(raw_output),
        expected_count,
        str(request["forbidden_elements"]),
    )

    result = {
        "schema_version": "1.0",
        "model": "Qwen3-VL-8B-Instruct",
        "quantization": "NF4_4bit_double_quant",
        "review_protocol": "blind_observation_v2",
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "image_original_size": original_size,
        "image_inference_size": list(image.size),
        "expected_repeat_count": expected_count,
        "counted_feature": str(request.get("counted_feature", "主要重复结构")),
        "visual_checks": review_rules.string_list(request.get("visual_checks")),
        "review": review,
        "raw_model_output": raw_output,
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "total_seconds": round(time.perf_counter() - total_started, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }
    write_json(result_path, result)
    print(f"检查完成，最终等级：{review['decision']}", flush=True)
    print(f"结果已保存：{result_path}", flush=True)


if __name__ == "__main__":
    main()
