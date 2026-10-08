from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

import api_server


class IndependentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="industrial-ai-api-test-")
        self.root = Path(self.temporary.name)
        self.runs = self.root / "workspace/runs"
        for model_name in ("ernie-image-ms", "ernie-image-turbo-ms"):
            model_dir = self.root / "models" / model_name
            for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
                (model_dir / component).mkdir(parents=True, exist_ok=True)
            (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
            for component in ("text_encoder", "transformer", "vae"):
                (model_dir / component / "model.safetensors").write_bytes(b"test")
        (self.root / "models/hunyuan3d-2.1-shape-ms").mkdir(parents=True)
        (self.root / "models/hunyuan3d-2mv-ms").mkdir(parents=True)
        self.project_patch = patch.object(api_server, "PROJECT_ROOT", self.root)
        self.runs_patch = patch.object(api_server, "RUNS_ROOT", self.runs)
        self.project_patch.start()
        self.runs_patch.start()

    def tearDown(self) -> None:
        self.runs_patch.stop()
        self.project_patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def _relative(root: Path, path: Path) -> str:
        return str(path.relative_to(root))

    def fake_ernie(self, **kwargs):
        run_dir = kwargs["run_dir"]
        image = run_dir / "images/ernie_attempt_001/candidate_01.png"
        request = run_dir / "requests/ernie_attempt_001.json"
        result = run_dir / "reports/ernie_attempt_001_result.json"
        log = run_dir / "logs/ernie_attempt_001.log"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"test-image")
        request.write_text("{}\n", encoding="utf-8")
        log.write_text("ok\n", encoding="utf-8")
        payload = {
            "model": "ERNIE-Image-Turbo",
            "total_seconds": 1.2,
            "peak_allocated_gib": 1.0,
            "peak_reserved_gib": 1.1,
            "outputs": [
                {
                    "path": self._relative(self.root, image),
                    "sha256": api_server.sha256_file(image),
                }
            ],
        }
        result.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "request_path": self._relative(self.root, request),
            "result_path": self._relative(self.root, result),
            "log_path": self._relative(self.root, log),
            "gpu_preflight": {"free_mib": 10000},
            "result": payload,
        }

    def fake_hunyuan(self, **kwargs):
        run_dir = kwargs["run_dir"]
        output = run_dir / "meshes/hunyuan_attempt_001_raw.glb"
        request = run_dir / "requests/hunyuan_attempt_001.json"
        result = run_dir / "reports/hunyuan_attempt_001_result.json"
        log = run_dir / "logs/hunyuan_attempt_001.log"
        output.write_bytes(b"glTF-test")
        request.write_text("{}\n", encoding="utf-8")
        log.write_text("ok\n", encoding="utf-8")
        payload = {
            "model": "Hunyuan3D-2.1 Shape",
            "output_sha256": api_server.sha256_file(output),
            "total_seconds": 2.5,
            "peak_allocated_gib": 6.4,
            "peak_reserved_gib": 6.7,
            "mesh": {"components": 1, "watertight": True},
        }
        result.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "output_path": self._relative(self.root, output),
            "request_path": self._relative(self.root, request),
            "result_path": self._relative(self.root, result),
            "log_path": self._relative(self.root, log),
            "gpu_preflight": {"free_mib": 10000},
            "result": payload,
        }

    def fake_hunyuan_multiview(self, **kwargs):
        execution = self.fake_hunyuan(**kwargs)
        execution["result"].update(
            {
                "interface_name": "Hunyuan3D-2.1-compatible",
                "model": "Hunyuan3D-2.1 兼容多视图（2mv 后端）",
                "backend_model": "Hunyuan3D-2mv",
                "input_mode": "multi_view",
                "inputs": {
                    view: {
                        "path": self._relative(self.root, path),
                        "sha256": kwargs["image_sha256"][view],
                    }
                    for view, path in kwargs["image_paths"].items()
                },
            }
        )
        Path(self.root / execution["result_path"]).write_text(
            json.dumps(execution["result"]), encoding="utf-8"
        )
        return execution

    def test_health_reports_model_identity_without_gpu_access(self) -> None:
        body = api_server.health()
        self.assertEqual(body["queue_concurrency"], 1)
        self.assertEqual(body["models"]["task1"]["default_backend"], "ERNIE-Image")
        self.assertTrue(body["models"]["task1"]["ernie_image_available"])
        self.assertTrue(body["models"]["task1"]["turbo_fallback_available"])

    def test_text_to_image_job_is_persisted_and_downloadable(self) -> None:
        request = {
            "prompt": "生成一个六孔法兰",
            "model": "ernie-image-turbo",
            "base_seed": 7,
            "candidate_count": 1,
            "width": 768,
            "height": 768,
            "inference_steps": 8,
        }
        job_id, _ = api_server._new_job("text_to_image", request)
        with patch.object(api_server, "run_ernie", side_effect=self.fake_ernie):
            api_server._execute_text_to_image(job_id)
        job = api_server._read_job(job_id)
        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["metrics"]["model"], "ERNIE-Image-Turbo")
        self.assertEqual(job["artifacts"][0]["role"], "candidate_01")

        response = api_server.download_artifact(job_id, "candidate_01")
        self.assertEqual(Path(response.path).read_bytes(), b"test-image")

    def test_hunyuan_job_keeps_raw_output_waiting_for_review(self) -> None:
        job_id, run_dir = api_server._new_job(
            "image_to_3d",
            {
                "filename": "input.png",
                "seed": 1234,
                "inference_steps": 30,
                "octree_resolution": 256,
                "num_chunks": 8000,
            },
        )
        image = run_dir / "masks/api_input_rgba_v1.png"
        Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(image)
        job = api_server._read_job(job_id)
        job["request"].update(
            {
                "image_path": self._relative(self.root, image),
                "image_sha256": api_server.sha256_file(image),
            }
        )
        api_server._save_job(job)
        with patch.object(api_server, "run_hunyuan", side_effect=self.fake_hunyuan):
            api_server._execute_image_to_3d(job_id)
        completed = api_server._read_job(job_id)
        self.assertEqual(completed["status"], "COMPLETED_RAW_WAITING_REVIEW")
        self.assertEqual(
            [item["role"] for item in completed["artifacts"][:2]],
            ["input_rgba", "raw_glb"],
        )
        self.assertIn("必须继续完成网格检查", completed["notice"])

    def test_multiview_job_records_real_2mv_backend(self) -> None:
        job_id, run_dir = api_server._new_job(
            "multiview_to_3d",
            {
                "interface_name": "Hunyuan3D-2.1-compatible",
                "backend_model": "Hunyuan3D-2mv",
                "input_mode": "multi_view",
                "seed": 1234,
                "inference_steps": 30,
                "octree_resolution": 256,
                "num_chunks": 8000,
                "images": {},
            },
        )
        job = api_server._read_job(job_id)
        for view in ("front", "left", "back"):
            image = run_dir / "masks" / f"{view}.png"
            Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(image)
            job["request"]["images"][view] = {
                "path": self._relative(self.root, image),
                "sha256": api_server.sha256_file(image),
            }
        api_server._save_job(job)
        with patch.object(
            api_server,
            "run_hunyuan_multiview",
            side_effect=self.fake_hunyuan_multiview,
        ):
            api_server._execute_multiview_to_3d(job_id)
        completed = api_server._read_job(job_id)
        self.assertEqual(completed["status"], "COMPLETED_RAW_WAITING_REVIEW")
        self.assertEqual(completed["metrics"]["backend_model"], "Hunyuan3D-2mv")
        self.assertEqual(completed["metrics"]["views"], ["front", "left", "back"])

    def test_gpu_block_is_recorded_instead_of_lost(self) -> None:
        job_id, _ = api_server._new_job(
            "text_to_image",
            {
                "prompt": "test",
                "model": "ernie-image-turbo",
                "base_seed": 1,
                "candidate_count": 1,
                "width": 768,
                "height": 768,
                "inference_steps": 8,
            },
        )
        with patch.object(
            api_server,
            "run_ernie",
            side_effect=api_server.GpuUnavailableError("GPU 正在使用"),
        ):
            api_server._execute_text_to_image(job_id)
        job = api_server._read_job(job_id)
        self.assertEqual(job["status"], "BLOCKED_GPU")
        self.assertEqual(job["error"], "GPU 正在使用")

    def test_artifact_tampering_is_rejected(self) -> None:
        request = {
            "prompt": "test",
            "model": "ernie-image-turbo",
            "base_seed": 1,
            "candidate_count": 1,
            "width": 768,
            "height": 768,
            "inference_steps": 8,
        }
        job_id, _ = api_server._new_job("text_to_image", request)
        with patch.object(api_server, "run_ernie", side_effect=self.fake_ernie):
            api_server._execute_text_to_image(job_id)
        job = api_server._read_job(job_id)
        image = self.root / job["artifacts"][0]["path"]
        image.write_bytes(b"changed")
        with self.assertRaises(HTTPException) as context:
            api_server.download_artifact(job_id, "candidate_01")
        self.assertEqual(context.exception.status_code, 409)

    def test_whitespace_prompt_is_rejected_before_queueing(self) -> None:
        with patch.object(api_server.JOB_EXECUTOR, "submit") as submit:
            with self.assertRaises(HTTPException) as context:
                api_server.submit_text_to_image(
                    api_server.TextToImageRequest(prompt="   ")
                )
        self.assertEqual(context.exception.status_code, 422)
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
