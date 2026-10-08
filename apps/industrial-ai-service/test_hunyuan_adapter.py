from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import hunyuan_adapter


class _GpuSnapshot:
    def to_dict(self) -> dict[str, int]:
        return {"free_mib": 10000}


class HunyuanAdapterCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hunyuan-adapter-test-")
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "workspace/runs/RUN-20260922-120000-abcdef"
        for name in ("requests", "masks", "meshes", "reports", "logs"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        python_path = self.root / "apps/Hunyuan3D-2.1/.venv/bin/python"
        worker_path = self.root / "apps/Hunyuan3D-2.1/hy3dshape/run_hunyuan_service_task.py"
        python_path.parent.mkdir(parents=True)
        worker_path.parent.mkdir(parents=True)
        python_path.write_text("", encoding="utf-8")
        worker_path.write_text("", encoding="utf-8")
        for model_dir, subfolder in (
            ("hunyuan3d-2.1-shape-ms", "hunyuan3d-dit-v2-1"),
            ("hunyuan3d-2mv-ms", "hunyuan3d-dit-v2-mv"),
        ):
            folder = self.root / "models" / model_dir / subfolder
            folder.mkdir(parents=True)
            (folder / "config.yaml").write_text("test: true\n", encoding="utf-8")
            (folder / "model.fp16.ckpt").write_bytes(b"checkpoint")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _image(self, view: str) -> tuple[Path, str]:
        path = self.run_dir / "masks" / f"{view}.png"
        image = Image.new("RGBA", (8, 8), (255, 255, 255, 0))
        image.putpixel((4, 4), (20, 20, 20, 255))
        image.save(path)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def _fake_subprocess(self, command, **kwargs):
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        output = self.root / request["output_path"]
        result_path = self.root / request["result_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"glTF-test")
        result = {
            "interface_name": request["interface_name"],
            "model": "test",
            "backend_model": request["backend_model"],
            "input_mode": request["input_mode"],
            "inputs": request["images"],
            "output_path": request["output_path"],
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "output_bytes": output.stat().st_size,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    def test_multiview_request_uses_2mv_and_records_views(self) -> None:
        inputs = {view: self._image(view) for view in ("front", "left", "back", "right")}
        paths = {view: item[0] for view, item in inputs.items()}
        hashes = {view: item[1] for view, item in inputs.items()}
        with (
            patch.object(hunyuan_adapter, "project_gpu_lock", return_value=nullcontext()),
            patch.object(hunyuan_adapter, "require_available_gpu", return_value=_GpuSnapshot()),
            patch.object(hunyuan_adapter.subprocess, "run", side_effect=self._fake_subprocess),
        ):
            execution = hunyuan_adapter.run_hunyuan_multiview(
                project_root=self.root,
                run_dir=self.run_dir,
                image_paths=paths,
                image_sha256=hashes,
            )
        request = json.loads((self.root / execution["request_path"]).read_text(encoding="utf-8"))
        self.assertEqual(request["interface_name"], "Hunyuan3D-2.1-compatible")
        self.assertEqual(request["backend_model"], "Hunyuan3D-2mv")
        self.assertEqual(request["model_subfolder"], "hunyuan3d-dit-v2-mv")
        self.assertEqual(list(request["images"]), ["front", "left", "back", "right"])

    def test_multiview_rejects_missing_required_view(self) -> None:
        inputs = {view: self._image(view) for view in ("front", "left")}
        with self.assertRaisesRegex(ValueError, "缺少"):
            hunyuan_adapter.run_hunyuan_multiview(
                project_root=self.root,
                run_dir=self.run_dir,
                image_paths={view: item[0] for view, item in inputs.items()},
                image_sha256={view: item[1] for view, item in inputs.items()},
            )

    def test_single_view_contract_remains_2_1(self) -> None:
        image_path, digest = self._image("front")
        with (
            patch.object(hunyuan_adapter, "project_gpu_lock", return_value=nullcontext()),
            patch.object(hunyuan_adapter, "require_available_gpu", return_value=_GpuSnapshot()),
            patch.object(hunyuan_adapter.subprocess, "run", side_effect=self._fake_subprocess),
        ):
            execution = hunyuan_adapter.run_hunyuan(
                project_root=self.root,
                run_dir=self.run_dir,
                image_path=image_path,
                image_sha256=digest,
            )
        request = json.loads((self.root / execution["request_path"]).read_text(encoding="utf-8"))
        self.assertEqual(request["backend_model"], "Hunyuan3D-2.1")
        self.assertEqual(request["input_mode"], "single_view")


if __name__ == "__main__":
    unittest.main()
