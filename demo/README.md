# CPU mesh-repair demo

This demo exercises the real detached-artifact cleanup logic without model weights or a GPU.

It deterministically creates a watertight main body plus one tiny detached solid artifact, runs the guarded cleanup, and verifies:

- the input contains two connected components;
- the candidate contains one connected component;
- the candidate remains watertight and winding-consistent;
- boundary, non-manifold, and degenerate-face counts are zero;
- the original GLB SHA256 does not change;
- the removed component satisfies the configured area, volume, and size limits.

From the repository root:

```bash
python3 -m venv .demo-venv
.demo-venv/bin/python -m pip install -r requirements-geometry.txt
.demo-venv/bin/python demo/run_mesh_repair_demo.py
```

Outputs are written to `demo/output/`:

```text
demo/output/
├── damaged_with_detached_artifact.glb
├── repaired_candidate.glb
├── repair_report.json
└── before_after.png
```

The script refuses to overwrite a non-empty output directory. Use a different path for another run:

```bash
.demo-venv/bin/python demo/run_mesh_repair_demo.py --output-dir /tmp/mesh-repair-demo
```

The generated candidate is a demonstration of guarded topology cleanup, not a print approval.
