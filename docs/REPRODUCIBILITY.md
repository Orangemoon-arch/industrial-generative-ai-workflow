# Reproducibility and scope

## Included

- First-party Gradio/FastAPI orchestration code.
- Model adapters and short-lived worker entry points.
- CPU mesh inspection, comparison, repair and gear-analysis code.
- Unit tests written for the workflow.
- A small set of representative, sanitized result images.

## Not included

- Model weights and model caches.
- Python virtual environments.
- The full upstream ComfyUI or Hunyuan3D repositories.
- Raw experiment runs, equipment configuration, credentials or internal platform data.
- Company or customer materials.

## Environment layout used during development

The original workstation used four isolated Python environments:

1. Service environment for Gradio/FastAPI.
2. ERNIE environment for text-to-image inference.
3. Qwen environment for NF4 visual-language inference.
4. Hunyuan environment for image-to-3D inference and geometry tooling.

GPU jobs were never loaded concurrently. Each adapter created a versioned request, executed a model-specific Python subprocess, captured logs and results, and waited for the process to exit before another model could acquire the project GPU lock.

## Why the complete application is not one-command reproducible here

The original experiment depended on roughly 90 GB of separately licensed local model weights and GPU-specific environments. Those assets are intentionally excluded. This repository is therefore a portfolio snapshot of the first-party engineering work, not a redistribution of the model stack.

The CPU geometry modules are the most portable part of the repository. Their tests document the expected safety properties: no silent overwrite, guarded topology changes, explicit candidate status, and human review before selection.
