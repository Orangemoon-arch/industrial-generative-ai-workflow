# 工业生成式 AI 独立 HTTP API V1

本服务为竞赛任务一“文生图模型部署与验证”和任务二“图生三维模型部署与呈现”提供独立 HTTP 接口。它与 Gradio 页面相互独立，外部程序不需要进入模型目录或运行模型脚本。

## 当前准确边界

- 任务一默认连接 0904 竞赛规程指定的 `ERNIE-Image`；`ERNIE-Image-Turbo` 只保留为可回退后端。健康检查会分别报告两套权重是否完整。
- 任务二单图接口连接 `Hunyuan3D-2.1 Shape`；多图接口使用统一兼容包装，但真实后端明确记录为 `Hunyuan3D-2mv`。两者都只返回原始 GLB，仍须经过网格检查、人工确认、毫米缩放和 Bambu Studio 切片门禁，不能直接打印。
- API 使用单工作线程队列；模型适配器还会执行 `nvidia-smi`、计算进程检查和项目 GPU 文件锁，因此 ERNIE 与 Hunyuan 不会由本服务同时加载。
- V1 任务和状态写入 `workspace/runs/RUN-*/reports/api_job_v1.json`。队列本身仍是单进程内存队列；服务异常退出后不会自动恢复尚未开始的任务。
- 默认只监听 `127.0.0.1`，不开放公网。局域网绑定、鉴权和公司平台接口格式应在取得公司标准后再定。

## 启动

在项目根目录执行：

```bash
./START_INDUSTRIAL_AI_API.sh
```

默认地址：

- 健康检查：`http://127.0.0.1:27866/health`
- Swagger 接口文档：`http://127.0.0.1:27866/docs`
- OpenAPI JSON：`http://127.0.0.1:27866/openapi.json`

启动 API 本身不会加载 GPU 模型。需要停止时，在启动终端按 `Ctrl+C`。

如果通过 VS Code Remote-SSH 使用，可转发远程端口 `27866`，然后在 Windows 浏览器打开 `http://127.0.0.1:27866/docs`。

## 任务一：提交文生图任务

```bash
curl --noproxy 127.0.0.1 \
  -X POST http://127.0.0.1:27866/api/v1/text-to-image/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "生成一个六孔圆形法兰工业零件设计图，纯白背景",
    "model": "ernie-image",
    "base_seed": 20260907,
    "candidate_count": 1,
    "width": 768,
    "height": 768,
    "inference_steps": 50
  }'
```

接口立即返回 HTTP `202` 和任务编号。用返回的 `job_id` 查询：

```bash
curl --noproxy 127.0.0.1 \
  http://127.0.0.1:27866/api/v1/jobs/RUN-YYYYMMDD-HHMMSS-abcdef
```

状态为 `COMPLETED` 后，响应中的 `artifacts[].download_url` 可以下载候选 PNG、请求、结果和日志。

## 任务二：提交图生三维任务

输入必须是已经人工检查的 RGBA PNG：

```bash
curl --noproxy 127.0.0.1 \
  -X POST http://127.0.0.1:27866/api/v1/image-to-3d/jobs \
  -F 'image=@/绝对路径/confirmed_rgba.png;type=image/png' \
  -F 'seed=1234' \
  -F 'inference_steps=30' \
  -F 'octree_resolution=256' \
  -F 'num_chunks=8000'
```

状态为 `COMPLETED_RAW_WAITING_REVIEW` 时可以下载 `raw_glb`。这个状态特意包含 `WAITING_REVIEW`，表示三维生成成功但尚未通过制造和打印审查。

## 任务二：提交多视图图生三维任务

本地权重须放在 `models/hunyuan3d-2mv-ms/hunyuan3d-dit-v2-mv/`。正面、左侧、背面必填，右侧可选；所有图片都必须是包含透明背景与不透明主体的 RGBA PNG：

```bash
curl --noproxy 127.0.0.1 \
  -X POST http://127.0.0.1:27866/api/v1/multiview-to-3d/jobs \
  -F 'front=@/绝对路径/front.png;type=image/png' \
  -F 'left=@/绝对路径/left.png;type=image/png' \
  -F 'back=@/绝对路径/back.png;type=image/png' \
  -F 'right=@/绝对路径/right.png;type=image/png' \
  -F 'seed=1234' \
  -F 'inference_steps=30' \
  -F 'octree_resolution=256' \
  -F 'num_chunks=8000'
```

任务请求、结果和状态中同时保留 `interface_name=Hunyuan3D-2.1-compatible` 与 `backend_model=Hunyuan3D-2mv`，用于兼容现有流程并保证实验可审计。

## 状态含义

| 状态 | 含义 |
|---|---|
| `QUEUED` | 已归档请求，等待单队列执行 |
| `RUNNING` | 模型任务正在执行 |
| `COMPLETED` | 文生图已完成 |
| `COMPLETED_RAW_WAITING_REVIEW` | 原始 GLB 已完成，等待网格与人工检查 |
| `BLOCKED_GPU` | GPU 有其他计算进程或显存门禁未通过，没有干扰其他进程 |
| `REJECTED_INPUT` | 上传的任务二图片格式或 Alpha 通道不合规 |
| `FAILED` | 工作进程失败，错误与项目内日志已保留 |

## 无 GPU 测试

```bash
cd <PROJECT_ROOT>/apps/industrial-ai-service
.venv/bin/python -m unittest -v test_api.py
```

测试使用 `/tmp` 临时目录和模拟适配器，不加载 ERNIE/Hunyuan，不修改正式任务。
