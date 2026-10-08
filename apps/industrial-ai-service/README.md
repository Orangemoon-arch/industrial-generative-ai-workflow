# 工业生成式 AI 工作台：ERNIE + Qwen + 遮罩 + Hunyuan

这是第二阶段服务原型。当前竞赛主线调整为行星轮/齿轮装配：任务一已按 0904 竞赛规程把默认后端切换为 ERNIE-Image，并保留 ERNIE-Image-Turbo 作为显式回退；其余已接入 Qwen3-VL 候选图结构检查、CPU 透明遮罩检查、Hunyuan3D-2.1 Shape 单图生三维、Hunyuan3D-2mv 多视图兼容接口，以及通用 CPU 网格诊断与人工确认修复工具。风扇历史规则化代码和历史资产保留用于追溯，但不再作为当前主线；打印机、机械臂、灵巧手和相机仍未真实接入。

独立 HTTP API V1 已新增，入口为 `api_server.py`，一键启动文件为项目根目录 `START_INDUSTRIAL_AI_API.sh`，详细接口、状态和调用示例见 `README_API.md`。API 与本 Gradio 页面可分别启动；当前尚未把 Gradio 回调改为通过 API 调用。

## 文件

- `app.py`：Gradio 页面和任务清单逻辑。
- `gpu_executor.py`：`nvidia-smi` 预检和项目内 GPU 独占锁。
- `ernie_adapter.py`：调用 ERNIE 独立环境的版本化子进程适配器。
- `qwen_adapter.py`：调用 Qwen 独立环境并保存结构化检查结果的适配器。
- `hunyuan_adapter.py`：统一 Hunyuan 接口；单图调用真实 2.1 后端，多图调用真实 2mv 后端，并在请求、结果和任务清单中保存实际模型身份。
- `part_profiles.py`：维护风扇、齿轮、法兰和自定义零件的提示词默认值、计数对象、范围及 Qwen 检查重点。
- `assembly_planning.py`：生成不含真机控制命令的版本化装配工艺草案、多模态采集节点、区域流转和设备安全缺项清单。
- `assembly_component.py`：校验三件式参数化装配的BOM、图片、GLB、STL和SHA256，并保存版本化实体试装记录。
- `build_gear_assembly_process_sheet.py`：使用确定性中文排版生成三件式装配工艺设计图、工艺说明和试装验收模板。
- `device_center.py`：维护已确认设备事实和四种明确标记的安全模拟场景；不会探测网络、打开串口、加载 SDK 或提供控制命令。
- `multimodal_capture.py`：把人工上传的图片/短视频、采集节点、零件 ID、区域和人工标签归档到当前任务，并记录 SHA256、安全边界和待质检状态。
- `build_parametric_gear_assembly.py`：使用 CPU 生成三件式“D 形孔齿轮 + D 形短轴 + 圆孔限位帽”教学装配、独立 STL 和装配/爆炸 GLB；齿形为非传动教学轮廓。
- `test_parametric_gear_assembly.py`：在 Hunyuan 环境中验证三件参数化网格的尺寸、水密和拓扑门禁。
- `preprocessing_adapter.py`：把已确认候选图的 CPU 遮罩生成，以及 Hunyuan 原始 GLB 的检查/保守整理封装为版本化任务。
- `mesh_optimization_adapter.py`：CPU 规则化、四视图检查和 STL 导出的版本化子进程适配器。
- `mesh_import_adapter.py`：把项目目录或 Gradio 临时目录中的 GLB 安全归档到当前 RUN；检查扩展名、glTF binary 文件头、200 MiB 大小上限，并生成版本化网格报告和四视图。
- `mesh_repair_adapter.py`：把当前 RUN 的 review GLB 交给 Hunyuan 专用环境做版本化 CPU 诊断/修复，校验任务路径和 SHA256，并生成候选复核报告与四视图。
- `mesh_repair_tool.py`：任意 GLB 的 CPU 诊断、疑似异常面标记、带回滚门禁的 `safe` / `standard` / `advanced` 修复和人工确认删除；默认不修改原始 GLB，不包含风扇或行星轮专用几何规则。
- `gear_analysis_adapter.py`：在 Hunyuan 专用环境中运行直齿轮径向轮廓分析，输出版本化 JSON、轮廓图和日志，不修改源 GLB。
- `extract_spur_gear_parameters.py`：从外啮合直齿轮网格提取齿数、外径、齿根径、孔径和厚度草案；可与参考网格按外径对齐比较，用于判断应继续修复还是转参数化重建。
- `../Hunyuan3D-2.1/hy3dshape/prepare_service_white_background_mask.py`：CPU 白背景分割、拓扑清理和遮罩报告脚本。
- `../Hunyuan3D-2.1/hy3dshape/run_hunyuan_service_task.py`：Hunyuan 低显存独立工作进程。
- `../Hunyuan3D-2.1/hy3dshape/inspect_service_mesh.py`：CPU 网格指标与三视图检查。
- `../Hunyuan3D-2.1/hy3dshape/prepare_service_mesh_review.py`：仅剔除独立零面积碎片并生成版本化 review GLB。
- `../Hunyuan3D-2.1/hy3dshape/build_regularized_fan_review.py`：根据任务约束生成可复现的单体七叶片制造规则化候选，不修改 Hunyuan 原始网格。
- `../Hunyuan3D-2.1/hy3dshape/export_regularized_print_stl.py`：校验批准 GLB 的 SHA、尺寸和拓扑，平移最低点到热床并导出版本化 STL；发现悬空面时必须显式标记为需要支撑。
- `../qwen3-vl/qwen_review_rules.py`：不加载模型的确定性 Qwen 结果归一化规则；区分法兰螺栓孔与已经安装的实体紧固件。
- `verify_ernie_integration.py`：创建一项真实任务并生成一张图片的集成验证脚本。
- `verify_qwen_integration.py`：对已有任务候选执行一次真实 Qwen 检查的集成验证脚本。
- `test_app.py`：不依赖 GPU 的标准库测试。
- `requirements.txt`：独立界面环境依赖。
- `workspace/runs/RUN-*/manifest.json`：用户点击“创建任务”后生成的任务档案。

三个模型仍分别使用自己的虚拟环境，服务界面不会混装模型依赖。实际工作入口为 `apps/ernie-image/run_ernie_service_task.py`、`apps/qwen3-vl/run_qwen_service_review.py` 和 `apps/Hunyuan3D-2.1/hy3dshape/run_hunyuan_service_task.py`。

## Web 模型修复流程

Gradio 使用独立的第 5 页“模型修复”。第 4 页只保留 Hunyuan 生成、基础网格检查，以及默认折叠的规则化/STL 可选出口；第 5 页按“选择修复源 → 自动诊断 → 原模型/候选对比 → 结果管理”组织：

1. 可先完成 Hunyuan 原始 GLB 的 CPU 网格检查，也可上传外部 GLB。外部文件只允许来自项目目录或 Gradio 临时上传目录，必须是内容与扩展名一致的 glTF 2.0 binary，归档后才成为当前任务的 `selected_mesh`；原上传文件不修改。
2. 点击“诊断并自动分流”，生成版本化 JSON、异常标记 GLB 和日志；诊断不会替换当前模型。
3. 默认点击“按推荐方案生成候选”；safe / standard / artifact cleanup / advanced / reconstruct 手动选择放在折叠的高级设置中。`artifact cleanup` 只用于预期为单零件、主体表面积占比至少 99.5%、其余独立部件总面积/体积和尺寸均低于严格阈值的场景；它可同时用不移动顶点的局部换边消除退化三角形，结果仍需人工确认。严重开放/碎裂且存在占表面积至少 95% 的明确主体时，`reconstruct` 才执行 CPU 体素闭合与 Marching Cubes 重建。
4. 对比修复前后部件、水密、边界边、非流形边、退化面及候选四视图。候选同时生成同相机、同缩放的整体/主体近景前后对照，避免各自自动缩放掩盖尺寸或远离碎片变化。没有操作通过门禁时不生成候选 GLB。
5. 只有报告状态、SHA256 和完整拓扑门禁全部通过，且用户点击“接受候选”后，新文件才成为当前 review GLB；原始 GLB 始终保留。部分修复候选不能被接受为当前模型。
6. “修复历史恢复”列出当前 RUN 的所有 `mesh_repair_vN` 报告。输入版本号可恢复诊断标记、候选、预览和 JSON 到页面；该操作只恢复显示，不会回滚或改变 `selected_mesh`。

该流程以数字化报告和人工视觉复核为验收依据，不要求导出 STL 或进行实物打印。严重模型重建即使得到单一水密网格，也可能封闭真实孔洞、改变薄壁或保留错误的原始轮廓，因此只生成等待确认的候选，不会自动替换当前模型。齿数、孔径、公差、渐开线和配合面不属于通用拓扑门禁，必要时应进入参数提取或结构/CAD 重建。

第 5 页折叠区中的“直齿轮专用数字化指标”只对齿轮任务开放。它从当前 GLB 提取 FFT 齿数估算、置信比、外径、齿根径、中心孔径、厚度和由外径反推的模数。默认显示模型单位；用户提供经图纸或实测确认的外径后，页面只做等比例毫米换算。该面板不能确认压力角、渐开线精度、侧隙或公差。

网格修复工具应使用 Hunyuan 环境中的 `trimesh`：

```bash
apps/Hunyuan3D-2.1/.venv/bin/python apps/industrial-ai-service/mesh_repair_tool.py \
  --input <input.glb> \
  --report <diagnosis.json> \
  --annotated-glb <diagnostic_marked.glb>
```

默认只诊断和标记，不删除面。只有人工确认后，才可额外提供包含 `{"face_indices": [...]}` 的 JSON 和 `--repaired-glb`；输出必须通过水密、法向、单连通、边界边、非流形边和正体积门禁。

低风险自动修复模式：

```bash
apps/Hunyuan3D-2.1/.venv/bin/python apps/industrial-ai-service/mesh_repair_tool.py \
  --input <input.glb> \
  --report <safe_repair.json> \
  --annotated-glb <diagnostic_marked.glb> \
  --safe-repair \
  --repaired-glb <safe_repaired.glb>
```

`safe` 模式依次尝试清理非有限值、未引用顶点、完全重复顶点、重复面、近零面积面和法向错误。每一步独立试算；如果造成水密性丢失、边界/非流形/连通部件增加、包围盒变化或水密体积异常变化，就回滚该步。若没有任何安全改动，报告写入 `NO_SAFE_CHANGE_AVAILABLE`，且不会伪造一个“已修复”GLB。

受限小孔修复模式：

```bash
apps/Hunyuan3D-2.1/.venv/bin/python apps/industrial-ai-service/mesh_repair_tool.py \
  --input <input.glb> \
  --report <standard_repair.json> \
  --annotated-glb <diagnostic_marked.glb> \
  --standard-repair \
  --repaired-glb <standard_repaired.glb>
```

`standard` 会先执行全部 safe 操作，再检查边界图。当前只填补无分叉、只有 3 或 4 条边、周长不超过模型包围盒对角线 15%、补片面积不超过总面积 0.2% 的小孔。大开口、中心孔、6 边以上边界、分叉边界和复杂非流形区域全部拒绝自动填补。输出状态为 `REPAIRED_CANDIDATE_REQUIRES_VISUAL_REVIEW` 或 `PARTIAL_REPAIR_NOT_PRINT_APPROVAL`，不能直接作为打印批准。

高级候选模式：

```bash
apps/Hunyuan3D-2.1/.venv/bin/python apps/industrial-ai-service/mesh_repair_tool.py \
  --input <input.glb> \
  --report <advanced_repair.json> \
  --annotated-glb <diagnostic_marked.glb> \
  --advanced-repair \
  --max-hole-size 4 \
  --repaired-glb <advanced_repaired.glb>
```

`advanced` 使用 PyMeshLab 尝试清理重复/空面、修复非流形边和顶点、封闭不超过指定边数的小孔，并统计自交面数量。自交面只检测、不自动删除；如果仍有复杂边界或自交，输出只能是部分修复候选。该模式可能改变拓扑，必须进行视觉检查、尺寸比较和打印前复核。

实验性曲面候选使用 `--surface-smooth-candidate`。它只对疑似细长三角形周围、避开锐边的顶点执行 1 次局部平滑，并在曲面质量指标变差时拒绝输出。该模式不是默认修复，尤其不能直接用于确认行星轮齿面。

保特征等距重网格候选：

```bash
apps/Hunyuan3D-2.1/.venv/bin/python apps/industrial-ai-service/mesh_repair_tool.py \
  --input <input.glb> \
  --report <surface_remesh.json> \
  --annotated-glb <surface_remesh_diagnostic.glb> \
  --surface-remesh-candidate \
  --surface-remesh-feature-angle 45 \
  --surface-remesh-iterations 1 \
  --surface-remesh-target-ratio 1.0 \
  --surface-remesh-max-distance-ratio 0.0005 \
  --repaired-glb <surface_remesh_candidate.glb>
```

该模式会重新布置整个三角网格，保护高于特征角的锐边，并把新顶点回投到原表面。只有水密、连通、边界/非流形、尺寸、面积、体积、双向点到三角面距离、三角形长宽比和锐边数量门禁全部通过才会输出。含纹理或 UV 的输入会被拒绝，因为当前 PyMeshLab 临时格式不能保证材质映射保持。输出仍是需要视觉复核的候选，不是打印批准，也不能恢复精确 CAD 齿形。

外啮合直齿轮参数草案提取：

```bash
apps/Hunyuan3D-2.1/.venv/bin/python apps/industrial-ai-service/extract_spur_gear_parameters.py \
  --input <candidate.glb> \
  --reference <known_reference.stl> \
  --report <parameter_draft.json> \
  --profile-preview <radial_profile.png>
```

该工具用径向包络 FFT 估算齿数，并报告外径、齿根径、孔径和厚度比例。提供参考网格时，会先按外径对齐，再比较其他尺寸。它不能从生成式网格可靠恢复压力角、齿侧间隙、公差或材料收缩量，输出固定为工程确认前的参数草案，不能直接批准打印。

stereoframe 评估的简化入口是项目根目录的 `RUN_STEREOFRAME_EVAL.sh`：

```bash
./RUN_STEREOFRAME_EVAL.sh <模型.glb> [输出名称]
```

它会自动在 `workspace/docs/stereoframe_runs/` 下创建不覆盖旧结果的版本化目录。

不传参数时脚本会列出 `workspace/runs` 和 `workspace/meshes` 中最近修改的 GLB，输入编号即可；使用 `./RUN_STEREOFRAME_EVAL.sh --list` 可查看全部候选路径。`workspace/docs/stereoframe_*` 下的 GLB 是评估副本，不是原始模型。

## 环境

界面使用独立虚拟环境，不能把 Gradio 安装到三个模型环境中：

```bash
cd <PROJECT_ROOT>/apps/industrial-ai-service
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 启动

推荐使用项目根目录的一键启动文件。在 VS Code 已打开项目根目录时，按 `Ctrl+Shift+B`，会自动执行默认任务“启动工业 AI 本地服务”：

```bash
<PROJECT_ROOT>/START_INDUSTRIAL_AI_SERVICE.sh
```

一键脚本使用当前默认端口 `127.0.0.1:27865`，只启动网页，不会立即加载三个 GPU 模型。服务在当前 VS Code 终端前台运行；需要停止时在该终端按 `Ctrl+C`。如果检测到页面已经运行，脚本会直接显示访问地址；如果端口有其他服务响应，脚本只报错退出，不会停止共享机器上的进程。

为避免用户会话代理把 Gradio 的 localhost 启动回调错误转发到外部网络，一键脚本只在本次服务进程内把 `127.0.0.1,localhost` 加入 `NO_PROXY/no_proxy`，并关闭 Gradio 遥测；`app.py` 同时关闭 SSR。它不会修改 `/etc/environment`、shell 配置或用户的全局代理。若浏览器长期显示“正在加载”，先在远程机确认 `curl --noproxy 127.0.0.1 http://127.0.0.1:27865/` 是否返回，再检查 VS Code 端口转发。

如果需要临时使用其他端口，可在终端运行：

```bash
INDUSTRIAL_AI_SERVICE_PORT=27866 <PROJECT_ROOT>/START_INDUSTRIAL_AI_SERVICE.sh
```

手动启动方式仍然保留：

```bash
cd <PROJECT_ROOT>/apps/industrial-ai-service
.venv/bin/python app.py
```

直接运行 `app.py` 时默认只监听远程机本地地址 `127.0.0.1:17860`；一键脚本明确使用 `127.0.0.1:27865`。两种方式都不会生成公网分享链接。

如果此前已启动旧版界面，需要在确认该终端进程属于自己后按 `Ctrl+C` 停止，再重新启动，才能加载最新代码。不要结束共享机器上的未知进程。

如果使用 VS Code Remote-SSH，通常会自动发现端口；可以在 VS Code 的“端口”面板确认 `27865` 已转发，然后点击面板显示的本地地址。如果没有自动转发，也可在 Windows PowerShell 手动建立：

```powershell
ssh -L 27865:127.0.0.1:27865 用户名@远程机地址
```

保持该 PowerShell 窗口开启，然后在 Windows 浏览器访问：

```text
http://127.0.0.1:27865
```

如果提示端口被占用，可在远程机使用 `--port 17861` 启动，并把 SSH 命令和浏览器地址中的端口同步改为 `17861`。不要结束共享机器上的未知进程。

不建议在共享实验机上直接使用 `--host 0.0.0.0`，除非已经确认局域网访问控制和现场网络要求。

## 测试

```bash
cd <PROJECT_ROOT>/apps/industrial-ai-service
.venv/bin/python -m unittest -v test_app.py
```

测试使用临时目录，不会创建正式任务或修改已有图片、GLB、STL。

多视图兼容适配器测试：

```bash
cd <PROJECT_ROOT>/apps/industrial-ai-service
.venv/bin/python -m unittest -v test_hunyuan_adapter.py
```

多视图权重已安装在 `models/hunyuan3d-2mv-ms/hunyuan3d-dit-v2-mv/`；当前使用 `model.fp16.safetensors`，SHA256 为 `d36f5881bcdc56726b73e517cd444c13c60732431622da7268145355c8d38e9c`。worker 会为官方旧命名空间生成任务内兼容配置，不改写官方 `config.yaml` 或现有 2.1 权重。实际推理前仍必须通过 `nvidia-smi` GPU 门禁。

参数化齿轮装配的 CPU 几何测试使用已有 Hunyuan 环境中的 `trimesh`，不会加载模型或使用 GPU：

```bash
cd <PROJECT_ROOT>/apps/industrial-ai-service
../Hunyuan3D-2.1/.venv/bin/python -m unittest -v test_parametric_gear_assembly.py
```

齿轮轴装配的配合测试片也有独立 CPU 几何测试：

```bash
cd <PROJECT_ROOT>/apps/industrial-ai-service
../Hunyuan3D-2.1/.venv/bin/python -m unittest -v test_gear_assembly_fit_coupons.py
```

## 使用 ERNIE、Qwen 和 Hunyuan

1. 在“1 任务与提示词”页面选择零件类型；风扇、齿轮和法兰会自动填写各自默认结构、禁止项和计数对象，也可直接输入其他零件名称进入自定义配置。自定义零件的重复结构数量填 `0` 时不会强制旋转对称。若结构文字写出的数量与数量字段冲突（例如“16齿”与 `12`），页面会在创建任务前阻止提交；
2. 创建任务并检查系统组合出的最终提示词；
3. 点击“确认最终提示词”；
4. 在“2 文生图与检查”页面选择生成 1–4 张，并选择 512、768 或 1024 分辨率；首次批量筛选推荐 768；
5. 点击“启动 ERNIE 文生图”；
6. 等待 ERNIE 完成，在候选编号中选择要检查的图片；
7. 点击“使用 Qwen 检查候选”，先查看中文检查摘要；原始结构化 JSON 保留在折叠的高级信息中；
8. 人工检查数量、结构、背景和额外零件；
9. 候选经人工确认后即可进入遮罩步骤；Qwen 检查为可选风险参考，其 `PASS`、`REVIEW` 或 `REJECT` 不作为流程门禁。人工主动拒绝的候选仍不得进入后续步骤。
10. 在“3 透明遮罩”页点击“CPU 生成新遮罩”；页面会同时显示原图、RGBA 和棋盘格预览。检查背景是否透明、内部孔洞是否保留、底部阴影是否误入主体；只有已归档且基础检查通过的遮罩才能人工确认。参数不合适时可生成新版本，旧版本不会被覆盖。
11. 确认遮罩后才允许点击“启动 Hunyuan Shape”；输出先标记为原始 GLB，不能直接确认或打印。如需多视图，可在第 4 页展开“Hunyuan 多视图输入”，上传同一零件的 `front/left/back` 三张 RGBA PNG，`right` 可选。该入口对外保持统一工作流，档案会明确记录真实后端为 `Hunyuan3D-2mv`，不会把 2mv 权重冒充为 2.1 权重。
12. 点击“CPU 检查并保守整理 Hunyuan 原始 GLB”，检查异常平面、碎片、连通部件、水密、法向、边界边和非流形边，并另存版本化 review GLB。已合格的原始 GLB 只做逐字节副本；只有独立零面积碎片允许自动剔除，存在有实际面积的次部件时会停止并要求人工判断。
13. 点击“加载当前网格检查”，同时查看 GLB、正视/侧视/斜视/背面检查图和中文摘要；原始 JSON 收在折叠的高级信息中。
14. 如果外形可参考但制造质量不足，可展开“七叶片制造规则化”，调整外径、涵道、轮毂、叶片厚度、掠角和弯度；点击后只运行 CPU，自动生成新版本 GLB、四视图和网格报告。
15. 新规则化版本会取代旧版本成为当前检查对象，但不会覆盖或删除旧 GLB/STL；只有 SHA256 与归档一致且检查指标全部通过，才能人工确认三维。
16. 人工确认当前 GLB 后，在独立的“通用 STL 导出”区域填写本次 STL 目标最长尺寸并勾选切片检查说明。页面会按当前网格比例预览预计 X/Y/Z；尺寸允许为 20–200 mm，默认继承任务创建值，也可为同一已确认 GLB 改填新值并导出版本化 STL，不覆盖旧文件或篡改任务原始目标。导出会等比缩放无单位 Hunyuan 网格、把最低点对齐到 `Z=0` 并重新加载验证；不会连接 Bambu Studio、打印机或自动发送打印。

如果需要继续历史任务，可以在页面顶部输入完整 `RUN-...` 编号并点击“加载已有任务”，也可以在历史任务页直接点击任意一行。加载本身是只读的，会恢复表单、候选图、Qwen 结果以及经过路径和 SHA 校验的当前遮罩、review GLB、多视图、报告和 STL；任务没有对应产物时会清空活动区域，不会用历史涡扇占位。

“当前任务摘要”默认以普通表格展示零件、状态、候选数量、中文阶段进度和建议下一步；每页顶部同步显示最近一次操作的成功、等待、拦截或失败信息。完整 `manifest.json` 仍保留在折叠的“高级信息”中，便于实验追溯。历史任务页使用表格显示任务编号、零件类型、中文状态和时间。

ERNIE、Qwen、Hunyuan、遮罩、网格检查、七叶片规则化和 STL 导出等耗时按钮会在点击后立即显示预计等待说明，并临时禁用和改名为“处理中，请勿重复点击”；任务结束后恢复按钮。该反馈只能说明请求已经提交或正在运行，不代表模型结果通过门禁。

三维页会根据零件类型显示制造能力提示。只有风扇任务可以在当前网页使用“七叶片制造规则化”面板。齿轮装配使用独立 CPU 参数化脚本，并已接入第 7 页的受验证装配组件入口；法兰和自定义零件仍没有专用确定性制造生成器。

## 三件套装配验收与后续装配规划

竞赛指导要点没有规定多个零件必须在同一打印板同时打印。任务 2 要求生成合规的工业零部件三维模型并打印成型；任务 3 要求实施典型零部件装配作业。因此装配项目应在建模前先定义 BOM、配合接口、基准尺寸和制造间隙，再为每个零件分别保留可检查的 STL。“同板打印”只是为了节省一次切片和打印操作，只有零件朝向兼容、间距足够且切片预览通过时才使用；分别打印不影响它们构成同一装配组件。

第 7 页新增“按功能自动规划组件（试用）”。选手只选择教学用途、已验证的装配效果和平台打印配置，再用一句话描述需求；不需要输入齿轮外径、轴径、孔径或配合间隙。平台从 `RUN-20260901-115822-a30c10` 的 A1/PLA 实物通过记录中选择内部参数，通过 CPU 创建新的版本化三件套任务，并生成三个独立 STL、齿轮与限位帽可选同板 STL、装配/爆炸 GLB、BOM、装配预览、工艺图和验收模板。内部毫米参数保存在折叠的高级规划档案中用于追溯，不要求选手修改。

当前自动规划器只支持这一种已验证模板，不使用自然语言大模型推测任意机械结构，也不能从无尺度图片猜测真实配合尺寸。每个新任务仍是打印候选：必须在 Bambu Studio 人工检查，并在新打印件上重新完成试装验收。创建过程只运行 CPU 参数化几何，不加载 ERNIE、Qwen 或 Hunyuan 模型，不连接打印机或机器人。

“7 三件套装配验收”页优先展示当前已验证的三件套入口；机械臂工艺草案收在“后续功能”折叠区，当前不要求填写。展开后可按当前零件配置编辑配合件、工艺步骤、候选抓取特征、配合特征、接近/装配方向和成功判据。用户还必须明确取料区、装配区、成品区和异常区；四个区域当前只保存名称，不保存或推测现场坐标。

每次保存生成 `assembly/assembly_plan_vN.json`，旧版本不会覆盖。方案同时记录：

- `取件 → 搬运 → 对准 → 装配 → 视觉复核 → 成品/异常分拣`状态机；
- 抓取前、抓取后、装配前、装配后和最终分拣的图像/视频、时间戳及标签采集点；
- Qwen 只负责观察零件、区域、抓取/对准/装配状态，禁止直接产生控制命令；
- 尚缺的机械臂、灵巧手、相机、坐标标定、工作区、零件工程数据和安全资料；
- 禁止输出关节角、笛卡尔位姿、轨迹、速度、力/力矩或夹爪命令。

人工确认装配方案只表示工艺草案内容可以继续研究，状态仍是 `PLAN_CONFIRMED_SIMULATION_REQUIRED`。真正执行必须依次完成设备资料、现场坐标标定、离线仿真、空载验证和现场人工授权的低速联调。

同一页面新增“三件式参数化装配组件（MVP）”。它当前只支持“16齿教学齿轮 + D形短轴 + 可拆限位帽”一个模板，不把单零件Hunyuan输出冒充多零件装配。输入对应参数化装配 `RUN-...` 后，页面会重新校验任务内路径和SHA256，并显示：

- 三项BOM及每项独立打印文件；
- 装配完成预览和分步骤装配工艺设计图；
- 齿轮、短轴、限位帽独立STL；
- 齿轮与限位帽同板打印STL；
- 装配/爆炸GLB、工艺说明和验收模板；
- D孔/帽孔间隙、打印状态和短轴是否需要重打；
- 打印后的齿轮空转角度、插入情况、帽倒置脱落、拆卸和缺陷记录。

试装记录保存为任务内版本化 `reviews/assembly_fit_review_vN.json`。建议通过判据为齿轮剩余空转不大于2°、限位帽倒置轻晃不脱落、齿轮和限位帽可手动装入并无损拆卸且无可见打印缺陷。该判据用于教学装配验证，不是通用机械配合标准。

## 设备中心与多模态采集 V1

“8 设备中心与多模态采集”是任务 3 的第一阶段开发入口，不是真机控制台。设备中心提供四种场景：当前真实边界、只读在线演示、相机采集演示和安全故障拦截演示。除“当前真实边界”外均带“模拟”标记；所有快照固定为 `READ_ONLY_NO_DEVICE_COMMANDS`，`control_allowed=false`，保存后进入当前任务的 `devices/device_snapshot_vN.json`。刷新和保存均不会执行网络扫描、串口访问或厂商 SDK 调用，也不会改变文生图/三维/STL 的主流程状态。

多模态采集 V1 接受单个不超过 20 MiB 的 PNG、JPG、WEBP、MP4、MOV 或 WEBM，采集节点限定为 `before_pick / after_grasp / before_assembly / after_assembly / final_sort`，区域限定为取料区、装配区、成品区和异常区。每次归档创建独立的 `multimodal/captures/CAP-.../`，其中包含原媒体和 `record.json`；任务清单同时记录两者路径和 SHA256。记录明确标记为 `MANUAL_WEB_UPLOAD / MANUAL_LABEL_UNVERIFIED`，相机连接、标定和 Qwen 检查均为未完成，也不含任何设备运动命令。

使用时先创建或加载一个 `RUN-...`，在设备中心选择场景并可选保存快照；然后上传一张现场图片或短视频，填写采集节点、零件 ID、区域和人工标签，点击“归档到当前任务”。“加载并校验已有采集”会重新检查所有记录与媒体的 SHA256。当前适合验证数据格式、交接和页面流程，不能作为任务 3 真机联调完成证据。

首个参数化装配样例为 `RUN-20260828-145136-b7c4e2`：16 齿教学齿轮外径 60 mm、厚 10 mm，使用 12 mm D 形轴和 12.5 mm D 形孔形成 0.50 mm 直径间隙；顶部 8 mm 圆柱与限位帽 8.3 mm 盲孔形成 0.30 mm 直径间隙。V1已完成实体打印，但齿轮仍可相对D轴转动约一个齿距，限位帽倒置会脱落，因此该配合未通过功能验收。齿形不是标准渐开线，禁止用于动力传动。

当前收紧版任务为 `RUN-20260901-115822-a30c10`：保持60 mm V1名义外形，D孔直径间隙改为0.30 mm，帽孔直径间隙改为0.10 mm。现有V1短轴可复用；`stl/gear_cap_v3_combined_print_candidate.stl` 将新齿轮和盲孔朝上的新限位帽以两个独立实体排在同一打印板上，尺寸约 `81 × 60 × 10 mm`。对应装配工艺图为 `reports/gear_shaft_cap_assembly_process_v2.png`。用户已确认实体试装通过；若用于正式归档，仍需在第7页填写实际观察值并保存版本化验收记录。

该任务的配合测试片 V1 位于 `fit_coupons/v1/`。组合 STL 包含 8 个独立实体：D 形孔测试 `0.30/0.50/0.70 mm` 三档、圆孔测试 `0.10/0.30/0.50 mm` 三档，以及两根测试轴。组合尺寸约 `98 × 53 × 11 mm`，设计为当前朝向无支撑打印；仍须先在 Bambu Studio 核对尺寸、8 个实体、首层和无支撑预览。具体识别方法与结果表见任务下的 `reports/gear_assembly_fit_coupon_print_guide_v1.md`。

如果 Hunyuan 网格通过拓扑检查但人工制造审查未通过，应保留原始网格作为模型输出证据，再生成独立的规则化制造候选。规则化文件必须明确记录为确定性后处理结果，不能冒充 Hunyuan 原始输出。当前七叶片 v3 使用 60 mm 外径、12 mm 外环深度、20° 扫掠和双面弯曲叶片；因用户取消平整背面，后续必须重新评估支撑，不能继承 v2 的无支撑结论。

Qwen 使用 `blind_observation_v2`：先让模型在不知道目标数量和禁止项的情况下独立观察，再由确定性代码把观察值和任务要求比较。这能降低模型迎合提示词目标的风险。真实涡扇样本证明它能发现中心孔和紧固件并正确拦截，但叶片精确计数仍可能出错，因此 Qwen 只能作为自动风险筛查，不能代替人工最终确认。

768 分辨率用于快速候选筛选，不代表最终固定输出。候选通过 Qwen 和人工检查后，再决定直接作为 Hunyuan 结构输入，或生成/放大高分辨率版本；不能重新生成后跳过复核，因为重新生成可能改变叶片数量和附加结构。

模型启动前会自动：

- 执行标准 `nvidia-smi`；
- 检查 GPU 0 是否存在其他计算进程；
- ERNIE 检查可用显存是否至少为 3500 MiB，Qwen 至少为 7500 MiB，Hunyuan 至少为 8000 MiB；
- 获取 `workspace/runs/.gpu-model.lock`，防止本服务并行启动两个 GPU 模型。

检测到其他计算进程时只会阻止本次启动，不会结束或干扰该进程。共享锁只能协调本项目服务内部任务，不能替代对整台共享机器的 `nvidia-smi` 检查。

每次尝试分别保存：

```text
workspace/runs/RUN-.../
├── requests/ernie_attempt_001.json
├── requests/qwen_review_001.json
├── requests/hunyuan_attempt_001.json
├── requests/candidate_03_seed_20260842_mask_auto_v1.json
├── requests/hunyuan_attempt_001_mesh_review_v1.json
├── images/ernie_attempt_001/*.png
├── masks/candidate_03_seed_20260842_rgba_auto_v1.png
├── masks/candidate_03_seed_20260842_mask_auto_v1_preview.png
├── masks/candidate_03_seed_20260842_mask_auto_v1_report.json
├── meshes/hunyuan_attempt_001_raw.glb
├── meshes/hunyuan_attempt_001_main_review_v1.glb
├── meshes/fan_7blade_regularized_review_v4.glb
├── requests/fan_regularization_v4.json
├── reports/fan_7blade_regularized_inspection_v4.json
├── reports/fan_7blade_regularized_projection_v4.png
├── reports/ernie_attempt_001_result.json
├── reports/hunyuan_attempt_001_result.json
├── reports/hunyuan_attempt_001_mesh_inspection_v2.json
├── reports/hunyuan_attempt_001_projection_review_v2.png
├── reviews/qwen_review_001.json
├── logs/ernie_attempt_001.log
├── logs/qwen_review_001.log
├── logs/hunyuan_attempt_001.log
└── manifest.json
```

## 当前边界

- 遮罩和三维活动区域初始为空，只显示当前任务产物；第一阶段涡扇候选与打印照片仅保留在明确标注的历史参考/打印证据区域。
- ERNIE、Qwen 和 Hunyuan 按钮会真实运行对应模型；三个模型严格顺序运行。
- 新任务已经能在页面内从已确认候选生成 CPU 遮罩，也能把 Hunyuan 原始 GLB 检查/保守整理成独立 review GLB；两个步骤都必须人工复核，不能仅凭自动指标确认外形正确。
- Hunyuan 原始 GLB 不能直接确认。当前网格检查与保守整理使用 CPU 脚本，生成独立 review GLB 后再由界面加载和人工确认。
- 已实现任务一/任务二独立 FastAPI V1，但 Gradio 仍直接调用适配器，尚未改成 API 客户端；API V1 使用单进程内存队列，服务退出后不会自动恢复尚未开始的任务，也尚未加入公司级鉴权和多用户隔离。
- 当前输出只用于设计与流程验证，不能直接作为机械部件工程图。
- 通用遮罩和通用网格门禁不限制零件类型；提示词和 Qwen 计数/检查重点已对风扇、齿轮、法兰、自定义零件分流，但 Qwen 仍是风险筛查而不是尺寸检测。七叶片制造规则化仍是风扇专用；齿轮装配只有独立的教学用参数化脚本，尚未成为网页通用模板；法兰、夹具等仍无确定性制造规则化模板。
- 当前使用 Gradio 前台队列而不是持久后台任务；服务进程退出时不能保证正在执行的任务继续运行。
- STL 导出后仍必须人工进入 Bambu Studio 做逐层、朝向、首层和支撑检查；服务不会自动批准或发送打印。
- 装配模块目前只做设备无关工艺规划、设备模拟/只读快照和人工上传采集。现场机械臂已从照片确认是 RM65-B，但 O7 控制协议、相机资料、坐标标定、完整安全回路和仿真仍不具备，因此不能声称已接入设备、已完成仿真或可执行真实装配。
