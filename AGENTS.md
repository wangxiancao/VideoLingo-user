# VideoLingo Agent 指南

## 项目概览

VideoLingo 是一个视频转录、翻译与字幕应用。默认入口是 **`videolingo.py`**（一条命令跑新六阶段管线 `pipeline/`：ingest→transcribe→segment→translate→subtitle→burn）。

`runs/<name>/` 下的产物在一次运行期间是跨阶段契约（可按阶段断点续跑）。旧编排（`st.py`/`translate_video.py`/`core/`）已下线，回退点是 git tag **`legacy-core-last`**。

## 仓库结构

- **`docs/agents/execution-chain.md`** — 动态流程：文件驱动的数据流、跨阶段契约（路径、SRT 集合、时间戳格式）、幂等实况与门控点。其中涉及旧管线的段落是历史记录。
- 顶层：`videolingo.py` 是默认入口；`pipeline/`（runner/stages/asr/llm/codec/registry）是新管线本体；`native/asr/` 是自研 FP16 CUDA ASR 引擎源码与构建（`native/llamacpp/` 为 Q4 翻译 LLM 的 llama.cpp STQ 构建脚本）；`bench/` 是一次性基准/取证 harness；`config.example.yaml` 是被跟踪的 schema（默认自包含本地栈：native ASR 引擎 + 托管 HY-MT2 LLM），`config.yaml` 可变、被忽略、可能含密钥；`docs/` 是独立的 pnpm/Next.js 站点。

## 开发流程

1. 改动产物路径、表格列、时间戳、排序规则或函数签名之前，先追清它的生产者与消费者；从 `docs/agents/execution-chain.md` 与 `pipeline/registry.py` 入手。
2. 不要在同一份 checkout 里并发跑流水线：它们共享 `runs/` 与显式 `--config`，锁只在进程内有效。
3. 新配置键加到 `config.example.yaml`。管线只通过薄包装读取只读 `--config`，运行期不写 `config.yaml`；无关工作中不要覆盖本地 `config.yaml`。
4. 改动保持聚焦。汇报受影响的阶段、验证证据，以及任何未测试的模型、GPU、API 或依赖媒体文件的路径。
5. 所有改动走 feature 分支 + PR；`main` 受保护，不直推。

## 编码准则

- LLM 调用走 `pipeline/llm/` 的客户端与托管服务（OpenAI 兼容；Q4 llama.cpp 为默认 runtime，BF16 transformers 为回退）；配置访问走 `pipeline/config.py` 薄包装。
- 把中间路径、JSON schema、SRT 顺序和时间戳格式当作 API 对待。保留有序的词级时间戳；对齐保持顺序进行、匹配不完美时尽力而为（best-effort）。
- UI/交互逻辑不进流水线阶段，供应商特定代码不进编排模块。

## 测试说明

`tests/` 是聚焦测试套件（runner/stages/llm 生命周期/服务多副本等；`.venv/bin/python -m pytest tests/ -q`；GPU/模型相关的按资产缺失自动跳过）。阶段级验证按 `docs/agents/verification.md`。

- Python：`.venv/bin/python -m compileall -q videolingo.py pipeline`
- 默认入口：`.venv/bin/python videolingo.py <video-or-url>`（产物落 `runs/`；阶段可 `--stages` 选择；`--validate-only` 只验产物）
- 字幕烧录质量门：`bench/v4_7_burn_gate.py`；Q4 探针：`bench/v4_7_q4_probe.py`
- 文档：`cd docs && pnpm build`

管线按产物存在性跳过已完成阶段、翻译缓存默认开启——重跑不等于干净重算。缺少所需服务与资产时，绝不声称端到端成功。

## 环境搭建

- 用 Python 3.10。FFmpeg 必需；字幕渲染需要 Noto 字体；GPU 推理需要 CUDA 12.x（llama.cpp 构建按 `native/llamacpp/build.sh`）。
- 项目 Python 环境是仓库根的 `.venv`（真实目录；唯一装全依赖集的解释器）。PATH 上的系统/conda 解释器缺依赖——绝不要信任 shell PATH 里的裸 `python3`/`pip`。
- 该 venv 没有 pip；用 `uv pip install --python .venv/bin/python <pkg>` 往里装包。
- 把 `config.example.yaml` 复制成 `config.yaml`；绝不提交或暴露其本地值。
- 模型权重一次性预下载：GLM-ASR 与 ForcedAligner（ModelScope 缓存 `~/.cache/modelscope`）、HY-MT2-1.8B GGUF（`_model_cache/HY-MT2-1.8B-GGUF/`）；运行期不下载。
- 使用 `docs/` 里已提交的 pnpm lockfile。

## 工具使用规则

- 不要直接修改 `runs/`、`output/`、`history/`、`queue_logs/`、`_model_cache/`、`bench/results/` 里的生成内容；通过产出它的那个阶段重新生成。
- 不要拿当前的 Docker 构建当本地改动的证据：Dockerfile 克隆远端仓库，而不是拷贝工作区。
- 只有任务明确要求时才新增依赖。

## 已知问题 / 约束

- 配音（TTS）能力不在新管线中：旧管线的 TTS 栈已随旧编排一并移除（历史见 tag `legacy-core-last`）；`--mode tts` 类能力如需恢复属于新特性。
- 新管线无 Web UI；`videolingo.py` 是唯一产品入口（如需 UI 属于新特性）。
- `docs/` 站点部分页面仍描述旧管线/Whisper/配音；以代码与 `config.example.yaml` 为准，文档翻新未排期。
- 完整验证可能需要网络访问、下载、凭据、FFmpeg、可观的 GPU 显存，以及一个真实媒体样本。
- 共享 GPU 上：native ASR 引擎按空闲显存动态下预算；正式基准按惯例独占 GPU。

## Agent skills

### Issue 跟踪

Issue 和 spec 在 GitHub Issues（`wangxiancao/VideoLingo-user`）；用 `gh issue` 读写，所有改动走 PR，`main` 受保护。见 `docs/agents/issue-tracker.md`。

### 分诊标签

五个 GitHub 标签：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。见 `docs/agents/triage-labels.md`。

### 领域文档

单上下文：仓库根一份 `GLOSSARY.md` 加 `docs/adr/`。见 `docs/agents/domain.md`。

### 验证

阶段级验证按 `docs/agents/verification.md`；工具层复用判定见 `docs/agents/tool-audit.md`；夹具约定见 `tests/fixtures/README.md`。
