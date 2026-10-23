# VideoLingo Agent 指南

## 项目概览

VideoLingo 是一个的视频转录、翻译、字幕与配音应用。Streamlit UI（`st.py`）和单视频 CLI（`translate_video.py`）共用同一条基于文件的流水线。

`output/` 下的文件在一次运行期间是跨阶段契约。

## 仓库结构

结构细节由两份文档承载，本节只是索引。

- **`docs/agents/execution-chain.md`** — 动态流程：文件驱动的数据流、并存的三套阶段编号、跨阶段契约（路径、表格列、SRT 集合、时间戳格式、`log_title` 缓存键）、当前的幂等实况、门控点、三个入口之间的差异，以及当前阻碍把阶段拆成独立进程的因素。

顶层：**默认入口是 `videolingo.py`**（薄封装，跑新六阶段管线 `pipeline/`：ingest→transcribe→segment→translate→subtitle→burn，产物落 `runs/<name>/`，`runs/` 被忽略）；旧管线的 `st.py`（Streamlit）与 `translate_video.py`（CLI）保留为**显式回退入口**，走 `core/_1_*.py`–`core/_12_*.py` 的 15 个有序阶段模块与 `output/`；`core/{utils,asr_backend,tts_backend,spacy_utils,st_utils}/` 存放旧管线共享代码；`translations/` 存放 7 份 UI locale JSON；`config.example.yaml` 是被跟踪的 schema（新管线默认自包含本地栈：native ASR 引擎 + 托管 HY-MT2 LLM），而 `config.yaml` 可变、被忽略、可能含密钥；`docs/` 是独立的 pnpm/Next.js 站点。

## 开发流程

1. 改动产物路径、表格列、时间戳、排序规则或函数签名之前，先追清它的生产者与消费者；从 `docs/agents/execution-chain.md` 入手。
2. 新管线（`pipeline/`）是默认路径（T9a 已切换）；旧管线保持阶段顺序和两个回退入口不动，直到 T9 后的独立下线里程碑。两套管线并存期间不要混用工作目录（新 `runs/`、旧 `output/`）。
3. 不要在同一份 checkout 里并发跑流水线。它们共享 `output/` 和 `config.yaml`，而 YAML 锁只在进程内有效。
4. 新配置键加到 `config.example.yaml`。旧阶段通过 `load_key`/`update_key` 访问；新 pipeline 只通过薄包装读取 `--config`，运行期不写 `config.yaml`。无关工作中不要覆盖本地 `config.yaml`。
5. 改动保持聚焦。汇报受影响的阶段、验证证据，以及任何未测试的模型、GPU、API 或依赖媒体文件的路径。
6. 所有改动走 feature 分支 + PR；`main` 受保护，不直推。

## 编码准则

- 旧阶段的 LLM 调用走 `core/utils/ask_gpt.py`；新 pipeline 使用新写的薄客户端（`ask_gpt` 只作行为参考）。旧阶段的配置访问留在 `core/utils/config_utils.py`；新 pipeline 通过薄包装支持只读的显式 `--config` 路径。
- 把中间路径、表格 schema、SRT 顺序和时间戳格式当作 API 对待。保留有序的 Whisper 词级时间戳；对齐必须保持顺序进行，并在匹配不完美的情况下尽力而为（best-effort）。
- 字幕产物要区分开：阶段 6 用阶段 5 的 split 产物做展示，用它的 remerged 产物做配音。
- 新增 UI 文案时更新全部 7 份 locale JSON，并保持它们的键集合对齐。
- UI 逻辑不进流水线阶段，供应商特定代码不进编排模块。

## 测试说明

`tests/` 已重建聚焦测试套件（runner/stages/llm 生命周期等，`.venv/bin/python -m pytest tests/ -q`；GPU/模型相关的会按资产缺失自动跳过）。阶段级验证按 `docs/agents/verification.md`；旧管线的 CLI/Streamlit 全流程命令只适用于回退路径或对照。

- Python：`.venv/bin/python -m compileall -q core st.py translate_video.py videolingo.py pipeline`
- 默认入口（新管线）：`.venv/bin/python videolingo.py <video-or-url>`（产物落 `runs/`；阶段可 `--stages` 选择）
- 回退入口（旧管线）：`.venv/bin/python translate_video.py <disposable-sample>`；`--mode tts` 仍不可用（`custom_tts` 空实现）
- Streamlit（旧管线）：`streamlit run st.py`，并实际操作受影响的那个 section
- 文档：`cd docs && pnpm build`
- 翻译：核对各 locale JSON 的键集合完全一致

新管线按产物存在性跳过已完成阶段、翻译缓存默认开启——重跑不等于干净重算。旧 CLI 在处理前会递归替换 `output/`。缺少所需服务与资产时，绝不声称端到端成功。

## 环境搭建

- 用 Python 3.10。FFmpeg 必需；字幕渲染需要 Noto 字体。
- 项目的 Python 环境是仓库根的 `.venv`（真实目录，约 16 GB；不是符号链接）。用 `.venv/bin/python` 跑流水线和各阶段（例如 `.venv/bin/python translate_video.py ...`）；它是唯一装全依赖集的解释器（torch、transformers/whisper、spacy、cv2、yt-dlp、streamlit……）。PATH 上的系统/conda 解释器缺这些依赖，会以 `ModuleNotFoundError` 失败——绝不要信任 shell PATH 里的裸 `python3`/`pip`。
- 该 venv 没有 pip；用 `uv pip install --python .venv/bin/python <pkg>` 往里装包。
- 把 `config.example.yaml` 复制成 `config.yaml`；绝不提交或暴露其本地值。
- 自动化场景优先手动搭建。`install.py` 是交互式的，会装包、可能调用系统包管理器、可能持久修改 pip 的全局 index URL，并启动 Streamlit。
- 使用 `docs/` 里已提交的 pnpm lockfile。

## 工具使用规则

- 不要直接修改 `output/`、`history/`、`queue_logs/`、`_model_cache/` 里的生成内容；通过产出它的那个阶段重新生成。
- 不要拿当前的 Docker 构建当本地改动的证据：Dockerfile 克隆远端仓库，而不是拷贝工作区。
- 只有任务明确要求时才新增依赖。

## 已知问题 / 约束

- Whisper 是唯一已实现的 ASR 路径，但 `requirements.txt`、translations 和 `docs/` 的部分内容仍提到 Qwen3-ASR；以当前代码和 `config.example.yaml` 为准。`qwen-asr` 仍是过期依赖。
- `setup.py` 发现不到 `VideoLingo` 包，且排除了真正的 `core` 包；可编辑执行目前依赖仓库根目录。
- 配音能力已归零。IndexTTS2 与 CosyVoice3 后端、`scripts/` 下的 worker、旧 `tests/` 测试套件和 `external/index-tts` submodule 都已删除，只剩 `core/tts_backend/custom_tts.py` 这个空实现（函数体是 `pass`，不落盘）。因此 `--mode tts` 会在阶段 10 重试 3 次后抛 `Failed to generate audio`，除非先实现 `custom_tts`。`tts_method` 只接受 `'custom_tts'`，其它值在 `tts_main` 入口直接抛 `ValueError`。
- `pause_before_translate` 用终端 `input()`，会阻塞 Streamlit。`burn_subtitles: false` 时，视频产物是黑色占位画面而不是原视频。
- 完整验证可能需要网络访问、下载、凭据、FFmpeg、可观的 GPU 显存，以及一个真实媒体样本。

## Agent skills

### Issue 跟踪

Issue 和 spec 在 GitHub Issues（`wangxiancao/VideoLingo-user`）；用 `gh issue` 读写，所有改动走 PR，`main` 受保护。见 `docs/agents/issue-tracker.md`。

### 分诊标签

五个 GitHub 标签：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。见 `docs/agents/triage-labels.md`。

### 领域文档

单上下文：仓库根一份 `GLOSSARY.md` 加 `docs/adr/`。见 `docs/agents/domain.md`。

### 验证

阶段级验证按 `docs/agents/verification.md`；工具层复用判定见 `docs/agents/tool-audit.md`；夹具约定见 `tests/fixtures/README.md`。
