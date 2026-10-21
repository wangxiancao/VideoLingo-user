# 工具层审计（T0）

范围：新 `pipeline` 可以复用的现有工具，以及必须新写的部分。结论只针对 v0.3 的 6 阶段重写；配音与 UI 已排除。

判定含义：

- **直接复用**：接口与行为都合适，不改代码即可调用。
- **包装后复用**：核心能力可用，但需要一层适配器隔离旧路径、旧缓存或全局状态。
- **新写**：现有实现与目标耦合太深，参照行为后重写。
- **不复用（仅参考）**：只把它当行为参考，不进入新管线。

## 审计清单

| 工具/模块 | 用途 | 判定 | 说明 |
|-----------|------|------|------|
| `core/utils/config_utils.py` | 读取与更新本地配置 | **包装后复用（只读）** | `load_key` 可读配置，但硬编码 `config.yaml`、不接受路径参数；新 runner 的 `--config` 需要一层薄包装或新 loader。`update_key` 会写 `config.yaml`，新管线禁止使用。锁是进程内 `threading.Lock`，不提供跨进程安全。 |
| `core/utils/ask_gpt.py` | LLM 调用、JSON 修复、缓存、重试 | **新写薄客户端** | `ask_gpt` 只作行为参考：缓存固定写到 `output/gpt_log/<log_title>.json`，与旧 `output` 生命周期绑定；没有 API 预检；占位 `api.key` 也会通过存在性检查。新客户端必须支持显式配置、独立缓存目录/命名空间、可注入假客户端，不复用旧缓存路径。 |
| `core/utils/decorator.py` | 重试/退避、文件存在跳过 | **直接复用** | 通用工具，无阶段耦合。`check_file_exists` 使用静态路径，配 runner 工作区时要显式传入。 |
| `core/utils/onekeycleanup.py` | 把 `output` 归档到 `history` | **包装后复用（收尾）** | 依赖 `find_video_files`，同名 history 会覆盖；不在关键路径，T9 再定归档策略。 |
| `core/_1_ytdlp.py` | 下载视频、查找源视频 | **新写（参考）** | YouTube 客户端回退可参考；`update_ytdlp()` 在运行时执行 `pip install --upgrade`，不能进新管线；`find_video_files` 假设 `output` 里只有一个视频。新 ingest 使用显式路径。 |
| `core/asr_backend/audio_preprocess.py` | 音频预处理、Whisper CJK 巨块修补 | **不复用** | 与 Whisper/pandas 的产物形状绑定；新 ASR 走 GLM-ASR + ForcedAligner，不需要这套修补。 |
| `core/asr_backend/whisper_asr.py` | Whisper 转写 | **不复用** | 已由官方 GLM-ASR 替换。 |
| `core/asr_backend/demucs_vl.py` | 人声/背景分离 | **不复用** | 配音已排除；ASR 直接用原音轨。 |
| `core/spacy_utils/` | NLP 断句 | **不复用** | 新 segment 阶段用 LLM 重切句，保留词级时间戳。 |
| `core/prompts.py` 与各阶段提示词 | 旧翻译/断句提示词 | **不复用** | 新 stage 自行撰写提示词，避免复制旧结构。 |
| `core/_6_gen_sub.py` | SRT 生成、时间戳单调与时长约束 | **参考后新写** | 时间戳单调、最大时长、SRT 顺序等不变量值得借鉴；实现与旧 xlsx/pandas 契约解耦。 |
| `core/_7_sub_into_vid.py` | FFmpeg 烧录、字体、nvenc | **参考后新写** | 字幕 filter 转义、Noto 字体与 nvenc 逻辑可参考；新 `pipeline/media` 自己封装。 |
| `core/utils/models.py` | 旧产物路径常量 | **不复用** | 新管线使用自己的 JSON 契约，不复用旧 xlsx/txt 路径。 |
| `translate_video.py` 的 `_sanitize_proxy_env()` | 清理不可用/失效代理变量 | **包装后复用** | 对 HF 下载与 LLM 调用都有用；只在当前进程内弹变量。提取为 runtime 工具，并保留“不改 shell 环境”的语义。 |
| ModelScope 下载 | 获取 GLM-ASR/ForcedAligner 权重 | **新写薄封装** | Python 包可用；CLI 入口 shebang 指向已删除的 `external/index-tts/.venv`，不可用。统一用 Python API，缓存不进入版本库。 |
| FFmpeg 通用封装 | 解码、烧录、媒体探测 | **新写** | 仓库没有通用 FFmpeg 工具；旧阶段直接拼命令。新 `pipeline/media` 负责转义、字体与错误处理。 |
| `requirements.txt` | Python 依赖清单 | **不复用** | 已过期：transformers 钉 4.39.3、仍列 `qwen-asr`/`demucs`/`streamlit`。新管线单独维护依赖，T2a 再补 `torchcodec`。 |
| `core/st_utils/`、`st.py`、`translations/` | Web UI | **不复用** | 已冻结，重构落地后按原计划删除。 |
| `external/video-miner/` | 参考实现 | **不复用** | 无 LICENSE，不接源码、二进制或模型；只重写其技术路线。 |
| `core/utils/pypi_autochoose.py`、`delete_retry_dubbing.py` | 安装器、TTS 清理 | **不复用** | 与旧安装/配音绑定。 |

## 结论

- **直接复用**：`decorator`。
- **包装后复用**：`config_utils`（只读路径适配）、`onekeycleanup`（收尾）、代理清理工具。
- **新写**：LLM 薄客户端（`ask_gpt` 仅作参考）、ingest 下载与显式路径、ASR 后端、segment/translate 的提示词、subtitle、burn 的 media 层、ModelScope 下载封装、JSON 契约与 runner。
- **不复用**：Whisper、demucs、spacy、旧提示词、旧产物路径常量、UI、video-miner。

## 需要带进实现的约束

1. **不写 `config.yaml`**：新配置键只进模板；运行期只读。旧 `load_key` 硬编码 `config.yaml`，新 runner 的 `--config` 通过薄包装解析，不改变旧阶段行为。
2. **缓存隔离**：旧 `ask_gpt` 的缓存与 `output/` 绑定；新 LLM 客户端必须使用独立缓存目录或命名空间，避免读到上一次运行的结果。
3. **禁止运行时 pip**：ingest 不能沿用 `update_ytdlp()` 的自动升级。
4. **归档覆盖**：`onekeycleanup` 会覆盖同名 history；T9 需要明确是否沿用。
5. **依赖清单分家**：新管线不直接复用旧 `requirements.txt`；新增依赖要单独记录。
6. **旧契约不回流**：新 JSON 契约不为兼容旧 xlsx/txt 而妥协；最终 SRT/MP4 保持兼容即可。
7. **代理处理**：只在当前进程内清理代理变量，不修改用户 shell 或全局环境。