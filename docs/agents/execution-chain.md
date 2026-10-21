# VideoLingo 核心执行链

一条**文件驱动**的单向流水线：阶段之间不传参数，只靠 `output/` 下的产物文件对接。结构索引见 [code-map.md](./code-map.md)。

## 数据流

```
输入(本地路径 | URL)
 └─ _1  下载/复制 → output/<video>                     find_video_files() 要求 output/ 里恰好 1 个源视频
     └─ _2  ffmpeg 抽音 → audio/raw.mp3
            [demucs?] → audio/vocal.mp3 + background.mp3
            split_audio 分段 → Whisper 逐段转录 → 合并 segments
            → log/cleaned_chunks.xlsx (词级: text/start/end) + log/sentences.xlsx
        └─ _3_1 spaCy 断句 → log/split_by_nlp.txt
            └─ _3_2 LLM 按义切长句 → log/split_by_meaning.txt   ⟳ 最多 3 轮重试，失败句子保持未切
                └─ _4_1 LLM 摘要+术语 → log/terminology.json    ⛔ pause_before_translate 在此
                    └─ _4_2 LLM 信/达两步翻译 → log/translation_results.xlsx
                        └─ _5   LLM 拆分对齐字幕 → …_for_subtitles.xlsx + …_remerged.xlsx  ⟳ 最多 3 轮
                            └─ _6 时间戳对齐 → 4 个展示 SRT + 2 个音频 SRT
                                └─ _7 ffmpeg 烧录 → output_sub.mp4        ★ srt-review 门控在此
                                    └─ [_8_1 → _8_2 → _9 → _10 → _11 → _12] → output_dub.mp4
```

`_5` 之后分岔出**两条字幕用途**，这是最容易改错的地方：`_for_subtitles.xlsx`（拆碎的，给 `_6` 写展示 SRT 烧进画面）与 `_remerged.xlsx`（整句的，给 `_6` 写音频 SRT 供配音）。`_6_gen_sub.py:356` 用前者、`:360` 用后者，注释明确写着"use remerged file to avoid unmatched lines when dubbing"。

## 三套编号对照（务必核对）

| skill 阶段 (`run_stage.py:17-29`) | `translate_video.py` step | 实际模块 |
|---|---|---|
| 1 输入 | Processing input file | `_1_ytdlp` |
| 2 转录 | Transcribing with Whisper | `_2_asr` |
| 3 断句 | Splitting sentences | `_3_1` **+** `_3_2` |
| 4 摘要+翻译 | Summarizing and translating | `_4_1` **+** `_4_2` |
| 5 字幕拆分+对齐 | Processing and aligning subtitles | `_5_split_sub` **+** `_6_gen_sub` |
| 6 烧录 | Merging subtitles to video | `_7_sub_into_vid` |
| 7 音频任务 | Generating audio tasks | `_8_1_audio_task` |
| 8 配音分块 | Generating dubbing chunks | `_8_2_dub_chunks` |
| 9 参考音 | Extracting reference audio | `_9_refer_audio` |
| 10 TTS | Generating TTS audio | `_10_gen_audio` |
| 11 合成整轨 | Merging full audio | `_11_merge_audio` |
| 12 混流 | Merging dubbing to video | `_12_dub_to_vid` |

skill 阶段号 ≥6 之后**整体错位一位**（阶段 6 是模块 `_7`，阶段 7 是模块 `_8_1`）。SKILL.md 的表格同样按 skill 编号书写，提到"stage 6 用 stage 5 的拆分产物"时指的是**模块 6 用模块 5 的产物**。

## 跨阶段契约（改动即破坏下游）

**产物路径**：除 `_8_1` 用哈希 sidecar 外，下游一律按字面路径读取上游产物，且 `models.py` 里只登记了一半（未登记的 SRT/视频名在各模块顶部重复字面量）。

**表格列名**：`log/cleaned_chunks.xlsx` 词级行（`text`/`start`/`end`）是时间戳对齐的唯一真源；`translation_results*.xlsx` 固定 `Source` / `Translation` 两列（`_4_2_translate.py:87`、`_5_split_sub.py:258-259`）；`_8_1` 之后 `tts_tasks.xlsx` 引入 `number`、`start_time`、`end_time`（`%H:%M:%S.%f` 截到毫秒，`_8_1:128-129`）。行数对齐靠**索引配对**，不是文本匹配——`_5_split_sub.py:149-157` 专门先 `_pad_equal_parts` 保证两侧等长，防止 zip 静默截断。

**SRT 文件名集合**：展示侧 `output/{src,trans,src_trans,trans_src}.srt`（`_6_gen_sub.py:11-16`），配音侧 `output/audio/{src,trans}_subs_for_audio.srt`（`:18-21`），配音产物侧 `output/dub.srt`（`_11:15`、`_12:15`）。

**时间戳格式**：`convert_to_srt_format`（`_6_gen_sub.py:74-85`）产出 `HH:MM:SS,mmm --> HH:MM:SS,mmm`。写入前有两道硬校验（`:229-247`）：非空字幕必须正时长、起始时间**不得回退**。对齐必须保持**顺序 + 尽力匹配**（AGENTS.md 明令），空行/`None` 填充行在 `align_timestamp` 里被跳过而非报错（`:293-299`）。

**LLM 出口**：只有 `core/utils/ask_gpt.py` 一个。7 个 `log_title` 同时是缓存文件名：`split_by_meaning`、`summary`、`translate_faithfulness`、`translate_expressiveness`、`align_subs`、`sub_trim`、`tts_correct_text`（后者 `use_cache=False`）。重试语义叠加三层：`@except_handler(retry=5)` 包裹 `ask_gpt` 本身，`translate_lines.py:30-39` 内部再重试 3 次，`_3_2`/`_5` 外层还有多轮 attempt。

## 幂等与重跑现状

真正的跳过装饰器只有 4 处（`_2:14`、`_3_1:5`、`_3_2:131`、`_4_2:49`），`check_file_exists` **只看存在、不看输入是否变过**。`_8_1` 是唯一严谨的实现：`tts_tasks_input.hash` 记录输入 SRT 的 SHA-256，一致才跳过，且写入走 `tmp.xlsx` + `os.replace` 原子替换（`_8_1:145-172`）——注释直接写明"plain file exists check would silently skip after the subtitles changed"。`_9` 用 `refers/1.wav` 是否存在自证。`_5`/`_6`/`_7`/`_10`/`_11`/`_12` **没有任何哨兵**。

LLM 缓存落在 `output/gpt_log/`，与产物同生命周期：`--fresh` / `prepare_output_folder` 的 `shutil.rmtree("output")` 会把缓存一起清掉，此时"重跑"等于全部重做。

## 门控点

| 门 | 位置 | 机制 | 归属 |
|---|---|---|---|
| `pause_before_translate` | `_4_1` 之后、`_4_2` 之前 | CLI: `translate_video.py:31-32` 提到终端 `input()` 阻塞；`st.py:63-69` 用 session_state 原生暂停。人工去改 `log/terminology.json` | 仓库 |
| **srt-review** | `_6` 写完 SRT、`_7` 烧录之前 | `export_review` → `retranscribe_clip` 取证 → `corrections.json` → `apply_corrections`（唯一 SRT 写入者） | skill |

## 三个入口的差异

| | `translate_video.py` | `st.py` | `skill run_stage.py` |
|---|---|---|---|
| 清空 `output/` | 总是 `rmtree`（`:328`） | 不清 | 仅 `--fresh` |
| 每步重试 | `run_steps` 3 次（`:255-276`） | 无 | **复用同一个 `run_steps`** |
| LLM 后端 | API | API | `--llm api` / `--llm agent`（猴补丁） |
| 归档 | 跑完自动 `cleanup("history")` | 点按钮 | `--archive` |
| TTS 关停 | `finally` 里 `stop_optional_tts_server()` | 不关 | 依赖被复用的 `tv` 逻辑，进程退出即失联 |
| 暂停 | 文档提及 | 原生 | 无 |

## 阻断"直接拆成独立进程"的现状

1. **进程级环境自愈写在 `translate_video.py` 模块顶层**：`LD_LIBRARY_PATH` 探测后 `os.execv` 自我重启（`:46-59`）、`_sanitize_proxy_env()` 在 import 时执行（`:145`）。任何新入口若忘记复用，Whisper 加载与 API 会当场失败——`retranscribe_clip.py:24` 已经在用"`import translate_video` 纯粹为了触发净化"这种丑陋补偿。
2. **驱动方式是 import 而非 subprocess**：`run_stage.py:39-41` 把 `translate_video` 拉进同进程，阶段边界只是函数调用。
3. **agent LLM 后端依赖猴补丁的可见性**：`agent_llm.install()` 必须先 `import core` 触发 `core/__init__.py:3-19` 的 15 个模块急加载，再扫 `sys.modules` 替换各模块里的 `ask_gpt` 名字（因为 `utils/__init__` 的 `from .ask_gpt import ask_gpt` 遮蔽了模块属性）。拆成分布式进程后，这个补丁无法跨进程传播。
4. **重资源常驻**：Whisper（`_2`）、spaCy nlp（`_3_1`/`_3_2`）、TTS HTTP 服务（`_10`）。按阶段切进程就要按次付费加载——skill 已经踩过这个坑，`retranscribe_clip --suspects` 特意批量一次跑完以免重复加载 Whisper。
5. **cwd 依赖**：`models.py` 全相对路径 + `find_video_files()` 靠扫目录找唯一源视频。
6. **并发不安全**：`output/` 单实例共享，`config.yaml` 的锁是 `threading.Lock`（进程内）。阶段一旦各自可独立调用，误并发概率上升。
7. **`exit(1)` 出现在阶段函数内部**（`_7_sub_into_vid.py:57`）：抛 `SystemExit` 属 `BaseException`，`run_steps` 的 `except Exception` 捕获不到，既不重试也绕过 `main` 的错误面板。

## 收尾

`cleanup("history")` 把 `output/` 归档进 `history/`（含 `output/review/` 的全部证据链：suspects、clips、二次转录、corrections、review_report.md）。
