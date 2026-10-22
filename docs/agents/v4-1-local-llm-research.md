# V4-1 本地小模型选型研究（翻译优先，2026-10-22）

## 1. 范围与结论

本报告只回答一件事：在 24 GB RTX 3090、与 GLM-ASR-Nano-2512 + Qwen3-ForcedAligner-0.6B 共卡、无云 API、OpenAI 兼容本地 HTTP 的约束下，2025 H2–2026 发布的哪个 ≤4B 模型最适合做 zh↔en 字幕翻译，以及如何验证。

结论先行：

1. **首选 HY-MT2-1.8B（2026-05-21，1.8B dense，Apache-2.0）**。它是唯一在 ≤4B 档同时给出 zh↔en WMT25 与 FLORES-200 分数的翻译专用模型，官方定位贴近视频字幕场景；有官方 GGUF、FP8、GPTQ。官方称其为 “fast-thinking”，但 chat template **没有 thinking 开关**，训练目标是只输出译文；spike 仍必须验证响应中无 reasoning/think 内容。风险是 JSON 包装与两步 refine 未经官方验证。
2. **次选 Qwen3.5-4B（2026-02，4B 级 hybrid，Apache-2.0）**。201 种语言、262K 上下文、指令跟随最强，接口风险最低（JSON 稳定性待 spike 验证）；但它是通用模型，WMT24++ 只有 55 语平均 66.6，zh↔en 单对未公布，且带视觉编码器（本任务用不到）。
3. **翻译质量对照选 TranslateGemma-4B（2026-01，4B dense，Gemma License）**。en→zh_CN 的 MetricX 2.66 明显优于同尺寸 Gemma 3 4B 的 3.27（论文自报，未独立复核），是 4B 档最强的英译中公开证据；但只有 2K 上下文、chat template 只接受 User/Assistant 与 `source_lang_code`/`target_lang_code` 字段，不适合直接套现有 system+JSON 调用，zh→en 也没有官方分数。
4. **7B 上限对照只保留 HY-MT2-7B（2026-05-21，Apache-2.0）**。WMT25 XCOMET 63.86，明显高于 1.8B 的 50.30；Q4_K_M 4.31 GiB，可作为“1.8B 到底差多少”的上界，不作为默认。

现有 4–8B 通用模型 brief 已过时；本报告不再把 Gemma 4 E4B、Ministral 3 3B、MiniCPM5-2B 这类“通用小模型”当作首选，只因它们缺少 zh↔en 翻译证据。

## 2. 方法与证据边界

- 只读：检查了 `/root/VideoLingo-refactor/pipeline/llm/{openai_client,factory,cache}.py`、`pipeline/stages/{segment,translate}.py`，确认当前 seam 是 **system+user 的 OpenAI chat.completions**，输出必须是 `{"translations":[{"id":1,"target":"..."}]}` 或 `{"sentences":[{"start":0,"end":3}]}` 的严格 JSON，temperature=0，没有 thinking 参数 [S34]。
- 显存：读取 `bench/results/t2b/zh_real_600s_pi-b20.json`，确认 ASR b20 峰值 **10,300 MB allocated / 13,224 MB reserved**；`docs/agents/rtfx-optimization.md` 记录同口径。
- 本机实况：2026-10-22 复核的 `nvidia-smi` 显示 24,576 MiB 中已用 9,993 MiB、空闲 14,140 MiB；占用者是 `infinity_emb`（7,414 MiB）、`mineru`（912 MiB）与另一个 `QM_cal` Python 进程（428 MiB），不是 ASR。**这意味着“24 GB 可全部给 ASR+LLM”在当前机器上不成立**，见第 6 节。
- 2026-10-22 复核：Hy-MT2-1.8B / Qwen3.5-4B / TranslateGemma-4B 模型卡已核对；Hy-MT2 官方称 “fast-thinking” 但 chat template 无 thinking 开关；Qwen3.5 模型卡日期为 2026-02；TranslateGemma 为 2026-01（论文 01-13/19）；llama.cpp 默认端口 9931 已按官方 README 核对（不是常见的 8080）。
- 翻译分数均为厂商自报（腾讯/Google/阿里），未采用独立第三方复现；测试集/metric 不可直接比。
- 未下载模型、未安装包、未改 `config.yaml`、未跑 GPU 基准；所有延迟/显存总量均为公式估算或引用，标注“估计”。

## 3. 发布时间线（新→旧）

| 日期 | 模型 | 规模/结构 | 许可 | 类型 | 本报告处置 |
|---|---|---|---|---|---|
| 2026-09-10 | Cohere North Small Translate [S22] | 218B total / 25B active MoE | CC BY-NC 4.0 | 翻译专用 | 拒绝：总参数与许可均不合 |
| 2026-09-07 | MiniCPM5-2B [S17] | 2.5B dense | Apache-2.0 | 通用，zh/en | 观察：无翻译分数 |
| 2026-05-21 | **HY-MT2-1.8B / 7B** [S9][S12] | 1.8B / 7B dense | Apache-2.0 | 翻译专用，33 语 | **入选** |
| 2026-03-31 | Gemma 4 E4B [S14][S2] | 8B total / 4.5B effective dense | Apache-2.0 | 通用多模态 | 拒绝：WMT25 明显弱于 HY-MT2-1.8B |
| 2026-02（模型卡；跟踪站 02-16） | **Qwen3.5-4B** [S5][S6] | 4B 级 hybrid，含视觉 | Apache-2.0 | 通用，201 语 | **入选** |
| 2026-01 | **TranslateGemma-4B** [S1][S3] | 4B dense | Gemma License | 翻译专用，55 语 | **入选（有条件）** |
| 2025-12-30 | HY-MT1.5-1.8B / 7B [S51][S54] | 1.8B / 7B dense | Tencent HY Community License（EU/UK/韩国除外） | 翻译专用 | 被 HY-MT2 取代 |
| 2025-12-02 | Ministral 3 3B [S15][S16] | 4B total（3.4B LM + 0.4B vision） | Apache-2.0 | 通用多模态，40+ 语 | 拒绝：无翻译证据 |
| 2025-09-01 | Hunyuan-MT-7B [S44][S53] | 7B dense | Tencent Hunyuan Community License | 翻译专用，33 语 | 拒绝：许可地域限制、被 HY-MT2 取代 |
| 2025-08-06 | Qwen3-4B-Instruct-2507 [S18][S19] | 4B dense | Apache-2.0 | 通用，非 thinking | 备选：稳定但代际较旧 |
| 2025-07-18 | Seed-X-7B [S20][S52] | 7B dense | OpenMDW（自定义） | 翻译专用，28 语 | 拒绝：被 HY-MT2-7B 取代 |
| 2025-07-08 | SmolLM3-3B [S40] | 3B dense | Apache-2.0 | 通用，6 主语言 | 拒绝：中文只是少量训练 |
| 2025-06-20 | TowerPlus-9B [S21] | 9B dense | CC BY-NC-SA 4.0 | 翻译专用 | 拒绝：非商用、>8B |
| 2025-06-06 | MiniCPM4-8B / 0.5B [S17] | 8B / 0.5B dense | Apache-2.0 | 通用 | 拒绝：被 MiniCPM5 取代 |
| 2025-04-29 | Qwen3-4B / 1.7B / 0.6B [S18] | dense | Apache-2.0 | 通用 | 拒绝：被 Qwen3.5 取代 |
| 2025-03-12 | Gemma 3 4B / 1B [S38] | 4B 多模态 / 1B 文本 | Gemma License | 通用，140+ 语 | 拒绝：被 TranslateGemma/Gemma 4 取代 |
| 2025-02-26 | Phi-4-mini [S39] | 3.8B dense | MIT | 通用，24 语 | 拒绝：无翻译证据、较旧 |
| 2025-01-15 | InternLM3-8B [S43] | 8B dense | Apache-2.0 | 通用，zh 强 | 拒绝：8B、较旧、无翻译证据 |
| 2024-09-25 | Llama 3.2 1B / 3B [S41] | dense | Llama 3.2 | 通用，8 语，无 zh | 拒绝：语种与代际均不合 |
| 2024 | TowerInstruct 2B / 7B [S21] | dense | CC BY-NC-SA 4.0 | 翻译专用 | 拒绝：非商用、较旧 |
| 2022-07 | NLLB-200 [S45] | 0.6B–54B | CC BY-NC 4.0 | 传统 NMT | 拒绝：非 instruct、无 JSON、非商用 |

## 4. 翻译优先候选

### 4.1 排序与证据

可用性：四个候选均有 HF 官方权重；HY-MT2 与 Qwen3.5 同步 ModelScope；TranslateGemma 为 gated manual，未见 ModelScope 官方镜像。

**第 1：HY-MT2-1.8B。** 官方论文在 WMT25 上给出 zh→en 的 XCOMET-XXL / CometKiwi / GEMBA = **50.30 / 64.59 / 70.36**，en→zh = **58.29 / 39.22 / 62.36**；FLORES-200 的 ZH⇔XX 为 **84.26 / 76.43 / 82.71** [S9][S10]。它支持 33 种语言（另含 5 种方言/少数民族变体，共 36），官方定位为真实场景翻译，场景匹配度高 [S8][S9]。官方称其为 fast-thinking，但 chat template 没有 thinking 开关；spike 必须验证输出无 reasoning/think 内容。1.8B dense、Apache-2.0、官方推荐 4096 max_tokens（config 写 262144，但未宣称长上下文）、官方 Q4_K_M 1.13 GB（约 1.05 GiB），是共卡压力最小的方案。注意：WMT25 XCOMET 略低于 HY-MT1.5-1.8B 的 53.08，但 GEMBA 从 63.58 升到 70.36，说明新版更偏“可读/自然”，不是全面指标胜出。JSON 与两步 refine 没有官方 benchmark，必须实测。

**第 2：Qwen3.5-4B。** 官方模型卡给出 WMT24++ 55 语平均 **66.6（XCOMET-XXL）**，Qwen3.5-9B 为 72.6；没有 zh↔en 单对分数 [S6]。它的优势不在“翻译专用”，而在 201 语覆盖、262K 上下文、指令跟随和 JSON 稳定性；接口风险最低，但 JSON 稳定性仍需 spike 验证。它是多模态（视觉编码器），本任务不用视觉，且 BF16 safetensors 9.32 GB（8.68 GiB），共卡成本高于 1.8B。官方卡称 “Gated DeltaNet + sparse MoE”，但 `config.json` 未见 expert 路由字段 [S7]；按 4B 级、约 4.7B（含视觉）处理，spike 时以实际权重为准。

**第 3：TranslateGemma-4B。** 官方技术报告给出 WMT24++ 平均 MetricX **5.32**、Comet22 **80.1**（Gemma 3 4B 为 6.97 / 77.2）；en→zh_CN 的 MetricX 为 **2.66**，Gemma 3 4B 为 **3.27**，是 4B 档最强的英译中公开数字（论文自报，未独立复核）[S3][S4]。但报告只评了 en→x，zh→en 没有官方分数；模型卡写 **2K tokens total input context**，chat template 只接受 User/Assistant，且每条 content 必须带 `source_lang_code`/`target_lang_code`。这意味着它无法直接吃现有 system prompt 与 JSON 包装，必须加 adapter；两步 refine 也不是它的训练目标，第二遍很可能只是重译。它是“质量对照”，不是默认。

**第 4（上限）：HY-MT2-7B。** 同一论文，WMT25 **63.86 / 71.21 / 82.24**，FLORES ZH⇔XX **89.45 / 78.97 / 88.89**，显著高于 1.8B [S10][S12]；Q4_K_M 4.31 GiB [S13]。它不是默认，因为 7B 会挤占共卡余量、延迟约 2 倍；只在 1.8B 的质量不达标时用来量化“差距值不值得”。

### 4.2 官方 claims 与独立证据

- HY-MT2、TranslateGemma、Qwen3.5 的分数全部由厂商自己跑、自己选 baseline；腾讯论文里对 Gemma 4 E4B 的 WMT25 评分（38.36 / 55.45 / 59.95）也是腾讯跑的，不是 Google 或第三方。
- 本报告未采用任何独立 WMT 官方提交结果作为证据；所引分数均为厂商论文/模型卡，不能视为独立结论。跨模型比较只能做“同论文内”比较：HY-MT2-1.8B 50.30 vs HY-MT2-7B 63.86 是同口径；HY-MT2 的 WMT25 与 Qwen3.5 的 WMT24++ 不可直接比。

### 4.3 JSON 与两步流程适配

| 模型 | JSON 包装 | 两步 faithful→refine | 判断 |
|---|---|---|---|
| HY-MT2-1.8B | 支持 system，官方强调复杂指令跟随（IFMTBench），但训练目标是“只输出译文”；JSON 需实测 | 可用“术语/上下文”提示模拟 refine，但非官方训练目标 | 最值得试，需 adapter 级 prompt 实验 |
| Qwen3.5-4B | 通用 instruct，JSON 与 id 覆盖最稳 | 天然支持 | 接口风险最低 |
| TranslateGemma-4B | 训练目标就是纯译文；无 system 角色；JSON 很可能失败 | 无 draft-conditioned refine | 需为它单独写 plain-text 单段 adapter，否则淘汰 |
| HY-MT2-7B | 同 1.8B | 同 1.8B | 质量上限对照 |

分段是次要能力：若 HY-MT2 的 JSON 分段不稳，保留规则/spaCy 分段即可，不因分段淘汰翻译模型；若必须用一个模型同时做 segment + translate，则 Qwen3.5-4B 的接口风险更低。

## 5. 运行时矩阵

| 维度 | llama.cpp server [S23] | Ollama [S24][S25] | vLLM [S26][S27][S28] | transformers serve [S30] |
|---|---|---|---|---|
| OpenAI 兼容 | `/v1/chat/completions`，`/v1/completions` | `/v1/chat/completions`、`/v1/responses` | `/v1/chat/completions` 等 | `/v1/chat/completions`、`/v1/responses` |
| 格式 | GGUF | GGUF | safetensors / AWQ / GPTQ / FP8 | safetensors（BF16） |
| GPU offload | `-ngl 99` / `--gpu-layers` | 自动 | 全卡/张量并行 | `--dtype bfloat16`，整模型上卡 |
| 监听 | 默认 `127.0.0.1:9931`，可 `--host/--port` | 默认 `127.0.0.1:11434` | `--host 127.0.0.1 --port 8000` | 默认 `localhost:8000` |
| 启停 | 单进程，SIGINT/SIGTERM；端口即生命周期 | 常驻 daemon；`keep_alive:0` 卸载，难做“项目独占” | 单进程，SIGTERM；启动约 20–60 s（估） | 单进程，SIGTERM；启动约 5–20 s（估） |
| 并发 | `-np N` 并行 slot；KV 随 slot 增 | `OLLAMA_NUM_PARALLEL` | continuous batching，吞吐最好 | `--continuous-batching`，中等负载 |
| thinking 关闭 | `--reasoning-budget 0`；请求 `reasoning_effort:"none"`；或 `--chat-template-kwargs '{"enable_thinking":false}'` | `think:false`（原生 API）；OpenAI 兼容用 `reasoning_effort:"none"` | `--default-chat-template-kwargs '{"enable_thinking": false}'`；请求 `extra_body={"chat_template_kwargs":{"enable_thinking":False}}` | `--reasoning off`（等价于 `enable_thinking=False`）；或 `--chat-template-kwargs` |
| 显存风格 | 按需分配，最可控 | 按需，daemon 有额外开销 | **默认 `gpu_memory_utilization=0.9` 预分配 KV**，共卡危险 | 按需，BF16 权重，无官方量化 |
| 许可 | MIT [S47] | MIT [S48] | Apache-2.0 [S49] | Apache-2.0 [S50] |

### 5.1 各候选的运行时可行性

- **HY-MT2-1.8B / 7B**：transformers 5.14.1 已原生支持 `HunYuanDenseV1ForCausalLM`（config/tokenizer/model class 已在 venv 中验证，无需 remote code）；但 `transformers serve` CLI 在该 venv 不可用，需用 FastAPI/uvicorn 包一层。llama.cpp 官方 GGUF 依赖自定义 STQ kernel，需 llama.cpp PR #22836 的构建 [S11]，不适合“仅 venv”路径；vLLM 模型卡写需 source build。**首选 venv 内 transformers + FastAPI 做质量基线；llama.cpp Q4 仅作非 venv 的低显存对照。**
- **Qwen3.5-4B**：vLLM 官方 recipe 支持 `--language-model-only` 跳过视觉编码器、`--default-chat-template-kwargs` 关 thinking [S27]；但早期 vLLM 0.16.0 的 `enable_thinking:false` 曾失效，2026-03-06 在 nightly `0.16.1rc1.dev85` 修复 [S29]，spike 必须用修复后的版本。venv 内 FastAPI/transformers 也可用。llama.cpp 有社区 GGUF（Q4_K_M 2.55 GiB [S31]），但 Qwen3.5 带视觉 mmproj，本次未发现 Ollama 官方模型，文本-only 加载需实测。**首选 vLLM 或 venv 内 transformers。**
- **TranslateGemma-4B**：transformers 原生支持 Gemma 3 架构；vLLM/llama.cpp 也能跑，但 chat template 的自定义 `source_lang_code` 字段无法通过标准 OpenAI message 表达，必须 `--chat-template-file` 或应用层拼 prompt。GGUF 只有社区量化（Q8 3.85 GiB）。**只建议在“允许 per-model adapter”的前提下试。**
- **HY-MT2-7B**：同 1.8B，但 BF16 13 GiB，共卡只能 Q4；llama.cpp 最合适。

### 5.2 推荐运行方式

当前 seam 是标准 OpenAI SDK，**对 A/B/C，本地 OpenAI 兼容服务都能零改客户端**（只需模型名对）。vLLM 吞吐最好，但默认预分配 KV，和 ASR 共卡时最容易 OOM。venv 内 FastAPI/uvicorn 单进程比 Ollama 常驻 daemon 更干净；llama.cpp 作为非 venv 的低显存对照。

### 5.3 venv 部署复核（HY-MT2-1.8B，2026-10-22）

- 现有 `.venv`：transformers 5.14.1 已包含 `hunyuan_v1_dense`，`AutoConfig`/`AutoTokenizer` 可加载，`AutoModelForCausalLM.from_config` 映射到 `HunYuanDenseV1ForCausalLM`；模型仓库没有 remote code，`trust_remote_code` 不是必需。
- 权重 `model.safetensors` 4,077,072,784 B（约 4.08 GB / 3.8 GiB，BF16）；下载一次后可离线。
- `AutoTokenizer` 的 `chat_template` 为 False；必须手动读取仓库里的 `chat_template.jinja` 并赋给 tokenizer，否则 `apply_chat_template` 不可用。
- `transformers serve` CLI 在该 venv 不可用（console script 指向已删除的 index-tts venv，且无 `transformers.commands`）；用 fastapi 0.136 + uvicorn 0.52 写最小 OpenAI 兼容服务即可，依赖已就绪。
- `bitsandbytes` 未安装；先跑 BF16。若 8 GB 共卡必须压缩，再评估 8-bit/4-bit；官方 GGUF Q4 需要自定义 llama.cpp PR #22836，不是纯 venv 路径。
- 环境坑：`httpx` 读取 `socks://127.0.0.1:7897` 代理会报错；本地服务/客户端需设 `NO_PROXY=127.0.0.1,localhost`，下载权重时也要处理代理。

## 6. 显存与共存

### 6.1 公式与权重

- 权重：`参数量 × 每参数字节`；GGUF 量化用官方文件大小（Q4_K_M≈4.5 bit/权重，Q5≈5.5，Q8≈8.5）。
- KV：`2 × n_layers × n_kv_heads × head_dim × ctx × bytes`；Qwen3.5 是 hybrid，只有 8 层 full attention，KV 按 8 层算，另加线性状态约 0.1 GiB。
- 运行时开销：CUDA context + compute buffer + 碎片，估计 0.5–1.0 GiB；**未实测**。
- ASR 基线：b20 **12.9 GiB reserved / 10.1 GiB allocated**；b1 **6.1 GiB reserved / 5.9 GiB allocated** [S32][S33]。

| 模型 | BF16 | Q8 | Q5（估） | Q4_K_M | KV/token | KV 4K | KV 8K | Q4+8K+overhead（估） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HY-MT2-1.8B | 3.80 | 1.78 | ~1.2 | 1.05 | 64 KiB | 0.25 | 0.50 | **2.2 GiB** |
| TranslateGemma-4B | 7.45 | 3.85 | ~2.8 | ~2.4 | 136 KiB | 0.54 | 1.09* | **3.3 GiB（2K KV 0.27）** |
| Qwen3.5-4B | 8.68 | 4.17 | 2.93 | 2.55 | 32 KiB | 0.13 | 0.25 | **3.5 GiB** |
| HY-MT2-7B | 13.0 | 7.43 | ~5.0 | 4.31 | 128 KiB | 0.50 | 1.00 | **6.0 GiB** |
| Qwen3-4B-2507（备选） | 7.45 | ~4.0 | ~2.7 | ~2.4 | 144 KiB | 0.56 | 1.13 | **4.1 GiB** |

\* TranslateGemma 官方卡片写 2K 上下文，8K 不可用；表中 1.09 仅为公式值，实际不应超过 2K。

### 6.2 能否共卡

- **按名义 24 GiB、ASR b20 12.9 GiB**：1.8B Q4 总约 15.1 GiB，可行；4B Q4 约 16.2–16.4 GiB，可行；7B Q4 约 18.9 GiB，可行但只剩约 5 GiB，遇到 CUDA 碎片/其他进程就危险；4B BF16 + ASR 约 20.4–21.6 GiB（TranslateGemma / Qwen3.5），太紧。
- **按本机实况 14.1 GiB free**（复核快照 9,993/14,140 MiB；infinity_emb 7.4 + mineru 0.9 + QM_cal 0.4）：ASR b20（12.9）+ 1.8B Q4（2.2）= 15.1 GiB，**已经放不下**；4B/7B 更不可能。若改用 ASR b1（6.1），1.8B 8.3、4B 9.4–9.6、7B 12.1，均可放进 14.1 GiB。
- **HY-MT2-1.8B BF16 约 4.8–5.2 GiB**（权重 3.8 GiB + 4K KV 0.25 GiB + 运行时/激活约 0.8–1.2 GiB，估计）。当前 14.1 GiB free 下：ASR b1 6.1 + LLM 5.0 = 11.1 GiB，可共卡；ASR b20 13.2 + LLM 5.0 = 18.2 GiB，放不下，停掉外部租户后可放下。8 GiB 卡上：b1 + BF16 共卡 11.1 GiB 不可行；串行峰值 max(6.1, 5.0) 可行，但 b20 的 13.2 GiB 不可行。
- 因此，**“24 GB 共卡”在当前机器上是一个需要先清理外部租户的假设**。`infinity_emb` 7.4 GiB + `mineru` 0.9 GiB 不停止，任何 b20 + LLM 的组合都不成立。
- vLLM 默认 `gpu_memory_utilization=0.9`，会在 24 GiB 卡上预留约 21.6 GiB；即使权重只有 2 GiB，也会把 ASR 挤掉。共卡场景应避免 vLLM，或显式 `--gpu-memory-utilization 0.4` / `--kv-cache-memory 2G`，但仍不如 llama.cpp 按需分配可控 [S28]。

### 6.3 确定性串行卸载

流水线本身是 `transcribe → segment → translate` 单向顺序，最稳的方案不是“同时驻留”，而是**阶段间串行**：

1. ASR 阶段结束（或独立 ASR worker 退出）后，`del model`、`gc.collect()`、`torch.cuda.empty_cache()`、`torch.cuda.synchronize()`；独立进程直接 SIGTERM。
2. 用 `nvidia-smi --query-gpu=memory.free` 确认空闲 ≥ LLM 预算 + 1 GiB；不满足就失败退出，不要“试试看”。
3. 启动 LLM 服务（llama.cpp/transformers serve），轮询 `/v1/models` 直到 200；跑 segment+translate；SIGTERM 停止并 wait。
4. 若 review/retranscribe 需要 ASR，再反向启动 ASR；用一个 lock 文件/端口检查防止两个服务重叠。
5. 不要只依赖 `empty_cache()`：PyTorch reserved 内存不一定还给 OS；跨进程退出最可靠。

如果必须共卡，选 **HY-MT2-1.8B Q4 + ASR b1 + 4K ctx + `-np 1`**，并把 `-c` 限制在 4096；这是当前 14.1 GiB free 下余量最大、最稳的组合。

### 6.4 video-miner 的 8GB 是什么

- README 的“建议 GPU 显存 ≥ 8GB”是 Docker 部署下限，不是“本地 LLM + ASR”的实测预算。video-miner 的默认翻译是云端 DeepSeek；其 GLM-ASR 引擎把 `GLMASR_VRAM_UTIL` 设为 0.9，FunASR-GGUF 路径约 1.5 GB。它没有把 4 GB 级本地 LLM 算进 8 GB。
- 它真正可借鉴的显存优化是阶段串行：ASR daemon 在 ForceAligner 前 `stop()`，ForceAligner 单独起进程；这使峰值是 max(ASR, aligner) 而不是 sum，并能把 CUDA reserved 内存随进程退出归还。
- 因此“8 GB 可部署”对我们的含义是：8 GB 单卡 + ASR b1 + 严格串行 + 无其它 GPU 租户；不是 ASR b20 与 LLM 同时驻留。

## 7. 延迟与吞吐预期

官方只给了碎片信息：HY-MT2 论文称 1.25-bit 版本 440 MB、在 Apple A15 上比 4-bit HY-MT1.5 快 1.5 倍 [S10]；没有任何 3090 上的 tok/s 或字幕 batch 延迟。以下均为**粗略估计，不是实测**：

| 配置 | 冷加载（估） | 解码（估） | 8 段字幕一批 p50（估） |
|---|---:|---:|---:|
| HY-MT2-1.8B Q4 / llama.cpp | 1–3 s | 150–250 tok/s | 1.5–3 s |
| HY-MT2-1.8B BF16 / transformers | 5–10 s | 80–150 tok/s | 2–4 s |
| Qwen3.5-4B Q4 / llama.cpp | 3–6 s | 80–120 tok/s | 3–5 s |
| Qwen3.5-4B BF16 / vLLM | 20–60 s | 40–80 tok/s | 4–7 s |
| TranslateGemma-4B Q8 | 3–6 s | 50–90 tok/s | 3–6 s（2K ctx） |
| HY-MT2-7B Q4 / llama.cpp | 5–10 s | 40–70 tok/s | 6–10 s |

冷 E2E 的测量口径必须是“进程 spawn → `/v1/models` 200 → 一次 warmup → 固定 200 段样本跑完”，包含服务启动、模型加载、首请求；不要只测 `model.generate()`。每配置至少 3 次冷启动取中位数，显存用 100 ms 采样 `nvidia-smi`，同时记录服务端 timings。

## 8. 许可与来源

- **HY-MT2-1.8B/7B：Apache-2.0**，仓库内有明确 LICENSE.txt [S9][S12]，可商用、可改、需保留声明。
- **Qwen3.5-4B：Apache-2.0** [S6]，可商用；模型卡带视觉，不影响文本使用。
- **TranslateGemma-4B：Gemma License** [S3][S35]，允许商用但含使用限制、Google 可终止、需遵守 Gemma Terms；HF 为 **gated manual**，下载要登录/接受条款，离线部署前需一次性获取。不是 OSI 许可，法务需确认。
- **HY-MT1.5**：Tencent HY Community License（EU/UK/韩国除外），非 Apache [S54]；HY-MT2 才改为 Apache-2.0。
- **Hunyuan-MT-7B**：Tencent Hunyuan Community License，含地域/商用条件；不如 HY-MT2 的 Apache 干净。
- **TowerPlus-9B：CC BY-NC-SA 4.0**，非商用；**North Small Translate：CC BY-NC 4.0**，非商用；两者直接排除。
- **Seed-X-7B：OpenMDW**（自定义），需逐条核对，不是 Apache。
- **Gemma 4 E4B：Apache-2.0** [S14]，但翻译质量不够，不是许可问题。
- **Ministral 3 3B：Apache-2.0** [S16]；**MiniCPM5-2B：Apache-2.0** [S17]；**Qwen3-4B：Apache-2.0** [S19]；**Phi-4-mini：MIT** [S39]；**SmolLM3：Apache-2.0** [S40]；**GLM-4-9B-0414：MIT** [S42]；**InternLM3-8B：Apache-2.0** [S43]；**Llama 3.2：Llama 3.2 License** [S41]。
- 运行时许可：llama.cpp MIT、Ollama MIT、vLLM Apache-2.0、transformers Apache-2.0 [S47–S50]。
- 归因：Apache 模型需保留 NOTICE/LICENSE；Gemma 需在分发/派生时附带 Gemma Terms 与使用限制；Cohere/Tower 的非商用许可即使只内部使用也应让法务确认“商业性”边界。

## 9. 推荐 spike 矩阵

统一验收口径（所有配置都测）：

- **JSON**：用 `translate.py` 的严格 parser，200 次调用（每次 8 段），JSON 解析成功率 ≥99%，id 无缺失/重复/越界；失败要能定位是模型还是 prompt。
- **thinking off**：200 次响应中 `reasoning_content` 为空、正文无 ` thinking`/` response`；服务端日志无 reasoning token。HY-MT2 也必须验证无 reasoning/think 内容（官方称 fast-thinking，但无开关）；Qwen3.5 必须同时验证 `enable_thinking:false` 与 `reasoning_effort:"none"` 两条路径。
- **翻译质量**：固定 100 段 zh→en + 100 段 en→zh（WMT25 对应语对 + 项目内 100 段字幕，含人名/数字/术语），盲 A/B 对比 HY-MT1.5-1.8B 或现有云基线；≥90% 段“无意义错误”，人名/数字/术语 100% 保留，字幕长度 ≤1.5× 源文。
- **两步 refine**：faithful-only 与 faithful+refine 各跑一遍；只有 refine 带来 ≥5% 盲评提升才保留，否则单步省一半延迟。
- **VRAM**：ASR b20 驻留时 LLM 增量 ≤6 GiB；串行时总峰值 <22 GiB；记录 `nvidia-smi` 与进程 RSS。C/D 是串行质量对照，不参与 b20 共卡验收；A/B 才比共卡。
- **启动/吞吐**：冷启动 ≤10 s（1.8B）/≤20 s（4B）/≤30 s（7B）；8 段批 p95 ≤8 s；1000 次调用无 OOM/崩溃。

| ID | 模型/量化 | 运行时与关键参数 | 要回答的问题 | 淘汰条件 |
|---|---|---|---|---|
| **A** | HY-MT2-1.8B BF16 | venv 内最小 FastAPI/uvicorn wrapper：`AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16, device_map='cuda')` + 手动加载 `chat_template.jinja`；监听 127.0.0.1:8000；不依赖 `transformers serve` | 翻译质量、JSON、两步 refine 的基线 | JSON<99%；盲评低于 HY-MT1.5；p95>8 s |
| **B** | HY-MT2-1.8B Q4_K_M | `llama-server -m Hy-MT2-1.8B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 -np 1 --jinja`（需 PR #22836 构建） | 低显存共卡、启动、吞吐 | 自定义 kernel 构建失败；增量>2.5 GiB；p95>8 s |
| **C** | Qwen3.5-4B BF16 | 串行（先停 ASR）：`vllm serve Qwen/Qwen3.5-4B --host 127.0.0.1 --port 8001 --language-model-only --max-model-len 8192 --default-chat-template-kwargs '{"enable_thinking": false}' --gpu-memory-utilization 0.5` | JSON/分段/两步最稳路径；thinking 关闭证明 | thinking 泄漏；JSON<99%；翻译低于 A 10%；vLLM 显存不可控/无法启动 |
| **D** | TranslateGemma-4B BF16/Q8 | `transformers serve google/translategemma-4b-it --port 8002`（需先接受 Gemma 条款），应用层按 `source_lang_code`/`target_lang_code` 拼 prompt；batch≤4、2K ctx | en→zh 质量上限；adapter 成本 | 无法稳定产出 JSON；2K 截断；en→zh 不优于 A；zh→en 明显更差 |
| **E** | HY-MT2-7B Q4_K_M | `llama-server -m Hy-MT2-7B-Q4_K_M.gguf --port 8083 -ngl 99 -c 8192 -np 1`（同 B 的 PR #22836 构建），ASR 停止后串行 | 1.8B 与 7B 的质量差/延迟差 | 相对 A 盲评提升<10%；增量>7 GiB；p95>15 s |

A/B/C 覆盖至少两个 ≤4B 选项；D 是翻译专用 4B 的对照；E 是唯一允许的 7B 上限。若 A 通过，B 只作为低显存部署候选；若 A 的 JSON 失败而 C 通过，则默认 C，A 仅作翻译质量对照。

## 10. 拒绝清单

**规模/许可硬伤**

- **North Small Translate**：218B total / 25B active MoE，CC BY-NC 4.0；总参数与许可均不合 [S22]。
- **TowerPlus-9B**：9B，CC BY-NC-SA 4.0；非商用，且未发现 TowerPlus 2B 开源权重 [S21]。
- **Hunyuan-MT-7B**：7B，Tencent Hunyuan Community License（地域/商用条件），且被 HY-MT2-7B 取代 [S44]。
- **Qwen-MT**：只有云 API，无开放权重，直接违反“无云 LLM API” [S46]。
- **NLLB-200**：2022 年传统 NMT，CC BY-NC 4.0，不能跟随 JSON/两步指令 [S45]。

**翻译证据不足或明显较弱**

- **Gemma 4 E4B**：8B total / 4.5B effective，通用多模态；腾讯同论文 WMT25 仅 38.36 / 55.45 / 59.95，低于 HY-MT2-1.8B 的 50.30 / 64.59 / 70.36；视觉用不到 [S14][S10]。
- **Ministral 3 3B**：4B total（3.4B LM + 0.4B vision），Apache、256K，但官方没有任何翻译 benchmark；多模态视觉用不到 [S16]。
- **MiniCPM5-2B**：2.5B dense、Apache、128K、zh/en，但官方评测集中在 code/math/agent，无翻译分数 [S17]。
- **Phi-4-mini**：3.8B、MIT、128K、24 语，但无翻译 benchmark，代际较旧 [S39]。
- **SmolLM3-3B**：3B、Apache、128K，主语言是英法西德意葡，中文只是少量训练，不适合 zh↔en [S40]。
- **Llama 3.2 1B/3B**：2024 年，8 种语言不含中文，代际与语种都不合 [S41]。
- **GLM-4-9B-0414**：9B、MIT、128K，超过 8B 上限 [S42]。
- **InternLM3-8B**：8B、Apache，中文强但无翻译专用证据，代际较旧 [S43]。

**被更新版本取代**

- Gemma 3 4B/1B → TranslateGemma-4B / Gemma 4；Qwen3-4B-Instruct-2507 → Qwen3.5-4B；Qwen3-1.7B/0.6B → 太小且旧；Seed-X-7B → HY-MT2-7B；HY-MT1.5 → HY-MT2。
- Qwen3.5-9B、Qwen3.8-27B、Gemma 4 31B、Hunyuan-MT-30B-A3B 均超上限，不作对照。

## 11. 来源表（访问日期均为 2026-10-22）

| ID | 来源 | URL |
|---|---|---|
| S1 | Gemma 发布说明（含 TranslateGemma 2026-01-15、Gemma 4 2026-03-31） | https://ai.google.dev/gemma/docs/releases |
| S2 | Gemma 4 总览 | https://ai.google.dev/gemma/docs/core |
| S3 | TranslateGemma-4B 模型卡 | https://huggingface.co/google/translategemma-4b-it |
| S4 | TranslateGemma 技术报告 | https://arxiv.org/abs/2601.09012 |
| S5 | Qwen3.5-4B ModelScope 镜像（发布日期 2026-02） | https://www.modelscope.cn/models/Qwen/Qwen3.5-4B |
| S6 | Qwen3.5-4B 模型卡 | https://huggingface.co/Qwen/Qwen3.5-4B |
| S7 | Qwen3.5-4B config（层数/头数/上下文） | https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json |
| S8 | Hy-MT2 官方仓库 | https://github.com/Tencent-Hunyuan/Hy-MT2 |
| S9 | Hy-MT2-1.8B 模型卡与 Apache LICENSE | https://huggingface.co/tencent/Hy-MT2-1.8B |
| S10 | Hy-MT2 技术报告（WMT25/FLORES 表） | https://arxiv.org/abs/2605.22064 |
| S11 | Hy-MT2-1.8B 官方 GGUF（Q4/Q6/Q8、STQ kernel 说明） | https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF |
| S12 | Hy-MT2-7B 模型卡 | https://huggingface.co/tencent/Hy-MT2-7B |
| S13 | Hy-MT2-7B 官方 GGUF | https://huggingface.co/tencent/Hy-MT2-7B-GGUF |
| S14 | Gemma 4 E4B 模型卡（8B total/4.5B effective、Apache） | https://huggingface.co/google/gemma-4-E4B |
| S15 | Mistral 3 / Ministral 3 发布页 | https://mistral.ai/news/mistral-3/ |
| S16 | Ministral 3 3B Instruct 模型卡（Apache、4B total） | https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512 |
| S17 | MiniCPM 仓库（MiniCPM5-2B 2026-09-07、MiniCPM5-1B 2026-05-19） | https://github.com/OpenBMB/MiniCPM |
| S18 | Qwen3 仓库（2025-04-29 首发、2025-08-06 4B-Instruct-2507） | https://github.com/QwenLM/Qwen3 |
| S19 | Qwen3-4B-Instruct-2507 模型卡 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| S20 | Seed-X-7B 仓库（OpenMDW、2025-07-18） | https://github.com/ByteDance-Seed/Seed-X-7B |
| S21 | Tower-Plus-9B 模型卡（CC BY-NC-SA 4.0） | https://huggingface.co/Unbabel/Tower-Plus-9B |
| S22 | North Small Translate 模型卡（218B/25B、CC BY-NC 4.0） | https://huggingface.co/CohereLabs/North-Small-Translate-1.0 |
| S23 | llama.cpp server README（`--reasoning-budget`、`reasoning_effort:"none"`、`--chat-template-kwargs`、`-np`） | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| S24 | Ollama OpenAI 兼容文档（`reasoning_effort:"none"`） | https://docs.ollama.com/api/openai-compatibility |
| S25 | Ollama Thinking 文档（`think:false`） | https://ollama.com/blog/thinking |
| S26 | vLLM Reasoning Outputs（`enable_thinking=False`） | https://docs.vllm.ai/en/latest/features/reasoning_outputs/ |
| S27 | vLLM Qwen3.5 recipe（`--language-model-only`、`--default-chat-template-kwargs`） | https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html |
| S28 | vLLM Optimization（`gpu_memory_utilization` 预分配、`--kv-cache-memory`） | https://docs.vllm.ai/en/stable/configuration/optimization/ |
| S29 | vLLM issue #35574（Qwen3.5 关 thinking 修复于 0.16.1rc1.dev85） | https://github.com/vllm-project/vllm/issues/35574 |
| S30 | transformers serve 文档（OpenAI 兼容、`--reasoning off`、`--continuous-batching`） | https://huggingface.co/docs/transformers/main/en/serving |
| S31 | Qwen3.5-4B 社区 GGUF 文件大小（Q4 2.55/Q8 4.17 GiB） | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF |
| S32 | 本仓库 ASR b20 基准 JSON（10,300 MB alloc / 13,224 MB reserved） | `/root/VideoLingo-refactor/bench/results/t2b/zh_real_600s_pi-b20.json` |
| S33 | 本仓库 RTFx 优化记录（b20 表、GPU 另有 9.5 GB 占用） | `/root/VideoLingo-refactor/docs/agents/rtfx-optimization.md` |
| S34 | 当前 LLM seam 与 JSON 契约 | `/root/VideoLingo-refactor/pipeline/llm/openai_client.py`、`pipeline/stages/translate.py`、`pipeline/stages/segment.py` |
| S35 | Gemma License 条款 | https://ai.google.dev/gemma/terms |
| S38 | Gemma 3 4B 模型卡 | https://huggingface.co/google/gemma-3-4b-it |
| S39 | Phi-4-mini 模型卡 | https://huggingface.co/microsoft/Phi-4-mini-instruct |
| S40 | SmolLM3-3B 模型卡 | https://huggingface.co/HuggingFaceTB/SmolLM3-3B |
| S41 | Llama 3.2 3B 模型卡（Llama 3.2 License） | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct |
| S42 | GLM-4-9B-0414 模型卡（MIT） | https://huggingface.co/THUDM/GLM-4-9B-0414 |
| S43 | InternLM3-8B 模型卡（Apache） | https://huggingface.co/internlm/internlm3-8b-instruct |
| S44 | Hunyuan-MT-7B 模型卡与 License.txt | https://huggingface.co/tencent/Hunyuan-MT-7B |
| S45 | NLLB-200 模型卡（CC BY-NC 4.0） | https://huggingface.co/facebook/nllb-200-distilled-600M |
| S46 | Qwen-MT（云 API，无开放权重） | https://qwenlm.github.io/blog/qwen-mt/ |
| S47 | Llama.cpp LICENSE（MIT） | https://github.com/ggml-org/llama.cpp/blob/master/LICENSE |
| S48 | Ollama LICENSE（MIT） | https://github.com/ollama/ollama/blob/main/LICENSE |
| S49 | vLLM LICENSE（Apache-2.0） | https://github.com/vllm-project/vllm/blob/main/LICENSE |
| S50 | transformers LICENSE（Apache-2.0） | https://github.com/huggingface/transformers/blob/main/LICENSE |
| S51 | Hy-MT1.5 技术报告 | https://arxiv.org/abs/2512.24092 |
| S52 | Seed-X 技术报告 | https://arxiv.org/abs/2507.13618 |
| S53 | Hunyuan-MT 技术报告 | https://arxiv.org/abs/2509.05209 |
| S54 | HY-MT1.5-1.8B 模型卡与 Tencent HY License | https://huggingface.co/tencent/HY-MT1.5-1.8B |

## 12. 待人类决策的问题

1. **是否必须商用？** 若是，TowerPlus/North/Hunyuan-MT 直接出局；Gemma License 需法务确认，HY-MT2/Qwen3.5 的 Apache 最省事。
2. **能否在跑批时停掉 `infinity_emb`、`mineru` 与 `QM_cal`？** 当前约 10.0 GiB 被它们占用；不停，ASR b20 + 任何 LLM 都无法共卡，只能串行或降 ASR 到 b1。
3. **是否允许 per-model prompt adapter？** TranslateGemma 需要 `source_lang_code`/`target_lang_code` 和无 system 的模板；若 LLM seam 必须保持通用，D 配置直接放弃。
4. **两步 refine 是否必须保留？** 若 HY-MT2 单步盲评已达标，refine 只是翻倍延迟；建议把“是否保留 refine”作为 spike 的决策项，而不是预设。
5. **7B 上限是否值得？** 只有 1.8B 盲评落后 ≥10% 才跑 E；否则 1.8B Q4 更省显存/启动。
