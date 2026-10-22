# V4-1 实测 spike：HY-MT2-1.8B 作为本地翻译 LLM（2026-10-23）

> 状态：**已完成，待维护者批准**。依据 #32 V4-1 与选型研究
> `docs/agents/v4-1-local-llm-research.md`（研究阶段已做、实测未做——本报告补上实测）。
> 测量环境：RTX 3090 24 GB、GPU 空闲 22.5 GiB（零计算进程，无需停任何服务，
> 维护者决策 #2）、`.venv`（transformers 5.14.1、fastapi 0.136、uvicorn 0.52），
> 权重 `Tencent-Hunyuan/HY-MT2-1.8B`（ModelScope，BF16 safetensors 3.9 GB，
> Apache-2.0）。原始数据：`bench/results/v4-1/spike.json`；脚本：
> `bench/v4_1_server.py`、`bench/v4_1_spike.py`。

## 0. 推荐组合（供批准）

| 项 | 选择 | 依据 |
|---|---|---|
| 模型 | **HY-MT2-1.8B**（BF16） | §3 质量；显存 4.2 GiB 共卡无压力 |
| Runtime | **venv transformers + FastAPI 最小 OpenAI 兼容服务** | 零新依赖；冷启 2.1 s；研究报告 §5.2 首选路径 |
| 解码 | **greedy + `repetition_penalty=1.0`** | §2.2：模型自带 1.05 在 greedy JSON 下确定性劣化 |
| 提示词 | **v2 示例锚定版**（faithful/refine 两步都用） | §2.1：orig 在 zh→en 下把译文塞进自造键 |
| 两步 refine | **建议砍掉**（保留为配置开关） | §4：24 例仅 1 例改动，纯付延迟 |
| 分段（T5） | **LLM 分段不可用，需规则分段回退**（V4-3 决策） | §5：两版提示词都不能完整覆盖词流 |

## 1. 性能与资源（实测）

| 指标 | 值 | 说明 |
|---|---:|---|
| 模型加载（冷） | **2.09 s** | 服务自报；含 tokenizer + chat_template.jinja（需手动挂载，tokenizer 自带 chat_template 为 False） |
| GPU 足迹 | **+4,340 MiB**（进程独跑，nvidia-smi 差值） | torch max alloc 4.03 GiB；权重 3.8 GiB BF16 + KV/运行时 |
| 解码吞吐 | **32.5 tok/s**（中位，6 次 12 段批） | greedy BF16 单并发（服务加锁串行） |
| 12 段批延迟 | **9.78 s**（decode 中位） | 60 min 视频 ≈ 380 条分割行 ≈ 32 次批 ≈ **5.3 min LLM 墙钟**（refine 砍掉后） |
| 共卡推算 | LLM 4.2 + native ASR 16 GB 预算 = **20.2 GiB < 22.5 GiB free** | 未做同时驻留实测，留 V4-2/V4-6 验证；无需停任何外部服务 |

## 2. JSON 契约稳定性（流水线同款提示词 + 流水线同款校验）

### 2.1 四格矩阵（12 段批，temperature=0）

| 提示词 | zh→en | en→zh |
|---|---|---|
| orig（现行 `SYSTEM_FAITHFUL`） | ❌ 译文塞进自造 `"extra"`/`"time"` 键，`target` 留空——流水线 `_parse_translations` 必拒 | ✅ 12/12 非空、无多余键 |
| v2（示例锚定） | ✅ 12/12 非空、无多余键 | ✅ 12/12 非空、无多余键 |

v2 保持 JSON 输入输出契约与语义不变，只在措辞中加入填好的示例（"full translation inside target, no additional keys"）。en→zh 对两种措辞都稳；**zh→en 必须用 v2**。

### 2.2 解码劣化取证（重要）

模型 `generation_config.json` 自带 `repetition_penalty=1.05`（为采样聊天调的）。在 greedy JSON 输出下，它会在**同一位置（char 363，id 10 处）确定性劣化**为 `{"id":10,"target","id":10,"target":"…"}`——两个方向、两版提示词都复现。**`repetition_penalty=1.0` 后四格全绿**。生产服务必须显式设 1.0。

另一次观测（合成 20 字窗口样例）：尾部半截句会导致模型缩译（"…"）并**丢尾部 id**。生产侧建议 V4-4 校验 id 完整性并单次重试（现行 `_parse_translations` 已拒绝缺 id，只需加重试）。

## 3. 翻译质量（盲评样本已存 `spike.json`，供维护者终审）

- **en→zh（12 条代表性字幕句）：12/12 准确自然**，字幕风格合适（如 "This is the last train to Stockholm tonight." → "这是今晚前往斯德哥尔摩的最后一班火车。"）。
- **zh→en（12 条真实产线分割行**，取自同一条 60 min 视频的旧管线 `output/log/split_by_nlp.txt`——即翻译阶段的真实 workload）：**12/12 译文通顺成句**，术语基本正确（溶剂重排/皮秒/轨迹）。
- **如实记录的瑕疵**：
  1. ASR 已把"皮秒"转成"匹秒/匹苗"时，模型会规范化成错误单位（"milliseconds"）——脏源传导，非模型幻觉；需要上游词典或人工复核兜底。
  2. 两行把频道赞助语（"请不吝点赞订阅……"）与正文混排的部分段译。
  3. 断句截半的残句上曾出现抢译/漂移——真实 segment 产物没有这么碎，但 V4-3 分段质量直接影响此项。

## 4. 两步 refine（决策项实测）

temperature=0 下：zh→en **0/12** 改动，en→zh **1/12** 改动（"谈谈"→"讨论"，横向平移）。refine 把 LLM 墙钟翻倍（≈5.3 → ≈10.6 min/60min 视频）换取 ≈4% 的行改动。**建议砍掉 refine**，保留为 `translation.refine: false` 配置开关；若维护者要求保守可留开。

## 5. 分段（T5）实测：LLM 分段不可用

60 词中文流（字符级词）：

- orig 提示词：8 句但**跳段**（0-3, **7**-11, **15**-19…），3-7、11-15 等区间丢失。
- v2 提示词：边界连续（0-3,3-7,7-11…）但**机械 4 词等分**、末尾不对齐总词数，无语义分组。

结论：HY-MT2-1.8B 不适合做 T5 分段。V4-3 的选择：(a) **规则分段**（停顿间隙 + 长度上限，研究报告 §4.3 既定回退），或 (b) Qwen3.5-4B 兼做分段+翻译（接口更稳但共卡 8.7 GiB）。建议 (a)，保持"翻译专用小模型"的显存优势。

## 6. thinking 验证（#32 D2 要求）

30+ 次调用输出扫描 `<think`/`</think`/`reasoning_content`/`As an AI`/`让我想一想` 等标记：**零命中**。chat template 本身无 thinking 分支（研究报告 §4.1），训练目标即直出译文——运行时与输出双口径均验证通过。

## 7. 待维护者批准（下一人工门控）

1. **选型终版**：HY-MT2-1.8B + venv transformers/FastAPI + greedy/rep1.0 + v2 提示词。
2. **砍 refine**（建议）或保留。
3. **T5 分段走规则回退**（建议）——这改 T5 实现但保持边界产物 JSON 契约不变，属 #32 D4 边界内的实现选择，需维护者点头。
4. 质量瑕疵兜底策略：上游术语词典（皮秒/匹秒类）是否纳入 V4-4 范围。

批准后进入 V4-2（服务生命周期：启动/健康检查/关闭/孤儿防护）。
