# 阶段验证：segment

## 范围

关联：Epic #1；T5 #9；spec #16；实现 commit `cb4f2d6`。

本阶段负责：

- `transcript.json` → `segments.json`。
- 用 LLM 把有序词流重切成自然句，保留原词、原时间戳和顺序。
- 新增薄 LLM 客户端接缝：支持真实 OpenAI 兼容客户端与固定考题夹具。
- LLM 失败、边界不完整或时间戳漂移时显式失败，不静默留下未切句。

本阶段不负责：

- 转写（T4）、翻译（T6）、字幕（T7）。
- 真实 LLM 质量验证。本轮使用“题目 + 答案”夹具，真实 API 路径标记未验证。
- 全流程运行。

## 前置条件

- Python：`/root/VideoLingo/.venv/bin/python`。
- 考题验证不需要 GPU、网络或 API key；`--llm-fixture` 指向固定响应 JSON。
- 真实 LLM 路径使用 `api.key/base_url/model` 配置；本轮未调用、未验证。
- 端口：不适用。

## 夹具

- `tests/fixtures/transcript/sample_6s.json`：10 个英文词，2 个 T4 chunk segment。
- `tests/fixtures/llm/segment_exam.json`：考题答案，半开区间 `[0,2)`、`[2,6)`、`[6,10)`。
- `tests/fixtures/llm/segment_exam_gap.json`：故意漏掉词 2，验证覆盖检查。

## 命令

```bash
cd /root/VideoLingo-refactor
PYTHON=/root/VideoLingo/.venv/bin/python
WORK=$(mktemp -d)
cp config.example.yaml "$WORK/config.yaml"
cp tests/fixtures/transcript/sample_6s.json "$WORK/transcript.json"

$PYTHON -m pipeline.runner --stages segment \
  --workdir "$WORK" --config "$WORK/config.yaml" \
  --llm-fixture tests/fixtures/llm/segment_exam.json

# 失败路径
$PYTHON -m pipeline.runner --stages segment \
  --workdir "$WORK" --config "$WORK/config.yaml" \
  --llm-fixture tests/fixtures/llm/segment_exam_gap.json
```

## 期望

- 退出 0；只生成 `segments.json`。
- 每条 sentence 的 `[start,end)` 半开区间连续、无重叠、无遗漏，覆盖全部词。
- 输出 3 段：`Hello world.`（0–1 s）、`This is a test.`（1–3 s）、`Another sentence follows here.`（3–5 s）。
- 所有词保留原 start/end/text 和顺序；句子时间戳取首词 start 与末词 end。
- `id` 从 1 连续；schema_version=1；无空句。
- 答案缺口时退出 1，stderr 指出 segment 与覆盖错误，不写 segments.json。
- 不运行 transcribe/translate/subtitle/burn 或全流程。

## 失败判定

- 段落缺失、为空、id 跳号、时间戳漂移或词序错乱。
- 边界漏词、重叠、越界或不是整数。
- LLM 返回无效 JSON/空响应/调用失败仍以 0 退出。
- 运行了前序阶段或全流程。
- 没有可复现命令/证据，或实现 agent 与验证 agent 相同。

## 证据

- 实测：验证时 `64 passed`（62 + T5 2）；当前全套 70。
- 考题输出 3 段，词数 10，时间 0–5 s；覆盖与顺序正确。
- 缺口答案退出 1，无 segments.json。
- 真实 LLM 未验证；只验证结构、时间戳、覆盖和失败行为。

## 独立验证

由未参与实现的 agent 执行：

1. 用 transcript 夹具 + 考题答案运行 segment，确认 3 段、10 词、时间正确。
2. 检查每段词都在原 transcript 中且顺序/时间戳不变。
3. 用 gap 考题运行，确认退出 1、无 segments.json。
4. 确认没有真实 LLM/API 调用，也没有跑前序阶段或全流程。
5. 跑全部测试，确认当前全套 70 passed。

## 验证记录

- Ticket: #9
- Stage: segment
- Implementer: 实现会话
- Verifier: 独立验证 agent（未参与实现）
- Commit: cb4f2d6
- Scope: transcript.json → segments.json（LLM 考题夹具）
- Preconditions: 无 GPU/网络/API key；fixture client
- Config path: 临时复制的 `config.example.yaml`
- Fixtures: transcript/sample_6s.json；llm/segment_exam.json、segment_exam_gap.json
- Command: 见“命令”小节；验证写入 `/tmp`
- Expected: 3 段、10 词、时间 0–1/1–3/3–5；缺口退出 1
- Result: PASS
  - 3 段，ids 1–3；文本与时间正确；词数 2/4/4，10 个词的 text/start/end 与输入完全一致
  - gap 考题退出 1，stderr 含覆盖错误，无 segments.json
  - 使用 fixture client，未读取/打印 API key，无真实 LLM 调用
  - 只跑 segment；验证时 64 tests passed；当前全套 70；仓库未修改
- Exit code: 0（正常）；1（缺口）
- Evidence: 上述段落/词/时间；验证输出在 `/tmp`
- Verdict: PASS
- 备注：真实 LLM 质量与真实 API 路径未验证，只验证结构、时间戳、覆盖和失败行为。