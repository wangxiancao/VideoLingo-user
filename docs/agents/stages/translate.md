# 阶段验证：translate

## 范围

关联：Epic #1；T6 #10；spec #16；实现 commit `bedad1c`。

本阶段负责：

- `segments.json` → `translation.json`。
- 两步翻译：先忠实直译，再意译/反思；提示词自行编写。
- 批量、并发、独立响应缓存；缓存目录为 `<workdir>/.llm-cache`。
- 源文/译文与非空校验；段 id、start、end 保持不变；失败显式报错。

本阶段不负责：

- 分段（T5）、字幕（T7）、烧录（T8）。
- 真实 LLM 质量验证。本轮使用“题目 + 答案”夹具；真实 API 路径未调用、未验证。
- 摘要、术语表、全流程。

## 前置条件

- Python：`/root/VideoLingo/.venv/bin/python`。
- 考题验证不需要 GPU、网络或 API key；`--llm-fixture` 提供两步固定答案。
- 真实 LLM 路径使用 `api.key/base_url/model`；本轮未验证。
- 配置只读；使用：

  ```yaml
  target_language: '简体中文'
  max_workers: 4
  translation:
    batch_size: 8
    cache: true
  ```

- `translation.cache` 默认 true；缓存位于工作区 `.llm-cache/`，不进入版本库。

## 夹具

- `tests/fixtures/segments/sample_6s.json`：2 段英文，0.5–4.0 s / 4.0–6.0 s。
- `tests/fixtures/llm/translate_exam.json`：第一步直译 + 第二步意译的固定答案。
- `tests/fixtures/llm/translate_exam_missing.json`：缺少 id 2，验证失败路径。

## 命令

```bash
cd /root/VideoLingo-refactor
PYTHON=/root/VideoLingo/.venv/bin/python
WORK=$(mktemp -d)
cp config.example.yaml "$WORK/config.yaml"
cp tests/fixtures/segments/sample_6s.json "$WORK/segments.json"

$PYTHON -m pipeline.runner --stages translate \
  --workdir "$WORK" --config "$WORK/config.yaml" \
  --llm-fixture tests/fixtures/llm/translate_exam.json

find "$WORK" -maxdepth 2 -type f -printf '%P\n' | sort
```

## 期望

- 退出 0；生成 `translation.json` 和 `.llm-cache/*.json`。
- `schema_version=1`；source_language 自动识别为 en；target_language 为配置值。
- 2 段；id 1/2；start/end 与 segments.json 完全一致。
- target 为第二步意译答案：`这是经过意译和反思后的自然中文版本。` / `短句。`
- 第一次运行两次 LLM 调用；第二次运行命中缓存，不再调用。
- `batch_size=2`、9 段时 5 批 × 2 步 = 10 次调用；所有段都有译文。
- 缺 id 时退出 1，stderr 指出 translate 与缺失 id，不写 translation.json。
- 不运行 segment/subtitle/burn 或全流程。

## 失败判定

- translation.json 缺失、为空、段数/id/时间戳与输入不一致。
- 任一 target 为空，或响应缺 id、重复 id、越界 id。
- LLM 失败/无效 JSON 仍以 0 退出。
- 缓存行为导致旧译文被错误复用，或第二次运行仍发起相同调用。
- 运行了前序阶段或全流程。
- 没有可复现命令/证据，或实现 agent 与验证 agent 相同。

## 证据

- 实测：验证时 `68 passed`（64 + T6 4）；当前全套 70。
- 考题输出：2 段，source en / target 简体中文，时间 0.5–4.0、4.0–6.0。
- 缓存：首轮 2 次调用；次轮 0 次新调用。
- 并发：9 段 / batch 2 → 10 次调用，9 段全部完成。
- 缺 id 退出 1，无 translation.json。
- 真实 LLM 未验证，只验证结构、时间戳、两步/缓存/并发行为与失败处理。

## 独立验证

由未参与实现的 agent 执行：

1. 用 segments 夹具 + 考题运行 translate，确认 2 段、译文来自第二步、时间戳不变。
2. 检查 `.llm-cache` 存在；同一工作区再跑一次，确认不发新请求（可用计数假客户端或缓存文件时间）。
3. 用缺 id 考题运行，确认退出 1、无 translation.json。
4. 检查 batch/并发配置路径；真实 API 路径标记“未验证”。
5. 确认没有真实 LLM 调用、没有跑前序阶段或全流程。
6. 跑全部测试，确认当前全套 70 passed。

## 验证记录

- Ticket: #10
- Stage: translate
- Implementer: 实现会话
- Verifier: 独立验证 agent（未参与实现）
- Commit: bedad1c
- Scope: segments.json → translation.json（两步翻译，考题夹具）
- Preconditions: 无 GPU/网络/API key；fixture client
- Config path: 临时复制的 `config.example.yaml`
- Fixtures: segments/sample_6s.json；translate_exam.json、translate_exam_missing.json
- Command: 见“命令”小节；验证写入 `/tmp`
- Expected: 2 段、target en→简体中文、时间戳不变、第二步译文；缓存命中；缺 id 退出 1
- Result: PASS
  - 2 段；id 1/2；时间 0.5–4.0 / 4.0–6.0；target 为第二步意译答案
  - `.llm-cache/` 2 个响应；第二次运行输出不变且无新调用（poison 响应测试也确认未调用）
  - 缺 id：退出 1，`missing translation id(s): [2]`，无 translation.json
  - fixture client，无网络/OpenAI 导入/密钥访问；只跑 translate
  - 验证时 68 tests passed（含缓存与 9 段 batch=2 并发）；当前全套 70；仓库未修改
- Exit code: 0（正常）；1（缺 id）
- Evidence: 上述译文/时间/缓存；验证输出在 `/tmp`
- Verdict: PASS
- 备注：真实 LLM 质量与真实 API 路径未验证；仅验证结构、时间戳、两步、缓存/并发与失败行为。