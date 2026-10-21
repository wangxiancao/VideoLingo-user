# 阶段验证：subtitle

## 范围

关联：Epic #1；T7 #11；spec #16；实现 commit `64ac103`。

本阶段负责：

- `translation.json` → `subs/source.srt`、`subs/target.srt`、`subs/bilingual.srt`。
- 按显示长度拆分长句；三份 SRT 共用同一 cue 时间线。
- bilingual cue 固定两行：源文 + 译文。
- 三份 SRT 编号、顺序和时间戳一致；显示长度与时长约束校验。

本阶段不负责：

- 翻译（T6）；段级时间戳由 translation.json 提供。
- 烧录（T8）。
- 全流程运行。
- 真实 LLM 翻译质量。

## 前置条件

- Python：`/root/VideoLingo/.venv/bin/python`。
- 不需要 GPU、模型权重、LLM 或网络。
- FFmpeg 不需要；本阶段只写 SRT。
- 配置只读；使用 `subtitle.*`：

  ```yaml
  subtitle:
    max_length: 75
    target_multiplier: 1.2
    min_duration: 0.5
    max_duration: 10.0
  ```

- `translation.json` 的输入校验还会读取 `segments.json` 作为时间戳/ID 参考；真实流水线中由 T5 产出。

## 夹具

- `tests/fixtures/translation/sample_6s.json`：2 段、3.5 s + 2 s；第一段源文超过 75 字符，强制切 3 条。
- `tests/fixtures/segments/sample_6s.json`：与 translation 的 id/时间戳一致，供输入校验。
- 夹具不提交音频或视频。

## 命令

```bash
cd /root/VideoLingo-refactor
PYTHON=/root/VideoLingo/.venv/bin/python
WORK=$(mktemp -d)
cp config.example.yaml "$WORK/config.yaml"
cp tests/fixtures/translation/sample_6s.json "$WORK/translation.json"
cp tests/fixtures/segments/sample_6s.json "$WORK/segments.json"

$PYTHON -m pipeline.runner --stages subtitle \
  --workdir "$WORK" --config "$WORK/config.yaml"

$PYTHON -m pipeline.runner --stages subtitle \
  --workdir "$WORK" --config "$WORK/config.yaml" --validate-only
```

## 期望

- 退出 0；只生成三份 `subs/*.srt`。
- 三份 SRT cue 数相同；编号从 1 连续；时间戳完全相同、单调、不重叠。
- source 行 ≤ `max_length`；target 行 ≤ `max_length * target_multiplier`。
- bilingual 每条恰好两行：source 行 + target 行。
- 每条 cue 的 `end > start`；长段按 `max_duration` 与显示长度拆分。
- 夹具结果：4 条 cue；第一段 3 条 0.5–4.0 s，第二段 1 条 4.0–6.0 s。
- 失败时显式报错；bilingual 不是两行时校验失败。
- 不运行 transcribe/translate/ingest/burn 或全流程。

## 失败判定

- 产物缺失、为空或 cue 数/时间线在三份文件间不一致。
- 编号跳号、时间戳回退或重叠、空文本。
- source/target 行超出长度限制。
- bilingual 不是恰好两行。
- 运行了前序阶段或全流程。
- 没有可复现命令/证据，或实现 agent 与验证 agent 相同。

## 证据

- 实测：`60 passed`（57 + T7 3）。
- 夹具输出：4 cues；bilingual 两行；三份文件时间戳一致。
- 输出示例（bilingual）：

  ```text
  1
  00:00:00,500 --> 00:00:01,667
  This is a deliberately long English sentence used
  这是一段故意写得很长的中文译文，用
  ```

## 独立验证

由未参与实现的 agent 执行：

1. 用夹具运行 subtitle，确认退出 0、三份 SRT、4 条 cue、时间戳一致。
2. 检查 source ≤75、target ≤90、bilingual 恰好两行、编号连续。
3. 将 bilingual 第一条改成三行，`--validate-only` 应退出 1。
4. 确认只跑 subtitle，未跑前序阶段或全流程。
5. 跑全部测试，确认 60 passed。

## 验证记录

- Ticket: #11
- Stage: subtitle
- Implementer: 实现会话
- Verifier: 独立验证 agent（未参与实现）
- Commit: 64ac103
- Scope: translation.json → 三份 SRT
- Preconditions: 无 GPU/模型/LLM；Python 3.10.19
- Config path: 临时复制的 `config.example.yaml`
- Fixtures: `tests/fixtures/translation/sample_6s.json`、`tests/fixtures/segments/sample_6s.json`
- Command: 见“命令”小节；验证写入 `/tmp`
- Expected: 4 cues；三份文件时间戳一致；source ≤75、target ≤90；bilingual 两行；三行时校验失败
- Result: PASS
  - 4 cues；编号 1–4；时间 0.5–4.0（3 条）+ 4.0–6.0
  - source 长度 49/53/57/11；target 17/18/17/3；bilingual 均两行
  - `--validate-only` 成功；bilingual 加第三行后退出 1
  - 60 tests passed；只跑 subtitle；仓库未修改
- Exit code: 0（正常）；1（三行 bilingual）
- Evidence: 上述 cue/长度/时间；验证输出在 `/tmp`
- Verdict: PASS