# 阶段级验证（T0 模板）

关联：Epic #1；T0 #2；spec #16；阶段票 #3–#13。

目标：验证 agent 能在**不跑全流程**的情况下接手任一阶段，按固定夹具与命令判断它是否真的工作；实现 agent 与验证 agent 必须是不同的 agent。

阶段 id：ingest、transcribe、segment、translate、subtitle、burn。性能票 T2a #4、T2b #5、T2c #6；收尾 T9 #13。新阶段 id 与旧 `_N_` 编号的对照由 T1 的阶段表维护，T0 不另立编号。

## 接缝

- **主接缝**：阶段级 runner。验证只通过“阶段 id + 工作区”进入，观察产物与退出码。
- **次接缝**：ASR 后端协议。真实 GLM-ASR/ForcedAligner 验证契约；假后端只验证阶段编排、偏移合并与失败处理。
- **契约接缝**：阶段产物校验器。检查 JSON 字段、时间戳单调性、SRT 顺序、MP4 时长。

不要测试阶段内部函数、私有结构或具体实现步骤。

## 角色与边界

- **实现 agent**：写代码，写本阶段的验证文档（范围、夹具、命令、期望、失败判定），在 ticket 上说明改动。
- **验证 agent**：只读验证文档与代码，独立执行命令，记录实际结果；不得修改实现来“让验证通过”。
- **汇报**：验证结论写在 ticket 评论中，包含 commit、命令、产物清单、PASS/FAIL。

## 环境

- 使用仓库记录的 Python 解释器，绝不使用 PATH 上的裸 `python`：

  ```bash
  # 本机示例；新 worktree 暂无 .venv，使用主 checkout 的 venv
  PYTHON=${PYTHON:-/root/VideoLingo/.venv/bin/python}
  ```

  如果当前 worktree 有自己的 `.venv`，用 `.venv/bin/python`；如果新建了独立 venv，把绝对路径记录到 ticket 和本文件。命令里的 `$PYTHON` 必须能被验证 agent 复现。
- 需要模型权重、FFmpeg、GPU/网络时，在验证文档的“前置条件”里写明；缺条件就记为“未验证”，不能猜测通过。
- 使用独立工作区；不要清空或覆盖他人的 `output/`。
- 不读取、不打印任何密钥；配置文件只读。

## 命令契约

T1 必须实现下面这个 runner 接口；如果 T1 选择不同接口，先更新本文件，再改验证文档。

```bash
$PYTHON -m pipeline.runner \
  --stages <stage-id>[,<stage-id>...] \
  --workdir <workdir> \
  --config <config-path> \
  [--source <path-or-url>] \
  [--llm-fixture <path>] \
  [--validate-only]
```

- `--stages`：逗号分隔的阶段 id，按流水线顺序执行；只跑列出的阶段，不自动补跑前置阶段。
- `--workdir`：阶段输入/输出的根目录；runner 不清空它。
- `--config`：显式配置路径；runner 只读，不写回 `config.yaml`。
- `--source`：ingest 的输入路径或 URL；其他阶段不需要。ingest 把它规范化为 `media/source.mp4` 与 `media/audio.wav`。
- `--llm-fixture`：LLM 阶段注入固定响应夹具；验证只校验结构、时间戳、失败行为与缓存命中，不断言具体措辞。
- `--validate-only`：只重跑目标阶段的产物校验，不执行阶段。
- 退出码：`0` 成功；`1` 阶段失败或产物校验失败；`2` 缺少输入或配置。
- 从仓库根运行，保证 `-m pipeline` 可导入；workdir 可以在仓库外。
- 这份接口是 T0 给 T1 的最小契约；T1 若要改，先更新本文件与 ticket #3。
- 不允许通过 CLI 入口跑全流程；全流程只用于 T9 的最终验收。

## Workdir 与夹具映射

阶段产物固定放在工作区根下：

```
<workdir>/
├── media/
│   ├── source.mp4       # ingest
│   └── audio.wav        # ingest: 16k mono
├── transcript.json      # transcribe
├── segments.json        # segment
├── translation.json     # translate
├── subs/
│   ├── source.srt       # subtitle
│   ├── target.srt
│   └── bilingual.srt
└── output.mp4           # burn
```

| 阶段 | 输入 | 输出 |
|------|------|------|
| ingest | 本地文件或 URL（`--source`） | `media/source.mp4`、`media/audio.wav` |
| transcribe | `media/audio.wav` | `transcript.json` |
| segment | `transcript.json` | `segments.json` |
| translate | `segments.json` | `translation.json` |
| subtitle | `translation.json` | `subs/source.srt`、`subs/target.srt`、`subs/bilingual.srt` |
| burn | `media/source.mp4` + `subs/source.srt` + `subs/target.srt` | `output.mp4` |

> 本表是 T0 的验证约定；T1 的阶段表是最终契约。T1 若要改文件名，先更新本文件与 ticket #3。
> 最终 SRT 兼容：新管线内部使用 source/target/bilingual 三份；切换时按 `docs/agents/execution-chain.md` 的旧集合做映射（src/trans/src_trans；`trans_src` 与音频 SRT 因配音移除不在范围）。

验证时先在 scratch 目录里铺好输入，例如：

```bash
cd /root/VideoLingo-refactor
WORK=$(mktemp -d)
mkdir -p "$WORK/media" "$WORK/subs"
cp config.example.yaml "$WORK/config.yaml"
cp tests/fixtures/audio/zh_30s.wav "$WORK/media/audio.wav"
$PYTHON -m pipeline.runner --stages transcribe --workdir "$WORK" --config "$WORK/config.yaml"

# ingest 示例：用夹具视频作为 --source
$PYTHON -m pipeline.runner --stages ingest --workdir "$WORK" --config "$WORK/config.yaml" \
  --source tests/fixtures/video/sample_5s.mp4
```

其他阶段的夹具复制方式相同：把 `tests/fixtures/<type>/<name>` 复制到表中对应的输入文件名。T3–T8 负责添加实际夹具；T0 只定义目录、文件名与命令契约（见 `tests/fixtures/README.md`）。

## 配置

- 非 LLM 阶段：把 `config.example.yaml` 复制到 scratch 路径，作为 `--config` 传入；不写入仓库，不使用真实密钥。
- LLM 阶段：使用验证者提供的临时配置（含有效 key，不提交）或在边界注入固定假客户端；只校验结构、时间戳、失败行为与缓存命中，不断言具体措辞。
- 任何情况下都不打印 key，不修改 `config.yaml`，不把配置提交进仓库。
- `config.example.yaml` 是 schema 来源；新配置键只进模板。

## 通过标准

- 目标阶段退出码为 0。
- 目标产物存在，且通过契约校验（字段、顺序、时间戳、时长）。
- 关键不变量成立：时间戳单调、在媒体范围内、SRT 编号/顺序正确、无空段、无静默回退。
- 失败路径也验证：缺输入、缺模型、坏夹具时应显式报错，而不是产出半成品。
- 性能任务额外记录：RTFx、显存峰值、批量大小、冷/热启动。
- burn 的输出 MP4 时长与源视频相差不超过 0.5 秒，音视频流存在。

## 失败判定

出现任一项即 FAIL：

- 目标产物缺失或为空。
- 退出码为 0 但产物不完整（静默成功）。
- 时间戳回退、越界、词序错乱、SRT 编号跳号。
- 缺少前置产物却自动补跑，或悄悄换用未声明的后端。
- 验证命令实际跑了全流程。
- 没有可复现的命令或证据。
- 实现 agent 与验证 agent 是同一个 agent。

## 阶段检查清单

| 阶段 | 必查 |
|------|------|
| ingest | `media/source.mp4` 与 16k mono `media/audio.wav` 存在；时长/采样率正确；不做人声分离。 |
| transcribe | 语言、分段、词级时间戳完整；时间戳单调且在音频范围内；失败显式报错。 |
| segment | 句子边界合理；词级时间戳不漂移；LLM 失败不静默留下未切句。 |
| translate | 源文/译文结构完整；时间戳不变；两步流程可复现；失败显式报错。 |
| subtitle | 三份 SRT 均生成；编号/顺序/时间戳正确；显示长度约束成立。 |
| burn | MP4 存在；时长/流正确；CJK 字体渲染正常；失败不输出黑色占位视频。 |
| 性能（T2a/T2b） | 30s/1min/10min 的冷/热 RTFx、显存、批量；独立复跑；不改变阶段契约。T2a 报告：`docs/agents/rtfx-baseline.md`。 |
| 人类门（T2c/T9） | 决策记录、真实样本产物、人工确认；只跑一次全流程。 |

## 阶段验证文档骨架

实现 agent 为每个阶段写一份文档，至少包含：

```markdown
# 阶段验证：<stage-id>

## 范围
本阶段负责什么，不负责什么；对应 ticket 与 commit。

## 前置条件
Python/venv、模型、依赖、GPU、网络、配置路径；缺什么。

## 夹具
使用的 tests/fixtures 文件、复制到 workdir 的目标文件名、来源与语言。

## 命令
完整可复制的 runner 命令、工作目录、环境变量（不含密钥）。

## 期望
产物、字段、时间戳/顺序不变量、退出码。

## 失败判定
什么情况算 FAIL，包含失败路径。

## 证据
命令输出、产物清单、RTFx/显存（如适用）；不放大段日志或密钥。
```

## 验证记录模板

复制到 ticket 评论或验证文档末尾：

```markdown
## 验证记录

- Ticket:
- Stage:
- Implementer:
- Verifier:
- Commit:
- Scope:
- Preconditions:
- Config path:
- Fixtures:
- Command:
- Expected:
- Result:
- Exit code:
- Evidence:
- Verdict: PASS / FAIL
```