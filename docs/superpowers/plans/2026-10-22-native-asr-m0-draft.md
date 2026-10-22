# Native ASR M0 minimal complete engine — Implementation Plan (DRAFT)

> **状态：草稿，待维护者批准后方可开工。** 依据 replay gate 的实测选择（见
> `docs/agents/native-asr-replay.md` §0.5）与设计 spec
> `docs/superpowers/specs/2026-10-22-native-asr-540-design.md`。wayfinder 决策票
> #36（MVP 范围）拍板前，本计划不进入执行。

**Goal:** 实现最小完整垂直切片 `PCM → mel → encoder → projector → greedy decode → 文本`，
在单个 30 s 块上验证 GLM-ASR-Nano-2512 的 FP16/FP16-累加自研推理链的正确性，并给出
分量计时。**M0 不承诺 RTFx，不接产品流水线。**

**Architecture:** C++/CUDA 单进程可执行文件（延续 `native/asr/`），复用 replay 的
cuBLAS FP16-累加 GEMM 封装；权重从官方 safetensors 一次性加载为 FP16 设备缓冲；
mel 前端确定性实现；encoder/decoder 逐层串行；单序列贪心解码。PyTorch/transformers
只出现在对照测试中，不进入构建与运行时依赖。

**Tech Stack:** C++17/CUDA 12.8（`/usr/local/cuda-12.8`）、cuBLAS、cuFFT（如 mel 需要）、
HF `tokenizers`（Rust 绑定，非 `transformers`）；Python 3.10 仅用于加载与对照脚本。

## Global Constraints

- 延续 replay 计划全部约束：单张共享 RTX 3090（diagnostic 标注）、独占口径留给正式
  验收；禁止 vLLM/TensorRT/PyTorch/libtorch 进入运行时；不新增依赖；不动
  `config.yaml`/`output/`/流水线。
- 权重与 config 只读自 `/root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512/`，
  SHA256 记录进每个结果 JSON。
- BF16→FP16 转换必须显式检查范围（|w| ≥ 65504 记为溢出并统计），禁止静默饱和。
- 每个结果 JSON 携带 `measurement_kind`、命令、哈希与场景假设；不输出 ASR RTFx。
- 对照基准：同输入走 `pipeline/asr` 现有 transformers BF16 路径，文本逐字比对。

## Task 1: 权重与配置加载器

**Files:** `native/asr/src/model_loader.{h,cu}`、`native/asr/src/main_m0.cu`。

- [ ] 解析 safetensors 头（纯标准 C++，无第三方库），按官方 config 枚举全部张量，
      缺失/多余张量为硬错误。
- [ ] BF16→FP16 转换在设备端完成；记录 `overflow_count`、`max_abs_weight`、逐张量
      SHA256（转换后字节）。
- [ ] `--dump-plan` 干跑模式：只加载并打印形状/哈希，不初始化 CUDA 上下文。

## Task 2: mel 前端

- [ ] 16 kHz mono PCM（f32）→ 128 mel × 1500 帧，与 librosa/torch 参考实现比对：
      逐元素相对误差 ≤ 1e-3（确定性输入），记录最大偏差。
- [ ] 30 s 定长分块、0 overlap；不足 30 s 补零并记录填充长度。

## Task 3: encoder + projector（单块）

- [ ] conv 前端 + 32 层 encoder：QKV/O/MLP 用 cuBLAS `CUBLAS_COMPUTE_16F`；注意力
      与 norm 先用朴素 CUDA kernel（正确性优先，性能留给 M1）。
- [ ] projector（375 token、5120→4096→2048）。
- [ ] 中间张量落盘开关 `--dump-activations`：与 transformers BF16 逐层对照（余弦
      相似度阈值 0.999，超出即停）。

## Task 4: decoder 贪心解码（单序列）

- [ ] prefill（375+prompt）+ 逐步贪心，`max_new_tokens=256`，EOS 终止；GQA、RoPE、
      RMSNorm、SiLU 与官方 config 对齐。
- [ ] 输出 token id 经 HF `tokenizers` 解码为文本（进程外 Python 包装亦可，但不得
      import `transformers`）。
- [ ] 与 transformers BF16 输出比对：要求字符一致率 100%（单块、贪心、无采样；
      若不一致，逐 token 定位首个分歧层并记录）。

## Task 5: 验证、报告与提交

- [ ] 独立验证代理（实现方之外）复跑：加载器干跑、mel 对照、端到端文本比对、
      `ldd` 依赖检查、pytest（新增对照脚本测试）。
- [ ] 报告 `docs/agents/native-asr-m0.md`：分量计时（mel/encoder/projector/prefill/
      decode）、共享 GPU 标注、限制与遗漏；原始输出进忽略目录
      `bench/results/native-asr-m0/`。
- [ ] 提交 `feat/native-asr-540`，更新本文件 checkbox；M1（多块 batch + KV）另立
      计划，先过维护者审批。

## Self-review rulings

- M0 的通过标准是**单块文本与 BF16 参考逐字一致** + 分量计时落盘；性能优化
  （CUDA graphs、批处理、KV 管理）全部后置到 M1+。
- replay 的 3757.8 ms（FP16-累加 GEMM 线性层，1 h 场景）不是 M0 的预算承诺；
  禁止乘法外推端到端吞吐。
- 若 BF16→FP16 转换出现溢出或文本不一致，按 spec 停止规则回到 #38 三岔口，
  不擅自换精度路线。
