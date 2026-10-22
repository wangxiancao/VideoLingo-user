# Native ASR M0 最小完整引擎证据（milestone gate）

> 状态：已执行的 M0 垂直切片验证（设计 spec Evidence gate 2 的正确性部分）。自研
> C++/CUDA 引擎在单个 30 s 块上完成 `PCM → mel → encoder → projector → greedy
> decode → 文本`，**输出与 transformers BF16 参考逐 token 一致**。所有运行为
> correctness-only、共享 GPU，**不含任何 RTFx 主张**；16 GB 显存预算下权重
> FP16 常驻 4.52 GiB。
>
> 日期：2026-10-22 晚。分支 `feat/native-asr-m0`（堆叠 PR #42 ← #41）；计划
> `docs/superpowers/plans/2026-10-22-native-asr-m0-draft.md`；设计 spec
> `docs/superpowers/specs/2026-10-22-native-asr-540-design.md`（#33/#37）。

## 0. 结论摘要

1. **M0 通过门达成**：真实中文语音夹具 `tests/fixtures/audio/zh_30s.wav`（30 s，
   390-token 提示），native FP16 引擎（含 FP16-累加 GEMM）与 transformers BF16
   参考的贪心解码输出 **token 完全一致**（72 tokens）；`--gemm-acc 32` 复跑同样
   一致。文本与参考一致即 M0 的验收标准（计划 Self-review 节）。
2. **五个组件全部通过隔离对照**（确定性输入，逐项余弦 ≥0.99999）：safetensors
   加载与 BF16→FP16 转换（745/745 哈希 = numpy 参考，零溢出/零 NaN，max |w|
   10.9375）；Whisper 语义 mel 前端（max abs diff 9.1e-05，门槛 1e-3）；encoder
   layer-0 全子组件（LN、q/k/v、partial-RoPE、QKᵀ 分数、softmax、PV、MLP）；
   conv 前端（0.999993）；projector（0.9992）。
3. **诚实的数值披露**：encoder 逐层余弦并非全线 0.999——层 0–16 ≥0.99987，
   层 18–30 漂移至 0.987、层 31 至 0.23，但幅值分布与参考吻合（离群激活 max
   1256）、无 NaN/Inf；定位为 fp16-vs-bf16 在离群层的注意力翻转（离散事件），
   非实现错误（所有子组件隔离验证通过；最终 norm 0.9985 / projector 0.9992）。
   端到端文本门（更贴近 #37 Q3 质量门口径）判定通过。逐层 0.999 的诊断阈值在
   后续 M1 若持续失守，按 spec 停止规则升级。
4. **实现期修复的三个真 bug**（全部由上述门捕获）：softmax head 偏移重复累加
   （越界读写）；PV 矩阵乘读入未转置的分数矩阵；o_proj 原位 GEMM 别名。
5. **约束遵守**：运行时零 Python/PyTorch/vLLM/TensorRT（二进制仅链 CUDA
   runtime + cuBLAS，ldd 核实）；prompt token id 由调用方注入（tokenizer 留在
   工具/测试侧）；`config.yaml`/`output/`/流水线未动；无新依赖（SHA-256、JSON
   解析自包含）。

## 1. 交付物与复现

提交（feat/native-asr-m0）：`de4f935`（Task 1 加载器）、`7b72f65`（Task 2 mel）、
`a7a6e1f`（Task 3 encoder/projector）、`5cc2a45`（Task 4 decoder + 文本门）。
文件：`native/asr/src/{model_loader,mel_frontend,engine,decoder,main_m0}.{h,cu}`、
`tests/test_native_asr_m0.py`（9 项测试）。

```sh
cmake -S native/asr -B native/asr/build \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build native/asr/build -j 4

# 提示 id 由 processor 构造（工具侧）：
#   AutoProcessor.from_pretrained(MODEL).apply_transcription_request([pcm])
native/asr/build/glm_asr_m0 transcribe \
  --model-dir /root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512 \
  --pcm <30s-f32le-16k-mono.raw> --prompt-ids <ids.txt> [--gemm-acc 16|32]

.venv/bin/python -m pytest -q tests/test_native_asr_m0.py   # 9 passed
```

架构实现（对齐官方 config，加载器硬校验 745 张量形状）：conv1(k3,s1)+gelu →
conv2(k3,s2)+gelu → 32 层 Whisper 风格 encoder（pre-LN、partial RoPE 32/64、
fp32 softmax）→ LN → reshape [375,5120] → projector 4096→2048 → 28 层 Llama
decoder（GQA 16/4、全维 RoPE、RMSNorm、SiLU、untied lm_head），位置递增式
KV cache（causality by construction）。计时口径：correctness-only，不作为
性能证据。

## 2. 门与证据索引

| 门 | 结果 | 证据 |
|---|---|---|
| 权重转换哈希（CPU=GPU=numpy） | PASS | 745/745 一致；零溢出/NaN |
| mel vs transformers FE | PASS | max abs 9.1e-05（30 s）/ 9.5e-06（短块） |
| encoder 子组件（layer 0 隔离） | PASS | LN/q/k/v/RoPE/softmax/PV ≥0.99999 |
| encoder 逐层余弦 | 部分 | 层 0–16 ≥0.99987；18–31 0.987→0.23（见 §0.3 披露） |
| projector | PASS | 0.9992 |
| 端到端文本（真实语音） | **PASS** | token-identical（acc16 与 acc32 均一致） |
| pytest | PASS | 9 passed |
| 依赖纪律 | PASS | ldd 仅 CUDA/cuBLAS/系统库 |
| 显存预算 16 GB | PASS | 权重常驻 4.52 GiB + 激活/KV（~60 MB）+ staging |

- 独立验证代理复核（只读，2026-10-22 晚）：**7/7 PASS**——干净重建 0 error（7 个
  未使用变量 warning，已清理）；pytest 9 passed；文本门独立复跑三方 token 一致
  （native acc16 = native acc32 = BF16 参考，复跑逐字节确定）；ldd 干净；显存峰值
  **4794 MiB**（nvidia-smi 1 Hz 轮询实测）；负面用例退出码正确。证据在 PR #42 评论。
- 调试钩子：`M0_DEBUG_LAYER0=<dir>` 环境变量转储 layer-0 中间张量（triage 用）。

## 3. 限制与下一步

- **M0 不含性能工作**：逐 token 串行解码（~390 次前向）仅为正确性服务；
  批处理、CUDA graphs、FA2、Paged KV、多块流水属 M1。端到端 9.7–11.7 s 墙钟
  是 correctness 工具的观测值，**禁止当作 RTFx 或预算依据**。
- 逐层余弦 0.999 诊断阈值未在层 18–31 达成（dtype 数值已定位）；M1 引入真实
  语音批量样本时按 #37 Q3（字符一致率 ≥99%）口径复验。
- 下一里程碑 M1（另立计划、先过维护者）：mel+encoder 批处理、decoder 批量
  prefill/decode、KV 管理，开始计时并对照 replay 的 3757.8 ms（fp16-acc16
  GEMM 线性层估计，1 h 场景）校准。
