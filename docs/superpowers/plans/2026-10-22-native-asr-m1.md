# Native ASR M1 batching & timing Implementation Plan

> **状态：执行中**（维护者 2026-10-22 晚指令"不要停止"，本计划承接 M0 里程碑，
> PR #42）。硬件预算：RTX 3090、**显存 16 GB**；ASR-only ≥ 540 RTFx 目标归 M2
> 验收（#35 合同：1 h 音频 3 次热跑中位数 ≤ 6.67 s，独占 GPU）；本阶段计时全部
> 为共享 GPU **diagnostic** 口径。

**Goal:** 把 M0 的单块正确性引擎升级为多块批处理引擎：mel+encoder 批处理、
decoder 批量 prefill 与跨块批量 decode、每块 KV 管理，并加入 CUDA-event 分量
计时，与 replay 的 3757.8 ms（fp16-acc16 GEMM 线性层估计，1 h 场景）校准。

**预算数学（校准基准）：** 1 h = 120 块 × 30 s；540 门 = 6.67 s 总预算 =
**55.6 ms/块**。replay：GEMM 线性层 31.3 ms/块（encoder 16.7、prefill 9.0、
decode 5.4、projector 0.2），非 GEMM 预算 **~24 ms/块**。M0 现状：逐 token
串行 decode ~7 s/块——必须消灭。

## Global Constraints（延续 M0）

- 仅 CUDA runtime + cuBLAS；无 PyTorch/vLLM/TensorRT 于运行时；无新依赖。
- 权重 FP16 常驻 4.52 GiB + KV + 激活 ≤ 16384 MiB 硬校验。
- 正确性门：批处理输出与 M0 单块路径**逐 token 一致**（确定性贪心）。
- 计时用 CUDA events 包裹被测 GPU 工作（warmup/校验在外），JSON 落
  `measurement_kind: "diagnostic_shared_gpu"`；不输出 RTFx 验收主张。

## Task 1: decoder 批量 prefill（单块）

- [ ] 把逐 token prefill 循环替换为序列维 GEMM（复用 encoder 的 linear 机制，
      m=序列长），因果 mask + 每行 softmax 的批量注意力（strided-batched），
      KV 一次写入。
- [ ] 门：`transcribe` 输出与 M0 逐 token 路径 token-identical（zh_30s 夹具，
      acc16/acc32）。

## Task 2: mel + encoder 多块批处理

- [ ] mel kernel 增加 chunk 维（grid.z）；encoder GEMM m=batch×1500、注意力
      batch=chunk×head、LayerNorm/激活按行扩展；sub-batch（默认 8 块）控制峰值。
- [ ] 门：批处理 encoder 输出与单块路径 cosine ≥0.9999（同输入确定性比对）；
      批处理 transcribe 文本 token-identical。

## Task 3: 跨块批量 decode + 每块 KV

- [ ] KV 布局 [chunk][layer][2][seq][512]；prefill 按 sub-batch 写入；decode
      步进批量 m=active_chunks，每块独立 EOS（完结块 logits 强制 EOS），步进至
      全部完结或 max_new_tokens。
- [ ] 显存核算：120 块 × 28 层 × 2 × 646 × 512 × 2 B ≈ 4.42 GiB + 权重 4.52
      GiB + 激活 < 16 GB（加载器预算内校验）。
- [ ] 门：N 块批 decode 每块文本 = 单块 M0 输出（N≥4，含不同完结时间步）。

## Task 4: bench 计时与校准

- [ ] `glm_asr_m0 bench --model-dir --pcm <30s> --chunks 120`：同一块循环 120
      次（合成 1 h 场景，标注 synthetic-loop diagnostic），CUDA events 分量计时
      mel/encoder/projector/prefill/decode，3 次热跑中位数，记录共享 GPU 进程
      快照（compute PIDs）。
- [ ] 校准：分量之和 vs replay 3757.8 ms（GEMM-only 估计）对比表，GEMM 实测/
      估计比值与非 GEMM 开销占比落 JSON 与报告；给出距 55.6 ms/块的剩余差距。

## Task 5: 验证、报告与提交

- [ ] 独立验证（实现方之外，只读）：干净重建、全套件、批处理 token 门复跑、
      计时 JSON schema 校验。
- [ ] 报告 `docs/agents/native-asr-m1.md` + PR #42 更新；M2 计划（1 h 真实
      音频、独占 GPU、#35 合同验收）另立。

## Self-review rulings

- decode 批量的"完结块继续算"浪费在 M1 接受（简单性优先）；active-set 压缩
  留 M2 优化清单。
- mel 的 chunk 循环若实测超预算，先批处理 kernel 化再考虑融合。
- 任何分量计时不得乘法外推 RTFx；540 判定只归 M2 独占口径。
