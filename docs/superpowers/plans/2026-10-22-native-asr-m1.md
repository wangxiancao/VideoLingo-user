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

- [x] 序列维 GEMM + 因果 mask + 批量注意力 + KV 一次写入（`--prefill
      batched|sequential` 双路径）；GQA 组内 K/V stride=0。门：zh_30s 夹具
      token-identical（72 tokens），单块墙钟 11.5 s → 5.6 s。
- [x] 门随附修复：GQA stride bug（组内 K/V 应共享）。

## Task 2: mel + encoder 多块批处理

- [x] mel chunk 维（grid.z，每块独立归一化）；encoder 行级 m=chunks×1500、
      注意力逐块循环复用单块 scores 缓冲；sub-batch 8。
- [x] 门：batch-4 chunk0 vs 单块——mel 字节一致、conv cos=1.000000、norm
      0.999985、projector 0.999992、最差层 0.999985。

## Task 3: 跨块批量 decode + 每块 KV

- [x] KV [chunks, layers, 2, max_seq, 512]；prefill 8 块子批；decode 全量
      lockstep（统一长度），每块独立 EOS（宿主小环冻结完结块）。
- [x] 显存：120 块 KV 4.42 GiB + 权重 4.52 GiB + 激活，实测通过（16 GB 内）。
- [x] 门：4 块全部 token-identical（72 tokens、72 步、EOS 位置正确）。120 块：
      116/120 token-identical；4 块在 token 6 同一近平局翻转（20370↔1935）后
      立即重新同步——跨次运行发散块不同（cuBLAS 非确定性）、acc32 下 119/120，
      判定为数值观察而非实现 bug，#37 Q3 CER 门在 M2 裁决。
      修复 3 个真 bug：decode KV 追加漏 layer 偏移、首 token 行提取少乘
      kTextHidden、组内 RoPE 未对 390 取模。

## Task 4: bench 计时与校准

- [x] `bench` 子命令（CUDA events、3 次中位、diagnostic_shared_gpu 标注、
      replay 参考值内嵌）；原始 JSON 归档 `bench/results/native-asr-m1/`。
- [x] 校准（120 块、fp16-acc16、共享 GPU、中位）：mel 1801.8 / encoder
      42685.3 / prefill 2793.4 / decode 2271.4，**总 49551.9 ms** vs replay
      GEMM-only 3757.8 ms（13.2×）vs 540 预算 6666.7 ms（7.4×）。瓶颈定位：
      encoder 朴素 strided softmax ~40 s；其次 prefill 逐块注意力、decode
      launch 开销（72 步 × 28 层）、mel 朴素 400-tap。

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
