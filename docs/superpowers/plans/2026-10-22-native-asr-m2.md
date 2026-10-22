# Native ASR M2 throughput sprint Implementation Plan

> 状态：执行中（承接 M1 校准，PR #42）。目标：**ASR-only 540 RTFx 验收**——#35
> 合同：真实 1 h 音频、独占 GPU、3 次热跑中位数 ≤ 6.6667 s；附 #37 Q3
> FP16-vs-BF16 CER 质量门。硬件：RTX 3090 / 16 GB。

**M1 基线（120 块合成循环，共享 GPU diagnostic，中位）：** mel 1801.8 /
encoder 42685.3 / prefill 2793.4 / decode 2271.4 = **49551.9 ms**（预算
6666.7，差 7.4×）。

## Task 1: 注意力分数行主序化（softmax 重构）

- [ ] scores GEMM 对调 q/k 操作数（计算 Sᵀ；col-major 落盘即行主序 S），softmax
      kernel 改连续行访存，PV GEMM opB 改 N；encoder（单块+批处理）与 decoder
      prefill（单块+组批）四处同步；causal mask kernel 索引随行主序更新。
- [ ] 门：transcribe（单块、4 块）token-identical；pytest 全绿。
- [ ] bench 复测并记录收益（预期 encoder 42.7 s → 个位数 s）。

## Task 2: mel kernel 效率

- [ ] 当前 400-tap 朴素 DFT + 逐 tap reflect 映射（1.8 s）。优化：寄存器化 tap
      循环 + 预计算边界外的反射索引区间，或分块共享内存。目标 ≤ 0.5 s。
- [ ] 门：mel 单块/批处理输出与现实现逐字节一致（确定性不变）。

## Task 3: decode/prefill launch 开销

- [ ] 视 Task 1/2 后余量决定投入：decode 步 CUDA graph 捕获（72 步 × 28 层 ×
      ~10 kernel），prefill 注意力与 encoder 同源收益自动获得。
- [ ] 门：token-identical 保持。

## Task 4: 真实 1 h 验收（#35 合同）

- [ ] 寻找/构造真实 1 h 音频（非循环拼接；优先仓库/本机既有长音频），切 120 块。
- [ ] 独占 GPU：记录 compute PID 快照，外部进程清零后跑 3 次热跑（首次预热
      排除），中位数 ≤ 6666.7 ms → **540 RTFx 验收**；超标则记录差距并按
      #38 三岔口升级。
- [ ] #37 Q3 质量门：FP16 vs BF16 在固定语料上 CER 对照（字符一致率 ≥99%）。
- [ ] 报告 `docs/agents/native-asr-m2.md` + PR/地图留痕。

## Rulings

- 所有优化必须先过 token/字节级门再进 bench；数值行为改变（如 softmax 融合
  改变累加顺序）需在报告中披露。
- 独占 GPU 前不得宣称 540；合成循环数字永远标 diagnostic。
