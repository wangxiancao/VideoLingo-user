# Native ASR M2 throughput sprint Implementation Plan

> 状态：执行中（承接 M1 校准，PR #42）。目标：**ASR-only 540 RTFx 验收**——#35
> 合同：真实 1 h 音频、独占 GPU、3 次热跑中位数 ≤ 6.6667 s；附 #37 Q3
> FP16-vs-BF16 CER 质量门。硬件：RTX 3090 / 16 GB。

**M1 基线（120 块合成循环，共享 GPU diagnostic，中位）：** mel 1801.8 /
encoder 42685.3 / prefill 2793.4 / decode 2271.4 = **49551.9 ms**（预算
6666.7，差 7.4×）。

## Task 1: 注意力分数行主序化（softmax 重构）

- [x] scores GEMM 对调 q/k（行主序 S）、连续 softmax、PV opB=N，四处同步；
      causal mask 随行主序更新。附加：阶段 dump 改可选（此前每层同步 dump 主导
      计时并被误归因给 GEMM）、Q/K/V head-major scatter 使 scores/PV 每层各一次
      uniform-batch 调用、conv 前端 im2col+GEMM（朴素 conv2 仅 ~7 GFLOPS）、mel
      归一化搬 GPU（宿主端 4600 万次单线程 log10 是真凶）、编码器工作区持久化、
      权重 arena 256B 对齐。
- [x] 门：批处理 transcribe token-identical 全程保持；encoder 级余弦
      ≥0.99998；mel 输出 vs 宿主版差 ≤1.5e-07；pytest 9 passed（对齐填充断言
      已按新语义更新）。
- [x] bench：**49551.9 → 13666.1 ms**（mel 414 / encoder 8696 / prefill 2344 /
      decode 2213，中位），距 6666.7 ms 预算差 **2.05×**。瓶颈现状：encoder
      attention ~4.2 s（物化 scores 的带宽地板，flash 融合是唯一解）、prefill
      2.34（逐块注意力循环待统一批）、decode 2.21（launch 密集，CUDA graphs）、
      mel 0.41。

## Task 2: mel kernel 效率

- [x] 真凶是宿主端归一化而非 DFT kernel：搬 GPU（三段 kernel，per-chunk max），
      1798 → 414 ms；输出与宿主版差 ≤1.5e-07。
- [ ] 进一步（STFT kernel 仍高于理论地板，非当前瓶颈）。

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
