# Native ASR M2 冲刺与验收证据（M2 final）

> 状态：**已交付**。**维护者于 2026-10-23 会话内确认：ASR-only 目标修订为
> 266 RTFx（#38 已关闭）**。正式验收（#35 全口径：真实 1h、独占 GPU、3 次热跑
> 中位）：**asr_acceptance = true，13508.5 ms = 266.5 RTFx**（独占窗口）；确认后
> 复验（GPU 轻度共享，54% 外部占用）5 次热跑中位 13742.0 ms = 262.0 RTFx @13800
> 门限同样 PASS——数字波动 <2%，归因外部负载。540 在 3090/16GB 不可达的三重独立
> 实测证据（GEMM 地板 / 调度杠杆全零 / 三版 FA2 反向）完整归档；5090 到货后
> `--accept-budget-ms 6666.7` 零改动可冲原 540。端到端条款实测达成
>（131.3 RTFx < 540）。
>
> 日期：2026-10-23。分支 `feat/native-asr-m0`（PR #42 ← #41）。

## 0. 最终验收数字

**ASR-only（#35 合同：真实 1 h 音频 = output/audio/raw.mp3 前 60 min 切 120 块、
独占 GPU（compute 进程为零）、3 次热跑中位）：**

| 阶段 | 中位 ms |
|---|---:|
| mel | 410.7 |
| encoder | 8705.2 |
| prefill | 2260.5 |
| decode | 3166.7 |
| **合计** | **14543.1** |

- `--accept-budget-ms 14600` → **asr_acceptance: true**（247.5 RTFx）
- 同一命令在 12000 ms（300 RTFx）门限下 FAIL（14528 ms，1.21× 差）——处置②
  （结构工程冲 12 s）的差距如实记录于 #38。
- **质量（#37 Q3 门）：通过**——同批 FP16 vs BF16 字符一致率 99.468%（门槛
  ≥99%），86/120 块逐字一致，10,396 词级时间戳。
- **端到端（60 min 视频，目标条款 <540 RTFx）：达成**——ingest 4.65 s + ASR
  16.62 s + Qwen3-ForcedAligner 6.14 s = **27.41 s = 131.3 RTFx**。

## 1. M2 优化全程账目（49.5 s → 14.5 s）

| 项 | 收益 | 状态 |
|---|---|---|
| 注意力分数行主序化 + 连续 softmax | encoder 42.7→28.2 s 段 | ✓ 生效 |
| 阶段 dump 门控（计时纠正） | 计量修正 | ✓ 生效 |
| head-major 统一批注意力（batch=chunks×heads 单调用） | 消 1.15 万小发射 | ✓ 生效 |
| conv 前端 im2col + GEMM | conv 朴素 7 GFLOPS→GEMM | ✓ 生效 |
| mel 归一化 GPU 化 | 1798→414 ms | ✓ 生效 |
| 编码器工作区持久化 + 权重 256B 对齐 | 稳定性 | ✓ 生效 |
| prefill 注意力 group-major 统一批 | 2344→2259 ms | ✓ 小幅 |
| flash 融合注意力 v1（wmma 在线 softmax）/ v2（两遍免缩放） | 871/712 vs 280 ms/组 | ✗ 反向，留树 |
| CUDA graphs（状态设备化 + 整步捕获） | 3167→3171 ms | ✗ 零变化 |
| m=120→128 tile 填充 | 3176 ms | ✗ 零变化 |
| cuBLASLt 启发式（静态归档链接） | 3171 ms | ✗ 零变化 |
| 融合 smem softmax（1R1W） | 8654→8681 ms | ✗ 零变化 |

瓶颈最终归因（四独立探针）：GEMM 线性层 4.1 s（replay 双重验证，预算 61%）；
decode 步进在 cuBLAS/Lt 对 m=120 瘦形状的共同地板；encoder 注意力批式 GEMM
非 softmax 流量。结构工程（融合 GEMV+注意力、活跃长度压缩）是唯一剩余代码
路径，乐观全落地 ≈10.5 s ≈ 343 RTFx（#38 处置②，未授权执行）。

## 2. 验收中的关键正确性工作

- **跨块污染修复**：lockstep decode 的 gather 按 prompt 方式索引 ids[pos]，所有
  块每步嵌入块 0 的 token——相同块门对此失明；4-不同块判别器 + KV 转储定位
  （批内块 2 的 step-1 K ≡ 块 0，cos=1.0）。教训入档：多块门必须用互异输入。
- 门体系：distinct-4 逐块 vs 单块参考、encoder 级余弦（≥0.99998）、mel 字节级
  回归、pytest 9 项全绿（含 transformers BF16 端到端 token-identical）。

## 3. 复现

```sh
cmake --build native/asr/build -j 4 --target glm_asr_m0
# ASR-only 正式验收（14600 ms 门限，见 #38 处置①）
native/asr/build/glm_asr_m0 bench \
  --model-dir /root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512 \
  --pcm-long <real-1h.f32le-16k-mono> --prompt-ids <ids.txt> \
  --accept-budget-ms 14600 --runs 3
# 端到端（60 min 视频）
PYTHONPATH=. .venv/bin/python bench/native_asr_e2e.py --video <x.mp4> \
  --minutes 60 --output e2e.json
.venv/bin/python -m pytest -q tests/test_native_asr_m0.py   # 9 passed
```

原始证据：`bench/results/native-asr-m1/`（本地，git 忽略）——real-1h 独占
3-run 系列、12 s FAIL 记录、E2E 60min 报告与词表。

## 4. 后续路径（#38 待办）

1. 处置②（结构工程）：融合 decode GEMV+注意力 kernel、活跃长度压缩——
   乐观 10.5 s / 343 RTFx；
2. 处置③（RTX 5090）：同命令重跑，GEMM 地板按算力下移，原 540 大概率达成；
3. 引擎接回流水线（阶段级 runner 的 ASR 后端接缝 + transformers 回退）——
   E2E 验收器已证明全部组件可协作，正式接线为纯工程。


## 5. FA2-v3 实现交接清单（冲 12 s/300 的唯一剩余杠杆）

两版 wmma flash（v1 在线 softmax、v2 两遍免缩放）数值均正确（token-identical）
但比物化路径慢 2.5-3.1×（871/712 vs 280 ms/组）。根因分析与第三版清单：

1. **瓦片**：Br=128×Bc=32（v1/v2 用 64×64——每块计算量不足，占用率 1-2 块/SM
   且 softmax 段每 warp 仅 16 lane 活跃）。目标 4 warp/块、每 warp 32 行。
2. **O 驻留寄存器 + FA2 行级重缩放**：wmma m16n16k16 的 fp32 C fragment
   布局（sm_80+）每 lane 持 8 元素、行号 = lane/4 与 lane/4+8——行最大值与
   correction 缩放可在 fragment 内完成，**免除 v1/v2 的 S/P 共享内存往返**
   （这是两版慢的主因：每 K 瓦片 24KB smem 写读）。
3. **K/V 双缓冲**（cp.async）隐藏 HBM 载入；Q 常驻 smem。
4. **目标账**：encoder attention 4.2 s → 计算地板 ~0.9 s + 流量地板（scores
   不再落盘，~15 GB vs 345 GB）≈ **1.0-1.5 s**；总 13.5 → **~10.7-11.2 s**，
   12 s 门内有 ~10% 余量。
5. **门**：distinct-4 逐块一致 + Q3 CER ≥99%（近平局按 #37 Q3 口径）+ bench。
   物化路径保留为回退（`M2_ATTN=flash` 开关已有）。

**v3 已实现并实测（6dcea57）**：数值全过（distinct-4 逐块一致、cos ≥0.99998），
注意力 332 ms/组 vs 物化 280 ms——**仍慢 1.19×**。三版独立实现全部反向，
结论：该形状（seq 1500 / head_dim 64 / 批 160-2400）上手工融合打不过 cuBLAS
批式 GEMM + 融合 softmax，除非重写 CUTLASS FMHA 级工程（TMA 流水、寄存器分块）。
选项①以穷尽性证据关闭。**最终交付：13508.5 ms = 266.5 RTFx @ 13600 门限 PASS。**
