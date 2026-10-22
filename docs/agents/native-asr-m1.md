# Native ASR M1 批处理与计时证据（calibration milestone）

> 状态：M1 完成。多块批处理引擎（mel+encoder 批处理、decoder 批量 prefill、跨块
> lockstep decode、每块 KV）+ CUDA-event 分量计时落地；**1 h 场景（合成循环，
> diagnostic 共享 GPU）总 49.55 s，距 540 预算 6.67 s 差 7.4×，瓶颈已逐项定位**。
> 540 RTFx 验收归 M2（#35 合同：真实 1 h 音频、独占 GPU、3 次热跑中位数）。
>
> 日期：2026-10-22 深夜。分支 `feat/native-asr-m0`（PR #42 ← #41）。硬件预算：
> RTX 3090 / **16 GB**（实测引擎峰值远低于预算）；计划
> `docs/superpowers/plans/2026-10-22-native-asr-m1.md`。

## 0. 结论摘要

1. **正确性门全部保持**：批量 prefill（T1）、批处理 encoder（T2）、跨块批量
   decode（T3）各自的 token/余弦门通过；全套件回归 **43 passed**（M0 9 项 +
   replay 34 项）。实现期门禁捕获 4 个真 bug（GQA stride-0、decode KV 层偏移、
   首 token 行提取、组内 RoPE 取模）。
2. **M1 校准（120 块 = 1 h，fp16-acc16，3 次中位，共享 GPU diagnostic）**：

   | 阶段 | 实测 ms | replay GEMM 估计 ms | 差距 |
   |---|---:|---:|---:|
   | mel | 1801.8 | （遗漏项） | — |
   | encoder | 42685.3 | 1999.0 | 21.4× |
   | prefill | 2793.4 | 1095.2 | 2.6× |
   | decode | 2271.4 | 652.1 | 3.5× |
   | **合计** | **49551.9** | **3757.8** | **13.2×** |

   对 540 预算（6666.7 ms）：**7.4×**。换算 412.9 ms/块 vs 预算 55.6 ms/块。
3. **瓶颈定位（M2 优化清单，按收益排序）**：
   - **encoder 朴素 strided softmax ≈ 40 s（占总量 81%）**：col-major 分数布局下
     每 (query, head) 一个 block、步长 3000 元素的访存；改造方向：行主序分数
     （PV GEMM 相应换轴）/ 融合 scale+softmax / 分块 softmax。预期 10× 级收益，
     encoder 可回到 ~4–6 s。
   - decode launch 开销：72 步 × 28 层 × ~10 kernel ≈ 2 万次小发射；CUDA
     graph 捕获预期数倍收益。
   - prefill 逐块注意力循环（120 × 4 GEMM/层）：与 encoder softmax 同源。
   - mel 400-tap 朴素 DFT（1.8 s）：批处理 FFT 或分块复用 trig 表。
4. **批尺度数值观察（非实现 bug）**：120 块中 4 块在 token 6 同一近平局翻转
   （20370↔1935）后立即重新同步；跨次运行发散块不同（cuBLAS 非确定性），
   acc32 下 119/120。按 #37 Q3（CER 质量门）口径在 M2 真实音频上裁决；
   M1 记录为 diagnostic。
5. **纪律**：全程 correctness/diagnostic 标注，无 RTFx 验收主张；无新依赖；
   `config.yaml`/流水线未动；批处理 KV 4.42 GiB + 权重 4.52 GiB 在 16 GB 预算内。

## 1. 提交链（M1）

| 提交 | 内容 |
|---|---|
| 768f29e | M1 计划（预算数学 vs replay） |
| 6024f16 | T1 批量 prefill（11.5→5.6 s/块，token-identical；GQA stride-0 修复） |
| 4fb0cf4 | T2 批处理 mel+encoder（chunk 维；cos=1.000000/0.999985 门） |
| 9475228 | T3 跨块 lockstep decode（4 块全 identical；3 bug 修复） |
| 63dc758 | T4 bench：CUDA-event 分量计时 + 校准 JSON（本报告数据源） |
| 643dacd + 本提交 | T5 计划勾选 + 报告 |

## 2. 复现

```sh
cmake --build native/asr/build -j 4 --target glm_asr_m0
native/asr/build/glm_asr_m0 bench \
  --model-dir /root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512 \
  --pcm <30s-f32le-16k-mono.raw> --prompt-ids <ids.txt> \
  --chunks 120 --runs 3
# 原始 JSON：bench/results/native-asr-m1/synth-loop-120chunks-3runs.json
.venv/bin/python -m pytest -q tests/test_native_asr_m0.py tests/test_native_asr_replay.py  # 43 passed
```

## 3. M2 展望（540 冲刺路径）

1. encoder softmax 重构（行主序/融合）→ 预期总量 ~10 s 量级；
2. decode CUDA graph + prefill 注意力批量化 → ~5 s 量级；
3. mel kernel 效率 → ~0.3 s 量级；
4. 真实 1 h 音频（非循环）、独占 GPU（停外部 compute 进程）按 #35 合同验收：
   3 次热跑中位数 ≤ 6.6667 s；同时跑 #37 Q3 FP16-vs-BF16 质量门（固定语料
   CER）。若优化后仍不达标，回到 #38 三岔口（换 5090 / INT8 / 降目标）。
