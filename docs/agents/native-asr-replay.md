# Native ASR 原语回放证据（replay gate）

> 状态：已执行的验证门（设计 spec 的 Evidence gate 1）。交付原生 CUDA 可执行文件与确定性 GLM
> shape planner，在真实 GLM 矩阵形状上比较 FP16/FP32-累加、FP16/FP16-累加与 W8A8 GEMM。
> **所有数字都是孤立 GEMM 估计（`isolated_gemm_estimate`），不是 ASR RTFx，不能作为 540
> 验收证据。** 结果在共享 GPU 上测得，标注 diagnostic。
>
> 日期：2026-10-22。GPU：RTX 3090 24 GB（共享环境，测量前后外部进程占用约 1.3–1.4 GB 显存，
> 测后快照 46% util / 1392 MiB / SM 360 MHz / 33.74 W）。分支 `feat/native-asr-540`。
> 计划：`docs/superpowers/plans/2026-10-22-native-asr-replay.md`；设计：
> `docs/superpowers/specs/2026-10-22-native-asr-540-design.md`（wayfinder #33 / spec #37）。

## 0. 结论摘要

1. **三个子累加路径在真实 GLM 形状上的 GEMM 线性层工作量估计（1 小时音频场景，120 × 30 s 块）**
   为：FP16/FP32-累加 **6332.8 ms**、FP16/FP16-累加 **3757.8 ms**、INT8(W8A8) **3394.4 ms**。
   场景假设（chunks=120、encoder_batch=8、prefill_batch=32、decode_batch=120、
   output_tokens=120、prompt_tokens=20、decode 恒定 active 长度）随结果一并记录。
2. **FP32-累加被测量排除**：仅 GEMM 线性层 6.33 s 就几乎吃满 540 RTFx 的 6.667 s/小时预算，
   与 #34 的结构性判定一致。
3. **FP16/FP16-累加在真实形状上兑现 1.68×**（6332.8 → 3757.8 ms；#34 微基准为 2.00×）。
   **INT8 只比 FP16-累加再快约 10%**（3757.8 → 3394.4 ms），且每计时迭代内含激活量化与
   反量化；encoder 部分 INT8（1927.4 ms）对 FP16-累加（1999.0 ms）优势仅约 3.6%。
4. **encoder 主导 GEMM 预算**（FP16-累加口径约 53%，INT8 口径约 57%），其次 prefill
   （约 27–29%）、decode（约 15–19%）、projector（<1%）。
5. **下一候选选择**：按 spec"replay gate 之后选定一个实测原语候选并实现完整垂直切片"，
   选 **FP16/FP16-累加** 进入最小完整引擎（M0：PCM → mel → encoder → projector → greedy
   decode → 文本）。理由：相对 INT8 免去量化/反量化与 W8A8 质量门，GEMM 差距仅 ~10%，
   而 spec 的停止规则要求候选失败时记录实测差距再评估剩余候选——INT8 保留为条件备选
   （若垂直切片预算不足，优先对 encoder 单独引入）。不得用本回放数字乘法外推端到端吞吐。
6. **纪律约束均满足**：无 vLLM/TensorRT/PyTorch/libtorch 进入原生 runtime（二进制仅链
   CUDA runtime + cuBLAS）；planner 仅标准库导入；汇总固定
   `metric_kind=isolated_gemm_estimate`、`asr_acceptance=false`、不输出 RTFx；原始结果在
   忽略目录 `bench/results/native-asr/`，不入库。

## 1. 交付物与复现命令

文件（本 PR 提交）：`native/asr/CMakeLists.txt`、`native/asr/src/gemm_replay.cu`（689 行）、
`native/asr/README.md`、`bench/native_asr_replay.py`（573 行，纯标准库）、
`tests/test_native_asr_replay.py`（419 行）。构建产物在忽略目录 `native/asr/build/`。

```sh
# 构建（本机 CMake 需指定 CUDA 12.8 与系统 g++，见 README）
cmake -S native/asr -B native/asr/build \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build native/asr/build -j 4

# shape 清单（官方 config → 21 个 GEMM case）
.venv/bin/python -m bench.native_asr_replay plan \
  --model-config /root/.cache/modelscope/ZhipuAI/GLM-ASR-Nano-2512/config.json \
  --manifest bench/results/native-asr/glm-shapes.tsv \
  --metadata bench/results/native-asr/glm-shapes.json

# 三模式回放（顺序执行；runs=3, warmups=2, iterations=5, 4 GiB 缓冲上限）
for mode in fp16_acc32 fp16_acc16 int8; do
  native/asr/build/glm_gemm_replay \
    --manifest bench/results/native-asr/glm-shapes.tsv --mode $mode \
    --runs 3 --warmups 2 --iterations 5 --max-buffer-mib 4096 \
    > bench/results/native-asr/glm-${mode//_/-}.json
done

# 汇总校验
.venv/bin/python -m bench.native_asr_replay summarize \
  --metadata bench/results/native-asr/glm-shapes.json \
  --result bench/results/native-asr/glm-int8.json \
  --output bench/results/native-asr/glm-int8-summary.json
```

哈希链：

| 项 | SHA256（前 16 位） |
|---|---|
| 模型 config（`GLM-ASR-Nano-2512/config.json`） | `5e410db9ee1000a8`（全值记录于三份 summary 的 `model_config_sha256`） |
| shape 清单 | `08e228186a5f6e68`（`shape_sha256`） |
| 回放二进制 `glm_gemm_replay`（19:29:13 构建，产出结果后未重编） | `3ff6e2dc318bd936` |

## 2. 结果

加权估计线性毫秒（`median_ms × repeats` 求和；`repeats` 是该形状在场景工作负载中的出现
次数，计时复用同一 X/W 工作集）：

| 分量 | FP16/acc32 | FP16/acc16 | INT8 |
|---|---:|---:|---:|
| encoder（32 层 × 15 批） | 3529.3 | 1999.0 | 1927.4 |
| prefill（28 层 × 4 批） | 1895.8 | 1085.2 | 926.4 |
| decode（28 层 × 119 步） | 864.6 | 652.1 | 524.2 |
| projector | 43.1 | 21.5 | 16.5 |
| **合计** | **6332.8** | **3757.8** | **3394.4** |

数值校验：每个 case 记录 `error_check`（小输出全量、大输出 64 个等距确定性采样元素），
三模式全部 `finite=true`；实测 `max_abs_error=0`、`relative_l2_error=0`（确定性输入下
对照 FP32-累加参考 / 精确 INT32 点积）。INT8 口径：固定 scale 16、对称 `[-127,127]`
饱和、INT32 累加后 ×256 反量化，均在计时迭代内（静态权重量化在计时外）。

实现过程中发现并修复一个**测试预期错误**（非 kernel 错误）：INT8 自测首例输入 8 ×
scale 16 = 128 超出对称饱和上界 127，真实反量化值为 7.9375，旧手算期望误复用了未量化
FP16 结果。修复后为该饱和样例写入独立精确期望，误差容限未放宽，样例保留。

## 3. 限制与披露

- **共享 GPU（diagnostic）**：测量窗口外部进程占用约 1340 MiB（测前）/ 1392 MiB、46%
  util、SM 360 MHz（测后快照）；测量时刻的 compute PID 列表未被原样捕获，此为证据缺口。
  绝对毫秒值受共享影响，模式间相对比较方向可用；独占 GPU 复测留待 #33 的 Q4 口径。
- **工作集复用**：计时反复使用同一 X/W 分配，不测量逐层权重流量；README 明确小-M 的
  decoder GEMM 可能被高估，decoder 数字不可换算成整模型吞吐。
- **遗漏操作**（summary 的 `omitted_operations` 原样列出）：mel、convolutions、
  attention、norms、activations、residuals、KV、routing、tokenization、scheduling、
  model_loading。GEMM 之外的约 40–50% GPU 时间（#34 口径）不包含在上述毫秒中。
- **场景假设显式声明**：120/20 token 等是规划假设而非观测；decode 按 120 序列恒定
  active 长度假设建模，实际衰减会改变 decode 形状。
- 本报告不输出、不推算 ASR RTFx；540 判定归 #35 合同的完整引擎门。

## 4. 独立验证

（验证执行方与实现方分离，只读、只报告证据；全程未修改实现文件，临时产物仅写 /tmp。）

- 目标单测：`.venv/bin/python -m pytest -q tests/test_native_asr_replay.py` → **34 passed**
  （planner 合同、非整除批、布尔/越界参数、TSV 稳定性、缺失/重复/多余 case、非有限与
  负值拒绝、偶数中位数、CLI 缺路径等）；`compileall` 通过；导入
  `bench.native_asr_replay` 不加载 torch/transformers/vLLM（`sys.modules` 交集为空）。
- 独立验证代理复核（2026-10-22，8/8 项 PASS）：
  1. **GPU 状态**：确认共享（外部 python 进程占 972 MiB），验证全程只跑
     correctness-only self-test，未跑任何性能基准。
  2. **self-test**：退出码 0，`passed=true`，覆盖 fp16_acc32 / fp16_acc16 / int8。
  3. **负面用例 7/7**：无参数、manifest 不存在、不支持的模式、`--runs 0`、空
     manifest、重复 ID TSV、0 维度 TSV——全部退出码 2、错误进 stderr、stdout 无 JSON。
  4. **summarizer**：三份结果重跑 summarize 与已存 summary **逐字节等价**；篡改
     case（负 median_ms）被拒绝（退出码 2，报 `times_ms must be nonnegative`）。
  5. **依赖**：`ldd` 仅 CUDA runtime/cuBLAS/cublasLt 与系统库，无
     torch/vLLM/TensorRT/ONNX；planner 导入检查同上。
  6. **哈希链**：config SHA256 与三份 summary 逐一相等；二进制 SHA256
     `3ff6e2dc…` 且构建后未重编；三份结果各 21 个 case 与 glm-shapes.tsv 的
     id/m/n/k/repeats 在集合、顺序、字段值三个维度完全一致。
  7. **计时边界代码审查（引用行号）**：CUDA event 对只包裹被计时的 GEMM 批次，
     warmup（L500–502）与数值校验（L517）在计时外；int8 的激活量化（L391）与
     反量化（L399）在被计时迭代内、静态权重量化（L490–493）在计时外；median
     奇偶处理正确（L458–462）；三模式的 compute 类型与 alpha/beta 类型逐一对
     照 cuBLAS 调用核实（L369–398）。
  8. **附注**：结果 JSON 的 `timing_scope` 自行披露 `repeats` 仅为元数据、不在
     迭代内执行，与汇总的加权口径一致。

## 5. 证据索引

- 原始结果（忽略目录，不入库）：`bench/results/native-asr/glm-shapes.{tsv,json}`、
  `glm-fp16-acc32.json`、`glm-fp16-acc16.json`、`glm-int8.json` 及三份 `-summary.json`。
- 修复记录：INT8 饱和样例预期修正（源码 `native/asr/src/gemm_replay.cu` 最后修改
  19:29:05，先于二进制 19:29:13 与全部结果文件）。
- 上游证据：`docs/agents/asr-fp16-540-probe.md`（#34：FP16 精度 GO / PyTorch 吞吐
  NO-GO）、`docs/agents/asr-engine-feasibility-540.md`、wayfinder 地图 #33、spec #37。
