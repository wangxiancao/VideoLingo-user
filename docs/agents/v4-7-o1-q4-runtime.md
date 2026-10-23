# V4-7 O1：llama.cpp Q4 runtime 引入与质量门（2026-10-23）

> 状态：**完成，Q4 路线 GO**。依据 spec #54 / 票 #55（维护者约束：权重一律
> ModelScope；流水线显存峰值 ≤16 GiB）。构建脚本 `native/llamacpp/build.sh`
> （源码树与构建产物不入库）；探针 `bench/v4_7_q4_probe.py`；原始数据
> `bench/results/v4-7/q4-probe.json`。

## 0. 结论

| 项 | Q4（llama.cpp STQ，GPU 全卸载） | BF16（transformers，V4-2 基线） |
|---|---:|---:|
| 单副本 GPU 显存 | **1,680 MiB**（`-c 4096 -np 1`） | 4,340 MiB |
| 单副本吞吐 | **284.95 tok/s**（12 段批中位） | 32.5 tok/s |
| 契约矩阵 | **zh→en ✓ en→zh ✓ refine ✓（3/3）** | 3/3（en→zh 在本轮参考侧偶发一次失败，见 §3） |
| 盲评对照 | 与 BF16 输出基本逐字一致 | — |

**推算**：translate 全量 ≈ 43,000 completion tokens（T9b 实测 1332.8 s × 32.5 tok/s）
→ Q4 单副本 ≈ **152 s**，双副本 ≈ 80 s。**330 s 验收线单副本即达标**，
4 副本（6.7 GiB）远在 16 GiB 预算内。O2 的多副本从"必需"降级为"冗余配置"。

## 1. 构建（坑位记录，供复现）

STQ kernel 在 PR #22836（官方 GGUF README 明示依赖）。构建需绕开本机四个坑：
系统 cmake 3.16 过旧（用 venv cmake 4.4，其 shim 若指向已删除的 index-tts
venv 需 `--force-reinstall`）；系统 `/usr/bin/nvcc` 损坏（显式 CUDA 12.8）；
默认 libstdc++ 3.4.28 缺 GLIBCXX_3.4.29（`/usr/local/lib64` 有 3.4.29）；
**CMakeCache 会把 CUDA_cublas_LIBRARY 钉死在 /usr/lib 的 CUDA 10.1 残留上——
必须干净 build 目录重配**（`-DCUDAToolkit_ROOT=/usr/local/cuda-12.8`）。
全部固化进 `native/llamacpp/build.sh`。

## 2. 权重

`Tencent-Hunyuan/HY-MT2-1.8B-GGUF` 的 **Q4_K_M（1.13 GB）**，经 ModelScope
snapshot_download 落 `_model_cache/HY-MT2-1.8B-GGUF/`（运行期不下载）。

## 3. 契约矩阵与质量

- 端点：`llama-server -ngl 99 --jinja -c 4096 -np 1 --repeat-penalty 1.0`
  （OpenAI 兼容；greedy 由 temperature=0 请求侧保证）。
- 矩阵：V4-1 同款 12 段批 × 双方向 × refine，流水线同款校验
  （ids 完整、target 非空、无自造键）——**Q4 三项全过**。
- 盲评：zh→en 四个抽样点上 Q4 与 BF16 译文基本一致（含"匹秒→seconds/
  milliseconds"这一已知 ASR 脏源单位误差在两侧同样出现——非 Q4 退化）；
  en→zh 干净（"We need to talk…" → "我们需要谈谈昨天发生的事情。"等）。
- 参考侧备注：本轮 BF16 端 en→zh 出现一次矩阵失败（12 段批缺 id），属
  V4-1 已知的尾部 id 丢失败败模式（生产侧已有重试+二分升级兜底）；不影响
  Q4 判定，如实记录。

## 4. 遗留到 O2/O4 的事项

- 多副本仍实现（配置化 1..N，默认按 O4 实测定，预期 1–2 即足够）。
- llama-server 运行时 `LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/cuda-12.8/lib64`
  需由服务管理器统一注入。
- `-c 4096 -np 1` 为字幕服务形态（默认 262K 上下文会预分配 ~18 GiB KV，
  必须显式限制）。
