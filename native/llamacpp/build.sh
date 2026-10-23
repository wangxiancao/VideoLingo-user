#!/usr/bin/env bash
# Build llama-server with the STQ kernel for HY-MT2 GGUF weights (V4-7 O1).
#
# The official Tencent GGUFs depend on ggml-org/llama.cpp PR #22836 (STQ
# kernel); mainline does not carry it, so we fetch the PR branch.
#
# Environment notes (this box):
#   - system cmake 3.16 is too old for the CUDA backend: use the venv cmake
#     (.venv/bin/cmake, and reinstall it if its shim points at the removed
#     external/index-tts venv interpreter);
#   - system /usr/bin/nvcc is broken: always pass CUDA 12.8 explicitly;
#   - the default libstdc++ (3.4.28) lacks GLIBCXX_3.4.29 needed by CUDA 12.8
#     objects: /usr/local/lib64 carries 3.4.29;
#   - a stale CMakeCache can pin CUDA_cublas_LIBRARY to the CUDA 10.1
#     remnants in /usr/lib — build from a clean build dir only.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  git fetch origin pull/22836/head:stq
  git checkout stq
else
  cd llama.cpp
fi

export PATH="/root/VideoLingo/.venv/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
rm -rf build   # never reuse a cache from another toolkit
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda-12.8/lib64 -L/usr/local/lib64" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda-12.8/lib64 -L/usr/local/lib64"
cmake --build build --config Release -j16 --target llama-server llama-cli llama-bench

echo
echo "run a replica (subtitle-serving shape: 4K context, one slot):"
echo "LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/cuda-12.8/lib64 \\"
echo "  ./build/bin/llama-server -m <Q4_K_M.gguf> -ngl 99 --jinja -c 4096 -np 1 \\"
echo "  --host 127.0.0.1 --port <PORT> --repeat-penalty 1.0"
