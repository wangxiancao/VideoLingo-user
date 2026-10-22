cmake_minimum_required(VERSION 3.16)

# This host also carries an obsolete /usr/bin/nvcc. Prefer the documented
# CUDA 12.8 toolkit when the caller has not selected a compiler explicitly.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda-12.8/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "/usr/bin/g++")
  set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE FILEPATH "C++ compiler")
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++")
  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE FILEPATH "CUDA host compiler")
endif()

project(VideoLingoNativeAsr LANGUAGES CXX CUDA)

add_executable(glm_gemm_replay src/gemm_replay.cu)
target_compile_options(glm_gemm_replay PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17;--expt-relaxed-constexpr;-gencode=arch=compute_86,code=sm_86;--compiler-options=-Wall,-Wextra>
)
target_link_libraries(glm_gemm_replay PRIVATE cublas)

add_executable(glm_asr_m0 src/model_loader.cu src/mel_frontend.cu src/engine.cu src/main_m0.cu)
target_include_directories(glm_asr_m0 PRIVATE src)
target_compile_options(glm_asr_m0 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17;--expt-relaxed-constexpr;-gencode=arch=compute_86,code=sm_86;--compiler-options=-Wall,-Wextra>
)
target_link_libraries(glm_asr_m0 PRIVATE cublas)

