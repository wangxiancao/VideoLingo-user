// Whisper-semantics log-mel frontend (M0 Task 2).
//
// Contract (from transformers WhisperFeatureExtractor / audio_utils.spectrogram,
// which the GLM-ASR processor uses with feature_size=128, hop=160, n_fft=400):
//   PCM f32 (padded to 480000) -> reflect pad (200, 200) -> frame x window
//   (periodic Hann) -> rfft(400) power -> mel filter bank (slaney scale+norm,
//   128 x 201) -> max(1e-10) -> log10 -> drop last frame (3001 -> 3000) ->
//   clamp to (global max - 8) -> (x + 4) / 4 -> f32 [128, 3000].
//
// The window, trig tables and mel filter bank are computed on the host in
// double precision (matching the reference's float64 pipeline); the GPU only
// performs multiply-accumulate work. This is a correctness tool: it reports no
// performance and is not ASR RTFx evidence.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct MelOutput {
  std::vector<float> features;  // row-major [128][3000]
  int64_t input_samples = 0;    // samples before zero padding
};

// pcm must already be padded/truncated to exactly 480000 samples.
MelOutput compute_mel(const std::vector<float>& pcm);

// Batched variant: chunks * 480000 samples -> features [chunks, 128, 3000],
// with the log/normalize pass applied per chunk (matching the reference
// feature extractor, which processes each waveform independently).
MelOutput compute_mel_batched(const std::vector<float>& pcm, int chunks);

constexpr int kMelBins = 128;
constexpr int kFftLength = 400;
constexpr int kFrequencyBins = kFftLength / 2 + 1;  // 201
constexpr int kHopLength = 160;
constexpr int kSampleRate = 16000;
constexpr int kChunkSamples = 30 * kSampleRate;     // 480000
constexpr int kCenterPad = kFftLength / 2;          // 200
constexpr int kFrameCount = 1 + (kChunkSamples + 2 * kCenterPad - kFftLength) / kHopLength;  // 3001
constexpr int kOutputFrames = kFrameCount - 1;      // 3000
