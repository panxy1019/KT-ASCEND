#pragma once
#include <stdexcept>
#include <vector>

struct LlamafileTPSlice { int offset; int length; };

// Round9 qualifies BF16 weights with the existing ARM NEON/F32-input kernel.
// Keep the SVE BF16-input and other platform paths behind their existing guards.
inline constexpr bool llamafile_bf16_f32_tail_supported() {
#if defined(__aarch64__) && !defined(__ARM_FEATURE_SVE)
  return true;
#else
  return false;
#endif
}

// Unblocked F32/BF16 permit 32-aligned equal partitions. Block-quantized weights
// retain the 256-block distribution (including unequal block counts).
inline std::vector<LlamafileTPSlice> make_llamafile_tp_slices(int complete, int count, bool unblocked) {
  if (complete <= 0 || count <= 0) throw std::invalid_argument("invalid LLAMAFILE TP dimensions");
  std::vector<LlamafileTPSlice> slices;
  if (unblocked) {
    if (complete % count || (complete / count) % 32)
      throw std::invalid_argument("unblocked LLAMAFILE TP partitions require 32-element alignment");
    for (int i=0;i<count;++i) slices.push_back({i*(complete/count),complete/count});
  } else {
    if (complete % 256 || complete / 256 < count)
      throw std::invalid_argument("quantized LLAMAFILE TP requires complete 256-element blocks");
    int offset=0, blocks=complete/256;
    for (int i=0;i<count;++i) {
      int length=(blocks/count+(i<blocks%count))*256;
      slices.push_back({offset,length}); offset+=length;
    }
  }
  return slices;
}
