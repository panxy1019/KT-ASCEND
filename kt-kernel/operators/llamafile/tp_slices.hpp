#pragma once
#include <stdexcept>
#include <vector>

struct LlamafileTPSlice { int offset; int length; };

// Unblocked F32 permits 32-aligned equal partitions. Block-quantized weights
// retain the 256-block distribution (including unequal block counts).
inline std::vector<LlamafileTPSlice> make_llamafile_tp_slices(int complete, int count, bool all_f32) {
  if (complete <= 0 || count <= 0) throw std::invalid_argument("invalid LLAMAFILE TP dimensions");
  std::vector<LlamafileTPSlice> slices;
  if (all_f32) {
    if (complete % count || (complete / count) % 32)
      throw std::invalid_argument("F32 LLAMAFILE TP partitions require 32-element alignment");
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
