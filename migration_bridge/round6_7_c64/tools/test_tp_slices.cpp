#include "operators/llamafile/tp_slices.hpp"
#include <cassert>
#include <iostream>
#include <tuple>
int main() {
  for (int count:{1,2,4}) {
    auto slices=make_llamafile_tp_slices(1408,count,true);
    int end=0;
    for(auto s:slices){assert(s.offset==end);assert(s.length%32==0);end+=s.length;}
    assert(end==1408);
    if(count==2){assert(slices[0].length==704);assert(slices[1].offset==704);}
  }
  for(auto [size,count,f32]:{std::tuple{1408,2,false},std::tuple{1410,2,true},
                           std::tuple{1408,3,true},std::tuple{256,2,false},std::tuple{0,2,true}}) {
    bool rejected=false;
    try{make_llamafile_tp_slices(size,count,f32);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);
  }
  auto q=make_llamafile_tp_slices(1280,2,false);
  assert(q[0].length==768 && q[1].length==512 && q[1].offset==768);
  std::cout<<"PASS F32 exact coverage, 704+704, invalid shapes, quantized block contract\n";
}
