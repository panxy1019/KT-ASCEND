#include "llamafile/sgemm.h"
#include "llama.cpp/ggml.h"
// A: contiguous [M,K], B: contiguous [N,K], C: contiguous [N,M].
extern "C" int c64_sgemm(int m,int n,int k,const float* a,const float* b,float* c) {
  return llamafile_sgemm(m,n,k,a,k,b,k,c,m,0,1,GGML_TASK_TYPE_COMPUTE,
                         GGML_TYPE_F32,GGML_TYPE_F32,GGML_TYPE_F32,GGML_PREC_DEFAULT);
}
