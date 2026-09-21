// CPU-only gate: actual linked LLAMAFILE kernels against dequantized dot products.
#include "llama.cpp/ggml.h"
#include "llama.cpp/ggml-quants.h"
#include "llamafile/sgemm.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
int main() {
 std::setvbuf(stdout,nullptr,_IONBF,0);
 auto *ctx=ggml_init({1024*1024,nullptr,true});
 for(auto type:{GGML_TYPE_Q8_0,GGML_TYPE_MXFP4})
 for(auto shape: {std::pair<int,int>{160,2560},{64,2560},{256,160},{256,320}})
 for(int n:{1,10,32}) {
  int m=shape.first,k=shape.second,blocks=k/32,lda=blocks+1,ldb=blocks+2,ldc=m+3;
  std::printf("START type=%d m=%d n=%d k=%d\n",int(type),m,n,k);
  std::vector<float> x(k),deqA(m*k),deqB(n*k),out(ldc*n,123456.0f);
  std::vector<unsigned char>A(m*lda*ggml_type_size(type),0xAB);
  std::vector<block_q8_0>B(n*ldb);
  for(int i=0;i<m;i++) {
   auto *row=A.data()+i*lda*ggml_type_size(type);
   if(type==GGML_TYPE_Q8_0) {
    for(int l=0;l<k;l++)x[l]=std::sin(float(i*13+l*7)*.031f)*.3f;
    quantize_row_q8_0_reference(x.data(),reinterpret_cast<block_q8_0*>(row),k);
    dequantize_row_q8_0(reinterpret_cast<block_q8_0*>(row),deqA.data()+i*k,k);
   } else {
    auto *q=reinterpret_cast<block_mxfp4*>(row);
    for(int b=0;b<blocks;b++){q[b].e=123+(b+i)%4;for(int l=0;l<16;l++)q[b].qs[l]=((i+b+l)%16)|(((i+3*b+7*l)%16)<<4);}
    dequantize_row_mxfp4(q,deqA.data()+i*k,k);
   }
  }
  for(int j=0;j<n;j++) {
   for(int l=0;l<k;l++)x[l]=std::cos(float(j*17+l*11)*.027f)*.7f;
   quantize_row_q8_0_reference(x.data(),B.data()+j*ldb,k);
   dequantize_row_q8_0(B.data()+j*ldb,deqB.data()+j*k,k);
  }
  // Three disjoint native worker partitions exercise ith/nth as well as strides.
  for(int worker=0;worker<3;worker++)
   if(!llamafile_sgemm(m,n,blocks,A.data(),lda,B.data(),ldb,out.data(),ldc,
       worker,3,GGML_TASK_TYPE_COMPUTE,type,GGML_TYPE_Q8_0,GGML_TYPE_F32,GGML_PREC_DEFAULT))throw std::runtime_error("native tile rejected");
  double error=0,denom=0;
  for(int j=0;j<n;j++) {
   for(int i=0;i<m;i++) {
    double expected=0;for(int l=0;l<k;l++)expected+=double(deqA[i*k+l])*deqB[j*k+l];
    double delta=out[j*ldc+i]-expected;error+=delta*delta;denom+=expected*expected;
    if(!std::isfinite(out[j*ldc+i])){std::printf("NONFINITE i=%d j=%d expected=%g actual=%g\n",i,j,expected,out[j*ldc+i]);throw std::runtime_error("nonfinite output");}
   }
   for(int i=m;i<ldc;i++)if(out[j*ldc+i]!=123456.0f)throw std::runtime_error("output padding overwritten");
  }
  double rel=std::sqrt(error/std::max(denom,1e-30));
  std::printf("type=%d m=%d n=%d k=%d relative_l2=%.9g\n",int(type),m,n,k,rel);
  if(rel>1e-5)throw std::runtime_error("quantized dot reference mismatch");
 }
 ggml_free(ctx);std::puts("PASS_LOWBIT_NATIVE_TILES_24_CASES_NOT_MOE_QUALIFICATION");
}
