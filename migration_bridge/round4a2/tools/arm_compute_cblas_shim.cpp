#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>
#include <arm_compute/runtime/NEON/functions/NEGEMM.h>
#include <arm_compute/runtime/Tensor.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace {

constexpr int kCblasRowMajor = 101;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;

void copy_rows(float* destination, std::size_t destination_stride, const float* source,
               std::size_t source_stride, int rows, int columns) {
  for (int row = 0; row < rows; ++row) {
    std::memcpy(destination + row * destination_stride, source + row * source_stride,
                static_cast<std::size_t>(columns) * sizeof(float));
  }
}

}  // namespace

// Probe-only compatibility shim. The neutral Python harness invokes the CBLAS
// row-major contract as X[M,K] * W[N,K]^T -> Y[M,N]. ACL represents matrices
// with TensorShape(width, height), so W is explicitly materialized as B[K,N]
// without changing any logical operand value.
extern "C" void cblas_sgemm(int layout, int trans_a, int trans_b, int m, int n, int k, float alpha,
                            const float* x, int x_stride, const float* weight, int weight_stride, float beta,
                            float* output, int output_stride) {
  if (layout != kCblasRowMajor || trans_a != kCblasNoTrans || trans_b != kCblasTrans || alpha != 1.0f ||
      beta != 0.0f) {
    throw std::runtime_error("Round4A2 ACL shim only supports X * W^T with alpha=1 and beta=0");
  }

  using namespace arm_compute;
  NEScheduler::get().set_num_threads(1);

  Tensor a;
  Tensor b;
  Tensor result;
  a.allocator()->init(TensorInfo(TensorShape(k, m), 1, DataType::F32));
  b.allocator()->init(TensorInfo(TensorShape(n, k), 1, DataType::F32));
  result.allocator()->init(TensorInfo(TensorShape(n, m), 1, DataType::F32));

  NEGEMM gemm;
  gemm.configure(&a, &b, nullptr, &result, 1.0f, 0.0f);
  a.allocator()->allocate();
  b.allocator()->allocate();
  result.allocator()->allocate();

  copy_rows(reinterpret_cast<float*>(a.buffer()), a.info()->strides_in_bytes().y() / sizeof(float), x, x_stride,
            m, k);

  auto* b_data = reinterpret_cast<float*>(b.buffer());
  const std::size_t b_stride = b.info()->strides_in_bytes().y() / sizeof(float);
  for (int row = 0; row < k; ++row) {
    for (int column = 0; column < n; ++column) {
      b_data[row * b_stride + column] = weight[column * weight_stride + row];
    }
  }

  gemm.run();
  copy_rows(output, output_stride, reinterpret_cast<const float*>(result.buffer()),
            result.info()->strides_in_bytes().y() / sizeof(float), m, n);
}
