#include "acc_matmul.h"

#include <stdio.h>
#include <stdlib.h>

// matmul_base(
//     M : size,
//     N : size,
//     K : size,
//     C : f32[M, N] @DRAM,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM
// )
void matmul_base( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, float* C, const float* A, const float* B ) {
float acc;
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < N; j++) {
    acc = 0.0f;
    for (int_fast32_t k = 0; k < K; k++) {
      acc += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] += acc;
  }
}
}

