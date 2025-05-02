#include "accumulator.h"

#include <stdio.h>
#include <stdlib.h>

// matmul_base(
//     C : f32[16, 16] @DRAM,
//     A : f32[16, 16] @DRAM,
//     B : f32[16, 16] @DRAM
// )
void matmul_base( void *ctxt, float* C, const float* A, const float* B ) {
float acc;
for (int_fast32_t i = 0; i < 16; i++) {
  for (int_fast32_t j = 0; j < 16; j++) {
    acc = 0.0f;
    for (int_fast32_t k = 0; k < 16; k++) {
      acc += A[i * 16 + k] * B[k * 16 + j];
    }
    C[i * 16 + j] += acc;
  }
}
}

