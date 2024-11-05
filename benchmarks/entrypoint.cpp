#include <benchmark/benchmark.h>

#include "matmul.h"

#define RAND_MAT(N, M, matrix) \
    float matrix[N][M]; \
    for (int i = 0; i < N; i++) { \
        for (int j = 0; j < M; j++) { \
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX; \
        } \
    }

#define ZERO_MAT(N, M, matrix) \
    float matrix[N][M]; \
    for (int i = 0; i < N; i++) { \
        for (int j = 0; j < M; j++) { \
            matrix[i][j] = 0; \
        } \
    }

static void BM_sgemm16x16(benchmark::State& state) {
    RAND_MAT(16, 16, A);
    RAND_MAT(16, 16, B);
    ZERO_MAT(16, 16, C);

    for (auto _ : state) {
        sgemm(nullptr, 16, 16, 16, &C[0][0], &A[0][0], &B[0][0]);
    }
}
BENCHMARK(BM_sgemm16x16);


static void BM_sgemm128x128(benchmark::State& state) {
    RAND_MAT(128, 128, A);
    RAND_MAT(128, 128, B);
    ZERO_MAT(128, 128, C);

    for (auto _ : state) {
        sgemm(nullptr, 128, 128, 128, &C[0][0], &A[0][0], &B[0][0]);
    }
}
BENCHMARK(BM_sgemm128x128);

BENCHMARK_MAIN();
