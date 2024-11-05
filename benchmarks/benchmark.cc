#include <stdlib.h>
#include <time.h>

#include <benchmark/benchmark.h>

#include <sgemm.h>
#include <avx2_matmul.h>

#define RAND_MAT(Name, N, M)                                  \
    float *Name = (float *)malloc((N) * (M) * sizeof(float)); \
    for (int i = 0; i < (N); i++)                             \
        for (int j = 0; j < (M); j++)                         \
            Name[i * (M) + j] = (float)rand() / RAND_MAX * 2.0 - 1.0;

#define ZERO_MAT(Name, N, M)                                  \
    float *Name = (float *)malloc((N) * (M) * sizeof(float)); \
    for (int i = 0; i < (N); i++)                             \
        for (int j = 0; j < (M); j++)                         \
            Name[i * (M) + j] = 0.0;

static void BM_sgemm(benchmark::State &state) {
    int size = state.range(0);
    
    RAND_MAT(A, size, size);
    RAND_MAT(B, size, size);
    ZERO_MAT(C, size, size);

    for (auto _ : state)
    {
        sgemm(NULL, size, size, size, C, A, B);
    }
}

BENCHMARK(BM_sgemm)->RangeMultiplier(2)->Range(8, 8 << 7);

static void BM_x86_matmul(benchmark::State &state) {
    int size = state.range(0);

    RAND_MAT(A, 6, size);
    RAND_MAT(B, size, 16);
    ZERO_MAT(C, 6, 16);

    for (auto _ : state)
    {
        rank_k_reduce_6x16_scheduled(NULL, size, A, B, C);
    }
}

BENCHMARK(BM_x86_matmul)->RangeMultiplier(2)->Range(8, 8 << 7);

BENCHMARK_MAIN();
