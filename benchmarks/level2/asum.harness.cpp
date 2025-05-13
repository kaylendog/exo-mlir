#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

#include <exocc/level2/asum.h>

extern "C" void exomlir_exo_sasum_stride_1(int32_t n, const float *x, float *result);

static void BM_exo_sasum_stride_1(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> x(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	float result = 0.0f;

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &v : x) {
			v = dist(rng);
		}
		state.ResumeTiming();

		exo_sasum_stride_1(nullptr, n, {x.data(), {1}}, &result);
		benchmark::DoNotOptimize(result);
	}
}

BENCHMARK(BM_exo_sasum_stride_1)->RangeMultiplier(2)->Range(16, 1024);

static void BM_exomlir_exo_sasum_stride_1(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> x(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	float result = 0.0f;

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &v : x) {
			v = dist(rng);
		}
		state.ResumeTiming();

		exomlir_exo_sasum_stride_1(n, x.data(), &result);
		benchmark::DoNotOptimize(result);
	}
}

BENCHMARK(BM_exomlir_exo_sasum_stride_1)->RangeMultiplier(2)->Range(16, 1024);

BENCHMARK_MAIN();
