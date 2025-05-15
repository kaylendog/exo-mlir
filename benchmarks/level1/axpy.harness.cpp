#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

#include <exocc/level1/axpy.h>

extern "C" void exomlir_exo_saxpy_stride_1(int64_t n, const float alpha, float *x, float *y);

static void BM_exo_saxpy_stride_1(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> x(n);
	std::vector<float> y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (auto _ : state) {
		state.PauseTiming();
		for (auto &v : x) {
			v = dist(rng);
		}
		float alpha = dist(rng);
		state.ResumeTiming();

		exo_saxpy_stride_1(nullptr, n, &alpha, {x.data(), {1}}, {y.data(), {1}});
		benchmark::DoNotOptimize(y.data());
	}
}

// BENCHMARK(BM_exo_saxpy_stride_1)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

static void BM_exomlir_exo_saxpy_stride_1(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> x(n);
	std::vector<float> y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (auto _ : state) {
		state.PauseTiming();
		for (auto &v : x) {
			v = dist(rng);
		}
		float alpha = dist(rng);
		state.ResumeTiming();

		exomlir_exo_saxpy_stride_1(n, alpha, x.data(), y.data());
		benchmark::DoNotOptimize(y.data());
	}
}

BENCHMARK(BM_exomlir_exo_saxpy_stride_1)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

BENCHMARK_MAIN();
