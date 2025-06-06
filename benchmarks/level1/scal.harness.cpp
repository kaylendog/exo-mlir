#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <exocc/level1/scal.h>

extern "C" void exomlir_exo_sscal_stride_1(int64_t n, const float alpha, float *x);

static void BM_exo_sscal_stride_1(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data_x(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	exo_win_1f32 x = {data_x.data(), {1}};

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &v : data_x) {
			v = dist(rng);
		}
		float alpha = dist(rng);
		state.ResumeTiming();

		exo_sscal_stride_1(nullptr, n, &alpha, x);
		benchmark::DoNotOptimize(data_x.data());
	}
}

BENCHMARK(BM_exo_sscal_stride_1)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

static void BM_exomlir_exo_sscal_stride_1(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data_x(n);
	std::vector<float> data_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &v : data_x) {
			v = dist(rng);
		}
		float alpha = dist(rng);
		state.ResumeTiming();

		exomlir_exo_sscal_stride_1(n, alpha, data_x.data());
		benchmark::DoNotOptimize(data_x.data());
	}
}

BENCHMARK(BM_exomlir_exo_sscal_stride_1)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

BENCHMARK_MAIN();
