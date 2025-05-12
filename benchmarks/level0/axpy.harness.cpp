#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <exocc/level0/axpy.h>

extern "C" void exomlir_axpy_alpha_1(int32_t n, const float *x, float *y);

static void BM_axpy_alpha_1(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data_x(n);
	std::vector<float> data_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	exo_win_1f32c x = {data_x.data(), {1}};
	exo_win_1f32 y = {data_y.data(), {1}};

	for (auto _ : state) {
		state.PauseTiming();
		for (auto &v : data_x) {
			v = dist(rng);
		}
		for (auto &v : data_y) {
			v = dist(rng);
		}
		state.ResumeTiming();

		axpy_alpha_1(nullptr, n, x, y);
		benchmark::DoNotOptimize(data_y.data());
	}

	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_axpy_alpha_1)->RangeMultiplier(2)->Range(16, 1024);

static void BM_exomlir_asum(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data_x(n);
	std::vector<float> data_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &v : data_x) {
			v = dist(rng);
		}
		for (auto &v : data_y) {
			v = dist(rng);
		}
		state.ResumeTiming();

		exomlir_axpy_alpha_1(n, data_x.data(), data_y.data());
		benchmark::DoNotOptimize(data_y.data());
	}

	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_exomlir_asum)->RangeMultiplier(2)->Range(16, 1024);

BENCHMARK_MAIN();
