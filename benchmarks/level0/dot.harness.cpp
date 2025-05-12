#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <exocc/level0/dot.h>

extern "C" void exomlir_dot(int32_t n, const float *x, const float *y, float *result);

static void BM_dot(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data_x(n);
	std::vector<float> data_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	float result = 0.0f;
	exo_win_1f32c x = {data_x.data(), {1}};
	exo_win_1f32c y = {data_y.data(), {1}};

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

		dot(nullptr, n, x, y, &result);
		benchmark::DoNotOptimize(result);
	}

	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_dot)->RangeMultiplier(2)->Range(16, 1024);

static void BM_exomlir_dot(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data_x(n);
	std::vector<float> data_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	float result = 0.0f;

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
		exomlir_dot(n, data_x.data(), data_y.data(), &result);
		benchmark::DoNotOptimize(result);
	}

	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_exomlir_dot)->RangeMultiplier(2)->Range(16, 1024);

BENCHMARK_MAIN();
