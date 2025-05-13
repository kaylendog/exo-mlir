#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <exocc/level1/exo_copy.h>

extern "C" void exomlir_copy(int32_t n, const float *x, float *y);

static void BM_exo_copy(benchmark::State &state) {
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
		// fill data
		for (auto &v : data_x) {
			v = dist(rng);
		}
		state.ResumeTiming();

		copy(nullptr, n, x, y);
		benchmark::DoNotOptimize(data_y.data());
	}
}

BENCHMARK(BM_exo_copy)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

static void BM_exomlir_exo_copy(benchmark::State &state) {
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
		state.ResumeTiming();

		exomlir_copy(n, data_x.data(), data_y.data());
		benchmark::DoNotOptimize(data_y.data());
	}
}

BENCHMARK(BM_exomlir_exo_copy)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

BENCHMARK_MAIN();
