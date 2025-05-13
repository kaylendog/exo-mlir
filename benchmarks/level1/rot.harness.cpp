#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <exocc/level1/rot.h>

extern "C" void exomlir_rot(int32_t n, const float *x, const float *y, const float *c, const float *s);

static void BM_rot(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data_x(n);
	std::vector<float> data_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	float c = 0.5f;
	float s = 0.5f;

	exo_win_1f32 x = {data_x.data(), {1}};
	exo_win_1f32 y = {data_y.data(), {1}};

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

		rot(nullptr, n, x, y, &c, &s);
		benchmark::DoNotOptimize(data_x.data());
	}
}

BENCHMARK(BM_rot)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

static void BM_exomlir_rot(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data_x(n);
	std::vector<float> data_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	float c = 0.5f;
	float s = 0.5f;

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

		exomlir_rot(n, data_x.data(), data_y.data(), &c, &s);
		benchmark::DoNotOptimize(data_x.data());
	}
}

BENCHMARK(BM_exomlir_rot)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

BENCHMARK_MAIN();
