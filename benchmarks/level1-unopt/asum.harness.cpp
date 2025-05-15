#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include <exocc/level1-unopt/asum.h>

extern "C" void exomlir_asum(int32_t n, const float *x, float *result);

static void BM_asum(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	float result = 0.0f;
	exo_win_1f32c x = {data.data(), {1}};

	for (auto _ : state) {
		// fill data
		state.PauseTiming();
		for (auto &d : data) {
			d = dist(rng);
		}
		state.ResumeTiming();

		asum(nullptr, n, x, &result);
		benchmark::DoNotOptimize(result);
	}

	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_asum)->RangeMultiplier(2)->Range(16, 1 << 16)->Iterations(16);

static void BM_exomlir_asum(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> data(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	float result = 0.0f;

	for (auto _ : state) {
		state.PauseTiming();
		for (auto &d : data) {
			d = dist(rng);
		}
		state.ResumeTiming();

		exomlir_asum(n, data.data(), &result);
		benchmark::DoNotOptimize(result);
	}

	state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_exomlir_asum)->RangeMultiplier(2)->Range(16, 1 << 16)->Iterations(16);

BENCHMARK_MAIN();
