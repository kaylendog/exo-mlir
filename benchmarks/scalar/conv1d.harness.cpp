#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

#include <exocc/scalar/conv1d.h>

extern "C" void exomlir_conv1d_4(int32_t n, const int32_t *data, const int32_t *kernels, int32_t *out);

static void BM_conv1d_4(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<int32_t> data(n * 4);
	std::vector<int32_t> kernels(4 * 4);
	std::vector<int32_t> out(n * 4);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_int_distribution<int32_t> dist(0.0, 1.0);

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &d : data) {
			d = dist(rng);
		}
		for (auto &k : kernels) {
			k = dist(rng);
		}
		state.ResumeTiming();

		conv1d_4(nullptr, n, data.data(), kernels.data(), out.data());
		benchmark::DoNotOptimize(out.data());
	}
}

BENCHMARK(BM_conv1d_4)->RangeMultiplier(2)->Range(16, 512)->Iterations(16);

static void BM_exomlir_conv1d_4(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<int32_t> data(n * 4);
	std::vector<int32_t> kernels(4 * 4);
	std::vector<int32_t> out(n * 4);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_int_distribution<int32_t> dist(0.0, 1.0);

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &d : data) {
			d = dist(rng);
		}
		for (auto &k : kernels) {
			k = dist(rng);
		}
		state.ResumeTiming();

		exomlir_conv1d_4(n, data.data(), kernels.data(), out.data());
		benchmark::DoNotOptimize(out.data());
	}
}

BENCHMARK(BM_exomlir_conv1d_4)->RangeMultiplier(2)->Range(16, 512)->Iterations(16);

BENCHMARK_MAIN();
