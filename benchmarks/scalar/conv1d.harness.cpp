#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

#include <exocc/scalar/conv1d.h>

extern "C" void exomlir_conv1d(int32_t n, int32_t ic, int32_t oc, const int32_t *data, const int32_t *kernels,
							   int32_t *out);

static void BM_conv1d(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	int_fast32_t ic = state.range(1);
	int_fast32_t oc = state.range(2);
	std::vector<int32_t> data(n * ic);
	std::vector<int32_t> kernels(ic * oc);
	std::vector<int32_t> out(n * oc);

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

		conv1d(nullptr, n, ic, oc, data.data(), kernels.data(), out.data());
		benchmark::DoNotOptimize(out.data());
	}
}

// BENCHMARK(BM_conv1d)->RangeMultiplier(2)->Ranges({{16, 1024}, {4, 4}, {4, 4}});

static void BM_exomlir_conv1d(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	int_fast32_t ic = state.range(1);
	int_fast32_t oc = state.range(2);
	std::vector<int32_t> data(n * ic);
	std::vector<int32_t> kernels(ic * oc);
	std::vector<int32_t> out(n * oc);

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

		exomlir_conv1d(n, ic, oc, data.data(), kernels.data(), out.data());
		benchmark::DoNotOptimize(out.data());
	}
}

BENCHMARK(BM_exomlir_conv1d)->RangeMultiplier(2)->Ranges({{16, 1024}, {4, 4}, {4, 4}});

BENCHMARK_MAIN();
