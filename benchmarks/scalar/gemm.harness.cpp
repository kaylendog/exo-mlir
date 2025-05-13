#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

#include <exocc/scalar/gemm.h>

extern "C" void exomlir_gemm(int32_t m, int32_t n, int32_t k, const float *out, const float *a, float *b);

static void BM_gemm(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> out(n * n);
	std::vector<float> a(n * n);
	std::vector<float> b(n * n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &d : out) {
			d = dist(rng);
		}
		for (auto &a_ : a) {
			a_ = dist(rng);
		}
		for (auto &b_ : b) {
			b_ = dist(rng);
		}
		state.ResumeTiming();

		gemm(nullptr, n, n, n, out.data(), a.data(), b.data());
		benchmark::DoNotOptimize(b.data());
	}
}

// BENCHMARK(BM_gemm)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

static void BM_exomlir_gemm(benchmark::State &state) {
	int_fast32_t n = state.range(0);
	std::vector<float> out(n * n);
	std::vector<float> a(n * n);
	std::vector<float> b(n * n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (auto _ : state) {
		state.PauseTiming();
		// fill data
		for (auto &v : out) {
			v = dist(rng);
		}
		for (auto &v : a) {
			v = dist(rng);
		}
		for (auto &v : b) {
			v = dist(rng);
		}
		state.ResumeTiming();

		exomlir_gemm(n, n, n, out.data(), a.data(), b.data());
		benchmark::DoNotOptimize(out);
	}
}

BENCHMARK(BM_exomlir_gemm)->RangeMultiplier(2)->Range(16, 1 << 24)->Iterations(16);

BENCHMARK_MAIN();
