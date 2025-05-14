#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/scalar/gemm.h>

extern "C" void exomlir_gemm_lt_128(const int32_t n, const int32_t m, const int32_t k, float *out, const float *a,
									const float *b);

int main() {
	std::vector<float> a(128 * 128);
	std::vector<float> b(128 * 128);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// fill data
	for (auto &v : a) {
		v = dist(rng);
	}
	for (auto &v : b) {
		v = dist(rng);
	}

	std::vector<float> exocc_out(128 * 128);
	std::vector<float> exomlir_out(128 * 128);

	gemm_lt_128(nullptr, 128, 128, 128, exocc_out.data(), a.data(), b.data());

	std::cout << "exocc_out: " << std::endl;

	exomlir_gemm_lt_128(128, 128, 128, exomlir_out.data(), a.data(), b.data());

	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 128; j++) {
			if (std::abs(exocc_out[i * 128 + j] - exomlir_out[i * 128 + j]) > 1e-6f) {
				std::cerr << "Expected: " << exocc_out[i * 128 + j] << ", got: " << exomlir_out[i * 128 + j]
						  << std::endl;
				return 1;
			}
		}
	}
}
