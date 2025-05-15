#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/scalar/conv1d.h>

extern "C" void exomlir_conv1d_4(const int32_t n, int32_t *data, const int32_t *kernels, int32_t *out);

int main() {
	std::vector<int32_t> data(32768 * 4);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_int_distribution<int32_t> dist(-4, 4);

	for (auto &v : data) {
		v = dist(rng);
	}

	std::vector<int32_t> kernels(4 * 4 * 4);
	for (auto &v : kernels) {
		v = dist(rng);
	}

	std::vector<int32_t> exocc_out(32768 * 4);
	std::vector<int32_t> exomlir_out(32768 * 4);

	conv1d_4(nullptr, 32768, data.data(), kernels.data(), exocc_out.data());
	exomlir_conv1d_4(32768, data.data(), kernels.data(), exomlir_out.data());

	for (int i = 0; i < 32768 * 4; ++i) {
		if (std::abs(exocc_out[i] - exomlir_out[i]) > 1e-6f) {
			std::cerr << "Expected: " << exocc_out[i] << ", got: " << exomlir_out[i] << std::endl;
			return 1;
		}
	}

	return 0;
}
