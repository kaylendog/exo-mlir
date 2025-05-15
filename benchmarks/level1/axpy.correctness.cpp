#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level1/axpy.h>

extern "C" void exomlir_exo_saxpy_stride_1(int64_t n, const float alpha, float *x, float *y);

int main() {
	int_fast32_t n = 2048;
	std::vector<float> exocc_x(n);
	std::vector<float> exocc_y(n);
	std::vector<float> exomlir_x(n);
	std::vector<float> exomlir_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// fill both buffers with random data
	for (auto &v : exocc_x) {
		v = dist(rng);
	}
	for (auto &v : exocc_y) {
		v = dist(rng);
	}

	rng.seed(0);
	for (auto &v : exomlir_x) {
		v = dist(rng);
	}
	for (auto &v : exomlir_y) {
		v = dist(rng);
	}

	float alpha = dist(rng);

	exo_saxpy_stride_1(nullptr, n, &alpha, {exocc_x.data(), {1}}, {exocc_y.data(), {1}});
	exomlir_exo_saxpy_stride_1(n, alpha, exomlir_x.data(), exomlir_y.data());

	float precision = 1e-6f;

	for (int i = 0; i < n; ++i) {
		if (std::abs(exocc_y[i] - exomlir_y[i]) > precision) {
			std::cerr << "Expected: " << exocc_y[i] << ", got: " << exomlir_y[i] << std::endl;
			return 1;
		}
		std::cout << "exocc: " << exocc_y[i] << ", exomlir: " << exomlir_y[i] << std::endl;
	}

	return 0;
}
