#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level1-unopt/axpy.h>

extern "C" void exomlir_axpy_alpha_1(int32_t n, const float *x, float *y);

int main() {
	int_fast32_t n = 2048;
	std::vector<float> x(n);
	std::vector<float> y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	// fill data
	for (auto &v : x) {
		v = dist(rng);
	}
	for (auto &v : y) {
		v = dist(rng);
	}

	float result = 0.0f;

	// copy data
	std::vector<float> exomlir_x(x);
	std::vector<float> exomlir_y(y);

	exo_win_1f32c exocc_x = {x.data(), {1}};
	exo_win_1f32 exocc_y = {y.data(), {1}};

	axpy_alpha_1(nullptr, n, exocc_x, exocc_y);
	exomlir_axpy_alpha_1(n, exomlir_x.data(), exomlir_y.data());

	float precision = 1e-6f;

	for (int i = 0; i < n; ++i) {
		if (std::abs(y[i] - exomlir_y[i]) > precision) {
			std::cerr << "Expected: " << y[i] << ", got: " << exomlir_y[i] << std::endl;
			return 1;
		}
		std::cout << "exocc: " << y[i] << ", exomlir: " << exomlir_y[i] << std::endl;
	}

	return 0;
}
