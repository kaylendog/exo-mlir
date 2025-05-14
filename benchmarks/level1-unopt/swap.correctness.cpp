#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level1-unopt/swap.h>

extern "C" void exomlir_swap(int32_t n, const float *x, float *y);

int main() {
	int_fast32_t n = 2048;
	std::vector<float> x(n);
	std::vector<float> y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// fill data
	for (auto &v : x) {
		v = dist(rng);
	}
	for (auto &v : y) {
		v = dist(rng);
	}

	// buffers
	exo_win_1f32 exocc_x = {x.data(), {1}};
	exo_win_1f32 exocc_y = {y.data(), {1}};
	std::vector<float> exomlir_x(x);
	std::vector<float> exomlir_y(y);

	swap(nullptr, n, exocc_x, exocc_y);
	exomlir_swap(n, exomlir_x.data(), exomlir_y.data());

	float precision = 1e-6f;

	for (int i = 0; i < n; ++i) {
		if (std::abs(x[i] - exomlir_x[i]) > precision) {
			std::cerr << "Expected: " << x[i] << ", got: " << exomlir_x[i] << std::endl;
			return 1;
		}
		if (std::abs(y[i] - exomlir_y[i]) > precision) {
			std::cerr << "Expected: " << y[i] << ", got: " << exomlir_y[i] << std::endl;
			return 1;
		}

		std::cout << "exocc: " << x[i] << ", exomlir: " << exomlir_x[i] << std::endl;
	}

	return 0;
}
