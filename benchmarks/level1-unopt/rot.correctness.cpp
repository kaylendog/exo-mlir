#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level1-unopt/rot.h>

extern "C" void exomlir_rot(int32_t n, float *x, float *y, const float c, const float s);

int main() {
	int_fast32_t n = 2048;
	std::vector<float> exocc_x(n);
	std::vector<float> exocc_y(n);

	std::vector<float> exomlir_x(n);
	std::vector<float> exomlir_y(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// fill data
	for (auto &v : exocc_x) {
		v = 1.0;
	}
	for (auto &v : exocc_y) {
		v = 0.0;
	}
	for (auto &v : exomlir_x) {
		v = 1.0;
	}
	for (auto &v : exomlir_y) {
		v = 0.0;
	}

	float c = 0.5f;
	float s = 0.5f;

	// buffers
	rot(nullptr, n, {exocc_x.data(), {1}}, {exocc_y.data(), {1}}, &c, &s);
	exomlir_rot(n, exomlir_x.data(), exomlir_y.data(), c, s);

	float precision = 1e-6f;

	for (int i = 0; i < n; ++i) {
		if (std::abs(exocc_x[i] - exomlir_x[i]) > precision) {
			std::cerr << "Expected: " << exocc_x[i] << ", got: " << exomlir_x[i] << std::endl;
			return 1;
		}
		if (std::abs(exocc_y[i] - exomlir_y[i]) > precision) {
			std::cerr << "Expected: " << exocc_y[i] << ", got: " << exomlir_y[i] << std::endl;
			return 1;
		}
	}

	return 0;
}
