#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level1/dot.h>

extern "C" void exomlir_exo_sdot_stride_1(int64_t n, const float *x, const float *y, float *result);

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

	float result_exocc = 0.0f;
	float result_exomlir = 0.0f;

	exo_win_1f32c exocc_x = {x.data(), {1}};
	exo_win_1f32c exocc_y = {y.data(), {1}};

	exo_sdot_stride_1(nullptr, n, exocc_x, exocc_y, &result_exocc);
	exomlir_exo_sdot_stride_1(n, x.data(), y.data(), &result_exomlir);

	float precision = 1e-5f;

	if (std::abs(result_exocc - result_exomlir) > precision) {
		std::cerr << "Expected: " << result_exocc << ", got: " << result_exomlir
				  << " (delta: " << std::abs(result_exocc - result_exomlir) << ")" << std::endl;
		return 1;
	}

	std::cout << "exocc: " << result_exocc << ", exomlir: " << result_exomlir << std::endl;

	return 0;
}
