#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level1-unopt/asum.h>

extern "C" void exomlir_asum(int32_t n, const float *x, float *result);

int main() {
	int_fast32_t n = 2048;
	std::vector<float> x(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	// fill data
	for (auto &d : x) {
		d = dist(rng);
	}

	float result_exocc = 0.0f;
	float result_exomlir = 0.0f;

	exo_win_1f32c exocc_x = {x.data(), {1}};

	asum(nullptr, n, exocc_x, &result_exocc);
	exomlir_asum(n, x.data(), &result_exomlir);

	float precision = 1e-6f;

	if (std::abs(result_exocc - result_exomlir) > precision) {
		std::cerr << "Expected: " << result_exocc << ", got: " << result_exomlir << std::endl;
		return 1;
	}

	std::cout << "exocc: " << result_exocc << ", exomlir: " << result_exomlir << std::endl;

	return 0;
}
