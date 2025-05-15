#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level1/asum.h>

extern "C" void exomlir_exo_sasum_stride_1(int64_t n, const float *x, float *result);

int main() {
	int_fast32_t n = 2048;
	std::vector<float> x(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// fill data
	for (auto &d : x) {
		d = dist(rng);
	}

	float result_exocc = 0.0f;
	float result_exomlir = 0.0f;

	exo_win_1f32c exocc_x = {x.data(), {1}};

	exo_sasum_stride_1(nullptr, n, exocc_x, &result_exocc);
	exomlir_exo_sasum_stride_1(n, x.data(), &result_exomlir);

	float precision = 1e-6f;

	if (std::abs(result_exocc - result_exomlir) > precision) {
		std::cerr << "Expected: " << result_exocc << ", got: " << result_exomlir << std::endl;
		return 1;
	}

	std::cout << "exocc: " << result_exocc << ", exomlir: " << result_exomlir << std::endl;

	return 0;
}
