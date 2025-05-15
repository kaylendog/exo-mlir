#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level1/scal.h>

extern "C" void exomlir_exo_sscal_stride_1(int64_t n, const float alpha, float *x);

int main() {
	int_fast32_t n = 2048;
	std::vector<float> exocc_x(n);
	std::vector<float> exomlir_x(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// fill both buffers with the same data
	for (auto &v : exocc_x) {
		v = dist(rng);
	}
	rng.seed(0);
	for (auto &v : exomlir_x) {
		v = dist(rng);
	}

	float alpha = dist(rng);

	exo_sscal_stride_1(nullptr, n, &alpha, {exocc_x.data(), {1}});
	exomlir_exo_sscal_stride_1(n, alpha, exomlir_x.data());

	float precision = 1e-6f;

	for (int i = 0; i < n; ++i) {
		if (std::abs(exocc_x[i] - exomlir_x[i]) > precision) {
			std::cerr << "Expected: " << exocc_x[i] << ", got: " << exomlir_x[i] << std::endl;
			return 1;
		}
		std::cout << "exocc: " << exocc_x[i] << ", exomlir: " << exomlir_x[i] << std::endl;
	}

	return 0;
}
