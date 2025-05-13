#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <exocc/level2/scal.h>

extern "C" void exomlir_exo_sscal_stride_1(int32_t n, const float *alpha, const float *x);

int main() {
	int_fast32_t n = 1024;
	std::vector<float> x(n);

	// setup rng
	std::mt19937 rng(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// fill data
	for (auto &v : x) {
		v = dist(rng);
	}

	// buffers
	exo_win_1f32 exocc_x = {x.data(), {1}};
	std::vector<float> exomlir_x(x);

	float alpha = dist(rng);

	exo_sscal_stride_1(nullptr, n, &alpha, exocc_x);
	exomlir_exo_sscal_stride_1(n, &alpha, exomlir_x.data());

	float precision = 1e-6f;

	for (int i = 0; i < n; ++i) {
		if (std::abs(x[i] - exomlir_x[i]) > precision) {
			std::cerr << "Expected: " << x[i] << ", got: " << exomlir_x[i] << std::endl;
			return 1;
		}
		std::cout << "exocc: " << x[i] << ", exomlir: " << exomlir_x[i] << std::endl;
	}

	return 0;
}
