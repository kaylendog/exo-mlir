#include <base.h>
#include <benchmark.h>
#include <stdlib.h>

void matmul(matrix_t lhs, matrix_t rhs, matrix_t output) {
	for (int i = 0; i < lhs.width; i++) {
		for (int j = 0; j < rhs.height; j++) {
			float acc = 0.0f;
			for (int k = 0; k < lhs.height; k++) {
				acc += lhs.data[i * lhs.height + k] * rhs.data[k * rhs.height + j];
			}
			output.data[i * output.height + j] = acc;
		}
	}
}

void proxy_matmul_base(matrix_t lhs, matrix_t rhs, matrix_t output) {
	matmul_base(NULL, lhs.width, rhs.height, lhs.height, output.data, lhs.data, rhs.data);
}

#ifdef __SUPPORTS_AVX2
#include <avx2.H>
#endif

#ifdef __SUPPORTS_NEON
#include <neon.h>

#endif

int main() {
	benchmark_binary_procedure("matmul_base", 8, 8, matrix_alloc_square, matrix_init_uniform, matmul,
							   proxy_matmul_base);

#ifdef __SUPPORTS_AVX2
	benchmark_binary_procedure("matmul_avx2", 10, 8, matrix_alloc_square, matrix_init_uniform, matmul, matmul_avx2);
#endif
#ifdef __SUPPORTS_NEON
	benchmark_binary_procedure("matmul_neon", 10, 8, matrix_alloc_square, matrix_init_uniform, matmul, matmul_neon);
#endif

	return 0;
}
