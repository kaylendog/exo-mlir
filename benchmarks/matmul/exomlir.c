#include <benchmark.h>
#include <stdint.h>
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

#ifdef __TARGET_base

// declared in MLIR
void matmul_base(int32_t m, int32_t n, int32_t k, float *C, float *A, float *B);

void proxy_matmul_base(matrix_t lhs, matrix_t rhs, matrix_t output) {
	matmul_base(lhs.width, rhs.height, lhs.height, output.data, lhs.data, rhs.data);
}

#endif

int main() {
#ifdef __TARGET_base
	benchmark_binary_procedure("matmul_base", 8, 8, matrix_alloc_square, matrix_init_uniform, matmul,
							   proxy_matmul_base);
#endif

	return 0;
}
