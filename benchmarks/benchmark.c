#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "benchmark.h"

/// @brief Matrix equality.
int matrix_eq(matrix_t lhs, matrix_t rhs) {
	// size check
	if (lhs.width != rhs.width || lhs.height != rhs.height) {
		return 0;
	}
	// element-wise comparison
	for (int x = 0; x < lhs.width; x++) {
		for (int y = 0; y < lhs.height; y++) {
			if (lhs.data[x * lhs.height + y] != rhs.data[x * rhs.height + y]) {
				return 0;
			}
		}
	}
	return 1;
}

/// @brief Utility to compute the current time in nanoseconds.
long time_nanos() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e9 + ts.tv_nsec;
}

void benchmark_unary_procedure(long iterations, long repeats, matrix_alloc_t allocate_matrix, matrix_init_t init_matrix,
							   unary_procedure_t baseline, unary_procedure_t procedure) {
	for (long i = 0; i < iterations; i++) {
		long *deltas = malloc(repeats * sizeof(long));

		matrix_t input = allocate_matrix(i);
		matrix_t output_procedure = allocate_matrix(i);
		matrix_t output_baseline = allocate_matrix(i);

		for (long j = 0; j < repeats; j++) {
			init_matrix(&input);

			// run procedure and time
			long start = time_nanos();
			procedure(input, output_procedure);
			long end = time_nanos();
			deltas[j] = end - start;

			// verify correctness
			baseline(input, output_baseline);
			if (!matrix_eq(output_procedure, output_baseline)) {
				printf("Error: output of procedure does not match baseline\n");
				break;
			}
		}

		// tidy up
		free(input.data);
		free(output_procedure.data);
		free(output_baseline.data);
	}
}

void benchmark_binary_procedure(long iterations, long repeats, matrix_alloc_t allocate_matrix,
								matrix_init_t init_matrix, binary_procedure_t baseline, binary_procedure_t procedure) {
	for (long i = 0; i < iterations; i++) {
		long *deltas = malloc(repeats * sizeof(long));

		matrix_t input_lhs = allocate_matrix(i);
		matrix_t input_rhs = allocate_matrix(i);
		matrix_t output_procedure = allocate_matrix(i);
		matrix_t output_baseline = allocate_matrix(i);

		for (long j = 0; j < repeats; j++) {
			init_matrix(&input_lhs);
			init_matrix(&input_rhs);

			// run procedure and time
			long start = time_nanos();
			procedure(input_lhs, input_rhs, output_procedure);
			long end = time_nanos();
			deltas[j] = end - start;

			// verify correctness
			baseline(input_lhs, input_rhs, output_baseline);
			if (!matrix_eq(output_procedure, output_baseline)) {
				printf("Error: output of procedure does not match baseline\n");
				break;
			}
		}

		// tidy up
		free(input_lhs.data);
		free(input_rhs.data);
		free(output_procedure.data);
		free(output_baseline.data);
	}
}

matrix_t matrix_alloc_square(long iteration) {
	long size = 2 << iteration;
	matrix_t mat = {size, size, malloc(size * size * sizeof(float))};
	return mat;
}

void matrix_init_uniform(matrix_t *mat) {
	for (int x = 0; x < mat->width; x++) {
		for (int y = 0; y < mat->height; y++) {
			mat->data[x * mat->height + y] = (float)(rand() / RAND_MAX);
		}
	}
}
