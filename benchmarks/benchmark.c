#define _POSIX_C_SOURCE 199309L

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "benchmark.h"

#define NANOTIME_IMPLEMENTATION
#include "nanotime.h"

/// @brief Matrix equality.
int matrix_eq(matrix_t lhs, matrix_t rhs) {
	// size check
	if (lhs.width != rhs.width || lhs.height != rhs.height) {
		return 0;
	}
	// element-wise comparison
	for (int x = 0; x < lhs.width; x++) {
		for (int y = 0; y < lhs.height; y++) {
			if (fabs(lhs.data[x * lhs.height + y] - rhs.data[x * rhs.height + y]) > FLT_EPSILON) {
				return 0;
			}
		}
	}
	return 1;
}

void benchmark_print_stats(long *deltas, long size) {
	double acc = 0.0;
	double acc_squared = 0.0;
	long min = LONG_MAX;
	long max = 0;

	for (long i = 0; i < size; i++) {
		acc += (double)deltas[i];
		acc_squared += (double)deltas[i] * (double)deltas[i];
		min = deltas[i] < min ? deltas[i] : min;
		max = deltas[i] > max ? deltas[i] : max;
	}

	double mean = acc / size;
	double stddev = sqrt((acc_squared / size) - (mean * mean));

	printf("%ld,%ld,%ld,%f,%f\n", size, min, max, mean, stddev);
}

void benchmark_unary_procedure(char *name, long iterations, long repeats, matrix_alloc_t allocate_matrix,
							   matrix_init_t init_matrix, unary_procedure_t baseline, unary_procedure_t procedure) {
	fprintf(stderr, "----- Benchmark %s (iterations=%ld, repeats= %ld) -----\n", name, iterations, repeats);

	printf("size,min,max,mean,stddev\n");

	for (long i = 0; i < iterations; i++) {
		fprintf(stderr, "--- Iteration %ld ---\n", i);

		long *deltas = malloc(repeats * sizeof(long));

		matrix_t input = allocate_matrix(i);
		matrix_t output_procedure = allocate_matrix(i);
		matrix_t output_baseline = allocate_matrix(i);

		for (long j = 0; j < repeats; j++) {
			fprintf(stderr, "Repeat %ld\n", i);

			init_matrix(&input);

			// run procedure and time
			long start = nanotime_now();
			procedure(input, output_procedure);
			deltas[j] = nanotime_interval(start, nanotime_now(), nanotime_now_max());

			// verify correctness
			baseline(input, output_baseline);
			if (!matrix_eq(output_procedure, output_baseline)) {
				fprintf(stderr, "Error: output of procedure does not match baseline\n");
				break;
			}
		}

		// tidy up
		free(input.data);
		free(output_procedure.data);
		free(output_baseline.data);

		// print stats
		benchmark_print_stats(deltas, repeats);
		free(deltas);
	}
}

void benchmark_binary_procedure(char *name, long iterations, long repeats, matrix_alloc_t allocate_matrix,
								matrix_init_t init_matrix, binary_procedure_t baseline, binary_procedure_t procedure) {
	fprintf(stderr, "----- Benchmark %s (iterations=%ld, repeats= %ld) -----\n", name, iterations, repeats);

	printf("size,min,max,mean,stddev\n");

	for (long i = 0; i < iterations; i++) {
		fprintf(stderr, "--- Iteration %ld ---\n", i);

		long *deltas = malloc(repeats * sizeof(long));

		matrix_t input_lhs = allocate_matrix(i);
		matrix_t input_rhs = allocate_matrix(i);
		matrix_t output_procedure = allocate_matrix(i);
		matrix_t output_baseline = allocate_matrix(i);

		for (long j = 0; j < repeats; j++) {
			init_matrix(&input_lhs);
			init_matrix(&input_rhs);

			// run procedure and time
			long start = nanotime_now();
			procedure(input_lhs, input_rhs, output_procedure);
			deltas[j] = nanotime_interval(start, nanotime_now(), nanotime_now_max());

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

		// print stats
		benchmark_print_stats(deltas, repeats);
		free(deltas);
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
