/// @brief General purpose matrix structure. Stores width and height in addition to data.
typedef struct matrix {
	long width;
	long height;
	float *data;
} matrix_t;

/// @brief A matrix allocation function.
typedef matrix_t (*matrix_alloc_t)(long);

/// @brief A matrix initialization function.
typedef void (*matrix_init_t)(matrix_t *);

/// @brief A unary procedure that takes an input matrix and produces an output matrix.
typedef void (*unary_procedure_t)(matrix_t, matrix_t);

/// @brief A binary procedure that takes two input matrices and produces an output matrix.
typedef void (*binary_procedure_t)(matrix_t, matrix_t, matrix_t);

/// @brief Benchmarks a unary procedure over a range of matrix sizes, printing the results as CSV to stdout.
void benchmark_unary_procedure(char *name, long iterations, long repeats, matrix_alloc_t allocate_matrix,
							   matrix_init_t init_matrix, unary_procedure_t baseline, unary_procedure_t procedure);

/// @brief Benchmarks a binary procedure over a range of matrix sizes, printing the results as CSV to stdout.
void benchmark_binary_procedure(char *name, long iterations, long repeats, matrix_alloc_t allocate_matrix,
								matrix_init_t init_matrix, binary_procedure_t baseline, binary_procedure_t procedure);

/// @brief Generate random values for a matrix.
matrix_t matrix_alloc_square(long iteration);

/// @brief Initialize a matrix with uniformly distributed random values.
void matrix_init_uniform(matrix_t *mat);
