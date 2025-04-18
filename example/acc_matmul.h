
#pragma once
#ifndef ACC_MATMUL_H
#define ACC_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif



// matmul_base(
//     M : size,
//     N : size,
//     K : size,
//     C : f32[M, N] @DRAM,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM
// )
void matmul_base( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, float* C, const float* A, const float* B );



#ifdef __cplusplus
}
#endif
#endif  // ACC_MATMUL_H
