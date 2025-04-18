# example.py
from __future__ import annotations
from exo import *


@proc
def matmul_base(
    M: size,
    N: size,
    K: size,
    C: f32[M, N] @ DRAM,
    A: f32[M, K] @ DRAM,
    B: f32[K, N] @ DRAM,
):
    acc: f32 @ DRAM
    for i in seq(0, M):
        for j in seq(0, N):
            acc = 0.0
            for k in seq(0, K):
                acc += A[i, k] * B[k, j]

            C[i, j] += acc
