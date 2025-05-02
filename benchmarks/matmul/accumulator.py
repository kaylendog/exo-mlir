from __future__ import annotations
from exo import *


@proc
def matmul_base(
    C: f32[16, 16] @ DRAM,
    A: f32[16, 16] @ DRAM,
    B: f32[16, 16] @ DRAM,
):
    acc: f32[1] @ DRAM
    for i in seq(0, 16):
        for j in seq(0, 16):
            acc[0] = 0.0
            for k in seq(0, 16):
                acc[0] += A[i, k] * B[k, j]

            C[i, j] += acc[0]
