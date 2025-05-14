from __future__ import annotations


from exo import proc
from exo.libs.memories import *


@proc
def gemm_lt_128(
    N: size,
    M: size,
    K: size,
    out: f32[128, 128] @ DRAM,
    a: f32[128, 128] @ DRAM,
    b: f32[128, 128] @ DRAM,
):
    assert N <= 128
    assert M <= 128
    assert K <= 128

    for i in seq(0, N):
        for j in seq(0, K):
            out[i, j] = 0.0
            for k in seq(0, M):
                out[i, j] += a[i, k] * b[k, j]
