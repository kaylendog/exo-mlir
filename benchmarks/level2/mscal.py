from __future__ import annotations

from exo import *

from exoblas.codegen_helpers import *
from exoblas.blaslib import *


@proc
def mscal_rm(M: size, N: size, alpha: R, A: [R][M, N]):
    assert stride(A, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            A[i, j] = A[i, j] * alpha


@proc
def trmscal_rm(Uplo: size, N: size, alpha: R, A: [R][N, N]):
    assert stride(A, 1) == 1

    for i in seq(0, N):
        for j in seq(0, N):
            if (Uplo == CblasUpperValue and j >= i) or (
                Uplo == CblasLowerValue and j < i + 1
            ):
                A[i, j] = A[i, j] * alpha


variants_generator(optimize_level_1)(trmscal_rm, "j", 4, globals=globals())
variants_generator(optimize_level_1)(mscal_rm, "j", 4, globals=globals())
