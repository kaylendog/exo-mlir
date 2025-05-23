from __future__ import annotations

from exo import *

from exoblas.codegen_helpers import *
from exoblas.blaslib import *


@proc
def scal(n: size, alpha: R, x: [R][n]):
    for i in seq(0, n):
        x[i] = alpha * x[i]


@proc
def scal_alpha_0(n: size, x: [R][n]):
    for i in seq(0, n):
        x[i] = 0.0


for proc in scal, scal_alpha_0:
    variants_generator(optimize_level_1, targets=("avx2"), opt_precisions=("f32"))(
        proc, "i", 4, globals=globals()
    )
