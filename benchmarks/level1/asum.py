from __future__ import annotations

from exo import *
from exo.platforms.x86 import *
from exo.libs.externs import select

from exoblas.codegen_helpers import *
from exoblas.blaslib import *


@proc
def asum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])


variants_generator(optimize_level_1, targets=("avx2"), opt_precisions=("f32"))(
    asum, "i", 8, globals=globals()
)
