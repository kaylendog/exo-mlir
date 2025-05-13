from __future__ import annotations

from exo import *


@proc
def copy(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] = x[i]
