from __future__ import annotations

from exo import *


@proc
def swap(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        tmp: R
        tmp = x[i]
        x[i] = y[i]
        y[i] = tmp
