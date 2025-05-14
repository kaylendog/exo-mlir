from __future__ import annotations


from exo import proc
from exo.libs.memories import *


@proc
def conv1d_lt_32768(
    N: size,
    data: i32[4, 32768],
    kernels: i32[4, 4, 4],
    out: i32[4, 32768],
):
    assert N <= 32768

    # do the convolution
    for i in seq(0, 4):
        for j in seq(0, 16):
            # zero out the result memory
            out[i, j] = 0
            for c in seq(0, 4):
                for r in seq(0, 4):
                    y: i32
                    if j + r < 16:
                        y = data[c, j + r]
                    else:
                        y = 0
                    out[i, j] += kernels[i, c, r] * y
