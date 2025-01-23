# RUN: EXO-FILECHECK

from __future__ import annotations
from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @binary_noop(%0 : memref<16xf32, strided<[1]>>, %1 : memref<16xf32, strided<[1]>>) {
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def binary_noop(x: f32[16], y: f32[16]):
    pass
