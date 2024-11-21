func.func @sgemm(%c : memref<64x64xf32>, %a : memref<64x64xf32>, %b : memref<64x64xf32>) {
    %zero = arith.constant 0 : index
    %size = arith.constant 64 : index
    %step = arith.constant 1 : index

    scf.for %i = %zero to %size step %step {
        scf.for %j = %zero to %size step %step {
            %sum_0 = arith.constant 0.0 : f32
            %sum = scf.for %k = %zero to %size step %size iter_args(%sum_iter = %sum_0) -> (f32) {
                %a_ik = memref.load %a[%i, %k] : memref<64x64xf32>
                %b_kj = memref.load %b[%k, %j] : memref<64x64xf32>
                %mul = arith.mulf %a_ik, %b_kj : f32
                %sum_next = arith.addf %sum_iter, %mul : f32
                scf.yield %sum_next : f32
            }
            memref.store %sum, %c[%i, %j] : memref<64x64xf32>
        }
    }
    return
}
