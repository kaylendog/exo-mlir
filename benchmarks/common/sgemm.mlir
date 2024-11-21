func.func @sgemm(%c : memref<?x?xf32>, %a : memref<?x?xf32>, %b : memref<?x?xf32>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index

    // get dimensions
    %sc0 = memref.dim %c, %zero : memref<?x?xf32>
    %sc1 = memref.dim %c, %one : memref<?x?xf32>
    %sa0 = memref.dim %a, %zero : memref<?x?xf32>
    %sa1 = memref.dim %a, %one : memref<?x?xf32>
    %sb0 = memref.dim %b, %zero : memref<?x?xf32>
    %sb1 = memref.dim %b, %one : memref<?x?xf32>

    // check dimensions (M x N) * (N x K) = (M x K)
    // require sa0 == sc0, sa1 == sb0, sb1 == sc1
    %precondition0 = arith.cmpi "eq", %sc0, %sa0 : index
    %precondition1 = arith.cmpi "eq", %sc1, %sb1 : index
    %precondition2 = arith.cmpi "eq", %sa1, %sb0 : index
    
    // assert dimensions
    cf.assert %precondition0, "dimension mismatch"
    cf.assert %precondition1, "dimension mismatch"
    cf.assert %precondition2, "dimension mismatch"

    affine.for %i = 0 to %sc0 {
        affine.for %j = 0 to %sc1 {
            %sum_0 = arith.constant 0.0 : f32
            %sum = affine.for %k = 0 to %sa1 iter_args(%sum_iter = %sum_0) -> (f32) {
                %a_ik = affine.load %a[%i, %k] : memref<?x?xf32>
                %b_kj = affine.load %b[%k, %j] : memref<?x?xf32>
                %mul = arith.mulf %a_ik, %b_kj : f32
                %sum_next = arith.addf %sum_iter, %mul : f32
                affine.yield %sum_next : f32
            }
            affine.store %sum, %c[%i, %j] : memref<?x?xf32>
        }
    }
    return
}
