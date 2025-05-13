func.func @gemm(%0 : i32, %1 : i32, %2 : i32, %3 : memref<?x?xf32, "DRAM">, %4 : memref<?x?xf32, "DRAM">, %5 : memref<?x?xf32, "DRAM">) {
    %6 = arith.constant 0 : i32
    %7 = index.casts %6 : i32 to index
    %8 = exo.read %0 -> i32
    %9 = index.casts %8 : i32 to index
    %10 = arith.constant 1 : index
    scf.for %11 = %7 to %9 step %10 {
        %12 = arith.constant 0 : i32
        %13 = index.casts %12 : i32 to index
        %14 = exo.read %2 -> i32
        %15 = index.casts %14 : i32 to index
        %16 = arith.constant 1 : index
        scf.for %17 = %13 to %15 step %16 {
        %18 = exo.read %11 -> i32
        %19 = exo.read %17 -> i32
        %20 = index.casts %18 : i32 to index
        %21 = index.casts %19 : i32 to index
        %22 = arith.constant 0.000000e+00 : f32
        exo.assign %3[%20, %21], %22 : memref<?x?xf32, "DRAM">, f32
        %23 = arith.constant 0 : i32
        %24 = index.casts %23 : i32 to index
        %25 = exo.read %1 -> i32
        %26 = index.casts %25 : i32 to index
        %27 = arith.constant 1 : index
        scf.for %28 = %24 to %26 step %27 {
            %29 = exo.read %11 -> i32
            %30 = exo.read %17 -> i32
            %31 = index.casts %29 : i32 to index
            %32 = index.casts %30 : i32 to index
            %33 = exo.read %11 -> i32
            %34 = exo.read %28 -> i32
            %35 = index.casts %33 : i32 to index
            %36 = index.casts %34 : i32 to index
            %37 = exo.read %4 [(%35, %36)] -> f32
            %38 = exo.read %28 -> i32
            %39 = exo.read %17 -> i32
            %40 = index.casts %38 : i32 to index
            %41 = index.casts %39 : i32 to index
            %42 = exo.read %5 [(%40, %41)] -> f32
            %43 = arith.mulf %37, %42 : f32
            exo.reduce %3[%31, %32], %43 : memref<?x?xf32, "DRAM">, f32
        }
        }
    }
    func.return
}
