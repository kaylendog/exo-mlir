#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func @matmul_base(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}, %arg3: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = ["sspstrong", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(16 : i64) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %5 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %6 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg2, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg3, %8 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %1, %10 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb11
    %13 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> i64
    %14 = llvm.icmp "slt" %13, %2 : i64
    llvm.cond_br %14, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.store %1, %11 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb9
    %15 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %16 = llvm.icmp "slt" %15, %2 : i64
    llvm.cond_br %16, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    llvm.store %4, %9 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.store %1, %12 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb7
    %17 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %18 = llvm.icmp "slt" %17, %2 : i64
    llvm.cond_br %18, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %19 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %20 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> i64
    %21 = llvm.mul %20, %2 overflow<nsw> : i64
    %22 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %23 = llvm.add %21, %22 overflow<nsw> : i64
    %24 = llvm.getelementptr inbounds %19[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %25 = llvm.load %24 {alignment = 4 : i64} : !llvm.ptr -> f32
    %26 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %27 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %28 = llvm.mul %27, %2 overflow<nsw> : i64
    %29 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %30 = llvm.add %28, %29 overflow<nsw> : i64
    %31 = llvm.getelementptr inbounds %26[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %32 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr -> f32
    %33 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> f32
    %34 = llvm.intr.fmuladd(%25, %32, %33)  : (f32, f32, f32) -> f32
    llvm.store %34, %9 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    %35 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %36 = llvm.add %35, %3 overflow<nsw> : i64
    llvm.store %36, %12 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb5 {loop_annotation = #loop_annotation}
  ^bb8:  // pred: ^bb5
    %37 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> f32
    %38 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %39 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> i64
    %40 = llvm.mul %39, %2 overflow<nsw> : i64
    %41 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %42 = llvm.add %40, %41 overflow<nsw> : i64
    %43 = llvm.getelementptr inbounds %38[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %44 = llvm.load %43 {alignment = 4 : i64} : !llvm.ptr -> f32
    %45 = llvm.fadd %44, %37  : f32
    llvm.store %45, %43 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^bb9
  ^bb9:  // pred: ^bb8
    %46 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %47 = llvm.add %46, %3 overflow<nsw> : i64
    llvm.store %47, %11 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb3 {loop_annotation = #loop_annotation}
  ^bb10:  // pred: ^bb3
    llvm.br ^bb11
  ^bb11:  // pred: ^bb10
    %48 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> i64
    %49 = llvm.add %48, %3 overflow<nsw> : i64
    llvm.store %49, %10 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb12:  // pred: ^bb1
    llvm.return
  }
}
