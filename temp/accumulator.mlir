#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  llvm.func @matmul_base(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}, %arg3: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = ["sspstrong", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(16 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %6 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg2, %8 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg3, %9 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %14 = llvm.call @malloc(%1) : (i64) -> !llvm.ptr
    llvm.store %14, %10 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %2, %11 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb11
    %15 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %16 = llvm.icmp "slt" %15, %3 : i64
    llvm.cond_br %16, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.store %2, %12 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb9
    %17 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %18 = llvm.icmp "slt" %17, %3 : i64
    llvm.cond_br %18, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    %19 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %20 = llvm.getelementptr inbounds %19[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %5, %20 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.store %2, %13 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb7
    %21 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> i64
    %22 = llvm.icmp "slt" %21, %3 : i64
    llvm.cond_br %22, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %23 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %24 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %25 = llvm.mul %24, %3 overflow<nsw> : i64
    %26 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> i64
    %27 = llvm.add %25, %26 overflow<nsw> : i64
    %28 = llvm.getelementptr inbounds %23[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr -> f32
    %30 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %31 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> i64
    %32 = llvm.mul %31, %3 overflow<nsw> : i64
    %33 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %34 = llvm.add %32, %33 overflow<nsw> : i64
    %35 = llvm.getelementptr inbounds %30[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %36 = llvm.load %35 {alignment = 4 : i64} : !llvm.ptr -> f32
    %37 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %38 = llvm.getelementptr inbounds %37[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %39 = llvm.load %38 {alignment = 4 : i64} : !llvm.ptr -> f32
    %40 = llvm.intr.fmuladd(%29, %36, %39)  : (f32, f32, f32) -> f32
    llvm.store %40, %38 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    %41 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> i64
    %42 = llvm.add %41, %4 overflow<nsw> : i64
    llvm.store %42, %13 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb5 {loop_annotation = #loop_annotation}
  ^bb8:  // pred: ^bb5
    %43 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %44 = llvm.getelementptr inbounds %43[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %45 = llvm.load %44 {alignment = 4 : i64} : !llvm.ptr -> f32
    %46 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %47 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %48 = llvm.mul %47, %3 overflow<nsw> : i64
    %49 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %50 = llvm.add %48, %49 overflow<nsw> : i64
    %51 = llvm.getelementptr inbounds %46[%50] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.load %51 {alignment = 4 : i64} : !llvm.ptr -> f32
    %53 = llvm.fadd %52, %45  : f32
    llvm.store %53, %51 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^bb9
  ^bb9:  // pred: ^bb8
    %54 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %55 = llvm.add %54, %4 overflow<nsw> : i64
    llvm.store %55, %12 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb3 {loop_annotation = #loop_annotation}
  ^bb10:  // pred: ^bb3
    llvm.br ^bb11
  ^bb11:  // pred: ^bb10
    %56 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %57 = llvm.add %56, %4 overflow<nsw> : i64
    llvm.store %57, %11 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb12:  // pred: ^bb1
    %58 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    llvm.call @free(%58) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_unwind, passthrough = [["allocsize", "4294967295"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
  llvm.func @free(!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
}
