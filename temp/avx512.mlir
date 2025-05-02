module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func local_unnamed_addr @mm512_setzero_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(dense<0.000000e+00> : vector<16xf32>) : vector<16xf32>
    llvm.store %1, %arg0 {alignment = 64 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_add_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<16xf32> {llvm.noundef}, %arg2: vector<16xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fadd %arg1, %arg2  : vector<16xf32>
    llvm.store %0, %arg0 {alignment = 64 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_mask_add_ps(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: vector<16xf32> {llvm.noundef}, %arg3: vector<16xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.constant(-1 : i16) : i16
    %2 = llvm.shl %0, %arg0 overflow<nsw> : i32
    %3 = llvm.trunc %2 : i32 to i16
    %4 = llvm.xor %3, %1  : i16
    %5 = llvm.load %arg1 {alignment = 64 : i64} : !llvm.ptr -> vector<16xf32>
    %6 = llvm.fadd %arg2, %arg3  : vector<16xf32>
    %7 = llvm.bitcast %4 : i16 to vector<16xi1>
    %8 = llvm.select %7, %6, %5 : vector<16xi1>, vector<16xf32>
    llvm.store %8, %arg1 {alignment = 64 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_loadu_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.load %arg1 {alignment = 1 : i64} : !llvm.ptr -> vector<16xf32>
    llvm.store %0, %arg0 {alignment = 64 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_storeu_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<16xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    llvm.store %arg1, %arg0 {alignment = 1 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_maskz_loadu_ps(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.constant(-1 : i16) : i16
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(dense<0.000000e+00> : vector<16xf32>) : vector<16xf32>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.shl %0, %arg0 overflow<nsw> : i32
    %6 = llvm.trunc %5 : i32 to i16
    %7 = llvm.xor %6, %1  : i16
    %8 = llvm.bitcast %7 : i16 to vector<16xi1>
    %9 = llvm.intr.masked.load %arg2, %8, %3 {alignment = 1 : i32} : (!llvm.ptr, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
    llvm.store %9, %arg1 {alignment = 64 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_mask_storeu_ps(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg2: vector<16xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.constant(-1 : i16) : i16
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.shl %0, %arg0 overflow<nsw> : i32
    %4 = llvm.trunc %3 : i32 to i16
    %5 = llvm.xor %4, %1  : i16
    %6 = llvm.bitcast %5 : i16 to vector<16xi1>
    llvm.intr.masked.store %arg2, %arg1, %6 {alignment = 1 : i32} : vector<16xf32>, vector<16xi1> into !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_fmadd_ps(%arg0: vector<16xf32> {llvm.noundef}, %arg1: vector<16xf32> {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.load %arg2 {alignment = 64 : i64} : !llvm.ptr -> vector<16xf32>
    %1 = llvm.intr.fma(%arg0, %arg1, %0)  : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    llvm.store %1, %arg2 {alignment = 64 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_mask_fmadd_ps(%arg0: i32 {llvm.noundef}, %arg1: vector<16xf32> {llvm.noundef}, %arg2: vector<16xf32> {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.constant(-1 : i16) : i16
    %2 = llvm.shl %0, %arg0 overflow<nsw> : i32
    %3 = llvm.trunc %2 : i32 to i16
    %4 = llvm.xor %3, %1  : i16
    %5 = llvm.load %arg3 {alignment = 64 : i64} : !llvm.ptr -> vector<16xf32>
    %6 = llvm.intr.fma(%arg1, %arg2, %5)  : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %7 = llvm.bitcast %4 : i16 to vector<16xi1>
    %8 = llvm.select %7, %6, %arg1 : vector<16xi1>, vector<16xf32>
    llvm.store %8, %arg3 {alignment = 64 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm512_set1_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "512"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+avx512f", "+cmov", "+crc32", "+cx8", "+evex512", "+f16c", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.poison : vector<16xf32>
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.load %arg1 {alignment = 4 : i64} : !llvm.ptr -> f32
    %3 = llvm.insertelement %2, %0[%1 : i64] : vector<16xf32>
    %4 = llvm.shufflevector %3, %0 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32> 
    llvm.store %4, %arg0 {alignment = 64 : i64} : vector<16xf32>, !llvm.ptr
    llvm.return
  }
}

