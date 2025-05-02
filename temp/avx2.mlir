module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func local_unnamed_addr @mm256_setzero_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    llvm.store %1, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_setzero_pd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(dense<0.000000e+00> : vector<4xf64>) : vector<4xf64>
    llvm.store %1, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_loadu_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.load %arg1 {alignment = 1 : i64} : !llvm.ptr -> vector<8xf32>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_loadu_pd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.load %arg1 {alignment = 1 : i64} : !llvm.ptr -> vector<4xf64>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_storeu_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<8xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    llvm.store %arg1, %arg0 {alignment = 1 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_storeu_pd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<4xf64> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    llvm.store %arg1, %arg0 {alignment = 1 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_fmadd_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: vector<8xf32> {llvm.noundef}, %arg2: vector<8xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.load %arg0 {alignment = 32 : i64} : !llvm.ptr -> vector<8xf32>
    %1 = llvm.intr.fma(%arg1, %arg2, %0)  : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    llvm.store %1, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_fmadd_pd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: vector<4xf64> {llvm.noundef}, %arg2: vector<4xf64> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.load %arg0 {alignment = 32 : i64} : !llvm.ptr -> vector<4xf64>
    %1 = llvm.intr.fma(%arg1, %arg2, %0)  : (vector<4xf64>, vector<4xf64>, vector<4xf64>) -> vector<4xf64>
    llvm.store %1, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_broadcast_ss(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.poison : vector<8xf32>
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.load %arg1 {alignment = 1 : i64} : !llvm.ptr -> f32
    %3 = llvm.insertelement %2, %0[%1 : i64] : vector<8xf32>
    %4 = llvm.shufflevector %3, %0 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32> 
    llvm.store %4, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_broadcast_sd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.poison : vector<4xf64>
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.load %arg1 {alignment = 1 : i64} : !llvm.ptr -> f64
    %3 = llvm.insertelement %2, %0[%1 : i64] : vector<4xf64>
    %4 = llvm.shufflevector %3, %0 [0, 0, 0, 0] : vector<4xf64> 
    llvm.store %4, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_broadcast_ss_scalar(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: f32 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.poison : vector<8xf32>
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.insertelement %arg1, %0[%1 : i64] : vector<8xf32>
    %3 = llvm.shufflevector %2, %0 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32> 
    llvm.store %3, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_broadcast_sd_scalar(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.poison : vector<4xf64>
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.insertelement %arg1, %0[%1 : i64] : vector<4xf64>
    %3 = llvm.shufflevector %2, %0 [0, 0, 0, 0] : vector<4xf64> 
    llvm.store %3, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_fmadd_ps_broadcast(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: vector<8xf32> {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.poison : vector<8xf32>
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.load %arg2 {alignment = 1 : i64} : !llvm.ptr -> f32
    %3 = llvm.insertelement %2, %0[%1 : i64] : vector<8xf32>
    %4 = llvm.shufflevector %3, %0 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32> 
    %5 = llvm.load %arg0 {alignment = 32 : i64} : !llvm.ptr -> vector<8xf32>
    %6 = llvm.intr.fma(%arg1, %4, %5)  : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    llvm.store %6, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_mul_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<8xf32> {llvm.noundef}, %arg2: vector<8xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fmul %arg1, %arg2  : vector<8xf32>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_mul_pd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<4xf64> {llvm.noundef}, %arg2: vector<4xf64> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fmul %arg1, %arg2  : vector<4xf64>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_div_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<8xf32> {llvm.noundef}, %arg2: vector<8xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fdiv %arg1, %arg2  : vector<8xf32>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_div_pd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<4xf64> {llvm.noundef}, %arg2: vector<4xf64> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fdiv %arg1, %arg2  : vector<4xf64>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_add_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<8xf32> {llvm.noundef}, %arg2: vector<8xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fadd %arg1, %arg2  : vector<8xf32>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_add_pd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<4xf64> {llvm.noundef}, %arg2: vector<4xf64> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fadd %arg1, %arg2  : vector<4xf64>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_sub_ps(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<8xf32> {llvm.noundef}, %arg2: vector<8xf32> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fsub %arg1, %arg2  : vector<8xf32>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<8xf32>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_sub_pd(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<4xf64> {llvm.noundef}, %arg2: vector<4xf64> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.fsub %arg1, %arg2  : vector<4xf64>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<4xf64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_loadu_si256(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.load %arg1 {alignment = 1 : i64} : !llvm.ptr -> vector<4xi64>
    llvm.store %0, %arg0 {alignment = 32 : i64} : vector<4xi64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_storeu_si256(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<4xi64> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    llvm.store %arg1, %arg0 {alignment = 1 : i64} : vector<4xi64>, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @mm256_add_epi16(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg1: vector<4xi64> {llvm.noundef}, %arg2: vector<4xi64> {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "256"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+avx", "+avx2", "+cmov", "+crc32", "+cx8", "+fma", "+fxsr", "+mmx", "+popcnt", "+sse", "+sse2", "+sse3", "+sse4.1", "+sse4.2", "+ssse3", "+x87", "+xsave"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.bitcast %arg1 : vector<4xi64> to vector<16xi16>
    %1 = llvm.bitcast %arg2 : vector<4xi64> to vector<16xi16>
    %2 = llvm.intr.uadd.sat(%0, %1)  : (vector<16xi16>, vector<16xi16>) -> vector<16xi16>
    llvm.store %2, %arg0 {alignment = 32 : i64} : vector<16xi16>, !llvm.ptr
    llvm.return
  }
}

