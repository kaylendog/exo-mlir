define dso_local void @mm256_setzero_ps(ptr nocapture noundef writeonly %dst) local_unnamed_addr #0 {
entry:
  store <8 x float> zeroinitializer, ptr %dst, align 32
  ret void
}

define dso_local void @mm256_setzero_pd(ptr nocapture noundef writeonly %dst) local_unnamed_addr #0 {
entry:
  store <4 x double> zeroinitializer, ptr %dst, align 32
  ret void
}

define dso_local void @mm256_loadu_ps(ptr nocapture noundef writeonly %dst, ptr nocapture noundef readonly %src) local_unnamed_addr #1 {
entry:
  %0 = load <8 x float>, ptr %src, align 1
  store <8 x float> %0, ptr %dst, align 32
  ret void
}

define dso_local void @mm256_loadu_pd(ptr nocapture noundef writeonly %dst, ptr nocapture noundef readonly %src) local_unnamed_addr #1 {
entry:
  %0 = load <4 x double>, ptr %src, align 1
  store <4 x double> %0, ptr %dst, align 32
  ret void
}

define dso_local void @mm256_storeu_ps(ptr nocapture noundef writeonly %dst, <8 x float> noundef %src) local_unnamed_addr #0 {
entry:
  store <8 x float> %src, ptr %dst, align 1
  ret void
}

define dso_local void @mm256_storeu_pd(ptr nocapture noundef writeonly %dst, <4 x double> noundef %src) local_unnamed_addr #0 {
entry:
  store <4 x double> %src, ptr %dst, align 1
  ret void
}

define dso_local void @mm256_fmadd_ps(ptr nocapture noundef %dst, <8 x float> noundef %src1, <8 x float> noundef %src2) local_unnamed_addr #1 {
entry:
  %0 = load <8 x float>, ptr %dst, align 32
  %1 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %src1, <8 x float> %src2, <8 x float> %0)
  store <8 x float> %1, ptr %dst, align 32
  ret void
}

define dso_local void @mm256_fmadd_pd(ptr nocapture noundef %dst, <4 x double> noundef %src1, <4 x double> noundef %src2) local_unnamed_addr #1 {
entry:
  %0 = load <4 x double>, ptr %dst, align 32
  %1 = tail call <4 x double> @llvm.fma.v4f64(<4 x double> %src1, <4 x double> %src2, <4 x double> %0)
  store <4 x double> %1, ptr %dst, align 32
  ret void
}

define dso_local void @mm256_broadcast_ss(ptr nocapture noundef writeonly %out, ptr nocapture noundef readonly %val) local_unnamed_addr #1 {
entry:
  %0 = load float, ptr %val, align 1
  %vecinit.i = insertelement <8 x float> poison, float %0, i64 0
  %vecinit8.i = shufflevector <8 x float> %vecinit.i, <8 x float> poison, <8 x i32> zeroinitializer
  store <8 x float> %vecinit8.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_broadcast_sd(ptr nocapture noundef writeonly %out, ptr nocapture noundef readonly %val) local_unnamed_addr #1 {
entry:
  %0 = load double, ptr %val, align 1
  %vecinit.i = insertelement <4 x double> poison, double %0, i64 0
  %vecinit4.i = shufflevector <4 x double> %vecinit.i, <4 x double> poison, <4 x i32> zeroinitializer
  store <4 x double> %vecinit4.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_broadcast_ss_scalar(ptr nocapture noundef writeonly %out, float noundef %val) local_unnamed_addr #0 {
entry:
  %vecinit.i = insertelement <8 x float> poison, float %val, i64 0
  %vecinit8.i = shufflevector <8 x float> %vecinit.i, <8 x float> poison, <8 x i32> zeroinitializer
  store <8 x float> %vecinit8.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_broadcast_sd_scalar(ptr nocapture noundef writeonly %out, double noundef %val) local_unnamed_addr #0 {
entry:
  %vecinit.i = insertelement <4 x double> poison, double %val, i64 0
  %vecinit4.i = shufflevector <4 x double> %vecinit.i, <4 x double> poison, <4 x i32> zeroinitializer
  store <4 x double> %vecinit4.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_fmadd_ps_broadcast(ptr nocapture noundef %dst, <8 x float> noundef %lhs, ptr nocapture noundef readonly %rhs) local_unnamed_addr #1 {
entry:
  %0 = load float, ptr %rhs, align 1
  %vecinit.i = insertelement <8 x float> poison, float %0, i64 0
  %vecinit8.i = shufflevector <8 x float> %vecinit.i, <8 x float> poison, <8 x i32> zeroinitializer
  %1 = load <8 x float>, ptr %dst, align 32
  %2 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %lhs, <8 x float> %vecinit8.i, <8 x float> %1)
  store <8 x float> %2, ptr %dst, align 32
  ret void
}

define dso_local void @mm256_mul_ps(ptr nocapture noundef writeonly %out, <8 x float> noundef %x, <8 x float> noundef %y) local_unnamed_addr #0 {
entry:
  %mul.i = fmul <8 x float> %x, %y
  store <8 x float> %mul.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_mul_pd(ptr nocapture noundef writeonly %out, <4 x double> noundef %x, <4 x double> noundef %y) local_unnamed_addr #0 {
entry:
  %mul.i = fmul <4 x double> %x, %y
  store <4 x double> %mul.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_div_ps(ptr nocapture noundef writeonly %out, <8 x float> noundef %x, <8 x float> noundef %y) local_unnamed_addr #0 {
entry:
  %div.i = fdiv <8 x float> %x, %y
  store <8 x float> %div.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_div_pd(ptr nocapture noundef writeonly %out, <4 x double> noundef %x, <4 x double> noundef %y) local_unnamed_addr #0 {
entry:
  %div.i = fdiv <4 x double> %x, %y
  store <4 x double> %div.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_add_ps(ptr nocapture noundef writeonly %out, <8 x float> noundef %x, <8 x float> noundef %y) local_unnamed_addr #0 {
entry:
  %add.i = fadd <8 x float> %x, %y
  store <8 x float> %add.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_add_pd(ptr nocapture noundef writeonly %out, <4 x double> noundef %x, <4 x double> noundef %y) local_unnamed_addr #0 {
entry:
  %add.i = fadd <4 x double> %x, %y
  store <4 x double> %add.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_sub_ps(ptr nocapture noundef writeonly %out, <8 x float> noundef %x, <8 x float> noundef %y) local_unnamed_addr #0 {
entry:
  %sub.i = fsub <8 x float> %x, %y
  store <8 x float> %sub.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_sub_pd(ptr nocapture noundef writeonly %out, <4 x double> noundef %x, <4 x double> noundef %y) local_unnamed_addr #0 {
entry:
  %sub.i = fsub <4 x double> %x, %y
  store <4 x double> %sub.i, ptr %out, align 32
  ret void
}

define dso_local void @mm256_loadu_si256(ptr nocapture noundef writeonly %dst, ptr nocapture noundef readonly %src) local_unnamed_addr #1 {
entry:
  %0 = load <4 x i64>, ptr %src, align 1
  store <4 x i64> %0, ptr %dst, align 32
  ret void
}

define dso_local void @mm256_storeu_si256(ptr nocapture noundef writeonly %dst, <4 x i64> noundef %src) local_unnamed_addr #0 {
entry:
  store <4 x i64> %src, ptr %dst, align 1
  ret void
}

define dso_local void @mm256_add_epi16(ptr nocapture noundef writeonly %out, <4 x i64> noundef %x, <4 x i64> noundef %y) local_unnamed_addr #0 {
entry:
  %0 = bitcast <4 x i64> %x to <16 x i16>
  %1 = bitcast <4 x i64> %y to <16 x i16>
  %elt.sat.i = tail call <16 x i16> @llvm.uadd.sat.v16i16(<16 x i16> %0, <16 x i16> %1)
  store <16 x i16> %elt.sat.i, ptr %out, align 32
  ret void
}

declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #2

declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>) #2

declare <16 x i16> @llvm.uadd.sat.v16i16(<16 x i16>, <16 x i16>) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
