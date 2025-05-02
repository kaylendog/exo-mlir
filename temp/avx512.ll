define dso_local void @mm512_setzero_ps(ptr nocapture noundef writeonly %dst) local_unnamed_addr #0 {
entry:
  store <16 x float> zeroinitializer, ptr %dst, align 64
  ret void
}

define dso_local void @mm512_add_ps(ptr nocapture noundef writeonly %out, <16 x float> noundef %x, <16 x float> noundef %y) local_unnamed_addr #0 {
entry:
  %add.i = fadd <16 x float> %x, %y
  store <16 x float> %add.i, ptr %out, align 64
  ret void
}

define dso_local void @mm512_mask_add_ps(i32 noundef %N, ptr nocapture noundef %out, <16 x float> noundef %x, <16 x float> noundef %y) local_unnamed_addr #1 {
entry:
  %notmask = shl nsw i32 -1, %N
  %0 = trunc i32 %notmask to i16
  %conv = xor i16 %0, -1
  %1 = load <16 x float>, ptr %out, align 64
  %add.i.i = fadd <16 x float> %x, %y
  %2 = bitcast i16 %conv to <16 x i1>
  %3 = select <16 x i1> %2, <16 x float> %add.i.i, <16 x float> %1
  store <16 x float> %3, ptr %out, align 64
  ret void
}

define dso_local void @mm512_loadu_ps(ptr nocapture noundef writeonly %dst, ptr nocapture noundef readonly %src) local_unnamed_addr #1 {
entry:
  %0 = load <16 x float>, ptr %src, align 1
  store <16 x float> %0, ptr %dst, align 64
  ret void
}

define dso_local void @mm512_storeu_ps(ptr nocapture noundef writeonly %dst, <16 x float> noundef %src) local_unnamed_addr #0 {
entry:
  store <16 x float> %src, ptr %dst, align 1
  ret void
}

define dso_local void @mm512_maskz_loadu_ps(i32 noundef %N, ptr nocapture noundef writeonly %dst, ptr nocapture noundef readonly %src) local_unnamed_addr #1 {
entry:
  %notmask = shl nsw i32 -1, %N
  %0 = trunc i32 %notmask to i16
  %conv = xor i16 %0, -1
  %1 = bitcast i16 %conv to <16 x i1>
  %2 = tail call <16 x float> @llvm.masked.load.v16f32.p0(ptr %src, i32 1, <16 x i1> %1, <16 x float> zeroinitializer)
  store <16 x float> %2, ptr %dst, align 64
  ret void
}

define dso_local void @mm512_mask_storeu_ps(i32 noundef %N, ptr nocapture noundef writeonly %dst, <16 x float> noundef %src) local_unnamed_addr #0 {
entry:
  %notmask = shl nsw i32 -1, %N
  %0 = trunc i32 %notmask to i16
  %conv = xor i16 %0, -1
  %1 = bitcast i16 %conv to <16 x i1>
  tail call void @llvm.masked.store.v16f32.p0(<16 x float> %src, ptr %dst, i32 1, <16 x i1> %1)
  ret void
}

define dso_local void @mm512_fmadd_ps(<16 x float> noundef %A, <16 x float> noundef %B, ptr nocapture noundef %C) local_unnamed_addr #1 {
entry:
  %0 = load <16 x float>, ptr %C, align 64
  %1 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %A, <16 x float> %B, <16 x float> %0)
  store <16 x float> %1, ptr %C, align 64
  ret void
}

define dso_local void @mm512_mask_fmadd_ps(i32 noundef %N, <16 x float> noundef %A, <16 x float> noundef %B, ptr nocapture noundef %C) local_unnamed_addr #1 {
entry:
  %notmask = shl nsw i32 -1, %N
  %0 = trunc i32 %notmask to i16
  %conv = xor i16 %0, -1
  %1 = load <16 x float>, ptr %C, align 64
  %2 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %A, <16 x float> %B, <16 x float> %1)
  %3 = bitcast i16 %conv to <16 x i1>
  %4 = select <16 x i1> %3, <16 x float> %2, <16 x float> %A
  store <16 x float> %4, ptr %C, align 64
  ret void
}

define dso_local void @mm512_set1_ps(ptr nocapture noundef writeonly %dst, ptr nocapture noundef readonly %src) local_unnamed_addr #1 {
entry:
  %0 = load float, ptr %src, align 4
  %vecinit.i = insertelement <16 x float> poison, float %0, i64 0
  %vecinit15.i = shufflevector <16 x float> %vecinit.i, <16 x float> poison, <16 x i32> zeroinitializer
  store <16 x float> %vecinit15.i, ptr %dst, align 64
  ret void
}

declare <16 x float> @llvm.masked.load.v16f32.p0(ptr nocapture, i32 immarg, <16 x i1>, <16 x float>) #2

declare void @llvm.masked.store.v16f32.p0(<16 x float>, ptr nocapture, i32 immarg, <16 x i1>) #3

declare <16 x float> @llvm.fma.v16f32(<16 x float>, <16 x float>, <16 x float>) #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "min-legal-vector-width"="512" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+avx512f,+cmov,+crc32,+cx8,+evex512,+f16c,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "min-legal-vector-width"="512" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+avx512f,+cmov,+crc32,+cx8,+evex512,+f16c,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="generic" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
