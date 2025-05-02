; ModuleID = './accumulator/accumulator.c'
source_filename = "./accumulator/accumulator.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: noinline nounwind optnone sspstrong uwtable
define dso_local void @matmul_base(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca float, align 4
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  %12 = alloca i64, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  store ptr %3, ptr %8, align 8
  store i64 0, ptr %10, align 8
  br label %13

13:                                               ; preds = %58, %4
  %14 = load i64, ptr %10, align 8
  %15 = icmp slt i64 %14, 16
  br i1 %15, label %16, label %61

16:                                               ; preds = %13
  store i64 0, ptr %11, align 8
  br label %17

17:                                               ; preds = %54, %16
  %18 = load i64, ptr %11, align 8
  %19 = icmp slt i64 %18, 16
  br i1 %19, label %20, label %57

20:                                               ; preds = %17
  store float 0.000000e+00, ptr %9, align 4
  store i64 0, ptr %12, align 8
  br label %21

21:                                               ; preds = %41, %20
  %22 = load i64, ptr %12, align 8
  %23 = icmp slt i64 %22, 16
  br i1 %23, label %24, label %44

24:                                               ; preds = %21
  %25 = load ptr, ptr %7, align 8
  %26 = load i64, ptr %10, align 8
  %27 = mul nsw i64 %26, 16
  %28 = load i64, ptr %12, align 8
  %29 = add nsw i64 %27, %28
  %30 = getelementptr inbounds float, ptr %25, i64 %29
  %31 = load float, ptr %30, align 4
  %32 = load ptr, ptr %8, align 8
  %33 = load i64, ptr %12, align 8
  %34 = mul nsw i64 %33, 16
  %35 = load i64, ptr %11, align 8
  %36 = add nsw i64 %34, %35
  %37 = getelementptr inbounds float, ptr %32, i64 %36
  %38 = load float, ptr %37, align 4
  %39 = load float, ptr %9, align 4
  %40 = call float @llvm.fmuladd.f32(float %31, float %38, float %39)
  store float %40, ptr %9, align 4
  br label %41

41:                                               ; preds = %24
  %42 = load i64, ptr %12, align 8
  %43 = add nsw i64 %42, 1
  store i64 %43, ptr %12, align 8
  br label %21, !llvm.loop !6

44:                                               ; preds = %21
  %45 = load float, ptr %9, align 4
  %46 = load ptr, ptr %6, align 8
  %47 = load i64, ptr %10, align 8
  %48 = mul nsw i64 %47, 16
  %49 = load i64, ptr %11, align 8
  %50 = add nsw i64 %48, %49
  %51 = getelementptr inbounds float, ptr %46, i64 %50
  %52 = load float, ptr %51, align 4
  %53 = fadd float %52, %45
  store float %53, ptr %51, align 4
  br label %54

54:                                               ; preds = %44
  %55 = load i64, ptr %11, align 8
  %56 = add nsw i64 %55, 1
  store i64 %56, ptr %11, align 8
  br label %17, !llvm.loop !8

57:                                               ; preds = %17
  br label %58

58:                                               ; preds = %57
  %59 = load i64, ptr %10, align 8
  %60 = add nsw i64 %59, 1
  store i64 %60, ptr %10, align 8
  br label %13, !llvm.loop !9

61:                                               ; preds = %13
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { noinline nounwind optnone sspstrong uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 19.1.7"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
