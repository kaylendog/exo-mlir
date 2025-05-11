from xdsl.context import Context
from xdsl.dialects import arith, llvm, memref, vector
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    UnrealizedConversionCastOp,
    VectorType,
    f32,
    f64,
    i32,
    i64,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from exomlir.dialects import exo, llvm_intrinsics


class ConvertSelect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ExternOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "select":
            return

        assert len(op.arguments) == 4
        assert op.arguments[0].type == op.arguments[1].type, (
            f"{op.arguments[0].type} != {op.arguments[1].type}"
        )
        assert op.arguments[2].type == op.arguments[3].type, (
            f"{op.arguments[2].type} != {op.arguments[3].type}"
        )
        assert op.arguments[2].type == op.result.type, (
            f"{op.arguments[2].type} != {op.result.type}"
        )

        rewriter.replace_matched_op(
            (
                cmp_op := arith.CmpfOp(op.arguments[0], op.arguments[1], "olt"),
                arith.SelectOp(
                    cmp_op.results[0],
                    op.arguments[2],
                    op.arguments[3],
                ),
            )
        )


class ConvertVecAbsF32x8(RewritePattern):
    """
    def vec_abs_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = select(0.0, src[i], src[i], -src[i])
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_abs_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                fabs_op := llvm_intrinsics.FAbsOp(
                    load_op.result,
                    VectorType(f32, [8]),
                ),
                vector.StoreOp.get(
                    fabs_op.result,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecAbsF32x8Pfx(RewritePattern):
    """
    def vec_abs_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                      src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = select(0.0, src[i], src[i], -src[i])
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_abs_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [8]), [0, 1, 2, 3, 4, 5, 6, 7]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                fabs_op := llvm_intrinsics.FAbsOp(
                    load_op.result,
                    VectorType(f32, [8]),
                ),
                vector.StoreOp.get(
                    load_op.result,
                    op.arguments[1],
                    [zero_op.result],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    fabs_op.result,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecAbsF64x4(RewritePattern):
    """
    def vec_abs_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_and_pd({src_data}, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = select(0.0, src[i], src[i], -src[i])
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_abs_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                fabs_op := llvm_intrinsics.FAbsOp(
                    load_op.result,
                    VectorType(f64, [4]),
                ),
                vector.StoreOp.get(
                    fabs_op.result,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecAbsF64x4Pfx(RewritePattern):
    """
    def vec_abs_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                      src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_and_pd({src_data}, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = select(0.0, src[i], src[i], -src[i])
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_abs_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i64, [4]), [0, 1, 2, 3]
                    ),
                ),
                sgext_op := arith.ExtSIOp(op.arguments[0], i64),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[sgext_op.result],
                    result_types=[VectorType(i64, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                fabs_op := llvm_intrinsics.FAbsOp(
                    load_op.result,
                    VectorType(f64, [4]),
                ),
                vector.StoreOp.get(
                    load_op.result,
                    op.arguments[1],
                    [zero_op.result],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    fabs_op.result,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecAddRedF32x8(RewritePattern):
    """
    def vec_add_red_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_ps({dst_data}, {src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_red_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[0], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                add_op := llvm.FAddOp(load0_op.result, load1_op.result),
                vector.StoreOp.get(
                    add_op.res,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecAddRedF32x8Pfx(RewritePattern):
    """
    def vec_add_red_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                          src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_ps({dst_data}, {src_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_red_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [8]), [0, 1, 2, 3, 4, 5, 6, 7]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                add_op := llvm.FAddOp(load0_op.result, load1_op.result),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    add_op.res,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecAddRedF64x4(RewritePattern):
    """
    def vec_add_red_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_pd({dst_data}, {src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_red_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[0], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                add_op := llvm.FAddOp(load0_op.result, load1_op.result),
                vector.StoreOp.get(
                    add_op.res,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecAddRedF64x4Pfx(RewritePattern):
    """
    def vec_add_red_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                          src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_pd({dst_data}, {src_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_red_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i64, [4]), [0, 1, 2, 3]
                    ),
                ),
                sgext_op := arith.ExtSIOp(op.arguments[0], i64),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[sgext_op.result],
                    result_types=[VectorType(i64, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                add_op := llvm.FAddOp(load0_op.result, load1_op.result),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    add_op.res,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecCopyF32x8(RewritePattern):
    """
    def vec_copy_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = {src_data};
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_copy_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                vector.StoreOp.get(
                    load_op.result,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecCopyF32x8Pfx(RewritePattern):
    """
    def vec_copy_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                       src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = {src_data};
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_copy_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [8]), [0, 1, 2, 3, 4, 5, 6, 7]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    load_op.result,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecCopyF64x4(RewritePattern):
    """
    def vec_copy_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = {src_data};
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_copy_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                vector.StoreOp.get(
                    load_op.result,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecCopyF64x4Pfx(RewritePattern):
    """
    def vec_copy_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                       src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = {src_data};
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_copy_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i64, [4]), [0, 1, 2, 3]
                    ),
                ),
                sgext_op := arith.ExtSIOp(op.arguments[0], i64),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[sgext_op.result],
                    result_types=[VectorType(i64, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    load_op.result,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


# Note: Alignment seems wrong - should be 1 here.
class ConvertVecLoadF32x8(RewritePattern):
    """
    def vec_load_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ DRAM):
    # @instr {dst_data} = _mm256_loadu_ps(&{src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_load_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                vector.StoreOp.get(
                    load_op.result,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecLoadF32x8Pfx(RewritePattern):
    """
    def vec_load_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                       src: [f32][8] @ DRAM):
    # @instr {dst_data} = _mm256_maskload_ps(&{src_data}, mm256_prefix_mask_epi32({m}));
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_load_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [8]), [0, 1, 2, 3, 4, 5, 6, 7]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := vector.LoadOp(
                    operands=[
                        op.arguments[2],
                        [zero_op.result],
                    ],
                    result_types=[VectorType(f32, [8])],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    load_op.result,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


# Note: Same as above.
class ConvertVecLoadF64x4(RewritePattern):
    """
    def vec_load_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ DRAM):
    # @instr {dst_data} = _mm256_loadu_pd(&{src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_load_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                vector.StoreOp.get(
                    load_op.result,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecLoadF64x4Pfx(RewritePattern):
    """
    def vec_load_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                       src: [f64][4] @ DRAM):
    # @instr {dst_data} = _mm256_maskload_pd(&{src_data}, mm256_prefix_mask_epi64x({m}));
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_load_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i64, [4]), [0, 1, 2, 3]
                    ),
                ),
                sgext_op := arith.ExtSIOp(op.arguments[0], i64),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[sgext_op.result],
                    result_types=[VectorType(i64, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := vector.LoadOp(
                    operands=[
                        op.arguments[2],
                        [zero_op.result],
                    ],
                    result_types=[VectorType(f64, [4])],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    load_op.result,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecReduceAddSclF32x8(RewritePattern):
    """
    def vec_reduce_add_scl_f32x8(dst: f32 @ DRAM, src: [f32][8] @ VEC_AVX2):
    # @instr *{dst_data} = mm256_reduce_add_ps({src_data});
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_reduce_add_scl_f32x8":
            return

        assert len(op.arguments) == 2

        # dst: f32 @ DRAM
        assert op.arguments[0].type == f32, op.arguments[0].type
        assert isinstance(dst_load_op := op.arguments[0].owner, memref.LoadOp), (
            op.arguments[0].owner
        )

        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                reduce_op := vector.ReductionOp(
                    load_op.result,
                    vector.CombiningKindFlag.ADD,
                    f32,
                    acc=op.arguments[0],
                ),
                memref.StoreOp.get(
                    reduce_op.result, dst_load_op.memref, [zero_op.result]
                ),
            )
        )


class ConvertVecReduceAddSclF64x4(RewritePattern):
    """
    def vec_reduce_add_scl_f64x4(dst: f64 @ DRAM, src: [f64][4] @ VEC_AVX2):
    # @instr *{dst_data} = mm256_reduce_add_pd({src_data});
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_reduce_add_scl_f64x4":
            return

        assert len(op.arguments) == 2

        # dst: f64 @ DRAM
        assert op.arguments[0].type == f64, op.arguments[0].type
        assert isinstance(dst_load_op := op.arguments[0].owner, memref.LoadOp), (
            op.arguments[0].owner
        )

        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                reduce_op := vector.ReductionOp(
                    load_op.result,
                    vector.CombiningKindFlag.ADD,
                    f64,
                    acc=op.arguments[0],
                ),
                memref.StoreOp.get(
                    reduce_op.result, dst_load_op.memref, [zero_op.result]
                ),
            )
        )


class ConvertVecZeroF32x8(RewritePattern):
    """
    def vec_zero_f32x8(dst: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_setzero_ps();
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        dst[i] = 0.0
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_zero_f32x8":
            return

        assert len(op.arguments) == 1
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                const_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_float(
                        VectorType(f32, [8]), [0.0] * 8
                    )
                ),
                vector.StoreOp.get(
                    const_op.result,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecZeroF64x4(RewritePattern):
    """
    def vec_zero_f64x4(dst: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_setzero_pd();
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = 0.0
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_zero_f64x4":
            return

        assert len(op.arguments) == 1
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                const_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_float(
                        VectorType(f64, [4]), [0.0] * 4
                    )
                ),
                vector.StoreOp.get(
                    const_op.result,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecAddF32x8(RewritePattern):
    """
    def vec_add_f32x8(dst: [f32][8] @ VEC_AVX2, src1: [f32][8] @ VEC_AVX2,
                  src2: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_ps({src1_data}, {src2_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 8):
        dst[i] = src1[i] + src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_f32x8":
            return

        assert len(op.arguments) == 3
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                add_op := llvm.FAddOp(load0_op.result, load1_op.result),
                vector.StoreOp.get(
                    add_op.res,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecAddF32x8Pfx(RewritePattern):
    """
    def vec_add_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                      src1: [f32][8] @ VEC_AVX2, src2: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_ps({src1_data}, {src2_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src1[i] + src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, MemRefType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [8]), [0, 1, 2, 3, 4, 5, 6, 7]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[3], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                add_op := llvm.FAddOp(load0_op.result, load1_op.result),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    add_op.res,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecAddF64x4(RewritePattern):
    """
    def vec_add_f64x4(dst: [f64][4] @ VEC_AVX2, src1: [f64][4] @ VEC_AVX2,
                  src2: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_pd({src1_data}, {src2_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 4):
        dst[i] = src1[i] + src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_f64x4":
            return

        assert len(op.arguments) == 3
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                add_op := llvm.FAddOp(load0_op.result, load1_op.result),
                vector.StoreOp.get(
                    add_op.res,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecAddF64x4Pfx(RewritePattern):
    """
    def vec_add_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                      src1: [f64][4] @ VEC_AVX2, src2: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_pd({src1_data}, {src2_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src1[i] + src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, MemRefType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [4]), [0, 1, 2, 3]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[3], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                add_op := llvm.FAddOp(load0_op.result, load1_op.result),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    add_op.res,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecBrdcstSclF32x8(RewritePattern):
    """
    def vec_brdcst_scl_f32x8(dst: [f32][8] @ VEC_AVX2, src: f32 @ DRAM):
    # @instr {dst_data} = _mm256_set1_ps(*{src_data});
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        dst[i] = src
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_brdcst_scl_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src: f32 @ DRAM
        assert op.arguments[1].type == f32, op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                broadcast_op := vector.BroadcastOp(
                    operands=[op.arguments[1]],
                    result_types=[VectorType(f32, [8])],
                ),
                vector.StoreOp.get(
                    broadcast_op.vector,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecBrdcstSclF32x8Pfx(RewritePattern):
    """
    def vec_brdcst_scl_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                             src: f32 @ DRAM):
    # @instr {dst_data} = _mm256_set1_ps(*{src_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_brdcst_scl_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: f32 @ DRAM
        assert op.arguments[2].type == f32, op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [8]), [0, 1, 2, 3, 4, 5, 6, 7]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                broadcast_op := vector.BroadcastOp(
                    operands=[op.arguments[2]],
                    result_types=[VectorType(f32, [8])],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    broadcast_op.vector,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecBrdcstSclF64x4(RewritePattern):
    """
    def vec_brdcst_scl_f64x4(dst: [f64][4] @ VEC_AVX2, src: f64 @ DRAM):
    # @instr {dst_data} = _mm256_set1_pd(*{src_data});
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_brdcst_scl_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src: f64 @ DRAM
        assert op.arguments[1].type == f64, op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                broadcast_op := vector.BroadcastOp(
                    operands=[op.arguments[1]],
                    result_types=[VectorType(f64, [4])],
                ),
                vector.StoreOp.get(
                    broadcast_op.vector,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecBrdcstSclF64x4Pfx(RewritePattern):
    """
    def vec_brdcst_scl_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                             src: f64 @ DRAM):
    # @instr {dst_data} = _mm256_set1_pd(*{src_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_brdcst_scl_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: f64 @ DRAM
        assert op.arguments[2].type == f64, op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [4]), [0, 1, 2, 3]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                broadcast_op := vector.BroadcastOp(
                    operands=[op.arguments[2]],
                    result_types=[VectorType(f64, [4])],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    broadcast_op.vector,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecFmadd2F32x8(RewritePattern):
    """
    def vec_fmadd2_f32x8(dst: [f32][8] @ VEC_AVX2, src1: [f32][8] @ VEC_AVX2,
                     src2: [f32][8] @ VEC_AVX2, src3: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 8):
        dst[i] = src3[i] + src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd2_f32x8":
            return

        assert len(op.arguments) == 4
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type
        # src3: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, MemRefType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                load2_op := vector.LoadOp(
                    operands=[op.arguments[3], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                fma_op := llvm_intrinsics.FMAOp(
                    load0_op.result,
                    load1_op.result,
                    load2_op.result,
                ),
                vector.StoreOp.get(
                    fma_op.res,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecFmadd2F32x8Pfx(RewritePattern):
    """
    def vec_fmadd2_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                         src1: [f32][8] @ VEC_AVX2, src2: [f32][8] @ VEC_AVX2,
                         src3: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src3[i] + src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd2_f32x8_pfx":
            return

        assert len(op.arguments) == 4
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, MemRefType), op.arguments[3].type
        # src3: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[4].type, MemRefType), op.arguments[4].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [8]), [0, 1, 2, 3, 4, 5, 6, 7]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[3], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                load2_op := vector.LoadOp(
                    operands=[op.arguments[4], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                fma_op := llvm_intrinsics.FMAOp(
                    load0_op.result,
                    load1_op.result,
                    load2_op.result,
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    fma_op.res,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecFmadd2F64x4(RewritePattern):
    """
    def vec_fmadd2_f64x4(dst: [f64][4] @ VEC_AVX2, src1: [f64][4] @ VEC_AVX2,
                     src2: [f64][4] @ VEC_AVX2, src3: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 4):
        dst[i] = src3[i] + src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd2_f64x4":
            return

        assert len(op.arguments) == 4
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type
        # src3: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, MemRefType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                load2_op := vector.LoadOp(
                    operands=[op.arguments[3], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                fma_op := llvm_intrinsics.FMAOp(
                    load0_op.result,
                    load1_op.result,
                    load2_op.result,
                ),
                vector.StoreOp.get(
                    fma_op.res,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecFmadd2F64x4Pfx(RewritePattern):
    """
    def vec_fmadd2_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                         src1: [f64][4] @ VEC_AVX2, src2: [f64][4] @ VEC_AVX2,
                         src3: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src3[i] + src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd2_f64x4_pfx":
            return

        assert len(op.arguments) == 5
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, MemRefType), op.arguments[3].type
        # src3: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[4].type, MemRefType), op.arguments[4].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [4]), [0, 1, 2, 3]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                load1_op := vector.LoadOp(
                    operands=[op.arguments[3], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                load2_op := vector.LoadOp(
                    operands=[op.arguments[4], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                fma_op := llvm_intrinsics.FMAOp(
                    load0_op.result,
                    load1_op.result,
                    load2_op.result,
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    fma_op.res,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecStoreF32x8(RewritePattern):
    """
    def vec_store_f32x8(dst: [f32][8] @ DRAM, src: [f32][8] @ VEC_AVX2):
    # @instr _mm256_storeu_ps(&{dst_data}, {src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_store_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ DRAM
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                vector.StoreOp.get(
                    load_op.res,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecStoreF32x8Pfx(RewritePattern):
    """
    def vec_store_f32x8_pfx(m: size, dst: [f32][8] @ DRAM,
                        src: [f32][8] @ VEC_AVX2):
    # @instr _mm256_maskstore_ps(&{dst_data}, mm256_prefix_mask_epi32({m}), {src_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_store_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f32][8] @ DRAM
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [8]), [0, 1, 2, 3, 4, 5, 6, 7]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f32, [8])],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    load_op.res,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class ConvertVecStoreF64x4(RewritePattern):
    """
    def vec_store_f64x4(dst: [f64][4] @ DRAM, src: [f64][4] @ VEC_AVX2):
    # @instr _mm256_storeu_pd(&{dst_data}, {src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_store_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ DRAM
        assert isinstance(op.arguments[0].type, MemRefType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                vector.StoreOp.get(
                    load_op.res,
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertVecStoreF64x4Pfx(RewritePattern):
    """
    def vec_store_f64x4_pfx(m: size, dst: [f64][4] @ DRAM,
                        src: [f64][4] @ VEC_AVX2):
    # @instr _mm256_maskstore_pd(&{dst_data}, mm256_prefix_mask_epi64x({m}), {src_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_store_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i32, op.arguments[0].type
        # dst: [f64][4] @ DRAM
        assert isinstance(op.arguments[1].type, MemRefType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, MemRefType), op.arguments[2].type
        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(
                        VectorType(i32, [4]), [0, 1, 2, 3]
                    ),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := vector.LoadOp(
                    operands=[op.arguments[2], [zero_op.result]],
                    result_types=[VectorType(f64, [4])],
                ),
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    llvm.LLVMPointerType.opaque(),
                ),
                llvm_intrinsics.MaskedStoreOp(
                    load_op.res,
                    ptr_cast_op.results[0],
                    mask_op.res,
                ),
            )
        )


class InlineBLASPass(ModulePass):
    name = "inline-avx2"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertSelect(),
                    ConvertVecAbsF32x8(),
                    ConvertVecAbsF32x8Pfx(),
                    ConvertVecAbsF64x4(),
                    ConvertVecAbsF64x4Pfx(),
                    ConvertVecAddRedF32x8(),
                    ConvertVecAddRedF32x8Pfx(),
                    ConvertVecAddRedF64x4(),
                    ConvertVecAddRedF64x4Pfx(),
                    ConvertVecCopyF32x8(),
                    ConvertVecCopyF32x8Pfx(),
                    ConvertVecCopyF64x4(),
                    ConvertVecCopyF64x4Pfx(),
                    ConvertVecLoadF32x8(),
                    ConvertVecLoadF32x8Pfx(),
                    ConvertVecLoadF64x4(),
                    ConvertVecLoadF64x4Pfx(),
                    ConvertVecReduceAddSclF32x8(),
                    ConvertVecReduceAddSclF64x4(),
                    ConvertVecZeroF32x8(),
                    ConvertVecZeroF64x4(),
                    ConvertVecAddF32x8(),
                    ConvertVecAddF32x8Pfx(),
                    ConvertVecAddF64x4(),
                    ConvertVecAddF64x4Pfx(),
                    ConvertVecBrdcstSclF32x8(),
                    ConvertVecBrdcstSclF32x8Pfx(),
                    ConvertVecBrdcstSclF64x4(),
                    ConvertVecBrdcstSclF64x4Pfx(),
                    ConvertVecFmadd2F32x8(),
                    ConvertVecFmadd2F32x8Pfx(),
                    ConvertVecFmadd2F64x4(),
                    ConvertVecFmadd2F64x4Pfx(),
                    ConvertVecStoreF32x8(),
                    ConvertVecStoreF32x8Pfx(),
                    ConvertVecStoreF64x4(),
                    ConvertVecStoreF64x4Pfx(),
                ]
            )
        ).rewrite_module(m)
