from typing import cast
from xdsl.context import Context
from xdsl.dialects.builtin import (
    ModuleOp,
    MemRefType,
    IntegerAttr,
    IndexType,
    Float32Type,
    VectorType,
    UnrealizedConversionCastOp,
)
from xdsl.dialects import memref, vector, arith
from xdsl.passes import ModulePass

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    RewritePattern,
    op_type_rewrite_pattern,
)

from exomlir.dialects import exo


class ConvertAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AllocOp, rewriter):
        pass


class ConvertFreeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.FreeOp, rewriter):
        pass


class ConvertMM256StoreuPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_storeu_ps":
            return

        # preconditions
        assert len(op.arguments) == 2
        assert isinstance(op.arguments[0].type, MemRefType)

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                cast_op := UnrealizedConversionCastOp.get(
                    [op.arguments[1]],
                    [VectorType(Float32Type(), [8])],
                ),
                vector.StoreOp.get(
                    op.arguments[0], cast_op.results[0], [zero_op.result]
                ),
            )
        )


class ConvertMM256FmaddPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_fmadd_ps":
            return

        assert len(op.arguments) == 3
        assert isinstance(op.arguments[0].type, MemRefType)

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                cast_op := UnrealizedConversionCastOp.get(
                    op.arguments[1:],
                    [VectorType(Float32Type(), [8]), VectorType(Float32Type(), [8])],
                ),
                load_op := vector.LoadOp(
                    operands=[op.arguments[0], [zero_op.result]],
                    result_types=[VectorType(Float32Type(), [8])],
                ),
                fma_op := vector.FMAOp.get(
                    cast_op.results[0],
                    cast_op.results[1],
                    load_op,
                ),
                vector.StoreOp.get(fma_op.res, op.arguments[0], [zero_op.result]),
            )
        )


class ConvertMM256BroadcastSsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_broadcast_ss":
            return


class ConvertMM256LoaduPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_loadu_ps":
            return

        assert len(op.arguments) == 2
        assert isinstance(op.arguments[0].type, MemRefType)
        assert isinstance(op.arguments[1].type, MemRefType)

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := vector.LoadOp(
                    operands=[op.arguments[1], [zero_op.result]],
                    result_types=[VectorType(Float32Type(), [8])],
                ),
                vector.StoreOp.get(
                    load_op.result,
                    op.arguments[1],
                    [zero_op.result],
                ),
            )
        )


class InlineAVX2Pass(ModulePass):
    name = "inline-avx2"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertFreeOp(),
                    ConvertAllocOp(),
                    ConvertMM256StoreuPsOp(),
                    ConvertMM256FmaddPsOp(),
                    ConvertMM256BroadcastSsOp(),
                    ConvertMM256LoaduPsOp(),
                ]
            )
        ).rewrite_module(m)
