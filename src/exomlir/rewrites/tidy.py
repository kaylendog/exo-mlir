from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, MemRefType
from xdsl.dialects import memref
from xdsl.passes import ModulePass
from xdsl.ir import Region, Use

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    TypeConversionPattern,
    RewritePattern,
    op_type_rewrite_pattern,
    attr_type_rewrite_pattern,
)

from exomlir.dialects import index


class ConvertMemRefCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CastOp, rewriter: PatternRewriter):
        if not op.source.type == op.dest.type:
            return

        op.results[0].replace_by(op.source)


class RemoveMemorySpacePattern(TypeConversionPattern):
    """
    Replaces `ptr_dxdsl.ptr` with `llvm.ptr`.
    """

    @attr_type_rewrite_pattern
    def convert_type(self, typ: MemRefType) -> MemRefType:
        return MemRefType(
            element_type=typ.element_type,
            shape=typ.shape,
            layout=typ.layout,
        )


class ConvertCastsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: index.CastsOp, rewriter: PatternRewriter):
        if op.input.type == op.result.type:
            # replace x -> x cast with x
            rewriter.replace_matched_op((), (op.input,))


class TidyPass(ModulePass):
    name = "tidy"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ConvertMemRefCastOp(), RemoveMemorySpacePattern(), ConvertCastsOp()]
            )
        ).rewrite_module(m)
