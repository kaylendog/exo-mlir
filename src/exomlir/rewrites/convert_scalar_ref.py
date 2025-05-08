from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, IndexType, MemRefType
from xdsl.dialects import arith
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.ir import BlockArgument

from exomlir.dialects import exo


class ConvertRedundantReadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        # convert scalar reads only
        if isinstance(op.input.type, MemRefType):
            return

        # replace index -> x with index cast
        if isinstance(op.input.type, IndexType) and not isinstance(
            op.result.type, IndexType
        ):
            rewriter.replace_matched_op(arith.IndexCastOp(op.input, op.result.type))

        if op.input.type == op.result.type and isinstance(op.input, BlockArgument):
            # replace x -> x with x
            op.result.replace_by(op.input)
            rewriter.erase_matched_op()


class ReconcileIndexCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.IndexCastOp, rewriter: PatternRewriter):
        # replace x -> x cast with x
        if op.input.type == op.result.type:
            op.result.replace_by(op.input)

        # replace x -> y -> x cast with x
        for use in tuple(op.result.uses):
            if not isinstance(use.operation, arith.IndexCastOp):
                continue

            if len(use.operation.result.uses) != 1:
                continue

            if use.operation.result.type != op.input.type:
                continue

            op.result.replace_by(op.input)


class ConvertScalarRefPass(ModulePass):
    name = "convert-scalar-ref"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ConvertRedundantReadOp(), ReconcileIndexCasts()]
            )
        ).rewrite_module(m)
