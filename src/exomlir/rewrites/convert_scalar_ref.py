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

from exomlir.dialects import exo, index


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
            rewriter.replace_matched_op(index.CastsOp(op.input, op.result.type))

        if op.input.type == op.result.type and isinstance(op.input, BlockArgument):
            # replace x -> x with x
            rewriter.replace_matched_op((), (op.input,))


class ReconcileIndexCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: index.CastsOp, rewriter: PatternRewriter):
        # replace x -> y -> x cast with x
        if not isinstance(op.input.owner, index.CastsOp):
            return

        if op.input.owner.input.type != op.result.type:
            return

        rewriter.replace_matched_op((), (op.input.owner.input,))


class ConvertScalarRefPass(ModulePass):
    name = "convert-scalar-ref"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ConvertRedundantReadOp(), ReconcileIndexCasts()]
            ),
            walk_reverse=True,
        ).rewrite_module(m)
