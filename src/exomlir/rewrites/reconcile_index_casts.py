from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from exomlir.dialects import index


class ReconcileIndexCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: index.CastsOp, rewriter: PatternRewriter):
        # replace x -> y -> x cast with x
        if not isinstance(op.input.owner, index.CastsOp):
            return

        if op.input.owner.input.type != op.result.type:
            return

        rewriter.replace_matched_op((), (op.input.owner.input,))


class ReconcileIndexCastsPass(ModulePass):
    name = "reconcile-index-casts"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ReconcileIndexCasts()]),
            walk_reverse=True,
        ).rewrite_module(m)
