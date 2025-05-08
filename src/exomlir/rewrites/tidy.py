from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, MemRefType
from xdsl.dialects import memref
from xdsl.passes import ModulePass
from xdsl.ir import Region, Use

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ConvertMemRefCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CastOp, rewriter: PatternRewriter):
        if not op.source.type == op.dest.type:
            return

        op.results[0].replace_by(op.source)


class TidyPass(ModulePass):
    name = "tidy"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertMemRefCastOp()])
        ).rewrite_module(m)
