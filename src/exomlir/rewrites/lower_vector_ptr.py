from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)


class ConvertVectorPtrToLLVMPass(ModulePass):
    name = "convert-vector-ptr-to-llvm"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([])).rewrite_module(m)
