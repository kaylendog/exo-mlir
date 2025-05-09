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


class ConvertVectorPtrToLLVMPass(ModulePass):
    name = "convert-vector-ptr-to-llvm"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([])).rewrite_module(m)
