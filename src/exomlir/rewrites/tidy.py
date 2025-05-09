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


class TidyPass(ModulePass):
    name = "tidy"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveMemorySpacePattern(),
                ]
            )
        ).rewrite_module(m)
