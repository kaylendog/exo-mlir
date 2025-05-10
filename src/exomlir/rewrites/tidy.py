from xdsl.context import Context
from xdsl.dialects.builtin import MemRefType, ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
)


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
                    RemoveMemorySpacePattern(recursive=True),
                ]
            )
        ).rewrite_module(m)
